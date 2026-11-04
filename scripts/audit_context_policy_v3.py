"""Independent, read-only causal prompt reconstruction for the v3 experiment.

Never invokes a model, re-scores by tuning rules, or updates original results.
Writes only its independent audit receipt. Use after the writer exits.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.config import settings
from app.core.context_policy import EvidenceState, assemble
from app.core.prompts import COREF_REWRITE_PROMPT, SUMMARY_PROMPT
from app.core.summarizer import compute_boundary
from app.graph.nodes import _history_text, _agent_messages
from scripts.context_benchmark_strategies import raw_messages
from scripts.eval_context_policy_v3 import (ARMS, ROOT, Summary, ServicePlan, READER_SYSTEM,
                                          REPLY_SYSTEM, LIMITS, EXTRA_BODY, digest, score, tool_result, write_json)


def audit(out):
    read=lambda path:json.loads(path.read_text(encoding='utf-8'))
    manifest=read(out/'manifest.json')
    data=read(out/'dataset.json')
    issues=[];expected_keys=set();checked=0;maintenance_count=0
    unknown_usage=0;model_names={};reasoning_tokens=0
    if digest(data)!=manifest['dataset_sha256']:issues.append('dataset changed')
    for name,expected in manifest['sources'].items():
        for path in (ROOT/name,out/'source_snapshot'/name):
            if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:issues.append('source changed '+str(path))
    original={k:getattr(settings,k) for k in manifest['config']}
    for k,v in manifest['config'].items():setattr(settings,k,v)

    def check_call(key,stage,messages,schema=None,model=None):
        nonlocal unknown_usage,reasoning_tokens
        expected_keys.add(key)
        path=out/'calls'/(key+'.json')
        if not path.exists():issues.append('missing call '+key);return None
        r=read(path)
        if schema:
            instruction='请调用 '+schema.__name__+' 工具返回结构化结果，不要用正文代替工具参数。'
            messages=[SystemMessage(messages[0].content+'\n'+instruction),*messages[1:]]
        expected={'messages':[m.model_dump(mode='json') for m in messages],'stage':stage,
                  'model':model or manifest['config']['chat_model'],'max_tokens':LIMITS[stage],
                  'schema':schema.model_json_schema() if schema else None,'tool_choice':'auto' if schema else None,
                  'extra_body': EXTRA_BODY}
        if r['contract']!=expected or r['input_sha256']!=digest(expected):issues.append('causal prompt mismatch '+key)
        for i,a in enumerate(r['attempts'],1):
            ap=out/'attempts'/(key+f'-{i}.json')
            if not ap.exists() or read(ap)!=a:issues.append('attempt receipt mismatch '+key)
            if a['input_sha256']!=digest(expected):issues.append('attempt input mismatch '+key)
            if i>1 and not r['attempts'][i-2].get('retryable'):issues.append('forbidden semantic retry '+key)
            usage=a.get('usage',{})
            if usage.get('input_tokens') is None or usage.get('output_tokens') is None:unknown_usage+=1
            reasoning_tokens+=(usage.get('output_token_details') or {}).get('reasoning',0) or 0
            if a.get('model'):model_names[a['model']]=model_names.get(a['model'],0)+1
        if len(r['attempts'])>2:issues.append('too many attempts '+key)
        if r['value']!=r['attempts'][-1]['value']:issues.append('final attempt value mismatch '+key)
        return r

    try:
        for case in data['cases']:
            summary='';boundary=0;ingested=0;state=EvidenceState();events=[]
            for probe in case['probes']:
                rows=case['history'][:2*probe['at_turn']]
                state.advance(rows)
                path=out/'states'/f'{case["id"]}-{probe["id"]}.json'
                if not path.exists() or read(path)!=state.snapshot():issues.append('state mismatch '+str(path))
                for end in range(ingested+2,len(rows)+1,2):
                    if end-boundary<settings.summary_trigger_messages:continue
                    objects=[SimpleNamespace(id=i,role=r['role']) for i,r in enumerate(rows[:end],1)]
                    new_boundary=compute_boundary(objects,settings.context_window_turns)
                    if not new_boundary or new_boundary<=boundary:continue
                    dialog='\n'.join(f"{'用户' if r['role']=='user' else '客服'}:{r['content']}" for r in rows[boundary:new_boundary])
                    key=f'{case["id"]}-summary-{end}'
                    r=check_call(key,'summary',SUMMARY_PROMPT.format_messages(old_summary=summary or '(无)',dialog=dialog),Summary,settings.summary_model or None)
                    if r is None:continue
                    maintenance_count+=1
                    events.append({'at_message':end,'old_boundary':boundary,'new_boundary':new_boundary,'call_key':key,'status':r['status']})
                    if r['ok'] and r['value']['summary'].strip():summary,boundary=r['value']['summary'],new_boundary
                ingested=len(rows)
                for arm in ARMS:
                    key=f'{case["id"]}-{probe["id"]}-{arm}'
                    path=out/'results'/(key+'.json')
                    if not path.exists():issues.append('missing result '+key);continue
                    r=read(path)
                    if arm=='C_current':
                        st={'messages':raw_messages(rows)+[HumanMessage(probe['query'],id=f'db-{len(rows)+1}')],
                            'summary':summary,'summary_upto_msg_id':boundary}
                        context={'body':_agent_messages(st)[1:],'coref_history':_history_text(st)}
                    else:context=assemble(rows,probe['query'],state if arm!='W_raw8k' else None,arm=='R_recall8k')
                    if digest([m.model_dump(mode='json') for m in context['body']])!=r['context_sha256']:issues.append('assembled context mismatch '+key)
                    cr=check_call(key+'-coref','coref',COREF_REWRITE_PROMPT.format_messages(query=probe['query'],history=context['coref_history'] or '(无)'))
                    if cr is None:continue
                    effective=cr['value'] if cr['ok'] else probe['query']
                    instruction=HumanMessage('本轮原话：'+probe['query']+'\n指代改写结果（需用历史核实）：'+effective)
                    plan=check_call(key+'-plan','plan',[SystemMessage(READER_SYSTEM),*context['body'],instruction],ServicePlan)
                    if plan is None:continue
                    tool=tool_result(case,plan['value']) if plan['ok'] else None
                    if tool!=r['tool_result']:issues.append('fixture/tool mismatch '+key)
                    prompt=[SystemMessage(REPLY_SYSTEM),*context['body'],instruction]
                    if tool is not None:
                        p=plan['value'];name='query_logistics' if p['action']=='logistics' else 'query_faq'
                        args={'order_id':p['order_id']} if p['action']=='logistics' else {'query':p['sku']+' 清洁方法'}
                        prompt += [AIMessage('',tool_calls=[{'id':'eval-readonly','name':name,'args':args,'type':'tool_call'}]),
                                   ToolMessage(json.dumps(tool,ensure_ascii=False),tool_call_id='eval-readonly')]
                    else:prompt.append(HumanMessage('本轮计划：'+json.dumps(plan['value'],ensure_ascii=False)+'\n请核实历史后作答。'))
                    reply=check_call(key+'-reply','reply',prompt)
                    if [cr,plan,reply]!=[r['coref'],r['plan'],r['reply']]:issues.append('embedded calls differ '+key)
                    if score(case,probe,r)!=r['score']:issues.append('score mismatch '+key)
                    checked+=1
            mp=out/'maintenance'/(case['id']+'.json')
            if not mp.exists() or read(mp)!=events:issues.append('maintenance replay mismatch '+case['id'])
    finally:
        for k,v in original.items():setattr(settings,k,v)
    actual_keys={p.stem for p in (out/'calls').glob('*.json')}
    if expected_keys!=actual_keys:issues.append('unexpected/missing cached calls '+str(sorted(actual_keys^expected_keys)))
    hashes={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('calls','attempts','results','states','maintenance') for p in sorted((out/folder).glob('*.json'))}
    write_json(out/'artifact_hashes.json',hashes)
    result={'passed':not issues and checked==len(data['cases'])*2*len(ARMS),'issues':issues,
            'probes_checked':checked,'maintenance_checked':maintenance_count,'calls_checked':len(expected_keys),
            'missing_usage_attempts':unknown_usage,'returned_model_names':model_names,
            'reported_reasoning_tokens':reasoning_tokens,'artifact_index_sha256':digest(hashes),
            'auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'Reconstructed all causal prompts and memory boundaries; no model calls; does not certify semantic grading quality.'}
    write_json(out/'independent_audit.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    args=p.parse_args();r=audit(Path(args.output));print(json.dumps(r,ensure_ascii=False,indent=2))
    raise SystemExit(0 if r['passed'] else 1)
