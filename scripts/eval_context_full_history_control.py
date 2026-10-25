"""Post-hoc full-history control, same frozen model/protocol/scoring as v3.

Separate results preserve the original four-arm study. This control runs later,
not interleaved, so provider/time effects are not controlled between studies.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import random
import statistics
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from app.core.context_policy import source_text
from app.core.prompts import COREF_REWRITE_PROMPT
from scripts.context_benchmark_strategies import raw_messages
from scripts.eval_context_policy_v3 import (Caller, ServicePlan, READER_SYSTEM, REPLY_SYSTEM,
    LIMITS, EXTRA_BODY, digest, score, tool_result, write_json, writer_lock, percentile,
    config_snapshot, get_store, ROOT, stamp)


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def prepare(parent,out):
    if out.exists():raise ValueError('New output required')
    reference=read(parent/'audited_report.json')
    if not reference['audit_passed'] or not read(parent/'independent_audit.json')['passed']:
        raise ValueError('Parent must be fully audited')
    manifest=read(parent/'manifest.json')
    if config_snapshot()!=manifest['config']:raise ValueError('Model/config changed')
    if get_store().policy().model_dump()!=manifest['pool_policy']:raise ValueError('Pool changed')
    out.mkdir(parents=True)
    write_json(out/'dataset.json',read(parent/'dataset.json'))
    sources={**manifest['sources']}
    for name in ['scripts/eval_context_full_history_control.py']:
        sources[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    for name,h in sources.items():
        b=(ROOT/name).read_bytes()
        if hashlib.sha256(b).hexdigest()!=h:raise ValueError('source changed '+name)
        dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
    write_json(out/'manifest.json',{'created_at':stamp(),'parent':str(parent.resolve()),
        'parent_manifest_hash':digest(manifest),'config':manifest['config'],
        'pool_policy':manifest['pool_policy'],'dataset_sha256':digest(read(out/'dataset.json')),
        'sources':sources,'limits':LIMITS,'extra_body':EXTRA_BODY,'concurrency':8,
        'scope':'Post-hoc complete visible history control, 48 known synthetic cases and 2 probes. Same model, protocol, output limits and scores; later time window than the four-arm study.'})


def prompts(case,probe,coref=None,plan=None):
    rows=case['history'][:2*probe['at_turn']]
    body=[*raw_messages(rows),HumanMessage(probe['query'])]
    cr=COREF_REWRITE_PROMPT.format_messages(query=probe['query'],history=source_text(rows,range(1,len(rows)+1)))
    if coref is None:return body,cr
    instruction=HumanMessage('本轮原话：'+probe['query']+'\n指代改写结果（需用历史核实）：'+(coref['value'] if coref['ok'] else probe['query']))
    pp=[SystemMessage(READER_SYSTEM),*body,instruction]
    if plan is None:return body,cr,pp
    tool=tool_result(case,plan['value']) if plan['ok'] else None
    rp=[SystemMessage(REPLY_SYSTEM),*body,instruction]
    if tool is not None:
        p=plan['value'];name='query_logistics' if p['action']=='logistics' else 'query_faq'
        args={'order_id':p['order_id']} if p['action']=='logistics' else {'query':p['sku']+' 清洁方法'}
        rp += [AIMessage('',tool_calls=[{'id':'eval-readonly','name':name,'args':args,'type':'tool_call'}]),ToolMessage(json.dumps(tool,ensure_ascii=False),tool_call_id='eval-readonly')]
    else:rp.append(HumanMessage('本轮计划：'+json.dumps(plan['value'],ensure_ascii=False)+'\n请核实历史后作答。'))
    return body,cr,pp,rp,tool


async def probe_call(out,caller,case,probe):
    key=f'{case["id"]}-{probe["id"]}-F_full'
    body,cr=prompts(case,probe)
    coref=await caller.call(key+'-coref','coref',cr)
    _,_,pp=prompts(case,probe,coref)
    plan=await caller.call(key+'-plan','plan',pp,ServicePlan)
    body,_,_,rp,tool=prompts(case,probe,coref,plan)
    reply=await caller.call(key+'-reply','reply',rp)
    result={'probe_id':probe['id'],'at_turn':probe['at_turn'],'coref':coref,'plan':plan,'reply':reply,
            'tool_result':tool,'body':[m.model_dump(mode='json') for m in body],
            'context_text':'\n'.join(m.content for m in body)}
    result['context_sha256']=digest(result['body']);result['score']=score(case,probe,result)
    write_json(out/'results'/(key+'.json'),result)
    print(key,'target',result['score']['target_pass'],'task',result['score']['task_rule_v3'],flush=True)


def audit(out):
    manifest=read(out/'manifest.json');data=read(out/'dataset.json');issues=[];entries=[];expected_keys=set()
    if digest(data)!=manifest['dataset_sha256']:issues.append('dataset hash')
    for name,h in manifest['sources'].items():
        if any(hashlib.sha256(p.read_bytes()).hexdigest()!=h for p in (ROOT/name,out/'source_snapshot'/name)):issues.append('source hash '+name)
    for case in data['cases']:
        for probe in case['probes']:
            key=f'{case["id"]}-{probe["id"]}-F_full';path=out/'results'/(key+'.json')
            if not path.exists():issues.append('missing '+key);continue
            r=read(path);body,cr,pp,rp,tool=prompts(case,probe,r['coref'],r['plan'])
            if r['body']!=[m.model_dump(mode='json') for m in body] or digest(r['body'])!=r['context_sha256']:issues.append('visible history '+key)
            if r['tool_result']!=tool or score(case,probe,r)!=r['score']:issues.append('tool/score '+key)
            for stage,ms,schema in [('coref',cr,None),('plan',pp,ServicePlan),('reply',rp,None)]:
                if schema:ms=[SystemMessage(ms[0].content+'\n请调用 '+schema.__name__+' 工具返回结构化结果，不要用正文代替工具参数。'),*ms[1:]]
                contract={'messages':[m.model_dump(mode='json') for m in ms],'stage':stage,
                          'model':manifest['config']['chat_model'],'max_tokens':LIMITS[stage],
                          'schema':schema.model_json_schema() if schema else None,
                          'tool_choice':'auto' if schema else None,'extra_body':EXTRA_BODY}
                name=key+'-'+stage;expected_keys.add(name);rec=read(out/'calls'/(name+'.json'))
                if rec!=r[stage] or rec['contract']!=contract or digest(contract)!=rec['input_sha256']:issues.append('contract '+name)
                for i,a in enumerate(rec['attempts'],1):
                    if a!=read(out/'attempts'/(name+f'-{i}.json')) or a['input_sha256']!=digest(contract):issues.append('attempt '+name)
                    if i>1 and not rec['attempts'][i-2].get('retryable'):issues.append('invalid retry '+name)
            entries.append(r)
    if expected_keys!={p.stem for p in (out/'calls').glob('*.json')}:issues.append('call set mismatch')
    attempts=[a for r in entries for s in ('coref','plan','reply') for a in r[s]['attempts']]
    seconds=[sum(r[s]['seconds'] for s in ('coref','plan','reply')) for r in entries]
    stats={'n':len(entries),'passed':sum(r['score']['task_rule_v3'] for r in entries),
        'legacy_passed':sum(r['score']['task_pass'] for r in entries),'target_passed':sum(r['score']['target_pass'] for r in entries),
        'usage':{k:sum(a.get('usage',{}).get(k,0) or 0 for a in attempts) for k in ('input_tokens','output_tokens','total_tokens')},
        'stages':{s:sum(r[s]['status']!='ok' for r in entries) for s in ('coref','plan','reply')},
        'p50_seconds':percentile(seconds,.5),'p95_seconds':percentile(seconds,.95),
        'transport_failed_probes':sum(r['score']['transport_failure'] for r in entries),
        'truncated_probes':sum(r['score']['truncated'] for r in entries),
        'missing_usage_attempts':sum(not a.get('usage') for a in attempts),
        'max_reported_input':max((a.get('usage',{}).get('input_tokens',0) or 0 for a in attempts),default=0),
        'by_kind':{k:{'passed':sum(read(out/'results'/f'{c["id"]}-{p["id"]}-F_full.json')['score']['task_rule_v3'] for c in data['cases'] if c['kind']==k for p in c['probes']),
                      'n':sum(2 for c in data['cases'] if c['kind']==k)} for k in {c['kind'] for c in data['cases']}}}
    parent=Path(manifest['parent']);differences={}
    for arm in ('C_current','S_state8k'):
        pairs=[]
        for c in data['cases']:
            if all((out/'results'/f'{c["id"]}-{p["id"]}-F_full.json').exists() for p in c['probes']):
                pairs.append(sum(read(out/'results'/f'{c["id"]}-{p["id"]}-F_full.json')['score']['task_rule_v3']-read(parent/'results'/f'{c["id"]}-{p["id"]}-{arm}.json')['score']['task_rule_v3'] for p in c['probes'])/2)
        rng=random.Random(20261023);draws=[sum(rng.choice(pairs) for _ in pairs)/len(pairs) for _ in range(4000)] if pairs else []
        differences[arm]={'mean':statistics.mean(pairs) if pairs else None,'paired_conversation_95ci':[percentile(draws,.025),percentile(draws,.975)]}
    report={'audit_passed':not issues and len(entries)==len(data['cases'])*2,'issues':issues,'stats':stats,'differences':differences,'scope':manifest['scope']}
    write_json(out/'audited_report.json',report)
    return report


async def run(out):
    m=read(out/'manifest.json')
    if config_snapshot()!=m['config'] or get_store().policy().model_dump()!=m['pool_policy']:raise ValueError('config changed')
    for name,h in m['sources'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h:raise ValueError('source changed '+name)
    data=read(out/'dataset.json')
    if digest(data)!=m['dataset_sha256']:raise ValueError('dataset changed')
    caller=Caller(out);gate=asyncio.Semaphore(8)
    async def one(case):
        async with gate:
            for probe in case['probes']:await probe_call(out,caller,case,probe)
    cases=list(data['cases']);random.Random(20261023).shuffle(cases)
    try:await asyncio.gather(*(one(c) for c in cases))
    finally:
        from app.core.model_transport import close_clients
        await close_clients()
    print('FINISHED',json.dumps(audit(out),ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--parent');parser.add_argument('--prepare',action='store_true');parser.add_argument('--audit',action='store_true')
    args=parser.parse_args();out=Path(args.output).resolve()
    if args.prepare:prepare(Path(args.parent),out)
    elif args.audit:print(json.dumps(audit(out),ensure_ascii=False))
    else:
        with writer_lock(out):asyncio.run(run(out))
