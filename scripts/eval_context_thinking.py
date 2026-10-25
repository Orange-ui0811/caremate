"""Paired thinking toggle experiment on synthetic customer-service histories.

No production mutation. Both modes receive an 8192-token total output ceiling;
reasoning usage is reported separately. Never retry semantic/format failures.
"""
import argparse
import asyncio
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import statistics
import time
from types import SimpleNamespace

import httpx
from openai import APIConnectionError, APIStatusError
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from app.config import settings
from app.core import llm, summarizer
from app.core.context_policy import EvidenceState, assemble, estimate
from app.core.model_pool import call_context, get_store, providers, PoolBusy
from app.core.prompts import SUMMARY_PROMPT, COREF_REWRITE_PROMPT
from app.graph.nodes import _agent_messages, _history_text
from scripts.context_benchmark_cases import make_suite
from scripts.context_benchmark_strategies import raw_messages
from scripts.eval_context_policy_v3 import (SOURCES as BASE_SOURCES, ROOT, Summary, ServicePlan,
    READER_SYSTEM, REPLY_SYSTEM, score, tool_result, digest, stamp, write_json, writer_lock, percentile, config_snapshot)

ARMS=('C_off','C_on','S_off','S_on')
MAX_OUTPUT=8192
CONCURRENCY=4
INTERVAL=1.0
SOURCES=list(dict.fromkeys([*BASE_SOURCES,'scripts/context_benchmark_cases.py','scripts/eval_context_thinking.py']))


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def mode_body(mode):
    return {'thinking':{'type':'enabled' if mode=='on' else 'disabled'},'enable_thinking':mode=='on'}


def wire_contract(mode,stage,messages,schema=None):
    if schema:
        if not isinstance(messages[0],SystemMessage):raise ValueError('Initial system required')
        messages=[SystemMessage(messages[0].content+'\n请调用 '+schema.__name__+' 工具返回结构化结果，不要用正文代替工具参数。'),*messages[1:]]
    return messages,{'messages':[m.model_dump(mode='json') for m in messages],
        'stage':stage,'model':settings.chat_model,'mode':mode,'extra_body':mode_body(mode),
        'reasoning_effort':None,'max_tokens':MAX_OUTPUT,
        'temperature':.3 if stage in ('summary','coref') else 0,
        'schema':schema.model_json_schema() if schema else None,'tool_choice':'auto' if schema else None}


class Caller:
    def __init__(self,out):
        self.out=out;self.gate=asyncio.Semaphore(CONCURRENCY);self.pace=asyncio.Lock();self.next_start=0

    async def call(self,key,mode,stage,messages,schema=None):
        messages,contract=wire_contract(mode,stage,messages,schema);sha=digest(contract)
        path=self.out/'calls'/(key+'.json')
        if path.exists():
            cached=read(path)
            if cached['input_sha256']!=sha:raise ValueError('cached input changed '+key)
            return cached
        approx=sum(estimate(m.content) for m in messages)+(estimate(json.dumps(contract['schema'],ensure_ascii=False)) if schema else 0)
        if approx>16000:raise ValueError('input budget exceeded')
        queued=time.monotonic()
        async with self.gate:
            queue_seconds=time.monotonic()-queued;attempts=[]
            for i in range(2):
                ap=self.out/'attempts'/(key+f'-{i+1}.json')
                if ap.exists():
                    a=read(ap)
                    if a['input_sha256']!=sha:raise ValueError('attempt changed')
                    attempts.append(a)
                    if i==0 and a.get('retryable'):continue
                    break
                async with self.pace:
                    await asyncio.sleep(max(0,self.next_start-time.monotonic()))
                    self.next_start=time.monotonic()+INTERVAL
                start=time.monotonic()
                a={'input_sha256':sha,'attempt':i+1,'started_at':stamp()}
                try:
                    with call_context(request_id=f'think-{self.out.name}-{key}-{i}',priority='background'):
                        model=llm.get_chat_model(streaming=True,temperature=contract['temperature'])
                        model.max_tokens=MAX_OUTPUT;model.reasoning_effort=None
                        model.extra_body={**(model.extra_body or {}),**mode_body(mode)}
                        chain=model.bind_tools([schema],tool_choice='auto') if schema else model
                        raw=await asyncio.wait_for(chain.ainvoke(messages),100)
                    parsed=None
                    if schema:
                        matches=[t for t in raw.tool_calls if t['name']==schema.__name__]
                        if len(matches)==1:
                            try:parsed=schema.model_validate(matches[0]['args']).model_dump()
                            except ValueError:pass
                    value=parsed if schema else raw.content
                    finish=raw.response_metadata.get('finish_reason')
                    status='ok'
                    if finish=='length':status='truncated'
                    elif schema and parsed is None:status='parse_error'
                    elif not schema and (not isinstance(value,str) or not value.strip()):status='empty'
                    elif stage=='summary' and not value['summary'].strip():status='empty'
                    a.update(status=status,transport_ok=True,value=value,finish_reason=finish,
                        model=raw.response_metadata.get('model_name'),usage=raw.usage_metadata or {},
                        raw_content=raw.content,tool_calls=raw.tool_calls,
                        seconds=round(time.monotonic()-start,4))
                except Exception as exc:
                    transport=isinstance(exc,(APIConnectionError,APIStatusError,httpx.TransportError,TimeoutError,PoolBusy))
                    code=getattr(exc,'status_code',None)
                    retryable=isinstance(exc,(APIConnectionError,httpx.TransportError,TimeoutError,PoolBusy)) or isinstance(exc,APIStatusError) and (code in (408,429) or code>=500)
                    a.update(status='transport_error' if transport else 'validation_error',transport_ok=False,
                        value=None,usage={},error_type=type(exc).__name__,http_status=code,retryable=retryable,
                        seconds=round(time.monotonic()-start,4))
                write_json(ap,a);attempts.append(a)
                if i==0 and a.get('retryable'):await asyncio.sleep(4)
                else:break
            a=attempts[-1]
            r={'contract':contract,'input_sha256':sha,'attempts':attempts,'status':a['status'],
               'ok':a['status']=='ok','value':a['value'],'finish_reason':a.get('finish_reason'),
               'seconds':sum(x['seconds'] for x in attempts),'queue_seconds':queue_seconds,
               'approx_input_tokens':approx,
               'calls':[{'input_tokens':x['usage'].get('input_tokens'),'output_tokens':x['usage'].get('output_tokens')} for x in attempts if x['transport_ok']]}
            write_json(path,r);return r


def workflow(case,arm):
    """Causal schedule used by execution and offline replay; only sees prefixes."""
    mode=arm.split('_')[1];cid=case['id'];summary='';boundary=0;ingested=0;events=[];state=EvidenceState()
    for probe in case['probes']:
        rows=case['history'][:2*probe['at_turn']];state.advance(rows)
        if arm.startswith('C'):
            for end in range(ingested+2,len(rows)+1,2):
                if end-boundary<settings.summary_trigger_messages:continue
                objects=[SimpleNamespace(id=i,role=r['role']) for i,r in enumerate(rows[:end],1)]
                new=summarizer.compute_boundary(objects,settings.context_window_turns)
                if not new or new<=boundary:continue
                dialog='\n'.join(f"{'用户' if r['role']=='user' else '客服'}:{r['content']}" for r in rows[boundary:new])
                key=f'{cid}-{arm}-summary-{end}'
                r=yield ('call',key,mode,'summary',SUMMARY_PROMPT.format_messages(old_summary=summary or '(无)',dialog=dialog),Summary)
                events.append({'at_message':end,'old_boundary':boundary,'new_boundary':new,'key':key,'status':r['status']})
                if r['ok']:summary,boundary=r['value']['summary'],new
        ingested=len(rows)
        if arm.startswith('C'):
            st={'messages':raw_messages(rows)+[HumanMessage(probe['query'],id=f'db-{len(rows)+1}')],
                'summary':summary,'summary_upto_msg_id':boundary}
            context={'body':_agent_messages(st)[1:],'coref_history':_history_text(st),'summary':summary,'boundary':boundary}
        else:context=assemble(rows,probe['query'],state,False)
        key=f'{cid}-{probe["id"]}-{arm}'
        cr=yield ('call',key+'-coref',mode,'coref',COREF_REWRITE_PROMPT.format_messages(query=probe['query'],history=context['coref_history'] or '(无)'),None)
        instruction=HumanMessage('本轮原话：'+probe['query']+'\n指代改写结果（需用历史核实）：'+(cr['value'] if cr['ok'] else probe['query']))
        plan=yield ('call',key+'-plan',mode,'plan',[SystemMessage(READER_SYSTEM),*context['body'],instruction],ServicePlan)
        tool=tool_result(case,plan['value']) if plan['ok'] else None
        prompt=[SystemMessage(REPLY_SYSTEM),*context['body'],instruction]
        if tool is not None:
            p=plan['value'];name='query_logistics' if p['action']=='logistics' else 'query_faq'
            args={'order_id':p['order_id']} if p['action']=='logistics' else {'query':p['sku']+' 清洁方法'}
            prompt += [AIMessage('',tool_calls=[{'id':'eval-readonly','name':name,'args':args,'type':'tool_call'}]),ToolMessage(json.dumps(tool,ensure_ascii=False),tool_call_id='eval-readonly')]
        else:prompt.append(HumanMessage('本轮计划：'+json.dumps(plan['value'],ensure_ascii=False)+'\n请核实历史后作答。'))
        reply=yield ('call',key+'-reply',mode,'reply',prompt,None)
        result={'probe_id':probe['id'],'at_turn':probe['at_turn'],'coref':cr,'plan':plan,'reply':reply,
            'tool_result':tool,'body':[m.model_dump(mode='json') for m in context['body']],
            'context_text':'\n'.join(m.content for m in context['body']),
            'context':{k:v for k,v in context.items() if k!='body'},'state':state.snapshot()}
        result['context_sha256']=digest(result['body']);result['score']=score(case,probe,result)
        yield ('save','results',key,result)
    yield ('save','maintenance',f'{cid}-{arm}',events)


def dataset():
    full=make_suite()
    # First 24-turn and first 64-turn variant in EVERY family; no score selection.
    return {**full,'cases':[c for i,c in enumerate(full['cases']) if i%6 in (0,2)],
            'selection':'variants 0 and 2 of each of 8 families, fixed before calls; 16 known synthetic histories'}


def prepare(out):
    if out.exists():raise ValueError('new output required')
    if settings.summary_model and settings.summary_model!=settings.chat_model:raise ValueError('all stages require same model')
    if settings.chat_failover_enabled is not False:raise ValueError('disable model failover in local config first')
    if any(p.fallback or p.model!=settings.chat_model for p in providers('chat',settings.chat_model)):raise ValueError('fixed model route required')
    out.mkdir(parents=True);write_json(out/'dataset.json',dataset())
    sources={}
    for name in SOURCES:
        data=(ROOT/name).read_bytes();sources[name]=hashlib.sha256(data).hexdigest()
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    old=read(ROOT/'data/ch07/context-policy/20261023-gitee-v3-initialsystem/manifest.json')
    write_json(out/'manifest.json',{'created_at':stamp(),'sources':sources,'dataset_sha256':digest(dataset()),
        'config':config_snapshot(),'pool_policy':get_store().policy().model_dump(),'arms':ARMS,
        'max_output':MAX_OUTPUT,'thinking_bodies':{m:mode_body(m) for m in ('off','on')},
        'reasoning_effort':None,'concurrency':CONCURRENCY,'min_request_interval':INTERVAL,
        'changed_sources_since_oct23':[n for n,h in old['sources'].items() if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h],
        'selection':dataset()['selection'],'scope':'All stages including summary toggle thinking; current implementation under shared test reader, not production end-to-end. Same 8192 total output ceiling, including reasoning. 90s pool deadline. No semantic retries.',
        'packages':{p:importlib.metadata.version(p) for p in ['langchain-core','langchain-openai','openai','pydantic','httpx']}})


def check(out):
    m=read(out/'manifest.json')
    if config_snapshot()!=m['config'] or get_store().policy().model_dump()!=m['pool_policy']:raise ValueError('config changed')
    if digest(read(out/'dataset.json'))!=m['dataset_sha256']:raise ValueError('data changed')
    for name,h in m['sources'].items():
        if any(hashlib.sha256(p.read_bytes()).hexdigest()!=h for p in (ROOT/name,out/'source_snapshot'/name)):raise ValueError('source changed '+name)


async def preflight(out):
    # Entirely synthetic fixture inputs, never production customer data.
    out.mkdir(parents=True,exist_ok=True);caller=Caller(out);case=make_suite()['cases'][0]
    rows=case['history'][:30]
    requests=[('coref',COREF_REWRITE_PROMPT.format_messages(query='那笔订单到哪里了？',history='用户：这次只查订单610010。\n客服：好的。'),None),
        ('summary',SUMMARY_PROMPT.format_messages(old_summary='(无)',dialog='\n'.join(r['role']+':'+r['content'] for r in rows[:14])),Summary),
        ('plan',[SystemMessage(READER_SYSTEM),*raw_messages(rows),HumanMessage(case['probes'][0]['query'])],ServicePlan)]
    records=[]
    try:
        for stage,msgs,schema in requests:
            for mode in ('off','on'):
                r=await caller.call('preflight-'+stage+'-'+mode,mode,stage,msgs,schema);records.append(r)
                reason=sum(a.get('usage',{}).get('output_token_details',{}).get('reasoning',0) or 0 for a in r['attempts'])
                print(stage,mode,r['status'],'reasoning',reason,'seconds',r['seconds'],flush=True)
    finally:
        from app.core.model_transport import close_clients
        await close_clients()
    summary={mode:{'all_valid':all(r['ok'] for r in records if r['contract']['mode']==mode),
        'reasoning':sum(a.get('usage',{}).get('output_token_details',{}).get('reasoning',0) or 0 for r in records if r['contract']['mode']==mode for a in r['attempts'])} for mode in ('off','on')}
    summary['toggle_verified']=summary['off']['all_valid'] and summary['on']['all_valid'] and summary['off']['reasoning']==0 and summary['on']['reasoning']>0
    write_json(out/'preflight_report.json',summary);print(summary,flush=True)


async def run(out):
    check(out);caller=Caller(out);case_gate=asyncio.Semaphore(4)
    async def one(case):
        async with case_gate:
            # Same case's mode/strategy workflows interleave without sharing memory.
            arms=list(ARMS);random.Random('think-'+case['id']).shuffle(arms)
            async def arm_run(arm):
                g=workflow(case,arm);value=None
                try:
                    while True:
                        item=g.send(value);value=None
                        if item[0]=='call':value=await caller.call(*item[1:])
                        else:
                            _,folder,key,result=item;write_json(out/folder/(key+'.json'),result)
                            if folder=='results':print(key,'task',result['score']['task_rule_v3'],'target',result['score']['target_pass'],flush=True)
                except StopIteration:pass
            await asyncio.gather(*(arm_run(a) for a in arms))
    cases=read(out/'dataset.json')['cases'];random.Random(20261025).shuffle(cases)
    try:await asyncio.gather(*(one(c) for c in cases))
    finally:
        from app.core.model_transport import close_clients
        await close_clients()
    print('FINISHED',json.dumps(audit(out),ensure_ascii=False),flush=True)


def audit(out):
    check(out);cases=read(out/'dataset.json')['cases'];issues=[];expected=set();scores={a:{} for a in ARMS}
    entries={a:[] for a in ARMS};records={a:[] for a in ARMS};missing=[]
    for c in cases:
        for arm in ARMS:
            g=workflow(c,arm);value=None
            try:
                while True:
                    item=g.send(value);value=None
                    if item[0]=='call':
                        _,key,mode,stage,msgs,schema=item;expected.add(key)
                        path=out/'calls'/(key+'.json')
                        if not path.exists():missing.append(key);break
                        r=read(path);_,contract=wire_contract(mode,stage,msgs,schema)
                        if r['contract']!=contract or r['input_sha256']!=digest(contract):issues.append('causal prompt '+key)
                        for i,a in enumerate(r['attempts'],1):
                            if read(out/'attempts'/(key+f'-{i}.json'))!=a or a['input_sha256']!=digest(contract):issues.append('attempt '+key)
                            if i>1 and not r['attempts'][i-2].get('retryable'):issues.append('semantic retry '+key)
                        if r['value']!=r['attempts'][-1]['value']:issues.append('final value '+key)
                        records[arm].append(r);value=r
                    else:
                        _,folder,key,result=item;path=out/folder/(key+'.json')
                        if not path.exists():missing.append(key);continue
                        if read(path)!=result:issues.append('replay mismatch '+key)
                        if folder=='results':entries[arm].append(result);scores[arm].setdefault(c['id'],[]).append(result['score']['task_rule_v3'])
            except StopIteration:pass
    if expected!={p.stem for p in (out/'calls').glob('*.json')}:issues.append('unexpected call set')
    for c in cases:
        for p in c['probes']:
            paths=[out/'results'/f'{c["id"]}-{p["id"]}-S_{m}.json' for m in ('off','on')]
            if all(x.exists() for x in paths) and read(paths[0])['body']!=read(paths[1])['body']:issues.append('S input context differs by mode')
    stats={}
    for arm in ARMS:
        rr=entries[arm];aa=[a for r in records[arm] for a in r['attempts']]
        usage={k:sum(a.get('usage',{}).get(k,0) or 0 for a in aa) for k in ('input_tokens','output_tokens','total_tokens')}
        reasoning=sum(a.get('usage',{}).get('output_token_details',{}).get('reasoning',0) or 0 for a in aa)
        latency=[sum(r[s]['seconds'] for s in ('coref','plan','reply')) for r in rr]
        stats[arm]={'n':len(rr),'passed':sum(r['score']['task_rule_v3'] for r in rr),
            'legacy_passed':sum(r['score']['task_pass'] for r in rr),'target_passed':sum(r['score']['target_pass'] for r in rr),
            'coref_passed':sum(r['score']['coref_pass'] for r in rr),'usage':usage,'reasoning_tokens':reasoning,
            'non_reasoning_output_tokens':usage['output_tokens']-reasoning,
            'p50_seconds':percentile(latency,.5),'p95_seconds':percentile(latency,.95),
            'transport_failed_probes':sum(r['score']['transport_failure'] for r in rr),
            'truncated_probes':sum(r['score']['truncated'] for r in rr),
            'missing_usage_attempts':sum(not a.get('usage') for a in aa),
            'models':dict(Counter(a.get('model') for a in aa)),
            'stages':{s:{'status':dict(Counter(r['status'] for r in records[arm] if r['contract']['stage']==s)),
                'input':sum(a.get('usage',{}).get('input_tokens',0) or 0 for r in records[arm] if r['contract']['stage']==s for a in r['attempts']),
                'output':sum(a.get('usage',{}).get('output_tokens',0) or 0 for r in records[arm] if r['contract']['stage']==s for a in r['attempts']),
                'reasoning':sum(a.get('usage',{}).get('output_token_details',{}).get('reasoning',0) or 0 for r in records[arm] if r['contract']['stage']==s for a in r['attempts'])} for s in ('coref','plan','reply','summary')},
            'by_kind':{kind:{'passed':sum(read(out/'results'/f'{c["id"]}-{p["id"]}-{arm}.json')['score']['task_rule_v3'] for c in cases if c['kind']==kind for p in c['probes'] if (out/'results'/f'{c["id"]}-{p["id"]}-{arm}.json').exists()),'n':sum(2 for c in cases if c['kind']==kind)} for kind in dict.fromkeys(c['kind'] for c in cases)}}
    pairs={}
    for policy in ('C','S'):
        diffs=[statistics.mean(scores[policy+'_on'][c])-statistics.mean(scores[policy+'_off'][c]) for c in sorted(scores[policy+'_off'].keys()&scores[policy+'_on'].keys()) if len(scores[policy+'_on'][c])==len(scores[policy+'_off'][c])==2]
        rng=random.Random(20261025);draws=[sum(rng.choice(diffs) for _ in diffs)/len(diffs) for _ in range(4000)] if diffs else []
        pairs[policy]={'difference':statistics.mean(diffs) if diffs else None,'conversation_bootstrap_95ci':[percentile(draws,.025),percentile(draws,.975)]}
    report={'complete':not missing and all(len(entries[a])==2*len(cases) for a in ARMS),'audit_passed':not missing and not issues,
        'issues':issues,'missing':missing,'strategies':stats,'thinking_on_minus_off':pairs,
        'scope':'16 known synthetic histories, 8 families x 24/64 turns, 2 correlated probes each; current code and shared reader; all C summary stages use assigned mode. Reasoning included in output and capped at 8192 total.'}
    write_json(out/'audited_report.json',report)
    lines=['# 思考开启/关闭：上下文配对实验','',f'完成：{report["complete"]}；因果回放审计：{report["audit_passed"]}。','',
        '|策略/模式|任务规则通过|对象/动作正确|输入含维护|输出含思考维护|其中思考Token|P50/P95秒|传输失败追问|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,s in stats.items():lines.append(f'|{arm}|{s["passed"]}/{s["n"]}|{s["target_passed"]}/{s["n"]}|{s["usage"]["input_tokens"]:,}|{s["usage"]["output_tokens"]:,}|{s["reasoning_tokens"]:,}|{s["p50_seconds"]}/{s["p95_seconds"]}|{s["transport_failed_probes"]}|')
    lines+=['','C为当前摘要窗口，S为原文窗口+来源状态。所有阶段同一模式；相同8192输出上限涵盖思考和最终答案；实际还受当前模型池90秒调用期限限制。',
        '按场景固定选择16条既有合成会话，未按旧分数选样。关闭组重新生成，不能与10月23日分数直接作因果比较。',
        '输入含全部摘要维护；不含生成已给定历史的费用。延迟为三阶段API等待之和，不含实验节流排队/后台维护，不是线上首字延迟。',
        '报告接口、解析及截断失败，保留语义答错；主指标仍有已知措辞误判，不是人工语义准确率。',
        '本实验未修改线上配置。源码、数据、调用契约、尝试与离线复算结果保留。']
    (out/'AUDITED_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--prepare',action='store_true');p.add_argument('--preflight',action='store_true');p.add_argument('--audit',action='store_true')
    a=p.parse_args();out=Path(a.output).resolve()
    if a.prepare:prepare(out)
    elif a.preflight:asyncio.run(preflight(out))
    elif a.audit:print(json.dumps(audit(out),ensure_ascii=False))
    else:
        with writer_lock(out):asyncio.run(run(out))
