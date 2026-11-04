"""Frozen three-policy comparison; unchanged prior baselines and scorer.

Only synthetic histories. No production graph changes, semantic retries or
reuse of old model outputs. Oversize local requests are recorded, not sent.
"""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import statistics

from langchain_core.messages import SystemMessage,HumanMessage,AIMessage,ToolMessage
from app.core.service_context import ServiceMemory,project_tool_result
from app.core.context_policy import estimate
from app.core.model_pool import providers,get_store
from app.config import settings
from scripts import eval_context_thinking as base
from scripts.service_context_cases import dataset
from scripts.eval_context_policy_v3 import ROOT,ServicePlan,READER_SYSTEM,REPLY_SYSTEM,COREF_REWRITE_PROMPT,tool_result,score,digest,stamp,write_json,writer_lock,percentile,config_snapshot

ARMS=('C_off','S_off','N_off')
LABELS={'C_off':'原摘要+滑窗','S_off':'旧原文+来源状态','N_off':'新业务状态+来源回查+预算控制'}
SOURCES=list(dict.fromkeys([*base.SOURCES,'app/core/service_context.py','scripts/service_context_cases.py',
    'scripts/eval_service_context.py','tests/test_service_context.py','tests/test_service_context_eval.py','docs/service-context-plan.md']))


def read(p):return json.loads(p.read_text(encoding='utf-8'))


class Caller(base.Caller):
    async def call(self,key,mode,stage,messages,schema=None):
        wired,contract=base.wire_contract(mode,stage,messages,schema)
        approx=sum(estimate(m.content) for m in wired)+(estimate(json.dumps(contract['schema'],ensure_ascii=False)) if schema else 0)
        if approx<=16000:return await super().call(key,mode,stage,messages,schema)
        path=self.out/'calls'/(key+'.json')
        r={'contract':contract,'input_sha256':digest(contract),'attempts':[],
            'status':'budget_blocked','ok':False,'value':None,'finish_reason':None,
            'seconds':0,'queue_seconds':0,'approx_input_tokens':approx,'calls':[],
            'local_error':'complete input estimate above 16000; not sent, no automatic content change'}
        if path.exists():
            if read(path)!=r:raise ValueError('blocked cache changed')
        else:write_json(path,r)
        return r


def workflow(case,arm):
    if arm!='N_off':
        yield from base.workflow(case,arm)
        return
    memory=ServiceMemory(case['id']);events=[]
    for probe in case['probes']:
        rows=case['history'][:2*probe['at_turn']];memory.advance(rows,case['id'])
        context=memory.assemble(probe['query'],budget=9800)
        events.append({'at_message':len(rows),'source_sha256':context['source_sha256'],
                       'evidence_refs':context['evidence_refs'],'protected_refs':context['protected_refs'],
                       'omitted_turns':context['omitted_turns'],'approx_context_tokens':context['approx_context_tokens']})
        key=f'{case["id"]}-{probe["id"]}-{arm}'
        cr=yield ('call',key+'-coref','off','coref',COREF_REWRITE_PROMPT.format_messages(query=probe['query'],history=context['coref_history']),None)
        instruction=HumanMessage('本轮原话：'+probe['query']+'\n指代改写结果（需用历史核实）：'+(cr['value'] if cr['ok'] else probe['query']))
        plan=yield ('call',key+'-plan','off','plan',[SystemMessage(READER_SYSTEM),*context['body'],instruction],ServicePlan)
        tool=tool_result(case,plan['value']) if plan['ok'] else None
        projection=project_tool_result(tool) if tool is not None else None
        prompt=[SystemMessage(REPLY_SYSTEM),*context['body'],instruction]
        if tool is not None:
            p=plan['value'];name='query_logistics' if p['action']=='logistics' else 'query_faq'
            args={'order_id':p['order_id']} if p['action']=='logistics' else {'query':p['sku']+' 清洁方法'}
            prompt += [AIMessage('',tool_calls=[{'id':'eval-readonly','name':name,'args':args,'type':'tool_call'}]),ToolMessage(json.dumps(projection['value'],ensure_ascii=False),tool_call_id='eval-readonly')]
        else:prompt.append(HumanMessage('本轮计划：'+json.dumps(plan['value'],ensure_ascii=False)+'\n请核实历史后作答。'))
        reply=yield ('call',key+'-reply','off','reply',prompt,None)
        result={'probe_id':probe['id'],'at_turn':probe['at_turn'],'coref':cr,'plan':plan,'reply':reply,
                'tool_result':tool,'tool_projection':projection,'body':[m.model_dump(mode='json') for m in context['body']],
                'context_text':'\n'.join(m.content for m in context['body']),
                'context':{k:v for k,v in context.items() if k!='body'},'state':memory.snapshot()}
        result['context_sha256']=digest(result['body']);result['score']=score(case,probe,result)
        yield ('save','results',key,result)
    yield ('save','maintenance',f'{case["id"]}-{arm}',events)


def prepare(out):
    if out.exists():raise ValueError('new output required')
    if settings.chat_failover_enabled or any(p.fallback or p.model!=settings.chat_model for p in providers('chat',settings.chat_model)):
        raise ValueError('fixed model without failover required')
    if settings.summary_model and settings.summary_model!=settings.chat_model:raise ValueError('same summary model required')
    out.mkdir(parents=True);d=dataset();write_json(out/'dataset.json',d)
    sources={}
    for name in SOURCES:
        data=(ROOT/name).read_bytes();sources[name]=hashlib.sha256(data).hexdigest()
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    write_json(out/'manifest.json',{'created_at':stamp(),'sources':sources,'dataset_sha256':digest(d),
        'config':config_snapshot(),'pool_policy':get_store().policy().model_dump(),'arms':ARMS,
        'max_output':base.MAX_OUTPUT,'input_limit_estimate':16000,'mode_body':base.mode_body('off'),
        'concurrency':base.CONCURRENCY,'start_interval':base.INTERVAL,
        'scope':'Current context components with common read-only reader. 16 known regression + 8 designed pressure histories, two causal probes. Not live end-to-end or blind holdout.',
        'fairness':'Same full input ceiling, model, nonthinking mode, reader, fixtures, output budget and frozen scoring. C keeps original 3000 window; S keeps 8000+1800; N caps complete context at 9800. Not equal actual input tokens.',
        'cost':'Includes all memory maintenance and two probes; given history generation excluded. No semantic retry. Only one transport retry.'})


def check(out):
    m=read(out/'manifest.json')
    if config_snapshot()!=m['config'] or get_store().policy().model_dump()!=m['pool_policy']:raise ValueError('configuration changed')
    if digest(read(out/'dataset.json'))!=m['dataset_sha256']:raise ValueError('dataset changed')
    for name,h in m['sources'].items():
        if any(hashlib.sha256(p.read_bytes()).hexdigest()!=h for p in (ROOT/name,out/'source_snapshot'/name)):
            raise ValueError('source changed '+name)


async def run(out):
    check(out);caller=Caller(out);gate=asyncio.Semaphore(4)
    async def one(case):
        async with gate:
            arms=list(ARMS);random.Random('service-'+case['id']).shuffle(arms)
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
    cases=read(out/'dataset.json')['cases'];random.Random(20261026).shuffle(cases)
    try:await asyncio.gather(*(one(c) for c in cases))
    finally:
        from app.core.model_transport import close_clients
        await close_clients()
    report=audit(out)
    print('FINISHED',json.dumps({k:v for k,v in report.items() if k!='strategies'},ensure_ascii=False),flush=True)


def audit(out):
    check(out);cases=read(out/'dataset.json')['cases'];issues=[];missing=[];expected=set()
    entries={a:[] for a in ARMS};calls={a:[] for a in ARMS};paired_scores={a:{} for a in ARMS}
    for c in cases:
        for arm in ARMS:
            g=workflow(c,arm);value=None
            try:
                while True:
                    item=g.send(value);value=None
                    if item[0]=='call':
                        _,key,mode,stage,msgs,schema=item;expected.add(key);path=out/'calls'/(key+'.json')
                        if not path.exists():missing.append(key);break
                        r=read(path);wired,contract=base.wire_contract(mode,stage,msgs,schema)
                        approx=sum(estimate(m.content) for m in wired)+(estimate(json.dumps(contract['schema'],ensure_ascii=False)) if schema else 0)
                        if r['contract']!=contract or r['input_sha256']!=digest(contract):issues.append('contract '+key)
                        if r['approx_input_tokens']!=approx:issues.append('input estimate '+key)
                        for i,a in enumerate(r['attempts'],1):
                            if read(out/'attempts'/(key+f'-{i}.json'))!=a or a['input_sha256']!=digest(contract):issues.append('receipt '+key)
                            if i>1 and not r['attempts'][i-2].get('retryable'):issues.append('semantic retry '+key)
                        if r['status']=='budget_blocked':
                            if r['attempts'] or r['approx_input_tokens']<=16000 or r['ok']:issues.append('invalid local block '+key)
                        elif not r['attempts'] or r['value']!=r['attempts'][-1]['value']:issues.append('result '+key)
                        calls[arm].append((c,r));value=r
                    else:
                        _,folder,key,result=item;path=out/folder/(key+'.json')
                        if not path.exists():missing.append(key);continue
                        if read(path)!=result:issues.append('causal replay '+key)
                        if folder=='results':
                            entries[arm].append((c,result));paired_scores[arm].setdefault(c['id'],[]).append(result['score']['task_rule_v3'])
                            if arm=='N_off':
                                if result['context']['approx_context_tokens']>9800:issues.append('context budget '+key)
                                for e in result['state']['evidence']:
                                    if e['seq']>2*result['at_turn'] or c['history'][e['seq']-1]['content'][e['start']:e['end']]!=e['text']:
                                        issues.append('source offset '+key)
            except StopIteration:pass
    if expected!={p.stem for p in (out/'calls').glob('*.json')}:issues.append('unexpected calls')
    stats={}
    for arm in ARMS:
        stats[arm]={}
        for cohort in ('all','regression','pressure'):
            rr=[r for c,r in entries[arm] if cohort=='all' or c['cohort']==cohort]
            cc=[r for c,r in calls[arm] if cohort=='all' or c['cohort']==cohort];aa=[a for r in cc for a in r['attempts']]
            latency=[sum(r[s]['seconds'] for s in ('coref','plan','reply')) for r in rr]
            s={'n':len(rr),'passed':sum(r['score']['task_rule_v3'] for r in rr),'target_passed':sum(r['score']['target_pass'] for r in rr),
               'legacy_passed':sum(r['score']['task_pass'] for r in rr),'coref_passed':sum(r['score']['coref_pass'] for r in rr),
               'usage':{k:sum(a.get('usage',{}).get(k,0) or 0 for a in aa) for k in ('input_tokens','output_tokens','total_tokens')},
               'reasoning_tokens':sum(a.get('usage',{}).get('output_token_details',{}).get('reasoning',0) or 0 for a in aa),
               'p50_api_seconds':percentile(latency,.5),'p95_api_seconds':percentile(latency,.95),
               'transport_failed_probes':sum(r['score']['transport_failure'] for r in rr),
               'budget_blocked_probes':sum(any(r[t]['status']=='budget_blocked' for t in ('coref','plan','reply')) for r in rr),
               'missing_usage_attempts':sum(not a.get('usage') for a in aa),
               'successful_attempts_without_reasoning_usage':sum(a.get('transport_ok') and 'reasoning' not in a.get('usage',{}).get('output_token_details',{}) for a in aa),
               'call_status':dict(Counter(r['status'] for r in cc)),
               'models':dict(Counter(a.get('model') for a in aa)),
               'summary':{'calls':sum(r['contract']['stage']=='summary' for r in cc),
                   'status':dict(Counter(r['status'] for r in cc if r['contract']['stage']=='summary')),
                   'input':sum(a.get('usage',{}).get('input_tokens',0) or 0 for r in cc if r['contract']['stage']=='summary' for a in r['attempts']),
                   'output':sum(a.get('usage',{}).get('output_tokens',0) or 0 for r in cc if r['contract']['stage']=='summary' for a in r['attempts'])}}
            stats[arm][cohort]=s
    comparisons={}
    for baseline in ('C_off','S_off'):
        comparisons[baseline]={}
        for cohort in ('all','regression','pressure'):
            ids=[c['id'] for c in cases if cohort=='all' or c['cohort']==cohort]
            diffs=[statistics.mean(paired_scores['N_off'][cid])-statistics.mean(paired_scores[baseline][cid]) for cid in ids if len(paired_scores['N_off'].get(cid,[]))==len(paired_scores[baseline].get(cid,[]))==2]
            rng=random.Random(20261026);draws=[statistics.mean(rng.choice(diffs) for _ in diffs) for _ in range(4000)] if diffs else []
            comparisons[baseline][cohort]={'difference':statistics.mean(diffs) if diffs else None,'bootstrap_95ci':[percentile(draws,.025),percentile(draws,.975)]}
    report={'complete':not missing and all(len(entries[a])==2*len(cases) for a in ARMS),'audit_passed':not missing and not issues,
            'issues':issues,'missing':missing,'nonthinking_verified':all(stats[a]['all']['reasoning_tokens']==0 and stats[a]['all']['successful_attempts_without_reasoning_usage']==0 and stats[a]['all']['n']>0 for a in ARMS),
            'strategies':stats,'new_minus_baseline':comparisons,
            'scope':'24 synthetic histories, 8 families; 16 known + 8 designed pressure. Two correlated probes per history, original rule metric. No live deployment.'}
    write_json(out/'audited_report.json',report)
    lines=['# 新客服上下文方案对照','',f'完整：{report["complete"]}；离线回放审计：{report["audit_passed"]}；非思考用量验证：{report["nonthinking_verified"]}。','']
    for cohort,label in [('regression','原有回归集'),('pressure','新增压力集'),('all','合计')]:
        lines += ['## '+label,'','|方案|规则通过|规划对象正确|输入含维护|输出含维护|API等待P50/P95秒|预算阻断追问|接口失败追问|','|---|---:|---:|---:|---:|---:|---:|---:|']
        for arm in ARMS:
            s=stats[arm][cohort]
            lines.append(f'|{LABELS[arm]}|{s["passed"]}/{s["n"]}|{s["target_passed"]}/{s["n"]}|{s["usage"]["input_tokens"]:,}|{s["usage"]["output_tokens"]:,}|{s["p50_api_seconds"]}/{s["p95_api_seconds"]}|{s["budget_blocked_probes"]}|{s["transport_failed_probes"]}|')
        lines.append('')
    lines += ['保留旧评分，含已知否定引用误判；规划正确不等于回答正确。不得把合成规则通过率写成线上准确率。',
        '三组同模型、非思考、相同完整输入上限16000估算Token、输出8192。C原窗口3000，S原窗口8000+状态1800，N完整上下文上限9800；不等于相同实际输入量。',
        '压力集是针对已知边界构造的回归测试，不是盲测；与原有集分开报告。当前评测不包含退款写操作、真实RAG召回、跨会话持久化或线上SSE。',
        'Token包含摘要维护和两次追问，不含生成给定历史。接口未返回用量时总量是下限。API等待含失败尝试、不含实验节流排队、退避和后台维护；本地阻断耗时为0，不能据此宣称更快。',
        'N借鉴业务状态、来源证据、工具结果投影和预算检查；未声称完整迁移QueryMind或Harness运行时。没有新增LLM摘要调用。']
    (out/'AUDITED_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--prepare',action='store_true');p.add_argument('--audit',action='store_true')
    a=p.parse_args();out=Path(a.output).resolve()
    if a.prepare:prepare(out)
    elif a.audit:print(json.dumps(audit(out),ensure_ascii=False))
    else:
        with writer_lock(out):asyncio.run(run(out))
