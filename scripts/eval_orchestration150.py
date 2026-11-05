"""Fresh paired baseline/current evaluation: 5x30 cases, exactly one round per arm."""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
HOME=Path('E:/Mewhelp')
OUT=Path(os.environ.get('ORCH150_OUTPUT',str(HOME/'data/multi-intent-eval/20261105-orchestration150-once')))
BASE=HOME/'data/multi-intent-eval/20261103-business120'
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(HOME))
from scripts.build_orchestration_dataset import assign_bucket,bucket_tags,BUCKETS,RULES
from scripts import rescore_multi120_evidence as scoring
from scripts.multi120_evidence_protocol import validate_labels
from scripts.score_multi120 import summarize


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str),encoding='utf-8');tmp.replace(p)


async def prepare():
    if OUT.exists():raise RuntimeError('Never overwrite frozen inputs/results')
    from scripts.orchestration150_cases import build_cases
    from app.config import settings
    from app.db import base,workflow
    from app.kb import milvus_client
    from app.core import model_pool
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.engine import make_url
    from sqlalchemy import text
    cases=build_cases();kb=await milvus_client.acall(lambda:milvus_client.get_client().query('knowledge',filter='',output_fields=['id','question','answer','section_path','content_type','category'],limit=16384));kb.sort(key=lambda k:k['id'])
    assert kb==read(BASE/'knowledge.json')
    OUT.mkdir(parents=True)
    save(OUT/'cases.json',cases);save(OUT/'knowledge.json',kb)
    save(OUT/'buckets.json',BUCKETS);save(OUT/'exclusion-rules.json',RULES)
    mapping={c['id']:{'bucket':assign_bucket(c),'tags':bucket_tags(c),'split':'main' if len(c['goals'])>1 else 'control'} for c in cases}
    save(OUT/'bucket-map.json',mapping)
    (OUT/'eval_orchestration150.jsonl').write_text('\n'.join(json.dumps({**c,**mapping[c['id']]},ensure_ascii=False) for c in cases)+'\n',encoding='utf-8')
    shutil.copy2(HOME/'tests/data/eval_ch04.jsonl',OUT/'original300.jsonl')
    shutil.copy2(HOME/'data/runtime_settings.json',OUT/'runtime_settings.json')
    for arm,src in [('disabled',BASE/'source/app'),('enabled',ROOT/'app')]:
        shutil.copytree(src,OUT/'source'/arm/'app',ignore=shutil.ignore_patterns('__pycache__'))
    names=['__init__.py','eval_orchestration150.py','orchestration150_cases.py','build_orchestration_dataset.py',
           'eval_multi120.py','score_multi120.py','rescore_multi120_evidence.py','multi120_evidence_protocol.py']
    for name in names:
        dest=OUT/'evaluator-source/scripts'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'scripts'/name,dest)
    db='caremate_orchestration150_20261105_once'
    url=make_url(settings.test_database_url);admin=create_async_engine(url.set(database=None),isolation_level='AUTOCOMMIT')
    async with admin.begin() as conn:await conn.execute(text(f'CREATE DATABASE {db} CHARACTER SET utf8mb4'))
    await admin.dispose();engine=create_async_engine(url.set(database=db))
    async with engine.begin() as conn:
        await conn.run_sync(base.Base.metadata.create_all)
        await conn.execute(text('ALTER TABLE conversations AUTO_INCREMENT = 2300000000'))
    old_engine=base.engine;base.engine=engine
    try:await workflow.ensure_tables()
    finally:base.engine=old_engine;await engine.dispose()
    config=read(BASE/'protocol.json')['model_config']
    assert all(getattr(settings,k)==v for k,v in config.items()),'Current model/context config changed; audit before comparison'
    # Copy only explicitly enabled primary credentials from this project's live registry.
    original_path=Path(settings.model_pool_db_path)
    if not original_path.is_absolute():original_path=HOME/original_path
    assert original_path.exists()
    prod=model_pool.Store(original_path)
    with prod.connect() as conn:disabled={r['resource'] for r in conn.execute('select resource from slots where disabled=1 or invalid=1')}
    entries={};provider_meta={}
    for ch,model in [('chat',config['chat_model']),('embed',config['embed_model']),('rerank',config['rerank_model'])]:
        specs=prod.credential_providers(ch,model)
        if specs is None:specs=[p for p in model_pool.environment_providers(ch,model) if not p.fallback]
        specs=[p for p in specs if not p.fallback and p.fingerprint not in disabled and p.base_url.rstrip('/')==config[ch+'_base_url'].rstrip('/')]
        assert specs,('No enabled primary key for',ch)
        provider_meta[ch]=[p.fingerprint for p in specs]
        for provider in specs:entries.setdefault(provider.api_key,set()).add(ch)
    store=model_pool.Store(OUT/'model_pool.sqlite3');store.set_credentials([{'api_key':key,'channels':sorted(chs)} for key,chs in entries.items()])
    policy=model_pool.Policy(global_limit=12,key_limit=2,background_limit=10,max_queue=512,queue_timeout=120,call_timeout=60,read_timeout=40,max_attempts=2)
    store.set_policy(policy)
    contract={'version':'orchestration150-once','repetitions':1,'arms':['disabled','enabled'],
              'main_case_ids':[c['id'] for c in cases if len(c['goals'])>1],
              'control_case_ids':[c['id'] for c in cases if len(c['goals'])==1],
              'main_goals':sum(len(c['goals']) for c in cases if len(c['goals'])>1),
              'control_goals':sum(len(c['goals']) for c in cases if len(c['goals'])==1),
              'scenario':'All required outcomes and hard constraints; unknown retained in denominator, never silently passed.',
              'write_success':'Exactly one matching SQL review/complaint ticket after independent confirmation; application is not funds refund.',
              'judge_prompt':scoring.PROMPT,'judge_schema':scoring.Verdict.model_json_schema(),
              'quality':'All300 get goal-coverage and factual-claim review in the same semantic call. No separate quality reruns.',
              'source_scope':'Retained83 unchanged plus67 newly authored from original300 and KB54. Post-hoc internal capability set; correlated families, not blind independent data.',
              'failures':'Keep API/execution/driver failures. No rerunning completed trajectories, no best-of repeats. Transport retries within a call remain recorded.'}
    save(OUT/'scoring.json',contract)
    p={'version':'orchestration150-once','repetitions':1,'arms':['disabled','enabled'],'candidate_shared_agent':True,
       'model_config':config,'test_db_name':db,'model_pool_db_path':str(OUT/'model_pool.sqlite3'),
       'evaluation_settings':{'model_pool_fallback_base_url':'','model_pool_fallback_api_key':'','model_pool_fallback_model':'',
                              'model_pool_turn_timeout':480,'model_pool_queue_timeout':120},
       'provider_policy':{'vendor_failover':False,'configured_fallback':False,'enabled_primary_key_fingerprints':provider_meta,'policy':policy.model_dump()},
       'baseline_definition':'Archived original single-intent source; multi_intent_enabled=false.',
       'candidate_definition':'Current v4 shared main_agent/agent_tools; multi_intent_enabled=true.',
       'candidate_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'scheduled_trajectories':300,'worker_concurrency':8,'judge_concurrency':4,
       'bucket_counts':dict(Counter(v['bucket'] for v in mapping.values())),
       'source_hashes':{str(f.relative_to(OUT/'source')):sha(f) for f in (OUT/'source').rglob('*.py')},
       'evaluator_hashes':{name:sha(OUT/'evaluator-source/scripts'/name) for name in names},
       'input_hashes':{n:sha(OUT/n) for n in ['cases.json','knowledge.json','scoring.json','buckets.json','bucket-map.json','exclusion-rules.json','original300.jsonl','runtime_settings.json']},
       'production_app_hashes':{str(f):sha(f) for f in (HOME/'app').rglob('*.py')},
       'production_runtime_hash':sha(HOME/'data/runtime_settings.json'),
       'old_artifact_hashes':{str(f):sha(f) for folder in ['20261103-business120','20261105-shared-agent120-v2','20261105-orchestration-five-buckets-v2'] for f in (HOME/'data/multi-intent-eval'/folder).glob('*.json')},
       'limits':['One stochastic observation per scenario/arm; no reliability estimate from repeats.',
                 'Real Gitee chat/embed/rerank; synthetic isolated business fixtures and SQL.',
                 'Graph/form API, no browser/SSE first-token or physical-process-crash benchmark.',
                 'Same selected primary registry keys for both arms; pool state/credentials isolated, no production credential edits.',
                 'Automated judge is not independent human truth; unsupported or inconsistent labels remain unknown.']}
    save(OUT/'protocol.json',p)
    lines=['# 五桶各30题：单轮完整业务场景','','保留原83题，新增67题；每版各150题，仅1轮，共300条。原300/120题及历史结果保留。','','| 桶 | 题数 | 多任务 | 单任务控制 |','|---|---:|---:|---:|']
    for b in BUCKETS:
        sub=[c for c in cases if mapping[c['id']]['bucket']==b['id']]
        lines.append(f'| {b["name"]} | {len(sub)} | {sum(len(c["goals"])>1 for c in sub)} | {sum(len(c["goals"])==1 for c in sub)} |')
    lines+=['','D桶五种情况各6题：取消退款表单、取消投诉确认、等待中重连、物流工具不可用、重复提交。新重复提交题包含明确投诉事由。E桶五种边界情况各6题，18个单任务控制单独报告。','',
            '评分沿用修复后的证据、时序及SQL核验。纯模型说已完成不算成功；接口故障保留；只运行1轮，不挑最好一次。原始标签所需条件仍须跟随答案说明。','','## 全部题目','']
    for b in BUCKETS:
        lines += [f'### {b["id"]} {b["name"]}','']
        for c in cases:
            if mapping[c['id']]['bucket']!=b['id']:continue
            goals='；'.join(g['kind']+(':'+g['order_id'] if g.get('order_id') else '')+'→'+g['expected_outcome'] for g in c['goals'])
            lines += [f'**{c["id"]}** {c["query"]}',f'来源：{", ".join(c["source_question_ids"])}；目标：{goals}；故障：{c["fault"] or "无"}。','']
    (OUT/'CASES.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Frozen150, 30 per bucket, fresh300 total; main132/control18; goals321 per arm',flush=True)


def verify_inputs():
    p=read(OUT/'protocol.json')
    for name,h in p['input_hashes'].items():assert sha(OUT/name)==h,name
    for name,h in p['source_hashes'].items():assert sha(OUT/'source'/name)==h,name
    for name,h in p['evaluator_hashes'].items():assert sha(OUT/'evaluator-source/scripts'/name)==h,name


def wire(arm='enabled'):
    sys.path.insert(0,str(OUT/'source'/arm))
    from app.config import settings
    p=read(OUT/'protocol.json')
    for k,v in p['model_config'].items():setattr(settings,k,v)
    for k,v in p['evaluation_settings'].items():setattr(settings,k,v)
    settings.model_pool_fallback_base_url='';settings.model_pool_fallback_api_key='';settings.model_pool_fallback_model=''
    settings.chat_failover_enabled=False;settings.model_pool_db_path=p['model_pool_db_path'];settings.langfuse_public_key=''
    from app.core import runtime_settings
    runtime_settings.SETTINGS_PATH=OUT/'runtime_settings.json'
    from scripts import eval_multi120 as old
    old.OUT=OUT;old.ROOT=HOME
    return old


async def worker(arm,cid):
    verify_inputs();await wire(arm).worker(arm,cid,1)


async def run():
    verify_inputs();cases=read(OUT/'cases.json');sem=asyncio.Semaphore(8)
    async def pair(index,c):
        async with sem:
            for arm in (['disabled','enabled'] if index%2==0 else ['enabled','disabled']):
                folder=OUT/'runs/r1'/arm/c['id'];folder.mkdir(parents=True,exist_ok=True)
                if (folder/'result.json').exists():continue
                with (folder/'worker.log').open('w',encoding='utf-8') as log:
                    proc=await asyncio.create_subprocess_exec(sys.executable,str(OUT/'evaluator-source/scripts/eval_orchestration150.py'),'worker','--arm',arm,'--case',c['id'],cwd=HOME,stdout=log,stderr=log)
                    try:rc=await asyncio.wait_for(proc.wait(),620)
                    except TimeoutError:proc.kill();await proc.wait();rc=-999
                if not (folder/'result.json').exists():save(folder/'result.json',{'arm':arm,'rep':1,'case_id':c['id'],'case':c,'status':'worker_error','exit_code':rc,'steps':[],'actor':[],'tickets':[],'authorized':[],'tool_audit':[]})
                print('runs',len(list((OUT/'runs').glob('*/*/*/result.json'))),'/300',arm,c['id'],flush=True)
    await asyncio.gather(*(pair(i,c) for i,c in enumerate(cases)))


async def judge():
    wire();from app.core import llm,model_pool
    kb=[{'source_id':'kb:'+str(k['id']),'section':k['section_path'],'question':k['question'],'answer':k['answer']} for k in read(OUT/'knowledge.json')]
    sem=asyncio.Semaphore(4)
    async def one(file):
        async with sem:
            raw=read(file);folder=OUT/'judges/r1'/raw['arm']/raw['case_id'];data=scoring.payload(raw,file.with_name('checkpoints.sqlite')) if file.with_name('checkpoints.sqlite').exists() else {'timeline':[],'unplaced_actions':[],'gold_issues':{},'source_ids':[],'goals':raw['case']['goals']}
            save(folder/'input.json',data)
            if not any(e['role']=='assistant' for e in data['timeline']):save(folder/'label.json',{'status':'no_answer','items':[],'claims':[]});return
            errors=[]
            for attempt in range(2):
                try:
                    with model_pool.call_context(f'orch150-judge:{raw["arm"]}:{raw["case_id"]}:{attempt}',priority='background'):
                        answer=await llm.structured(scoring.Verdict,temperature=0).ainvoke([{'role':'system','content':scoring.PROMPT},{'role':'user','content':json.dumps({'frozen_knowledge':kb,**data},ensure_ascii=False)}])
                    result=validate_labels(answer.model_dump(),data['timeline'],data['source_ids']+[x['source_id'] for x in kb],[g['id'] for g in data['goals']]);save(folder/f'attempt-{attempt}.json',result)
                    if any(not c['decision_consistent'] for c in result['claims']):raise ValueError('Risk decision contradicts reason')
                    save(folder/'label.json',{'status':'ok',**result,'errors':errors});return
                except Exception as exc:errors.append(str(exc))
            save(folder/'label.json',{'status':'judge_error','items':[],'claims':[],'errors':errors})
    while True:
        files=list((OUT/'runs').glob('*/*/*/result.json'))
        missing=[f for f in files if not (OUT/'judges/r1'/read(f)['arm']/f.parent.name/'label.json').exists()]
        await asyncio.gather(*(one(f) for f in missing))
        count=len(list((OUT/'judges').glob('*/*/*/label.json')));save(OUT/'progress.json',{'runs':len(files),'judges':count,'expected':300})
        print('judges',count,'runs',len(files),flush=True)
        if len(files)==count==300:break
        await asyncio.sleep(6)
    from app.core.model_transport import close_clients
    await close_clients()


def report():
    rows=[];mapping=read(OUT/'bucket-map.json')
    for f in (OUT/'runs').glob('*/*/*/result.json'):
        raw=read(f);folder=OUT/'judges/r1'/raw['arm']/raw['case_id']
        row=scoring.evaluate(raw,read(folder/'label.json'),read(folder/'input.json'));row.update(mapping[raw['case_id']]);rows.append(row)
    summary={arm:{split:summarize([r for r in rows if r['arm']==arm and (split=='all' or r['split']==split)]) for split in ['main','control','all']} for arm in ['disabled','enabled']}
    by_bucket={arm:{b['id']:summarize([r for r in rows if r['arm']==arm and r['bucket']==b['id']]) for b in BUCKETS} for arm in ['disabled','enabled']}
    save(OUT/'summary.json',{'summary':summary,'by_bucket':by_bucket,'rows':rows})
    save(OUT/'failure-index.json',[r for r in rows if r['success'] is not True])
    a,b=summary['disabled']['main'],summary['enabled']['main'];fmt=lambda n,d:f'{n}/{d}（{n/d:.1%}）'
    lines=['# 五桶150题：原单意图与当前共享Agent单轮对比','','每桶30题，两版各150条，仅一轮共300条新业务轨迹；132题多任务主集、18题单任务控制分开统计。没有三轮复测，没有复用旧业务结果。','','| 主集指标 | 单意图基线 | 当前多意图 |','|---|---:|---:|',
           f'| 场景通过 | {fmt(a["success_n"],a["n"])} | {fmt(b["success_n"],b["n"])} |',f'| 必需任务完成 | {fmt(a["terminal_n"],a["task_n"])} | {fmt(b["terminal_n"],b["task_n"])} |',
           f'| 场景待定 | {a["unknown_n"]} | {b["unknown_n"]} |',f'| 整条耗时中位数 | {a["median_seconds"]:.2f}秒 | {b["median_seconds"]:.2f}秒 |']
    for key,label in [('observed_selector_n','全部实际选单'),('unmapped_selector_n','用途无法映射的选单'),('extra_selection_n','旧规则显式订单额外选单'),('duplicate_ticket_n','重复工单'),('wrong_refund_order_n','错误订单退款'),('unauthorized_ticket_n','未经确认写入')]:lines.append(f'| {label} | {a["hard"][key]} | {b["hard"][key]} |')
    lines+=['','| 桶（各30题，含本桶控制） | 基线场景通过 | 多意图场景通过 |','|---|---:|---:|']
    for bucket in BUCKETS:
        x,y=by_bucket['disabled'][bucket['id']],by_bucket['enabled'][bucket['id']];lines.append(f'| {bucket["name"]} | {fmt(x["success_n"],x["n"])} | {fmt(y["success_n"],y["n"])} |')
    lines+=['',f'单任务控制：基线{fmt(summary["disabled"]["control"]["success_n"],18)}，多意图{fmt(summary["enabled"]["control"]["success_n"],18)}。', '',
            '未知标签不算通过、也不从分母删除。评分包含原始SQL、独立确认、完整知识和实际检索来源；模型评委不是独立人工真值。接口/执行/驱动故障完整保留，见failure-index.json。', '',
            '退款提交是审核工单，不是资金退款。D桶的工具不可用要求如实说明失败并处理其余事项，不代表物流查询成功。重连仅重建SQLite连接和执行图，未模拟操作系统崩溃；未做浏览器SSE/首Token评测。', '',
            '只有一次随机观察，不提供三轮稳定性或统计显著性结论。新集是作者可见的事后专项集，家族/变体相关，不能直接与原120整体分数比较。未根据本轮结果修改业务源码或挑题。', '',
            '题目与来源：CASES.md、cases.json、original300.jsonl；冻结配置：protocol.json；全部原始轨迹：runs/；评审输入与逐字证据：judges/；核验：verification.json。','']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


async def verify():
    verify_inputs();p=read(OUT/'protocol.json')
    for f,h in p['production_app_hashes'].items():assert sha(Path(f))==h,f
    for f,h in p['old_artifact_hashes'].items():assert sha(Path(f))==h,f
    assert sha(HOME/'data/runtime_settings.json')==p['production_runtime_hash']
    from app.kb import milvus_client
    kb=await milvus_client.acall(lambda:milvus_client.get_client().query('knowledge',filter='',output_fields=['id','question','answer','section_path','content_type','category'],limit=16384));kb.sort(key=lambda k:k['id']);assert kb==read(OUT/'knowledge.json')
    c=sqlite3.connect(OUT/'model_pool.sqlite3');models=[r[0] for r in c.execute('select distinct model from attempts')];c.close()
    assert set(models)<=set([p['model_config'][k] for k in ['chat_model','embed_model','rerank_model']])
    counts={a:len(list((OUT/'runs/r1'/a).glob('*/result.json'))) for a in ['disabled','enabled']};assert counts=={'disabled':150,'enabled':150}
    assert len(list((OUT/'judges').glob('*/*/*/label.json')))==300 and not (OUT/'runs/r2').exists()
    save(OUT/'verification.json',{'trajectories':counts,'judges':300,'repetitions':1,'original_artifacts_and_app_unchanged':True,'knowledge_unchanged':True,'runtime_config_unchanged':True,'observed_models':models,'no_fallback_model':True})


async def full():
    path=str(OUT/'evaluator-source/scripts/eval_orchestration150.py')
    with (OUT/'execution.log').open('a',encoding='utf-8') as log,(OUT/'judging.log').open('a',encoding='utf-8') as jl:
        worker=await asyncio.create_subprocess_exec(sys.executable,path,'run',cwd=HOME,stdout=log,stderr=log)
        scorer=await asyncio.create_subprocess_exec(sys.executable,path,'judge',cwd=HOME,stdout=jl,stderr=jl)
        if await worker.wait():scorer.terminate();await scorer.wait();raise RuntimeError('Worker scheduler failed')
        if await scorer.wait():raise RuntimeError('Judge scheduler failed')
    wire();report();await verify();save(OUT/'completion.json',{'complete':True,'business_trajectories':300,'repetitions':1})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','worker','run','judge','report','verify','full']);p.add_argument('--arm',choices=['disabled','enabled']);p.add_argument('--case');a=p.parse_args()
    if a.stage=='worker':asyncio.run(worker(a.arm,a.case))
    elif a.stage=='report':report()
    else:asyncio.run(globals()[a.stage]())
