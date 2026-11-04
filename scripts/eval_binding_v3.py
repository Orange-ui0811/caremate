"""Regression of task-local order binding on the unchanged120 scenarios.

Reuse the recorded enabled V2 baseline; run V3 freshly, three repetitions.
Legacy execution and chronological scoring functions are reused unchanged.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BASE=ROOT/'data/multi-intent-eval/20261103-business120'
OUT=ROOT/'data/multi-intent-eval/20261103-binding-v3'
TAG='mi120bindingv3'


def read(p):return json.loads(p.read_text('utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str),'utf-8');tmp.replace(p)


def wire():
    os.environ['MULTI120_OUTPUT']=str(OUT)
    sys.path.insert(0,str(OUT/'source'))
    from scripts import eval_multi120 as legacy
    legacy.OUT=OUT
    from app.core import model_pool
    original=model_pool.call_context
    def scoped(request_id=None,*args,**kwargs):
        if isinstance(request_id,str) and request_id.startswith('mi120formal'):
            request_id=TAG+request_id[len('mi120formal'):]
        return original(request_id,*args,**kwargs)
    model_pool.call_context=scoped
    return legacy


async def prepare():
    if (OUT/'protocol.json').exists():raise RuntimeError('Already frozen')
    from app.config import settings
    from app.kb import milvus_client
    from app.db import base,workflow
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy import text
    kb=await milvus_client.acall(lambda:milvus_client.get_client().query('knowledge',filter='',
        output_fields=['id','question','answer','section_path','content_type','category'],limit=16384))
    kb.sort(key=lambda r:r['id'])
    assert kb==read(BASE/'knowledge.json'),'Knowledge changed; stop comparison'
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ('cases.json','knowledge.json','scoring.json','knowledge-conflict-audit.json','CASES.md'):
        shutil.copy2(BASE/name,OUT/name)
    for name in ('summary-source-reviewed.json','summary-chronological.json'):
        shutil.copy2(BASE/name,OUT/('baseline-'+name))
    shutil.copytree(ROOT/'app',OUT/'source/app',ignore=shutil.ignore_patterns('__pycache__'))
    old=read(BASE/'protocol.json');dbname='caremate_binding_v3_20261103'
    server=settings.test_database_url.rsplit('/',1)[0]
    admin=create_async_engine(server,isolation_level='AUTOCOMMIT')
    async with admin.connect() as conn:await conn.execute(text(f'CREATE DATABASE IF NOT EXISTS {dbname} CHARACTER SET utf8mb4'))
    await admin.dispose();engine=create_async_engine(server+'/'+dbname)
    async with engine.begin() as conn:
        await conn.run_sync(base.Base.metadata.create_all)
        await conn.execute(text('ALTER TABLE conversations AUTO_INCREMENT = 3000000000'))
    live=base.engine;base.engine=engine
    try:await workflow.ensure_tables()
    finally:base.engine=live;await engine.dispose()
    helpers=['eval_multi120.py','score_multi120.py','score_multi120_chronological.py','eval_binding_v3.py']
    (OUT/'evaluator-source').mkdir(exist_ok=True)
    for name in helpers:shutil.copy2(ROOT/'scripts'/name,OUT/'evaluator-source'/name)
    save(OUT/'protocol.json',{
        'version':'order-binding-v3','model_config':old['model_config'],'test_db_name':dbname,
        'repetitions':3,'arms':['enabled'],'baseline_origin':str(BASE),'request_namespace':TAG,
        'cases_sha256':sha(OUT/'cases.json'),'knowledge_sha256':sha(OUT/'knowledge.json'),
        'scoring_sha256':sha(OUT/'scoring.json'),'original300_sha256':old['original300_sha256'],
        'source_hashes':{str(p.relative_to(OUT/'source')):sha(p) for p in (OUT/'source').rglob('*.py')},
        'source_asset_sha256':sha(OUT/'source/app/static/index.html'),
        'evaluation_script_hashes':{name:sha(ROOT/'scripts'/name) for name in helpers},
        'baseline_result_hashes':{str(p.relative_to(BASE)):sha(p) for p in (BASE/'runs').glob('*/enabled/*/result.json')},
        'baseline_summary_hashes':{name:sha(BASE/name) for name in ('summary-source-reviewed.json','summary-chronological.json')},
        'groups':old['groups'],'knowledge_rows':len(kb),
        'comparison':'V2 enabled recorded trajectories versus fresh V3 enabled. Same120 questions, goals, actor, fixtures, scoring and models. Baseline is NOT the disabled single-intent arm.',
        'limits':['Regression/development set has been examined to guide the fix, not an independent blind test.',
            'Baseline reused from earlier calls, not freshly randomized contemporaneous pairs.',
            'Known four data/label issues retained and sensitivity reported, not fixed for score gains.',
            'No production feature flag/configuration/knowledge change. No new browser/SSE acceptance.']})
    print('Frozen120, three fresh V3 repetitions; recorded V2 baseline retained',flush=True)


async def run():
    p=read(OUT/'protocol.json')
    for name,h in p['source_hashes'].items():assert sha(OUT/'source'/name)==h
    for name in ('cases','knowledge','scoring'):assert sha(OUT/(name+'.json'))==p[name+'_sha256']
    cases=sorted(read(OUT/'cases.json'),key=lambda c:(int(c['id'].split('-')[1]),c['group']))
    sem=asyncio.Semaphore(8)
    async def one(c,rep):
      async with sem:
        directory=OUT/'runs'/f'r{rep}'/'enabled'/c['id'];directory.mkdir(parents=True,exist_ok=True)
        if (directory/'result.json').exists():return
        with (directory/'worker.log').open('w',encoding='utf-8') as log:
            proc=await asyncio.create_subprocess_exec(sys.executable,str(Path(__file__).resolve()),'worker',
                '--case',c['id'],'--rep',str(rep),cwd=str(ROOT),stdout=log,stderr=log,
                env={**os.environ,'PYTHONIOENCODING':'utf-8'})
            rc=await proc.wait()
        if not (directory/'result.json').exists():save(directory/'result.json',{
            'case':c,'case_id':c['id'],'arm':'enabled','rep':rep,'status':'worker_error','exit_code':rc,
            'steps':[],'tickets':[],'actor':[],'authorized':[]})
        print(len(list((OUT/'runs').glob('*/*/*/result.json'))),'/360',rep,c['id'],'exit',rc,flush=True)
    for rep in range(1,4):await asyncio.gather(*(one(c,rep) for c in cases))


async def judge_follow():
    wire()
    from scripts import score_multi120_chronological as judge
    judge.OUT=OUT
    save(OUT/'judge-protocol.json',{'prompt':judge.PROMPT,
        'source_sha256':sha(ROOT/'scripts/score_multi120_chronological.py'),
        'method':'Unchanged chronological scorer, same goals/facts/actor. Request namespace adapter prevents replay of prior experiment model responses.',
        'source_reviews':'Recorded baseline source-reviewed score shown separately; no baseline review IDs copied onto new model labels.'})
    sem=asyncio.Semaphore(6)
    async def one(p):
        async with sem:await judge.judge(p)
    while True:
        files=list((OUT/'runs').glob('*/*/*/result.json'))
        missing=[p for p in files if not (OUT/'judges-chronological'/p.parents[2].name/'enabled'/(p.parent.name+'.json')).exists()]
        await asyncio.gather(*(one(p) for p in missing))
        print('judged',len(list((OUT/'judges-chronological').glob('*/*/*.json'))),'runs',len(files),flush=True)
        if len(files)>=360:break
        await asyncio.sleep(10)
    from app.core.model_transport import close_clients
    await close_clients()


def report():
    from scripts.score_multi120_chronological import evaluate
    from scripts.score_multi120 import summarize
    baseline=read(OUT/'baseline-summary-chronological.json')
    before=[r for r in baseline['rows'] if r['arm']=='enabled'];after=[]
    for p in (OUT/'runs').glob('*/*/*/result.json'):
        r=read(p);jp=OUT/'judges-chronological'/f'r{r["rep"]}'/'enabled'/(r['case_id']+'.json')
        after.append(evaluate(r,read(jp) if jp.exists() else None))
    a,b=summarize(before),summarize(after)
    index={(r['rep'],r['case_id']):r for r in before}
    wins=[r for r in after if r['success'] is True and index[(r['rep'],r['case_id'])]['success'] is False]
    losses=[r for r in after if r['success'] is False and index[(r['rep'],r['case_id'])]['success'] is True]
    save(OUT/'summary.json',{'before':a,'after':b,'before_rows':before,'after_rows':after,
        'paired_improved_n':len(wins),'paired_regressed_n':len(losses),
        'repetitions':{str(i):{'before':summarize([r for r in before if r['rep']==i]),
            'after':summarize([r for r in after if r['rep']==i])} for i in (1,2,3)}})
    def rate(s):return f"{s['success_n']}/{s['n']}"+(f"（待定{s['unknown_n']}）" if s['unknown_n'] else '')
    lines=['# 订单绑定与澄清V3：同120场景回归','',
        '对比双方均开启多任务。改进前复用已保存的V2三轮轨迹，V3三轮全部重新调用；不是原单主意图路径与多任务路径的对比。',
        f'固定120题及标注、业务数据、模型和时序评分规则；V3完成{len(after)}/360条。本套题已用于诊断和开发，是回归集。','',
        '| 指标 | V2改进前 | V3改进后 |','|---|---:|---:|',
        f"| 多余选单（实际进入选单操作） | {a['hard'].get('extra_selection_n',0)} | {b['hard'].get('extra_selection_n',0)} |",
        f'| 场景验收：相同时序自动评分 | {rate(a)} | {rate(b)} |',
        f"| 必需事项目标终态 | {a['terminal_n']}/{a['task_n']}，待定{a['terminal_unknown_n']} | {b['terminal_n']}/{b['task_n']}，待定{b['terminal_unknown_n']} |"]
    for key,label in [('wrong_refund_order_n','错订单退款工单'),('wrong_preview_n','错误/未要求的预览'),
        ('duplicate_ticket_n','重复工单'),('unauthorized_ticket_n','未确认的写入'),('forbidden_read_n','禁止的查询'),('forbidden_ticket_n','禁止类型工单')]:
        lines.append(f"| {label} | {a['hard'].get(key,0)} | {b['hard'].get(key,0)} |")
    lines+=['','| 分组 | V2通过 | V3通过 |','|---|---:|---:|']
    for g,label in read(OUT/'protocol.json')['groups'].items():
        x=a['by_group'].get(g,{'success_n':0,'n':0});y=b['by_group'].get(g,{'success_n':0,'n':0})
        lines.append(f"| {label} | {x['success_n']}/{x['n']} | {y['success_n']}/{y['n']} |")
    old=read(OUT/'baseline-summary-source-reviewed.json')['summary']['enabled']
    lines+=['','## 口径与限制','',
        f"- 上轮对外的{old['success_n']}/360包含11条助手来源复核；相同自动评分基线是{a['success_n']}/360。此处主表用自动评分对自动评分，不把旧复核标签套在新答复上。", 
        '- 旧7次来自单意图关闭模式，88次来自V2开启模式，处理事项数量不同；这次直接比较V2与V3均开启多任务。',
        '- 88次根因：73次任务原文有订单但短引用不含动作词被清空，9次同消息明确指代未解析，6次并列退款号码被清空。固定旧任务输出的离线重放可恢复88次；它不是新真实接口的成绩。',
        '- 新绑定仅从当前任务对应原文小句取实体，支持相邻订单主语、明确“这笔订单”及显式并列订单；积分/金额/电话/日期、否定、其他任务号码不作为退款目标。缺参、歧义或非本人订单继续选单。',
        '- 用户选择的订单按任务保存，且仍做归属复验与独立提交确认。没有把整句唯一数字或唯一本人订单当默认退款目标。',
        '- 旧基线来自较早调用，未同期重新采样；模型/API有波动。回归结果不是独立盲测，也不能全部差异归因于一个规则。',
        '- 原题及已知4个资料/标注问题保留，不按新版结果改题或答案；详见knowledge-conflict-audit.json。模型裁判不是人工真值，待定不当成功。',
        '- 业务订单物流为同一固定夹具，模型/向量/重排是真实接口，写入隔离MySQL。生产配置、知识库及功能开关未改；本轮未新增完整浏览器/SSE故障验收。',
        f'- 逐次配对改善{len(wins)}条、退步{len(losses)}条；不挑最好一次。','',
        '## 退步配对','']
    for r in losses:lines.append(f"- 第{r['rep']}次 {r['case_id']}：{r['terminal']}")
    lines+=['','文件：cases.json、protocol.json、offline-binding-replay.json、summary.json、runs/、judges-chronological/、verification.json。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n','utf-8')
    print({'before':(a['success_n'],a['unknown_n'],a['hard'].get('extra_selection_n')),
           'after':(b['success_n'],b['unknown_n'],b['hard'].get('extra_selection_n'))})


async def verify():
    from app.config import settings
    from app.kb import milvus_client
    p=read(OUT/'protocol.json')
    kb=await milvus_client.acall(lambda:milvus_client.get_client().query('knowledge',filter='',
        output_fields=['id','question','answer','section_path','content_type','category'],limit=16384))
    kb.sort(key=lambda r:r['id'])
    result={'runs':len(list((OUT/'runs').glob('*/*/*/result.json'))),
        'judges':len(list((OUT/'judges-chronological').glob('*/*/*.json'))),
        'cases_identical_to_previous':sha(OUT/'cases.json')==sha(BASE/'cases.json'),
        'scoring_identical_to_previous':sha(OUT/'scoring.json')==sha(BASE/'scoring.json'),
        'original300_unchanged':sha(ROOT/'tests/data/eval_ch04.jsonl')==p['original300_sha256'],
        'knowledge_unchanged':kb==read(OUT/'knowledge.json'),'production_feature_enabled':settings.multi_intent_enabled,
        'baseline_changed':[n for n,h in p['baseline_result_hashes'].items() if sha(BASE/n)!=h],
        'frozen_app_changed':[n for n,h in p['source_hashes'].items() if sha(OUT/'source'/n)!=h],
        'current_app_differs_from_snapshot':[str(f.relative_to(OUT/'source')) for f in (OUT/'source/app').rglob('*.py') if f.read_bytes()!=(ROOT/f.relative_to(OUT/'source')).read_bytes()],
        'harness_changed':[n for n,h in p['evaluation_script_hashes'].items() if sha(ROOT/'scripts'/n)!=h],
        'tests':'250 passed full affected business/API regression before final quoted-product refinement; final affected core+graph tests recorded separately',
        'production_restarted':False,'knowledge_or_customer_config_modified':False}
    save(OUT/'verification.json',result);print(result)
    assert result['runs']==result['judges']==360
    assert result['cases_identical_to_previous'] and result['scoring_identical_to_previous'] and result['original300_unchanged'] and result['knowledge_unchanged']
    assert not(result['baseline_changed'] or result['frozen_app_changed'] or result['current_app_differs_from_snapshot'] or result['harness_changed'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','worker','judge','report','verify'])
    p.add_argument('--case');p.add_argument('--rep',type=int,default=1);a=p.parse_args()
    if a.stage=='report':report()
    elif a.stage=='worker':
        legacy=wire();asyncio.run(legacy.worker('enabled',a.case,a.rep))
    else:asyncio.run({'prepare':prepare,'run':run,'judge':judge_follow,'verify':verify}[a.stage]())
