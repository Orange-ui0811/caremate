"""Original120 x3: archived single-intent baseline and fresh shared-Agent candidate only."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
HOME=Path('E:/Mewhelp')
BASE=HOME/'data/multi-intent-eval/20261103-business120'
OUT=Path(os.environ.get('SHARED120_OUTPUT',str(HOME/'data/multi-intent-eval/20261105-shared-agent120-v2')))
sys.path.insert(0,str(ROOT))
if (OUT/'evaluator-source/scripts').is_dir():
    sys.path.insert(0,str(OUT/'evaluator-source'))

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8');tmp.replace(p)

async def prepare():
    if OUT.exists():raise RuntimeError('Do not overwrite a frozen experiment')
    from app.config import settings
    from app.db import base,workflow
    from app.kb import milvus_client
    from sqlalchemy import text
    from sqlalchemy.engine import make_url
    from sqlalchemy.ext.asyncio import create_async_engine
    prior=read(BASE/'protocol.json')
    assert sha(ROOT/'tests/data/eval_ch04.jsonl')==prior['original300_sha256']
    kb=await milvus_client.acall(lambda:milvus_client.get_client().query('knowledge',filter='',output_fields=['id','question','answer','section_path','content_type','category'],limit=16384))
    kb.sort(key=lambda x:x['id']);assert kb==read(BASE/'knowledge.json')
    baseline_files=[*BASE.glob('runs/*/disabled/*/result.json'),*BASE.glob('judges-chronological/*/disabled/*.json')]
    assert len(baseline_files)==720
    OUT.mkdir(parents=True)
    for name in ['cases.json','knowledge.json','scoring.json','CASES.md']:shutil.copy2(BASE/name,OUT/name)
    shutil.copytree(ROOT/'app',OUT/'source/app',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(HOME/'data/runtime_settings.json',OUT/'runtime_settings.json')
    scripts=['__init__.py','eval_shared_agent_120.py','eval_multi120.py','multi120_evidence_protocol.py','score_multi120.py','score_multi120_chronological.py','eval_subflows_p4.py','eval_subflows_p4_reuse.py']
    (OUT/'evaluator-source/scripts').mkdir(parents=True)
    (OUT/'evaluator-source/scripts/__init__.py').write_text('',encoding='utf-8')
    for name in scripts:shutil.copy2(ROOT/'scripts'/name,OUT/'evaluator-source/scripts'/name)
    shutil.copy2(__file__,OUT/'evaluator-source/eval_shared_agent_120.py')
    db='caremate_'+OUT.name.replace('-','_');assert db.replace('_','').isalnum()
    url=make_url(settings.database_url)
    admin=create_async_engine(url.set(database=None),isolation_level='AUTOCOMMIT')
    async with admin.begin() as c:await c.execute(text(f'CREATE DATABASE {db} CHARACTER SET utf8mb4'))
    await admin.dispose();engine=create_async_engine(url.set(database=db))
    async with engine.begin() as c:
        await c.run_sync(base.Base.metadata.create_all)
        await c.execute(text('ALTER TABLE conversations AUTO_INCREMENT = 2200000000'))
    old=base.engine;base.engine=engine
    try:await workflow.ensure_tables()
    finally:base.engine=old;await engine.dispose()
    families={}
    for c in read(OUT/'cases.json'):families.setdefault(c['family_id'],c['id'])
    from scripts.eval_subflows_p4 import QUALITY_PROMPT
    config={k:getattr(settings,k) for k in prior['model_config']}
    assert all(config[k]==v for k,v in prior['model_config'].items())
    p={'version':'shared-agent120-baseline-reused','repetitions':3,'arms':['enabled'],
       'baseline_reused':360,'candidate_expected':360,'candidate_shared_agent':True,
       'candidate_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'model_config':config,'groups':prior['groups'],'test_db_name':db,
       'model_pool_db_path':str(OUT/'model_pool.sqlite3'),
       'evaluation_settings':{'model_pool_background_limit':3,'model_pool_queue_timeout':120,'model_pool_turn_timeout':480},
       'original300_sha256':prior['original300_sha256'],'request_namespace':'shared120-20261105',
       'quality':{'prompt':QUALITY_PROMPT,'sample_case_ids':list(families.values()),'rep':1,'sampling':'Same40 family representatives from original120; no extra cases'},
       'baseline_hashes':{str(f.relative_to(BASE)):sha(f) for f in baseline_files},
       'source_hashes':{str(f.relative_to(OUT/'source')):sha(f) for f in (OUT/'source').rglob('*') if f.is_file()},
       'evaluator_hashes':{'scripts/'+n:sha(OUT/'evaluator-source/scripts'/n) for n in scripts},
       'limits':['Historical baseline, not contemporaneous randomized pairs.',
                 'Original120 includes6 pending-text followups and3 cancel-order-selection cases. Unsupported outcomes remain failures, never excluded.',
                 'Same original deterministic business tool fixtures; real chat/embed/rerank; isolated database, checkpoints and model-pool state.',
                 'Model judge is not independent human ground truth. No best-of-three selection.',
                 'Graph/form API evaluation, not browser/SSE transport or first-token benchmark.']}
    for name in ['cases','knowledge','scoring']:p[name+'_sha256']=sha(OUT/(name+'.json'))
    p['runtime_settings_sha256']=sha(OUT/'runtime_settings.json')
    p['preflight_correction']='Prior20261105-shared-agent120 pilot had invalid queue_timeout180>120; no formal trajectories. Original records retained. This experiment uses valid120.'
    save(OUT/'protocol.json',p)
    old_quality=HOME/'data/multi-intent-eval/20261104-p4-subflows/quality/disabled'
    shutil.copytree(old_quality,OUT/'quality/disabled')
    print('Frozen:120 cases x3 candidate only; baseline360 reused; no auxiliary set',flush=True)

def verify_inputs():
    p=read(OUT/'protocol.json')
    for name in ['cases','knowledge','scoring']:assert sha(OUT/(name+'.json'))==p[name+'_sha256']
    assert sha(OUT/'runtime_settings.json')==p['runtime_settings_sha256']
    for name,h in p['source_hashes'].items():assert sha(OUT/'source'/name)==h,name
    for name,h in p['evaluator_hashes'].items():assert sha(OUT/'evaluator-source'/name)==h,name

def wire():
    sys.path.insert(0,str(OUT/'source'))
    sys.path.insert(1,str(OUT/'evaluator-source'))
    from app.config import settings
    p=read(OUT/'protocol.json')
    for k,v in p['model_config'].items():setattr(settings,k,v)
    for k,v in p['evaluation_settings'].items():setattr(settings,k,v)
    settings.multi_intent_shared_agent_enabled=True
    settings.langfuse_public_key='';settings.model_pool_db_path=p['model_pool_db_path']
    from app.core import runtime_settings
    runtime_settings.SETTINGS_PATH=OUT/'runtime_settings.json'
    from scripts import eval_multi120 as old
    old.OUT=OUT;old.ROOT=HOME
    return old

async def worker(case_id,rep,pilot):
    verify_inputs();await wire().worker('enabled',case_id,rep,pilot=pilot)

async def run(pilot=False):
    verify_inputs();sem=asyncio.Semaphore(6)
    cs=read(OUT/'cases.json')
    if pilot:cs=[c for c in cs if c['id'] in ['M1-05','M5-02','M6-01','M7-01']]
    async def one(c,rep):
        async with sem:
            directory=OUT/('pilot' if pilot else 'runs')/f'r{rep}'/'enabled'/c['id'];directory.mkdir(parents=True,exist_ok=True)
            if (directory/'result.json').exists():return
            with (directory/'worker.log').open('w',encoding='utf-8') as log:
                args=[sys.executable,str(OUT/'evaluator-source/eval_shared_agent_120.py'),'worker','--case',c['id'],'--rep',str(rep)]
                if pilot:args.append('--pilot')
                proc=await asyncio.create_subprocess_exec(*args,cwd=HOME,stdout=log,stderr=log,env={**os.environ,'SHARED120_OUTPUT':str(OUT),'PYTHONIOENCODING':'utf-8'})
                try:rc=await asyncio.wait_for(proc.wait(),620)
                except TimeoutError:proc.kill();await proc.wait();rc=-999
            if not (directory/'result.json').exists():save(directory/'result.json',{'case':c,'case_id':c['id'],'rep':rep,'arm':'enabled','status':'worker_error','exit_code':rc,'steps':[],'tickets':[],'actor':[],'authorized':[],'tool_audit':[]})
            print(len(list((OUT/('pilot' if pilot else 'runs')).glob('*/enabled/*/result.json'))),'/'+str(len(cs)*(1 if pilot else 3)),rep,c['id'],'exit',rc,flush=True)
    ordered=sorted(cs,key=lambda c:(int(c['id'].split('-')[1]),c['group']))
    for rep in ([1] if pilot else [1,2,3]):await asyncio.gather(*(one(c,rep) for c in ordered))

async def judge():
    wire()
    from scripts import score_multi120_chronological as scorer
    scorer.OUT=OUT;sem=asyncio.Semaphore(2)
    async def one(f):
        async with sem:await scorer.judge(f)
    while True:
        fs=list((OUT/'runs').glob('*/enabled/*/result.json'))
        missing=[f for f in fs if not (OUT/'judges-chronological'/f.parents[2].name/'enabled'/(f.parent.name+'.json')).exists()]
        await asyncio.gather(*(one(f) for f in missing))
        n=len(list((OUT/'judges-chronological').glob('*/enabled/*.json')))
        save(OUT/'progress.json',{'runs':len(fs),'judges':n,'baseline_reused':360})
        print('runs',len(fs),'judges',n,'baseline reused360',flush=True)
        if len(fs)==n==360:break
        await asyncio.sleep(8)
    from app.core.model_transport import close_clients
    await close_clients()

def reuse_module():
    from scripts import eval_subflows_p4 as core,eval_subflows_p4_reuse as reuse
    core.OUT=OUT;core.BASE=BASE;core.HOME=HOME;core.wire=wire;core.verify_inputs=verify_inputs
    reuse.OUT=OUT;reuse.BASE=BASE
    return reuse

async def quality():
    await reuse_module().quality()

def report():
    reuse_module().report()
    f=OUT/'REPORT.md';s=f.read_text(encoding='utf-8').replace('P3最新版','当前共享Agent版').replace('P3最新','当前共享Agent').replace('P3通过','新版通过').replace('P3 commit','当前冻结提交').replace('P3。','当前共享Agent版。')
    s += '\n本轮只新增120场景×3轮共360条业务轨迹；未运行disabled，也没有加入附加集。新版task_form按公开卡片经原表单API提交并恢复，取消与确认仍独立记录。当前不支持的交互仍纳入全量统计。\n'
    f.write_text(s,encoding='utf-8')

async def verify():
    verify_inputs();p=read(OUT/'protocol.json')
    assert all(sha(BASE/n)==h for n,h in p['baseline_hashes'].items())
    from app.kb import milvus_client
    kb=await milvus_client.acall(lambda:milvus_client.get_client().query('knowledge',filter='',output_fields=['id','question','answer','section_path','content_type','category'],limit=16384));kb.sort(key=lambda x:x['id'])
    value={'baseline_reused':360,'candidate_runs':len(list((OUT/'runs').glob('*/enabled/*/result.json'))),'candidate_judges':len(list((OUT/'judges-chronological').glob('*/enabled/*.json'))),'no_new_baseline':not list((OUT/'runs').glob('*/disabled/*/result.json')),'knowledge_unchanged':kb==read(OUT/'knowledge.json'),'baseline_artifacts_unchanged':True,'cases_unchanged':sha(OUT/'cases.json')==sha(BASE/'cases.json'),'scoring_unchanged':sha(OUT/'scoring.json')==sha(BASE/'scoring.json')}
    save(OUT/'verification.json',value)
    assert value['candidate_runs']==value['candidate_judges']==360 and all(value[k] for k in ['no_new_baseline','knowledge_unchanged','cases_unchanged','scoring_unchanged'])

async def full():
    with (OUT/'execution.log').open('a',encoding='utf-8') as log,(OUT/'judging.log').open('a',encoding='utf-8') as jl:
        env={**os.environ,'SHARED120_OUTPUT':str(OUT),'PYTHONIOENCODING':'utf-8'}
        path=str(OUT/'evaluator-source/eval_shared_agent_120.py')
        proc=await asyncio.create_subprocess_exec(sys.executable,path,'run',cwd=HOME,stdout=log,stderr=log,env=env)
        jp=await asyncio.create_subprocess_exec(sys.executable,path,'judge',cwd=HOME,stdout=jl,stderr=jl,env=env)
        rc=await proc.wait()
        if rc:jp.terminate();await jp.wait();raise RuntimeError('Execution scheduler failed')
        if await jp.wait():raise RuntimeError('Scorer failed')
    await quality();report();await verify();save(OUT/'completion.json',{'complete':True,'baseline_reused':360,'new_trajectories':360})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','pilot','worker','run','judge','quality','report','verify','full']);p.add_argument('--case');p.add_argument('--rep',type=int,default=1);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    if a.stage=='report':report()
    elif a.stage=='worker':asyncio.run(worker(a.case,a.rep,a.pilot))
    elif a.stage=='pilot':asyncio.run(run(True))
    else:asyncio.run(globals()[a.stage]())
