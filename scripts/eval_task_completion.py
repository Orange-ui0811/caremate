"""One new 150-case run; baseline/previous business trajectories are reused."""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
HOME=Path('E:/Mewhelp')
OUT=Path(os.environ.get('TASK_COMPLETION_OUTPUT',str(HOME/'data/verification/multi-intent-task-completion/20261105-200020')))
OLD=HOME/'data/multi-intent-eval/20261105-orchestration150-once'
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(HOME))
from scripts import rescore_multi120_evidence as scoring
from scripts import task_completion_scoring as revised
from scripts.score_multi120 import summarize


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.tmp')
    temp.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str),encoding='utf-8');temp.replace(p)


async def prepare():
    assert (OUT/'P0.json').exists() and not (OUT/'protocol.json').exists()
    from app.config import settings
    from app.db import base,workflow
    from app.core import model_pool
    from app.kb import milvus_client
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.engine import make_url
    from sqlalchemy import text
    assert read(OUT/'cases.json')==read(OLD/'cases.json')
    kb=await milvus_client.acall(lambda:milvus_client.get_client().query('knowledge',filter='',output_fields=['id','question','answer','section_path','content_type','category'],limit=16384))
    assert sorted(kb,key=lambda x:x['id'])==read(OUT/'knowledge.json'),'Knowledge changed since frozen150'
    shutil.copytree(ROOT/'app',OUT/'source/enabled/app',ignore=shutil.ignore_patterns('__pycache__'))
    (OUT/'evaluator-source/scripts').mkdir(parents=True)
    names=['__init__.py','eval_task_completion.py','task_completion_scoring.py','eval_multi120.py','multi120_cases.py',
           'score_multi120.py','rescore_multi120_evidence.py','multi120_evidence_protocol.py']
    for n in names:shutil.copy2(ROOT/'scripts'/n,OUT/'evaluator-source/scripts'/n)
    original=read(OLD/'protocol.json')
    config=original['model_config'];assert all(getattr(settings,k)==v for k,v in config.items())
    db='caremate_completion_'+OUT.name.replace('-','_')
    url=make_url(settings.test_database_url);admin=create_async_engine(url.set(database=None),isolation_level='AUTOCOMMIT')
    async with admin.begin() as conn:await conn.execute(text(f'CREATE DATABASE {db} CHARACTER SET utf8mb4'))
    await admin.dispose();engine=create_async_engine(url.set(database=db))
    async with engine.begin() as conn:
        await conn.run_sync(base.Base.metadata.create_all)
        await conn.execute(text('ALTER TABLE conversations AUTO_INCREMENT = 2400000000'))
    previous=base.engine;base.engine=engine
    try:await workflow.ensure_tables()
    finally:base.engine=previous;await engine.dispose()
    origin=Path(settings.model_pool_db_path)
    if not origin.is_absolute():origin=HOME/origin
    prod=model_pool.Store(origin)
    with prod.connect() as conn:disabled={r['resource'] for r in conn.execute('SELECT resource FROM slots WHERE disabled=1 OR invalid=1')}
    entries={}
    for ch,model in [('chat',config['chat_model']),('embed',config['embed_model']),('rerank',config['rerank_model'])]:
        specs=prod.credential_providers(ch,model)
        if specs is None:specs=model_pool.environment_providers(ch,model)
        specs=[p for p in specs if not p.fallback and p.fingerprint not in disabled and p.base_url.rstrip('/')==config[ch+'_base_url'].rstrip('/')]
        assert specs
        for p in specs:entries.setdefault(p.api_key,set()).add(ch)
    pool=model_pool.Store(OUT/'model_pool.sqlite3')
    pool.set_credentials([{'api_key':k,'channels':sorted(v)} for k,v in entries.items()])
    pool.set_policy(model_pool.Policy(**original['provider_policy']['policy']))
    settings_overrides={**original['evaluation_settings'],'multi_intent_completion_enabled':True}
    protocol={**original,'version':'task-completion-v1','arms':['enabled'],'scheduled_trajectories':150,
        'candidate_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'test_db_name':db,'model_pool_db_path':str(OUT/'model_pool.sqlite3'),'evaluation_settings':settings_overrides,
        'source_hashes':{str(p.relative_to(OUT/'source')):sha(p) for p in (OUT/'source').rglob('*.py')},
        'evaluator_hashes':{n:sha(OUT/'evaluator-source/scripts'/n) for n in names},
        'input_hashes':{n:sha(OUT/n) for n in ['cases.json','knowledge.json','bucket-map.json','runtime_settings.json','scoring-contract.json']},
        'reused':{'baseline':150,'previous':150},'judge_prompt':revised.PROMPT}
    save(OUT/'protocol.json',protocol)
    print('Frozen: 150 new business runs, original150 unchanged, baseline150 + previous150 reused.',flush=True)


def verify():
    p=read(OUT/'protocol.json')
    for n,h in p['source_hashes'].items():assert sha(OUT/'source'/n)==h,n
    for n,h in p['evaluator_hashes'].items():assert sha(OUT/'evaluator-source/scripts'/n)==h,n
    for n,h in p['input_hashes'].items():assert sha(OUT/n)==h,n
    for n,h in read(OUT/'P0.json')['old_input_hashes'].items():assert sha(Path(n))==h,n


def wire():
    sys.path.insert(0,str(OUT/'source/enabled'))
    from app.config import settings
    p=read(OUT/'protocol.json')
    for k,v in {**p['model_config'],**p['evaluation_settings']}.items():setattr(settings,k,v)
    settings.model_pool_db_path=p['model_pool_db_path'];settings.chat_failover_enabled=False;settings.langfuse_public_key=''
    from app.core import runtime_settings
    runtime_settings.SETTINGS_PATH=OUT/'runtime_settings.json'
    from scripts import eval_multi120 as old
    old.OUT=OUT;old.ROOT=HOME
    return old


async def worker(cid):
    verify();await wire().worker('enabled',cid,1)


async def run():
    verify();sem=asyncio.Semaphore(8)
    async def one(c):
        async with sem:
            dest=OUT/'runs/r1/enabled'/c['id'];dest.mkdir(parents=True,exist_ok=True)
            if (dest/'result.json').exists():return
            with (dest/'worker.log').open('w',encoding='utf-8') as log:
                proc=await asyncio.create_subprocess_exec(sys.executable,str(OUT/'evaluator-source/scripts/eval_task_completion.py'),'worker','--case',c['id'],cwd=HOME,stdout=log,stderr=log)
                try:rc=await asyncio.wait_for(proc.wait(),620)
                except TimeoutError:proc.kill();await proc.wait();rc=-999
            if not (dest/'result.json').exists():save(dest/'result.json',{'case_id':c['id'],'case':c,'arm':'enabled','rep':1,'status':'worker_error','exit_code':rc,'steps':[],'tickets':[],'actor':[],'authorized':[],'tool_audit':[]})
            print('runs',len(list((OUT/'runs').glob('*/*/*/result.json'))),'/150',c['id'],flush=True)
    await asyncio.gather(*(one(c) for c in read(OUT/'cases.json')))
    save(OUT/'business-complete.json',{'new_runs':150,'repetitions':1})


async def judge():
    wire();from app.core import llm,model_pool
    kb=[{'source_id':'kb:'+str(k['id']),'section':k['section_path'],'question':k['question'],'answer':k['answer']} for k in read(OUT/'knowledge.json')]
    kb_ids=[k['source_id'] for k in kb];sem=asyncio.Semaphore(4)
    async def one(arm,file):
        folder=OUT/'judges'/arm/file.parent.name
        if (folder/'label.json').exists():return
        async with sem:
            raw=read(file)
            old_arm='disabled' if arm=='baseline' else 'enabled'
            data=read(OLD/'judges/r1'/old_arm/file.parent.name/'input.json') if arm!='current' else scoring.payload(raw,file.with_name('checkpoints.sqlite'))
            save(folder/'input.json',data)
            if arm!='current':
                old_label=read(OLD/'judges/r1'/old_arm/file.parent.name/'label.json')
                if old_label.get('status')=='ok':
                    checked=revised.validate(old_label,data,kb_ids);checked['status']='ok'
                    if not revised.needs_review(checked,data):save(folder/'label.json',{**checked,'provenance':'reused_valid_label'});return
            if not any(e['role']=='assistant' for e in data['timeline']):save(folder/'label.json',{'status':'no_answer','items':[],'claims':[]});return
            errors=[];last=None
            for attempt in range(2):
                try:
                    with model_pool.call_context(f'completion-judge:{arm}:{raw["case_id"]}:{attempt}',priority='background'):
                        answer=await llm.structured(scoring.Verdict,temperature=0).ainvoke([
                            {'role':'system','content':revised.PROMPT},
                            {'role':'user','content':json.dumps({'frozen_knowledge':kb,**data},ensure_ascii=False)}])
                    last=revised.validate(answer.model_dump(),data,kb_ids);last.update(status='ok',errors=errors,provenance='new_review')
                    save(folder/f'attempt-{attempt}.json',last)
                    if not revised.needs_review(last,data):break
                except Exception as e:errors.append(str(e))
            save(folder/'label.json',last or {'status':'judge_error','items':[],'claims':[],'errors':errors})
    while True:
        files=[('baseline',f) for f in (OLD/'runs/r1/disabled').glob('*/result.json')]
        files += [('previous',f) for f in (OLD/'runs/r1/enabled').glob('*/result.json')]
        files += [('current',f) for f in (OUT/'runs/r1/enabled').glob('*/result.json')]
        await asyncio.gather(*(one(a,f) for a,f in files))
        count=len(list((OUT/'judges').glob('*/*/label.json')))
        save(OUT/'progress.json',{'judges':count,'new_runs':len(files)-300,'expected_judges':450})
        print('judges',count,'new runs',len(files)-300,flush=True)
        if count==450:break
        await asyncio.sleep(8)
    from app.core.model_transport import close_clients
    await close_clients()


def report():
    rows=[];mapping=read(OUT/'bucket-map.json')
    for arm,folder in [('baseline',OLD/'runs/r1/disabled'),('previous',OLD/'runs/r1/enabled'),('current',OUT/'runs/r1/enabled')]:
        for f in folder.glob('*/result.json'):
            r=read(f);labels=OUT/'judges'/arm/r['case_id']
            row=revised.evaluate(r,read(labels/'label.json'),read(labels/'input.json'));row.update(mapping[r['case_id']]);row['arm']=arm;rows.append(row)
    summaries={a:{split:summarize([r for r in rows if r['arm']==a and (split=='all' or r['split']==split)]) for split in ['all','main','control']} for a in ['baseline','previous','current']}
    buckets={a:{b:summarize([r for r in rows if r['arm']==a and r['bucket']==b]) for b in sorted({r['bucket'] for r in rows})} for a in summaries}
    save(OUT/'summary.json',{'summary':summaries,'by_bucket':buckets,'rows':rows})
    save(OUT/'failure-index.json',[r for r in rows if r['arm']=='current' and r['success'] is not True])
    save(OUT/'completion.json',{'complete':True,'business_runs_new':150,'business_runs_reused':300,'scored_trajectories':450})
    print(json.dumps(summaries,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','worker','run','judge','report','verify']);p.add_argument('--case');a=p.parse_args()
    if a.stage=='worker':asyncio.run(worker(a.case))
    elif a.stage in {'report','verify'}:globals()[a.stage]()
    else:asyncio.run(globals()[a.stage]())
