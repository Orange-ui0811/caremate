"""Frozen 120-scenario, paired real-model evaluation. Never imports live app in workers."""
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
OUT=Path(os.environ.get('MULTI120_OUTPUT',str(ROOT/'data/multi-intent-eval/20261103-business120')))
ARMS=('disabled','enabled')


def read(p):return json.loads(p.read_text('utf-8'))
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str),'utf-8');temp.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(s):return ''.join(s.split())


def ticket_sequence_namespace(arm,case_index,rep,case_count=120):
    if arm not in ARMS or not 0<=case_index<case_count or rep not in (1,2,3):raise ValueError('Invalid trajectory')
    return (((rep-1)*2+ARMS.index(arm))*case_count+case_index+1)*1000


SCORING={
 'version':'multi120-v1',
 'unit':'120 constructed scenarios in 40 semantic families; 8 groups of15; each arm3 repetitions.',
 'comparison':'Same frozen current app, MULTI_INTENT_ENABLED=false versus true; fresh trajectories for both.',
 'task_response':'Source-quoted model judgement of each user goal; no internal task_plan as success evidence.',
 'knowledge':'All predeclared requested facts must be correct and supported by frozen source sections. Semantic paraphrases accepted; mechanical point matching is diagnostic only.',
 'order_logistics':'Correct requested order facts in visible answers AND verified successful tool call for that target. No credit for user text or claims of DONE.',
 'refund_terminal':'Exactly one SQL refund-review ticket for each expected order, after corresponding explicit confirmation; not actual funds refund.',
 'complaint_terminal':'Queries explicitly request a complaint ticket. Exactly one SQL complaint ticket after independent confirmation; an apology/offer alone is not terminal.',
 'cancel_terminal':'No ticket for cancelled item, and an observed cancel action or explicit followup denial; sibling goals still required.',
 'normal_scenario':'All predeclared goals reach terminal and all hard constraints pass; source errors fail. Missing semantic judgement is unknown, reported as bounds, never dropped silently.',
 'fault_scenario':'Business completion is reported separately from correct failure handling. Injected unavailable-logistics goal requires honest unavailability and no fabricated success; complaint still must finish.',
 'hard_constraints':['No unconfirmed writes','No wrong-order refund ticket','No duplicate ticket per intended goal',
                     'No prohibited successful query or ticket type','No writes for consultation-only or quoted requests'],
 'actor':'Predeclared target per task; selection uses public task_kind when present or unambiguous visible context. Never reads task_plan to choose or rescue missing goals. Wrong previews rejected, not blindly approved.',
 'clarification':'Explicit, valid supplied order should not require selection; record extra selection separately from terminal success. Missing target is supplied through offered selection or one targeted textual clarification.',
 'followup':'Sent before any confirmation on initial response; explicit changed/cancelled refund and continued complaint are scored against final goals.',
 'recovery':'Reopen SQLite connection/compiled graph; duplicate real resume/form requests; injected logistics error. Not an OS crash, browser/SSE or full HTTP fault matrix.',
 'judge':'Same real model at temperature0, blind to arm/task_plan. Exact source quote required for positive semantic labels; raw attempts retained. Not independent human gold.',
 'repetitions':'Three fresh workflow runs per scenario per arm. Report per-repeat, pooled, by-group and paired win/loss; no best-of-three selection.',
 'latency':'Server processing and scripted interactions; no human think time. Includes queue/network and real RAG; not model-only compute.',
 'limits':['Constructed, author-visible acceptance set, not external blind test.',
           'Semantic families/ID variants are correlated; do not treat360 runs as360 independent cases.',
           'Business backends are deterministic fixtures; real chat/embedding/rerank and local isolated SQL.',
           'D_absent labels and selected facts validated against frozen knowledge, original300 unchanged.']}


async def prepare():
    from scripts.multi120_cases import build_cases,original,GROUPS
    from app.config import settings
    from app.kb import milvus_client
    from app.db import base,workflow
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy import text
    if (OUT/'protocol.json').exists():raise RuntimeError('Already frozen; do not overwrite inputs.')
    cases=build_cases();assert len(cases)==120 and len({c['query'] for c in cases})==120
    kb=await milvus_client.acall(lambda:milvus_client.get_client().query('knowledge',filter='',
        output_fields=['id','question','answer','section_path','content_type','category'],limit=16384))
    kb.sort(key=lambda r:r['id']);errors=[]
    for c in cases:
      assert c['source_question_ids'] and all(s in original() for s in c['source_question_ids'])
      for g in c['goals']:
        if g['kind']=='knowledge':
            source=g['source'];groups=source.get('expect_sections_all') or [source['expect_section']]
            evidence=[]
            for aliases in groups:
                matches=[r for r in kb if any(a in r['section_path'] for a in aliases)]
                if not matches:errors.append((c['id'],'missing section',aliases))
                evidence+=matches
            g['evidence']=list({r['id']:r for r in evidence}.values())
            corpus=norm(''.join(r['question']+r['answer'] for r in evidence))
            for p in g['required_points']:
                if norm(p) not in corpus:errors.append((c['id'],'unsupported gold point',p))
        elif g['kind']=='refuse':
            g['evidence']=[]
            g['gold_note']='Frozen KB has no supported answer for this original D-bucket service. Do not invent availability or price; allow clarification/human referral.'
    if errors:
        save(OUT/'draft-validation-errors.json',errors);raise ValueError(errors)
    # New isolated database; do not change application configuration or production tables.
    server=settings.test_database_url.rsplit('/',1)[0];dbname='caremate_multi120_formal_20261103'
    admin=create_async_engine(server,isolation_level='AUTOCOMMIT')
    async with admin.connect() as conn:await conn.execute(text(f'CREATE DATABASE IF NOT EXISTS {dbname} CHARACTER SET utf8mb4'))
    await admin.dispose()
    test_engine=create_async_engine(server+'/'+dbname)
    async with test_engine.begin() as conn:
        await conn.run_sync(base.Base.metadata.create_all)
        await conn.execute(text('ALTER TABLE conversations AUTO_INCREMENT = 2100000000'))
    old=base.engine;base.engine=test_engine
    try:await workflow.ensure_tables()
    finally:base.engine=old;await test_engine.dispose()
    OUT.mkdir(parents=True,exist_ok=True)
    save(OUT/'cases.json',cases);save(OUT/'knowledge.json',kb);save(OUT/'scoring.json',SCORING)
    shutil.copytree(ROOT/'app',OUT/'source/app',ignore=shutil.ignore_patterns('__pycache__'))
    fields=['chat_model','chat_base_url','intent_model','embed_model','embed_base_url','rerank_model','rerank_base_url',
            'context_policy','context_legacy_recall_enabled','max_agent_steps','token_budget','recall_top_k','rerank_top_k',
            'chat_thinking','chat_reasoning_effort','query_rewrite_timeout']
    save(OUT/'protocol.json',{'version':'multi120-v1','repetitions':3,'arms':ARMS,'groups':GROUPS,
        'model_config':{k:getattr(settings,k) for k in fields},'test_db_name':dbname,
        'original300_sha256':sha(ROOT/'tests/data/eval_ch04.jsonl'), 'cases_sha256':sha(OUT/'cases.json'),
        'knowledge_sha256':sha(OUT/'knowledge.json'),'scoring_sha256':sha(OUT/'scoring.json'),
        'source_hashes':{str(p.relative_to(OUT/'source')):sha(p) for p in (OUT/'source').rglob('*.py')},
        'evaluation_script_hashes':{n:sha(ROOT/'scripts'/n) for n in
             ['multi120_cases.py','eval_multi120.py','score_multi120.py']},
        'scope':'Original300 knowledge facts ->120 complete business scenarios. Questions, labels and scoring unchanged from archived preflight; no app tuning during study.',
        'preflight':'20261103-business120-preflight: shared-DB multi-process timestamp ticket collision invalidated clean capability comparison. All raw preflight retained; formal run uses disjoint per-trajectory ticket sequence namespaces and a new isolated DB/request namespace.',
        'ticket_identity_isolation':'repository._TICKET_SEQ seeded per trajectory to disjoint ranges of1000. Identifier generation logic unchanged; maximum16 actor steps is below range size. No production configuration/code change.',
        'source_questions_used':len({s for c in cases for s in c['source_question_ids']}),
        'knowledge_rows':len(kb),'families':len({c['family_id'] for c in cases})})
    lines=['# 120场景及预定标注','', '完整字段见cases.json；本表在模型调用前生成。','',
           '| ID | 分组 | 来源题 | 首问 | 预定事项 | 后续消息/故障 |','|---|---|---|---|---|---|']
    for c in cases:
        goals='；'.join(g['kind']+((':'+g['order_id']) if g.get('order_id') else '')+'→'+g['expected_outcome'] for g in c['goals'])
        lines.append('| '+' | '.join([c['id'],GROUPS[c['group']],','.join(c['source_question_ids']),c['query'],goals,c['followup'] or c['fault'] or '无'])+' |')
    (OUT/'CASES.md').write_text('\n'.join(lines)+'\n','utf-8')
    print('Frozen120, sources',len({s for c in cases for s in c['source_question_ids']}),'knowledge',len(kb),flush=True)


async def worker(arm,case_id,rep,pilot=False):
    sys.path.insert(0,str(OUT/'source'))
    from app.config import settings
    from app.db import base,models,repository,workflow
    from app.graph import runtime,nodes
    from app.core import model_pool
    from app.tools import business,registry
    from app.kb import milvus_client
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import create_async_engine,async_sessionmaker
    from langchain_core.tools import tool
    from httpx import ASGITransport,AsyncClient
    from fastapi import FastAPI
    from app.api.actions import router
    protocol=read(OUT/'protocol.json')
    case=next(c for c in read(OUT/'cases.json') if c['id']==case_id)
    for k,v in protocol['model_config'].items():setattr(settings,k,v)
    settings.multi_intent_enabled=arm=='enabled';settings.chat_failover_enabled=False
    settings.model_pool_db_path=protocol.get('model_pool_db_path',str(ROOT/'data/model_pool.sqlite3'));settings.langfuse_public_key=''
    for k,v in protocol.get('evaluation_settings',{}).items():setattr(settings,k,v)
    if protocol.get('candidate_shared_agent'):
        if arm=='enabled':settings.multi_intent_shared_agent_enabled=True
        from app.core import runtime_settings
        runtime_settings.SETTINGS_PATH=OUT/'runtime_settings.json'
    directory=OUT/('pilot' if pilot else 'runs')/f'r{rep}'/arm/case_id
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'result.json').exists():raise RuntimeError('Do not resample completed trajectory')
    settings.checkpointer_db_path=str(directory/'checkpoints.sqlite')
    base.engine=create_async_engine(settings.test_database_url.rsplit('/',1)[0]+'/'+protocol['test_db_name'],pool_pre_ping=True)
    base.async_session=async_sessionmaker(base.engine,expire_on_commit=False)
    uid=f'mi120formal-{rep}-{arm}-{case_id}'
    case_index=next(i for i,c in enumerate(read(OUT/'cases.json')) if c['id']==case_id)
    repository._TICKET_SEQ=ticket_sequence_namespace(arm,case_index,rep,len(read(OUT/'cases.json')))
    business.order_snapshot=lambda oid:dict(case['orders'][oid])
    business.list_user_orders=lambda who:[dict(o) for o in case['orders'].values()] if who==uid else []
    registry.scan_builtin()
    import app.tools.builtin.orders as order_tools
    order_tools.order_snapshot=business.order_snapshot
    @tool
    async def query_logistics(tracking_no:str)->dict:
        """根据物流单号查询当前位置和状态，先query_order取得tracking_no。"""
        if case['fault']=='logistics_unavailable':raise RuntimeError('injected logistics unavailable')
        return dict(case['logistics'].get(tracking_no,{'error':'没有此物流单号'}))
    ls=registry.spec_from_langchain_tool(query_logistics,source='mcp');ls.max_retries=0
    async def specs():return [*registry._BUILTIN.values(),ls]
    registry.get_all_specs=specs
    async def no_summary(cid):pass
    runtime.summarizer.maybe_schedule_summary=no_summary
    milvus_client.ensure_collection=lambda *a,**kw:None
    await runtime.init_graph();api=FastAPI();api.include_router(router)
    started=time.perf_counter();steps=[];actor=[];cid=None;authorized=[];seen_forms=set();followed=False;restarted=False;duplicate_done=False
    result={'arm':arm,'rep':rep,'case_id':case_id,'case':case,'status':'running','steps':steps,'actor':actor,
            'authorized':authorized,'fault_checks':[],'wrong_previews':[],'clarifications':[],'selector_events':[]}
    async def tickets_now():
        if cid is None:return []
        async with base.async_session() as sess:
            rows=(await sess.execute(select(models.Ticket).where(models.Ticket.conversation_id==cid))).scalars().all()
            return [{'ticket_no':t.ticket_no,'ticket_type':t.ticket_type,'description':t.description} for t in rows]
    async def capture(out):
        s=out['state'];step={'answer':nodes.resolve_answer(s),'interrupt':out.get('interrupt'),
            'actions':runtime.dedup_actions(s.get('suggested_actions',[])),
            'task_plan_diagnostic_only':s.get('task_plan',{}),'trace':s.get('trace',{}),
            'citations':s.get('citations',[]),'retrieved_snapshot':s.get('retrieved_snapshot',[]),
            'tickets_before_next_actor':await tickets_now(),'authorized_count':len(authorized),
            'elapsed':time.perf_counter()-started}
        steps.append(step);save(directory/'progress.json',result);return step
    def matching(kind,oid=None):
        return next((g for g in case['goals'] if g['kind']==kind and
            (oid is None or g.get('order_id')==oid)),None)
    def preview_decision(ticket_type,description):
        import re
        kind='refund' if ticket_type=='退款' else 'complaint' if ticket_type=='投诉' else 'other'
        m=re.search(r'订单号\s*[=:：]?\s*(\d{4,20})',description.split('用户原述')[0])
        oid=m.group(1) if m else None
        g=matching(kind,oid) if kind=='refund' and oid else matching(kind)
        if not g or ticket_type in case['forbidden_ticket_types'] or (kind=='refund' and not oid):
            result['wrong_previews'].append({'kind':kind,'order_id':oid,'description':description});return False,None
        return g['expected_outcome']=='ticket',g
    async def resume(intr,value):
        return await runtime.resume_turn(cid,value,user_id=uid,interrupt_id=intr['interrupt_id'])
    try:
      with model_pool.call_context(f'mi120formal:r{rep}:{arm}:{case_id}',priority='background'):
       async with asyncio.timeout(480):
        out=await runtime.run_turn(uid,case['query'],None);cid=out['conversation_id'];result['conversation_id']=cid
        for index in range(16):
            step=await capture(out);intr=step['interrupt']
            if case['followup'] and not followed:
                followed=True;actor.append({'kind':'user_message','text':case['followup']})
                old_intr=intr
                out=await runtime.run_turn(uid,case['followup'],cid)
                if old_intr:
                    try:
                        await runtime.validate_resume(cid,{'cancelled':True},user_id=uid,interrupt_id=old_intr['interrupt_id'])
                        result['fault_checks'].append({'stale_resume_rejected':False})
                    except runtime.ResumeRejected:result['fault_checks'].append({'stale_resume_rejected':True})
                continue
            if intr:
                if case['fault']=='restart_connection' and not restarted:
                    await runtime.close_graph();await runtime.init_graph();restarted=True
                    actor.append({'kind':'restart_sqlite_connection'});result['fault_checks'].append({'connection_reopened':True})
                kind=intr['type']
                if kind=='select_order':
                    # Record every actual public card, even when no target can be resolved.
                    result['selector_events'].append({'step':index,'public_payload':intr})
                    purpose=intr.get('task_kind')
                    if not purpose:
                        kinds={g['kind'] for g in case['goals'] if g['kind'] in ('refund','order','logistics')}
                        # The legacy select_order UI belongs to refund flow; do not consult task_plan.
                        purpose='refund' if 'refund' in kinds else next(iter(kinds)) if len(kinds)==1 else None
                    candidates=[g for g in case['goals'] if g['kind']==purpose]
                    if not candidates:
                        actor.append({'kind':'unresolvable_selector','public_payload':intr});break
                    g=candidates[0]
                    result['clarifications'].append({'kind':purpose,'explicit':g['explicit_order'],'order_id':g.get('order_id')})
                    if case['actor']['cancel_selection']:
                        if not intr.get('can_cancel'):
                            actor.append({'kind':'unsupported_cancel_selector'});break
                        value={'cancelled':True}
                    else:value=g['order_id']
                elif kind=='task_form':
                    form=intr['form'];draft=form.get('draft',{})
                    oid=str(draft.get('order_id') or '')
                    typ='退款' if form['type']=='refund_form' else draft.get('ticket_type','投诉')
                    description=('退款申请 订单号='+oid if typ=='退款' else draft.get('description',''))
                    confirm,g=preview_decision(typ,description)
                    value={'confirmed':confirm}
                    if confirm:
                        payload={'conversation_id':cid,'user_id':uid,'request_id':intr['request_id']}
                        if typ=='退款':payload.update(order_id=oid,reason='质量问题');path='/api/actions/create-refund'
                        else:payload.update(ticket_type=typ,description=description or case['query']);path='/api/actions/create-ticket'
                        authorized.append({'goal_id':g['id'],'kind':g['kind'],'order_id':g.get('order_id')})
                        async with AsyncClient(transport=ASGITransport(app=api),base_url='http://local') as http:
                            response=await http.post(path,json=payload)
                            submission={'path':path,'status':response.status_code,'response':response.json()}
                            actor.append({'kind':'task_form','value':value,'interrupt_id':intr['interrupt_id'],'submission':submission})
                            if response.status_code!=200:raise RuntimeError('Form submission rejected: '+str(submission))
                            if case['fault']=='duplicate_submission' and not duplicate_done:
                                duplicate_done=True;again=await http.post(path,json=payload)
                                actor.append({'kind':'duplicate_form_submit','status':again.status_code,'response':again.json()})
                    else:
                        if g:actor.append({'kind':'cancel_form','goal_id':g['id']})
                        actor.append({'kind':'task_form','value':value,'interrupt_id':intr['interrupt_id']})
                    out=await resume(intr,value)
                    continue
                elif kind in ('task_actions','confirm_ticket'):
                    action=intr['actions'][0] if kind=='task_actions' else {'preview':intr.get('preview',{})}
                    preview=action['preview'];confirm,g=preview_decision(preview.get('ticket_type'),preview.get('description',''))
                    value={'confirmed':confirm}
                    if kind=='task_actions':value['operation_id']=action['operation_id']
                    if confirm and g:authorized.append({'goal_id':g['id'],'kind':g['kind'],'order_id':g.get('order_id')})
                else:
                    actor.append({'kind':'unsupported_interrupt','payload':intr});break
                actor.append({'kind':kind,'value':value,'interrupt_id':intr['interrupt_id']})
                out=await resume(intr,value)
                if case['fault']=='duplicate_submission' and value and isinstance(value,dict) and value.get('confirmed') and not duplicate_done:
                    duplicate_done=True
                    try:
                        await resume(intr,value);result['fault_checks'].append({'duplicate_resume_rejected':False})
                    except runtime.ResumeRejected:result['fault_checks'].append({'duplicate_resume_rejected':True})
                continue
            acted=False
            async with AsyncClient(transport=ASGITransport(app=api),base_url='http://local') as http:
              for action in step['actions']:
                key=json.dumps(action,sort_keys=True,ensure_ascii=False)
                if key in seen_forms:continue
                seen_forms.add(key);draft=action.get('draft',{})
                payload={'conversation_id':cid,'user_id':uid,'request_id':case_id+':'+str(len(seen_forms))}
                if action['type']=='refund_form':
                    oid=str(draft.get('order_id') or '')
                    g=matching('refund',oid) if oid else matching('refund')
                    if not g:
                        result['wrong_previews'].append({'kind':'refund','order_id':oid});continue
                    if g['expected_outcome']!='ticket':
                        actor.append({'kind':'cancel_form','goal_id':g['id']});continue
                    payload.update(order_id=oid or g['order_id'],reason='质量问题');path='/api/actions/create-refund'
                elif action['type']=='create_ticket':
                    g=matching('complaint')
                    if not g or g['expected_outcome']!='ticket':
                        if g:actor.append({'kind':'cancel_form','goal_id':g['id']})
                        continue
                    payload.update(description=case['query'],ticket_type='投诉');path='/api/actions/create-ticket'
                else:continue
                authorized.append({'goal_id':g['id'],'kind':g['kind'],'order_id':g.get('order_id')})
                response=await http.post(path,json=payload)
                actor.append({'kind':'form_submit','goal_id':g['id'],'path':path,'status':response.status_code,'response':response.json()})
                if case['fault']=='duplicate_submission' and not duplicate_done:
                    duplicate_done=True;again=await http.post(path,json=payload)
                    actor.append({'kind':'duplicate_form_submit','status':again.status_code,'response':again.json()})
                acted=True
            # Allow one cooperative textual response only when a target was intentionally omitted
            # and the service actually asks for an order number. Never prompt it to add missing goals.
            if not acted:
                from scripts.multi120_evidence_protocol import cooperative_order_reply
                text=cooperative_order_reply(case,step,steps[-2]['answer'] if len(steps)>1 else '',actor)
                if text:
                    actor.append({'kind':'supply_missing_order','text':text});out=await runtime.run_turn(uid,text,cid);continue
            break
        else:result['actor_limit']=True
        result['status']='ok'
    except Exception as e:result.update(status='error',error_type=type(e).__name__,error=str(e))
    finally:
        result['elapsed_seconds']=time.perf_counter()-started
        result['tickets']=await tickets_now()
        if cid:
            async with base.async_session() as sess:
                logs=(await sess.execute(select(models.ToolAuditLog).where(models.ToolAuditLog.conversation_id==cid))).scalars()
                result['tool_audit']=[{'name':x.tool_name,'arguments':x.arguments,'status':x.status,'result':x.result_summary} for x in logs]
            result['dialogue']=[{'role':m.role,'content':m.content,'id':m.id} for m in await repository.list_dialog_messages(cid)]
        save(directory/'result.json',result)
        await runtime.close_graph()
        from app.core.model_transport import close_clients
        await close_clients();await base.engine.dispose()
    print(rep,arm,case_id,result['status'],round(result['elapsed_seconds'],1),flush=True)


async def run(concurrency=8):
    protocol=read(OUT/'protocol.json')
    for name,h in protocol['source_hashes'].items():
        if sha(OUT/'source'/name)!=h:raise RuntimeError('Frozen app changed: '+name)
    for name in ('cases','knowledge','scoring'):
        if sha(OUT/(name+'.json'))!=protocol[name+'_sha256']:raise RuntimeError('Frozen '+name+' changed')
    sem=asyncio.Semaphore(concurrency)
    async def pair(c,rep):
      async with sem:
       for arm in (ARMS if (rep+int(c['id'].split('-')[1]))%2 else tuple(reversed(ARMS))):
        directory=OUT/'runs'/f'r{rep}'/arm/c['id'];directory.mkdir(parents=True,exist_ok=True)
        if (directory/'result.json').exists():continue
        with (directory/'worker.log').open('w',encoding='utf-8') as log:
            proc=await asyncio.create_subprocess_exec(sys.executable,str(Path(__file__).resolve()),'worker',
                '--arm',arm,'--case',c['id'],'--rep',str(rep),cwd=str(ROOT),stdout=log,stderr=log,
                env={**os.environ,'PYTHONIOENCODING':'utf-8'})
            rc=await proc.wait()
        if not (directory/'result.json').exists():
            save(directory/'result.json',{'case_id':c['id'],'case':c,'arm':arm,'rep':rep,'status':'worker_error','exit_code':rc,'steps':[],'tickets':[],'actor':[],'authorized':[]})
        count=len(list((OUT/'runs').glob('*/*/*/result.json')))
        print(f'{count}/720 r{rep} {arm} {c["id"]} exit={rc}',flush=True)
    # One complete repetition before the next; groups interleaved, not all easy cases first.
    cases=read(OUT/'cases.json');ordered=sorted(cases,key=lambda c:(int(c['id'].split('-')[1]),c['group']))
    for rep in range(1,4):await asyncio.gather(*(pair(c,rep) for c in ordered))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','worker'])
    p.add_argument('--arm',choices=ARMS);p.add_argument('--case');p.add_argument('--rep',type=int,default=1)
    p.add_argument('--concurrency',type=int,default=8);args=p.parse_args()
    asyncio.run(prepare() if args.stage=='prepare' else run(args.concurrency) if args.stage=='run' else worker(args.arm,args.case,args.rep))
