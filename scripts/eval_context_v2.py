"""Candidate-only real API context acceptance, isolated SQL/checkpoint, frozen gold."""
import argparse
import asyncio
from contextlib import AsyncExitStack
import hashlib
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import httpx
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from sqlalchemy import select,text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine,async_sessionmaker
from app.config import settings
from app.core import task_planner
from app.db import base,models,repository,workflow
from app.graph import runtime,nodes,subflow_tasks
from app.graph.build import build_graph
from app.kb import milvus_client
from app.tools import business

OUT=Path('E:/Mewhelp/data/verification/multi-intent-context-v2/20261104-v1')
ACTIVE={}


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str),encoding='utf-8')


def record_response(row,body):
    if body.startswith(b'\x1f\x8b'):
        import gzip
        body=gzip.decompress(body)
    try:
        value=json.loads(body)
        row['usage']=value.get('usage')
    except (ValueError,UnicodeError):
        for line in body.decode('utf-8',errors='replace').splitlines():
            if line.startswith('data: {'):
                try:
                    value=json.loads(line[6:])
                    if value.get('usage'):row['usage']=value['usage']
                except ValueError:pass


class Tee(httpx.AsyncByteStream):
    def __init__(self,stream,row):self.stream=stream;self.row=row;self.parts=[]
    async def __aiter__(self):
        async for block in self.stream:
            self.parts.append(block);yield block
        record_response(self.row,b''.join(self.parts))
    async def aclose(self):
        try:await self.stream.aclose()
        finally:record_response(self.row,b''.join(self.parts))


def instrument():
    from app.core import task_context
    normalize=task_context.normalize
    def normalized(raw,items,context):
        ACTIVE['raw_proposals']=items
        try:return normalize(raw,items,context)
        except Exception as exc:
            ACTIVE['validation_error']={'type':type(exc).__name__,'message':str(exc)}
            raise
    task_context.normalize=normalized
    original=httpx.AsyncClient.send
    async def send(client,request,**kwargs):
        host=urlsplit(str(request.url)).hostname
        if host not in {'ai.gitee.com','api.moark.com'}:return await original(client,request,**kwargs)
        try:payload=json.loads(request.content)
        except ValueError:payload={}
        row={'endpoint':request.url.path,'request':payload,'started':time.monotonic()}
        request.headers['accept-encoding']='identity'
        ACTIVE.setdefault('http',[]).append(row)
        try:
            response=await original(client,request,**kwargs)
            row['status']=response.status_code
            if response.is_stream_consumed:record_response(row,response.content)
            else:response.stream=Tee(response.stream,row)
            return response
        except Exception as exc:row['error']=type(exc).__name__;raise
        finally:row['response_headers_seconds']=time.monotonic()-row.pop('started')
    httpx.AsyncClient.send=send
    extract=task_planner.extract
    async def planned(raw,*,context=None):
        row={'raw':raw,'context':context};ACTIVE.setdefault('planner',[]).append(row)
        result=await extract(raw,context=context);row['tasks']=result;return result
    task_planner.extract=planned
    retrieve=nodes.retrieve_knowledge
    async def retrieved(state):
        row={'query':state['resolved_query'],'task_context':state.get('task_context')}
        ACTIVE.setdefault('retrieval',[]).append(row)
        result=await retrieve(state)
        row['result']=result
        return result
    nodes.retrieve_knowledge=retrieved
    prepare=subflow_tasks.prepare
    async def prepared(state):
        if ACTIVE.get('case',{}).get('action')=='order_failure' and state['task_plan'][state['active_task']]['kind']=='knowledge':
            snapshot=business.order_snapshot
            def broken(oid):raise ConnectionError('synthetic order data outage')
            business.order_snapshot=broken
            try:return await prepare(state)
            finally:business.order_snapshot=snapshot
        return await prepare(state)
    subflow_tasks.prepare=prepared


async def seed(case,uid):
    cid=await repository.create_conversation(uid)
    for oid in case['focus']:
        msg=f'请确认订单{oid}，这是目前讨论的订单。'
        if case['id']=='B2' and oid=='2002':msg='更正：不再讨论1001，接下来讨论订单2002。'
        mid=await repository.append_message(cid,'user',msg)
        await repository.append_message(cid,'assistant',f'已核验订单{oid}，后续以这笔为准。')
        task={'task_id':f'seed:{cid}:{oid}','kind':'order','order_id':oid,'status':'DONE','source':msg,
              'binding':{'method':'user_selection','status':'resolved'}}
        await workflow.save_plan(cid,uid,f'seed:{cid}:{oid}',mid,{'order':task})
    if case['id']=='B4':
        await repository.append_message(cid,'user','你好')
        await repository.append_message(cid,'assistant','刚才查询的是订单1001。')
    if case['id']=='B1':
        for i in range(24):
            await repository.append_message(cid,'user',f'问一下宠物护理术语第{i}项的含义，与购买订单无关。')
            mid=await repository.append_message(cid,'assistant','术语介绍仅供参考，不代表办理了业务。'*18)
        await repository.update_conversation_summary(cid,'较早讨论并核验订单1001；其余均是通用术语咨询。',mid,expected_upto_msg_id=0)
    return cid


async def run_case(case,rep,folder,stack):
    global ACTIVE
    ACTIVE={'case':case,'rep':rep,'steps':[],'http':[],'planner':[],'retrieval':[],'checks':{}}
    started=time.monotonic();uid=f'context-v2-synthetic-r{rep}'
    cid=await seed(case,uid);ACTIVE['conversation_id']=cid
    checkpoint=str(folder/(case['id']+'.sqlite'))
    async def graph():
        saver=await stack.enter_async_context(AsyncSqliteSaver.from_conn_string(checkpoint))
        runtime._graph=build_graph(saver)
    await graph()
    out=await runtime.run_turn(uid,case['raw'],cid,multi_intent_enabled=True)
    ACTIVE['steps'].append(out)
    initial=out['state'].get('task_plan',{})
    selections=0;confirmations=[];old_submit=workflow.submit;lost=[]
    if case['action']=='lost_commit':
        async def submit(*args,**kwargs):
            receipt=await old_submit(*args,**kwargs);lost.append(receipt)
            if len(lost)==1:raise ConnectionError('synthetic response loss after SQL commit')
            return receipt
        workflow.submit=submit
    try:
        for _ in range(10):
            intr=out.get('interrupt')
            if not intr:break
            if intr['type']=='select_order':
                selections+=1
                if case['action']=='invalid_selection':
                    try:await runtime.resume_turn(cid,'9999',user_id=uid,interrupt_id=intr['interrupt_id'])
                    except runtime.ResumeRejected:ACTIVE['checks']['invalid_selection_rejected']=True
                    else:ACTIVE['checks']['invalid_selection_rejected']=False
                if case['action']=='supersede':
                    newer=await runtime.run_turn(uid,'改查订单2002的状态和物流，不再处理刚才的请求',cid,multi_intent_enabled=True)
                    ACTIVE['steps'].append(newer)
                    try:await runtime.resume_turn(cid,'1001',user_id=uid,interrupt_id=intr['interrupt_id'])
                    except runtime.ResumeRejected:ACTIVE['checks']['stale_card_rejected']=True
                    else:ACTIVE['checks']['stale_card_rejected']=False
                    out=newer;break
                if case['action']=='restart':await graph()
                value={'cancelled':True} if case['action']=='cancel' else '1001'
                out=await runtime.resume_turn(cid,value,user_id=uid,interrupt_id=intr['interrupt_id'])
                if case['action']=='duplicate_resume':
                    try:await runtime.resume_turn(cid,value,user_id=uid,interrupt_id=intr['interrupt_id'])
                    except runtime.ResumeRejected:ACTIVE['checks']['duplicate_resume_rejected']=True
                    else:ACTIVE['checks']['duplicate_resume_rejected']=False
            elif intr['type']=='task_actions':
                action=intr['actions'][0];confirmations.append(action['operation_id'])
                out=await runtime.resume_turn(cid,{'operation_id':action['operation_id'],'confirmed':True},
                                              user_id=uid,interrupt_id=intr['interrupt_id'])
                if case['action']=='lost_commit' and len(lost)==1:await graph()
            else:raise ValueError('Unexpected interrupt')
            ACTIVE['steps'].append(out)
    finally:workflow.submit=old_submit
    final=out['state'].get('task_plan',{})
    checks=ACTIVE['checks'];checks['selection_count']=selections==case['selections']
    checks['finished']=out.get('interrupt') is None
    checks['one_planner_per_turn']=len(ACTIVE['planner'])==(2 if case['action']=='supersede' else 1)
    for kind,expected in case['required'].items():
        ts=[t for t in initial.values() if t['kind']==kind]
        actual=[t.get('context_order_id') if kind=='knowledge' else t.get('order_id') for t in ts]
        checks['initial_'+kind]=len(ts)==1 and actual==[expected]
    # General history-recap fallback is checked only for no historical execution.
    checks['no_internal_state']=all(not any(x in s['state'].get('answer','') for x in
        ['本次事项处理结果','WAITING_DEPENDENCY','WAITING_USER','task_id']) for s in ACTIVE['steps'])
    checks['no_early_retrieval']=not any(r.get('task_context',{}).get('order_id') is None and
        any(t.get('knowledge_scope')=='product' for t in initial.values()) for r in ACTIVE['retrieval'])
    for t in final.values():
        if t['kind'] in {'order','logistics'}:checks['completed_'+t['task_id']]=t['status'] in {'DONE','CANCELLED'}
        if t['kind']=='knowledge' and t.get('knowledge_scope')=='product' and case['action'] not in {'cancel','order_failure'}:
            checks['product_query_'+t['task_id']]='智能猫砂盆' in t.get('retrieval_query','')
    if case['knowledge']=='must_not_retrieve':checks['no_retrieval_on_order_failure']=not ACTIVE['retrieval']
    if case['knowledge']=='must_refuse':checks['refusal']=any(t['kind']=='knowledge' and t['status']=='FAILED' for t in final.values())
    if case['action']=='cancel':checks['group_cancelled']=all(t['status']=='CANCELLED' for t in final.values() if t['kind'] in {'order','logistics','knowledge'})
    if case['id']=='B3':checks['no_replayed_tasks']=not any(t['kind'] in {'refund','complaint'} for t in final.values())
    async with base.async_session() as session:
        tickets=(await session.execute(select(models.Ticket).where(models.Ticket.conversation_id==cid))).scalars().all()
        audits=(await session.execute(select(models.ToolAuditLog).where(models.ToolAuditLog.conversation_id==cid))).scalars().all()
    ACTIVE['tickets']=[{'number':t.ticket_no,'type':t.ticket_type,'description':t.description} for t in tickets]
    ACTIVE['audit']=[{'tool':a.tool_name,'arguments':a.arguments,'status':a.status} for a in audits]
    expected_writes=sum(t['kind'] in {'refund','complaint'} and t.get('step')=='confirm_submission' for t in initial.values())
    checks['no_duplicate_or_unconfirmed_writes']=len(tickets)==len(set(t.ticket_no for t in tickets)) and len(tickets)<=len(set(confirmations))
    if case['action']=='lost_commit':checks['lost_commit_one_receipt']=len(lost)>=2 and lost[0]['ticket_no']==lost[1]['ticket_no'] and len(tickets)==2
    ACTIVE['elapsed_seconds']=time.monotonic()-started
    ACTIVE['passed']=all(checks.values())
    return ACTIVE


async def main(args):
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((OUT/'cases.json').read_bytes()).hexdigest()==protocol['cases_sha256']
    cases=json.loads((OUT/'cases.json').read_text(encoding='utf-8'))
    if args.ids:cases=[c for c in cases if c['id'] in args.ids.split(',')]
    folder=OUT/args.run/f'r{args.rep}';folder.mkdir(parents=True,exist_ok=True)
    # Keep the current context selector and all retrieval/gate settings intact.
    settings.multi_intent_context_v2_enabled=True;settings.multi_intent_subflows_enabled=True;settings.multi_intent_enabled=True
    from app.core import runtime_settings
    runtime_settings.SETTINGS_PATH=Path('E:/Mewhelp/data/runtime_settings.json')
    settings.langfuse_public_key='';settings.model_pool_db_path=str(folder/'pool.sqlite')
    url=make_url(settings.database_url);db=f'caremate_ctxv2_{args.run.replace("-","_")}_{args.rep}'
    if not db.replace('_','').isalnum():raise ValueError('invalid isolated database name')
    admin=create_async_engine(url.set(database=None),isolation_level='AUTOCOMMIT')
    async with admin.begin() as conn:await conn.execute(text(f'CREATE DATABASE IF NOT EXISTS {db} CHARACTER SET utf8mb4'))
    await admin.dispose()
    old=base.engine;base.engine=create_async_engine(url.set(database=db),pool_pre_ping=True)
    base.async_session=async_sessionmaker(base.engine,expire_on_commit=False)
    async with base.engine.begin() as conn:await conn.run_sync(base.Base.metadata.create_all)
    await workflow.ensure_tables()
    # Prevent production collection creation/upsert; the existing collection is read-only.
    def existing():
        client=milvus_client.get_client()
        if not client.has_collection('knowledge'):raise RuntimeError('Existing knowledge required')
        client.load_collection('knowledge');milvus_client._ensured.add('knowledge')
        return milvus_client.count(client)
    count=await milvus_client.acall(existing)
    save(folder/'manifest.json',{'database':db,'kb_count':count,'context_policy':settings.context_policy,
         'confidence':runtime_settings.get_confidence_settings(),
         'legacy_recall':settings.context_legacy_recall_enabled,'models':[settings.chat_model,settings.embed_model,settings.rerank_model],
         'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'app').rglob('*.py')}})
    instrument()
    async with AsyncExitStack() as stack:
        for case in cases:
            file=folder/(case['id']+'.json')
            if file.exists():continue
            try:result=await run_case(case,args.rep,folder,stack)
            except Exception as exc:
                import traceback
                result={**ACTIVE,'passed':False,'error':type(exc).__name__,'traceback':traceback.format_exc()}
            save(file,result)
            print(json.dumps({'rep':args.rep,'case':case['id'],'passed':result['passed'],
                'failed':[k for k,v in result.get('checks',{}).items() if not v],'error':result.get('error')},ensure_ascii=False),flush=True)
    await base.engine.dispose();await old.dispose()
    from app.core.model_transport import close_clients
    await close_clients()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rep',type=int,default=1);p.add_argument('--run',default='pilot1');p.add_argument('--ids')
    asyncio.run(main(p.parse_args()))
