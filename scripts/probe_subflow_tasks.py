"""Real P2 model/RAG/MCP + in-process HTTP, isolated MySQL and SQLite.

Keeps artifacts and the synthetic database for review; never replaces localhost.
"""
import asyncio
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import httpx
from fastapi import FastAPI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from sqlalchemy import select,text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine,async_sessionmaker

from app.config import settings
from app.db import base,models,workflow
from app.graph import runtime
from app.graph.build import build_graph
from app.kb import milvus_client
from app.api.chat import router as chat_router
from app.api.actions import router as actions_router
from app.api.conversations import router as conversations_router


async def main():
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix=sys.argv[1] if len(sys.argv)>1 else 'p2-live'
    if prefix not in {'p2-live','p3-live'}:raise ValueError('Unsupported probe prefix')
    folder=Path('data/verification')/(prefix+'-'+stamp);folder.mkdir(parents=True)
    dbname='caremate_subflow_p2_'+stamp
    settings.multi_intent_subflows_enabled=True
    settings.chat_failover_enabled=False
    settings.context_policy='legacy';settings.context_legacy_recall_enabled=False
    settings.langfuse_public_key=''
    settings.model_pool_db_path=str(folder/'pool.sqlite')
    settings.model_pool_turn_timeout=240
    async def no_summary(*args):pass
    runtime.summarizer.maybe_schedule_summary=no_summary
    url=make_url(settings.database_url)
    admin=create_async_engine(url.set(database=None),isolation_level='AUTOCOMMIT')
    async with admin.connect() as conn:
        await conn.execute(text(f'CREATE DATABASE {dbname} CHARACTER SET utf8mb4'))
    await admin.dispose()
    original_engine=base.engine
    base.engine=create_async_engine(url.set(database=dbname),pool_pre_ping=True)
    base.async_session=async_sessionmaker(base.engine,expire_on_commit=False)
    async with base.engine.begin() as conn:await conn.run_sync(base.Base.metadata.create_all)
    await workflow.ensure_tables()
    def existing_knowledge():
        client=milvus_client.get_client()
        if not client.has_collection('knowledge'):raise RuntimeError('Existing KB missing; refusing to create')
        client.load_collection('knowledge')
        milvus_client._ensured.add('knowledge')
        return client.get_collection_stats('knowledge')
    kb_before=await milvus_client.acall(existing_knowledge)
    code={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [
        'app/graph/subflow_tasks.py','app/graph/read_subflows.py','app/graph/build.py',
        'app/graph/runtime.py','app/graph/nodes.py','app/db/workflow.py']}
    for p in ['app/core/task_reply.py','app/graph/tasks.py','app/graph/state.py','app/static/index.html','app/static/chat.css']:
        if Path(p).exists():code[p]=hashlib.sha256(Path(p).read_bytes()).hexdigest()
    meta={'database':dbname,'code':code,'chat_model':settings.chat_model,
          'embedding':settings.embed_model,'reranker':settings.rerank_model,'kb_before':kb_before}
    (folder/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    app=FastAPI();app.include_router(chat_router);app.include_router(actions_router);app.include_router(conversations_router)
    records=[]; checks={}
    def save():
        (folder/'records.json').write_text(json.dumps(records,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
        (folder/'checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    async def post(client,path,body):
        response=await client.post(path,json=body)
        events=[json.loads(s[6:]) for s in response.text.splitlines() if s.startswith('data: {')]
        records.append({'path':path,'body':body,'status':response.status_code,'events':events})
        save()
        print(json.dumps({'step':len(records),'status':response.status_code,
                          'events':[e.get('event') for e in events]},ensure_ascii=False),flush=True)
        return response,events
    def card(events):return next(e for e in events if e.get('event')=='interrupt')
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://isolated') as client:
        filename=str(folder/'checkpoints.sqlite')
        async with AsyncSqliteSaver.from_conn_string(filename) as saver:
            runtime._graph=build_graph(saver)
            response,events=await post(client,'/api/chat',{'user_id':'p2-synthetic',
                'message':'我刚买的猫砂盆漏电了，我要退款，而且我要投诉商家。','multi_intent_enabled':True})
            assert response.status_code==200
            selection=card(events);cid=selection['conversation_id']
            checks['safety_before_selection']=events[0].get('event')=='task_event' and '停止使用' in events[0].get('text','')
            assert selection['kind']=='select_order'
        async with AsyncSqliteSaver.from_conn_string(filename) as saver:
            runtime._graph=build_graph(saver)
            body={'conversation_id':cid,'user_id':'p2-synthetic','interrupt_id':selection['interrupt_id'],'order_id':'1001'}
            _,events=await post(client,'/api/actions/resume',body)
            actions=card(events)
            checks['selected_once']=actions['kind']=='task_actions' and len(actions['actions'])==2
            response,_=await post(client,'/api/actions/resume',body)
            checks['stale_selection_rejected']=response.status_code==409
            refund=next(a for a in actions['actions'] if a['preview']['ticket_type']=='退款')
            confirm={'conversation_id':cid,'user_id':'p2-synthetic','interrupt_id':actions['interrupt_id'],
                     'operation_id':refund['operation_id'],'confirmed':True}
            _,events=await post(client,'/api/actions/resume',confirm)
            pending=card(events)
            checks['complaint_remains']=len(pending['actions'])==1 and pending['actions'][0]['preview']['ticket_type']=='投诉'
        async with AsyncSqliteSaver.from_conn_string(filename) as saver:
            runtime._graph=build_graph(saver)
            response,_=await post(client,'/api/actions/resume',confirm)
            checks['stale_refund_rejected']=response.status_code==409
            complaint=pending['actions'][0]
            confirm={'conversation_id':cid,'user_id':'p2-synthetic','interrupt_id':pending['interrupt_id'],
                     'operation_id':complaint['operation_id'],'confirmed':True}
            _,events=await post(client,'/api/actions/resume',confirm)
            checks['both_finish_after_restart']=events[-1].get('event')=='done'
            response,_=await post(client,'/api/actions/resume',confirm)
            checks['stale_complaint_rejected']=response.status_code==409
            _,events=await post(client,'/api/chat',{'conversation_id':cid,'user_id':'p2-synthetic',
                'message':'刚刚的订单是哪个，它现在物流到哪了？','multi_intent_enabled':True})
            checks['followup_no_selection']=not any(e.get('event')=='interrupt' for e in events)
            snapshot=await runtime.get_graph().aget_state({'configurable':{'thread_id':str(cid)}})
            tasks=list(snapshot.values['task_plan'].values())
            checks['followup_two_tasks_1001']=len(tasks)==2 and all(t.get('order_id')=='1001' and t['status']=='DONE' for t in tasks)
            if prefix=='p3-live':
                public=[e for e in events if e.get('event')=='task_event']
                reply='\n'.join(e.get('text','') for e in public)
                checks['p3_one_unified_reply']=len(public)==1
                checks['p3_verified_facts']=all(v in reply for v in ['1001','智能猫砂盆','成都','SF213502378238'])
                checks['p3_no_invites_or_internal']=not any(v in reply for v in ['需要我','随时告诉我','本次事项处理结果','PENDING','DONE','内部状态码'])
                history=await client.get(f'/api/conversations/{cid}/messages')
                restored=await client.get(f'/api/conversations/{cid}/task-events',params={'user_id':'p2-synthetic'})
                rows=history.json()['items'];replayed=restored.json()['items']
                checks['p3_history_matches_public_events']=len([m for m in rows if m['role']=='assistant'])==len(replayed)
                checks['p3_replay_deduplicated']=len({e['message_id'] for e in replayed})==len(replayed)
                checks['p3_no_internal_messages']=not any(m['content'].startswith('本次事项处理结果') for m in rows)
                (folder/'history.json').write_text(json.dumps({'messages':rows,'events':replayed},ensure_ascii=False,indent=2),encoding='utf-8')
            checks['one_selection_total']=sum(e.get('kind')=='select_order' for r in records for e in r['events'])==1
            async with base.async_session() as session:
                tickets=(await session.execute(select(models.Ticket).where(models.Ticket.conversation_id==cid))).scalars().all()
                audits=(await session.execute(select(models.ToolAuditLog).where(models.ToolAuditLog.conversation_id==cid))).scalars().all()
            checks['two_unique_tickets']=len(tickets)==2 and {t.ticket_type for t in tickets}=={'退款','投诉'}
            checks['fresh_business_tools']=len(audits)==3 and {a.tool_name for a in audits}=={'query_order','query_logistics'}
            checks['no_internal_task_dump']=not any('本次事项处理结果' in e.get('text','') for r in records for e in r['events'])
            checks['one_user_record_per_turn']=sum(m.role=='user' for m in await repository_messages(cid))==2
            details={'conversation_id':cid,'tickets':[{'type':t.ticket_type,'number':t.ticket_no,'description':t.description} for t in tickets],
                     'followup_tasks':tasks,'followup_results':snapshot.values.get('subflow_results',{}),
                     'audits':[{'tool':a.tool_name} for a in audits]}
            (folder/'details.json').write_text(json.dumps(details,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
        # A second synthetic conversation injects response loss AFTER actual SQL
        # commit, then resumes through the public HTTP endpoint after reconnect.
        async with AsyncSqliteSaver.from_conn_string(filename) as saver:
            runtime._graph=build_graph(saver)
            _,events=await post(client,'/api/chat',{'user_id':'p2-synthetic',
                'message':'我要为订单1001申请退款，还要投诉商家。','multi_intent_enabled':True})
            pending=card(events);cid2=pending['conversation_id']
            refund=next(a for a in pending['actions'] if a['preview']['ticket_type']=='退款')
            real_submit=workflow.submit; lost=[]
            async def lose_response(*args,**kwargs):
                receipt=await real_submit(*args,**kwargs)
                if receipt.get('ticket_no'):lost.append(receipt['ticket_no'])
                if len(lost)==1:raise ConnectionError('synthetic response loss after committed ticket')
                return receipt
            workflow.submit=lose_response
            body={'conversation_id':cid2,'user_id':'p2-synthetic','interrupt_id':pending['interrupt_id'],
                'operation_id':refund['operation_id'],'confirmed':True}
            _,events=await post(client,'/api/actions/resume',body)
            retry=card(events)
            checks['lost_response_exposes_retry_card']=any(a['operation_id']==refund['operation_id'] for a in retry['actions'])
        async with AsyncSqliteSaver.from_conn_string(filename) as saver:
            runtime._graph=build_graph(saver)
            body['interrupt_id']=retry['interrupt_id']
            _,events=await post(client,'/api/actions/resume',body)
            checks['lost_response_same_ticket']=len(lost)==2 and lost[0]==lost[1]
            workflow.submit=real_submit
            pending=card(events);complaint=pending['actions'][0]
            _,events=await post(client,'/api/actions/resume',{'conversation_id':cid2,'user_id':'p2-synthetic',
                'interrupt_id':pending['interrupt_id'],'operation_id':complaint['operation_id'],'confirmed':False})
            async with base.async_session() as session:
                ts=(await session.execute(select(models.Ticket).where(models.Ticket.conversation_id==cid2))).scalars().all()
            checks['lost_response_one_refund_no_complaint']=len(ts)==1 and ts[0].ticket_type=='退款'
            checks['cancel_complaint_finishes']=events[-1].get('event')=='done'
    checks['kb_count_unchanged']=await milvus_client.acall(lambda:milvus_client.get_client().get_collection_stats('knowledge'))==kb_before
    save()
    await base.engine.dispose();await original_engine.dispose()
    print(json.dumps({'folder':str(folder),'checks':checks}),flush=True)
    assert all(checks.values()),checks


async def repository_messages(cid):
    from app.db.repository import list_dialog_messages
    return await list_dialog_messages(cid)


if __name__=='__main__':asyncio.run(main())
