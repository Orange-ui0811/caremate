"""P2 integration: real MySQL receipts and SQLite checkpoints, isolated fixtures."""
import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from sqlalchemy import select

from app.config import settings
from app.core import task_planner
from app.db import repository, workflow
from app.db.models import Ticket
from app.graph import nodes, runtime, subflow_tasks
from app.graph.build import build_graph


@pytest.fixture
async def wired(db_session_factory, _test_engine, monkeypatch):
    async with _test_engine.begin() as conn:
        await conn.run_sync(lambda c: workflow.WorkflowRecord.__table__.drop(c,checkfirst=True))
    monkeypatch.setattr(workflow.base,'engine',_test_engine)
    await workflow.ensure_tables()
    monkeypatch.setattr(settings,'multi_intent_enabled',True)
    monkeypatch.setattr(settings,'multi_intent_subflows_enabled',True)
    async def policy(state):return {'citations':[{'id':1,'answer':'质量问题申请售后审核'}]}
    monkeypatch.setattr(nodes,'retrieve_policy',policy)
    async def no_summary(*a,**kw):pass
    monkeypatch.setattr(runtime.summarizer,'maybe_schedule_summary',no_summary)
    calls=[]
    async def read(state):
        task=dict(state['tasks'][0]); calls.append((task['kind'],task['order_id']))
        task['status']='DONE'
        return {'tasks':[task],'results':[{'answer':f"订单{task['order_id']}查询完成",'status':'DONE'}]}
    monkeypatch.setattr(subflow_tasks,'execute_read',read)
    return calls


def planner(monkeypatch, kinds, *, raw_source=None, oid=None):
    async def extract(raw):
        # Real binder attaches identical explicit referent groups from raw text.
        return [{'kind':k,'query':raw,'source':raw_source or raw,'order_id':oid,
                 'binding':{'status':'resolved','method':'task_clause'} if oid else {}}
                for k in kinds]
    monkeypatch.setattr(task_planner,'extract',extract)


async def tickets():
    async with workflow.base.async_session() as s:
        return (await s.execute(select(Ticket))).scalars().all()


async def invoke_new(graph, monkeypatch, raw):
    monkeypatch.setattr(runtime,'_graph',graph)
    return await runtime.run_turn('alice',raw,None,multi_intent_enabled=True)


async def resume(graph, monkeypatch, cid, value):
    monkeypatch.setattr(runtime,'_graph',graph)
    config={'configurable':{'thread_id':str(cid)}}
    snapshot=await graph.aget_state(config)
    intr=next(i for t in snapshot.tasks for i in t.interrupts)
    out=await runtime.resume_turn(cid,value,user_id='alice',interrupt_id=intr.id)
    return out,intr.id


async def test_select_once_reconnect_and_continue_all_tasks(wired,tmp_path,monkeypatch):
    planner(monkeypatch,['order','logistics','refund','complaint'])
    filename=str(tmp_path/'p2.sqlite')
    async with AsyncSqliteSaver.from_conn_string(filename) as saver:
        graph=build_graph(saver)
        first=await invoke_new(graph,monkeypatch,'这笔订单是什么，它物流到哪了，我还要为这笔订单退款，还要投诉商家')
        cid=first['conversation_id']
        assert first['interrupt']['type']=='select_order'
        # Complaint preparation continues before the order wait is exposed.
        assert first['state']['task_plan']['complaint']['step']=='confirm_submission'
        assert await tickets()==[]
        state=first['state']['task_plan']
        assert all(state[k]['status']=='WAITING_USER' for k in ('order','logistics','refund'))
    async with AsyncSqliteSaver.from_conn_string(filename) as saver:
        graph=build_graph(saver)
        out,old_selection=await resume(graph,monkeypatch,cid,'1001')
        assert out['interrupt']['type']=='task_actions'
        assert wired==[('order','1001'),('logistics','1001')]
        assert len(out['interrupt']['actions'])==2
        assert all(out['state']['task_plan'][k]['order_id']=='1001' for k in ('order','logistics','refund'))
        with pytest.raises(runtime.ResumeRejected):
            await runtime.validate_resume(cid,'1001',user_id='alice',interrupt_id=old_selection)
        refund=next(a for a in out['interrupt']['actions'] if a['preview']['ticket_type']=='退款')
        out,old_confirm=await resume(graph,monkeypatch,cid,{'operation_id':refund['operation_id'],'confirmed':True})
        assert len(await tickets())==1
    # Refund DONE survives restart; only complaint remains to be confirmed.
    async with AsyncSqliteSaver.from_conn_string(filename) as saver:
        graph=build_graph(saver)
        monkeypatch.setattr(runtime,'_graph',graph)
        snapshot=await graph.aget_state({'configurable':{'thread_id':str(cid)}})
        assert snapshot.values['task_plan']['refund']['status']=='DONE'
        with pytest.raises(runtime.ResumeRejected):
            await runtime.validate_resume(cid,{'operation_id':refund['operation_id'],'confirmed':True},user_id='alice',interrupt_id=old_confirm)
        action=snapshot.tasks[0].interrupts[0].value['actions'][0]
        out,_=await resume(graph,monkeypatch,cid,{'operation_id':action['operation_id'],'confirmed':True})
        assert out['interrupt'] is None
        assert all(t['status']=='DONE' for t in out['state']['task_plan'].values())
        assert len(await tickets())==2
        assert len(wired)==2
        assert sum(m.role=='user' for m in await repository.list_dialog_messages(cid))==1
        planner(monkeypatch,['order','logistics'])
        followup=await runtime.run_turn('alice','刚刚的订单是哪个，它现在物流到哪了',cid,multi_intent_enabled=True)
        assert followup['interrupt'] is None
        assert wired[-2:]==[('order','1001'),('logistics','1001')]
        assert len(await tickets())==2


@pytest.mark.parametrize('retry_confirmed',[True,False])
async def test_commit_response_lost_retries_receipt_without_duplicate(wired,tmp_path,monkeypatch,retry_confirmed):
    planner(monkeypatch,['refund','complaint'],oid='1001')
    filename=str(tmp_path/'lost.sqlite')
    async with AsyncSqliteSaver.from_conn_string(filename) as saver:
        graph=build_graph(saver)
        first=await invoke_new(graph,monkeypatch,'我要为订单1001退款，还要投诉商家')
        cid=first['conversation_id']; action=first['interrupt']['actions'][0]
        real=workflow.submit; attempts=[]
        async def lost(*args,**kwargs):
            receipt=await real(*args,**kwargs); attempts.append(receipt['ticket_no'])
            if len(attempts)==1:raise ConnectionError('injected response loss after SQL commit')
            return receipt
        monkeypatch.setattr(workflow,'submit',lost)
        retry,_=await resume(graph,monkeypatch,cid,{'operation_id':action['operation_id'],'confirmed':True})
        assert len(await tickets())==1
        assert any(a['operation_id']==action['operation_id'] for a in retry['interrupt']['actions'])
    async with AsyncSqliteSaver.from_conn_string(filename) as saver:
        graph=build_graph(saver)
        out,_=await resume(graph,monkeypatch,cid,{'operation_id':action['operation_id'],'confirmed':retry_confirmed})
        assert len(await tickets())==1 and attempts[0]==attempts[1]
        assert out['interrupt']['actions'][0]['preview']['ticket_type']=='投诉'


@pytest.mark.parametrize('cancel_selection',[True,False])
async def test_cancel_one_task_continues_complaint(wired,tmp_path,monkeypatch,cancel_selection):
    planner(monkeypatch,['refund','complaint'],oid=None if cancel_selection else '1001')
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'cancel.sqlite')) as saver:
        graph=build_graph(saver)
        first=await invoke_new(graph,monkeypatch,'我要退款，还要投诉商家'); cid=first['conversation_id']
        if cancel_selection:
            out,_=await resume(graph,monkeypatch,cid,{'cancelled':True})
        else:
            refund=next(a for a in first['interrupt']['actions'] if a['preview']['ticket_type']=='退款')
            out,_=await resume(graph,monkeypatch,cid,{'operation_id':refund['operation_id'],'confirmed':False})
        assert out['state']['task_plan']['refund']['status']=='CANCELLED'
        action=out['interrupt']['actions'][0]
        out,_=await resume(graph,monkeypatch,cid,{'operation_id':action['operation_id'],'confirmed':True})
        assert len(await tickets())==1
        assert (await tickets())[0].ticket_type=='投诉'


async def test_unrelated_orders_not_shared_and_invalid_resume_no_write(wired,tmp_path,monkeypatch):
    async def extract(raw):
        return [{'kind':'logistics','query':'查1001物流','source':'查订单1001物流','order_id':'1001'},
                {'kind':'refund','query':'另一笔退款','source':'我要为另一笔订单退款','order_id':None}]
    monkeypatch.setattr(task_planner,'extract',extract)
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'separate.sqlite')) as saver:
        graph=build_graph(saver)
        first=await invoke_new(graph,monkeypatch,'查订单1001物流，我要为另一笔订单退款'); cid=first['conversation_id']
        assert wired==[('logistics','1001')]
        assert first['state']['task_plan']['refund'].get('order_id') is None
        iid=first['interrupt']['interrupt_id']
        for invalid in ['9999',{'confirmed':True}]:
            with pytest.raises(runtime.ResumeRejected):
                await runtime.validate_resume(cid,invalid,user_id='alice',interrupt_id=iid)
        with pytest.raises(runtime.ConversationNotFound):
            await runtime.validate_resume(cid,'1001',user_id='bob',interrupt_id=iid)
        assert await tickets()==[]


async def test_policy_failure_does_not_block_complaint(wired,tmp_path,monkeypatch):
    planner(monkeypatch,['refund','complaint'],oid='1001')
    async def broken(state):raise TimeoutError('policy unavailable')
    monkeypatch.setattr(nodes,'retrieve_policy',broken)
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'failure.sqlite')) as saver:
        graph=build_graph(saver)
        out=await invoke_new(graph,monkeypatch,'我要为1001退款，还要投诉商家')
        assert out['state']['task_plan']['refund']['status']=='FAILED'
        assert len(out['interrupt']['actions'])==1
        assert out['interrupt']['actions'][0]['preview']['ticket_type']=='投诉'


async def test_http_sse_safety_selection_confirm_replay(wired,tmp_path,monkeypatch):
    import json
    import httpx
    from fastapi import FastAPI
    from app.api.chat import router as chat_router
    from app.api.actions import router as actions_router
    app=FastAPI();app.include_router(chat_router);app.include_router(actions_router)
    planner(monkeypatch,['refund','complaint'])
    def events(response):
        assert response.status_code==200,response.text
        return [json.loads(s[6:]) for s in response.text.splitlines() if s.startswith('data: {')]
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'http.sqlite')) as saver:
        monkeypatch.setattr(runtime,'_graph',build_graph(saver))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
            first=events(await client.post('/api/chat',json={'user_id':'alice','message':'我买的猫砂盆漏电了，我要退款，还要投诉商家','multi_intent_enabled':True}))
            card=next(e for e in first if e.get('event')=='interrupt');cid=card['conversation_id']
            assert first[0]['event']=='task_event' and '停止使用' in first[0]['text']
            assert card['kind']=='select_order' and await tickets()==[]
            body={'conversation_id':cid,'user_id':'alice','interrupt_id':card['interrupt_id'],'order_id':'1001'}
            second=events(await client.post('/api/actions/resume',json=body))
            assert (await client.post('/api/actions/resume',json=body)).status_code==409
            card=next(e for e in second if e.get('event')=='interrupt')
            for typ in ['退款','投诉']:
                action=next(a for a in card['actions'] if a['preview']['ticket_type']==typ)
                body={'conversation_id':cid,'user_id':'alice','interrupt_id':card['interrupt_id'],
                      'operation_id':action['operation_id'],'confirmed':True}
                reply=events(await client.post('/api/actions/resume',json=body))
                assert (await client.post('/api/actions/resume',json=body)).status_code==409
                if typ=='退款':card=next(e for e in reply if e.get('event')=='interrupt')
            assert len(await tickets())==2
            assert reply[-1]['event']=='done'
            assert sum(m.role=='user' for m in await repository.list_dialog_messages(cid))==1
            assert not any('本次事项处理结果' in e.get('text','') for e in first+second+reply)


async def test_new_message_invalidates_pending_confirmation(wired,tmp_path,monkeypatch):
    planner(monkeypatch,['refund','complaint'],oid='1001')
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'supersede.sqlite')) as saver:
        graph=build_graph(saver)
        first=await invoke_new(graph,monkeypatch,'为订单1001申请退款和投诉'); cid=first['conversation_id']
        old=first['interrupt']; action=old['actions'][0]
        planner(monkeypatch,['complaint'])
        await runtime.run_turn('alice','我现在只要投诉商家',cid,multi_intent_enabled=True)
        with pytest.raises(runtime.ResumeRejected):
            await runtime.validate_resume(cid,{'operation_id':action['operation_id'],'confirmed':True},
                user_id='alice',interrupt_id=old['interrupt_id'])
        assert await tickets()==[]


async def test_concurrent_duplicate_submissions_share_one_sql_receipt(wired):
    import asyncio
    cid=await repository.create_conversation('alice')
    args={'ticket_type':'投诉','description':'同一确认操作的并发重试'}
    receipts=await asyncio.gather(*(workflow.submit(cid,'alice',f'c{cid}:same-operation',args,confirmed=True)
                                    for _ in range(6)))
    assert len({r['ticket_no'] for r in receipts})==1
    assert len(await tickets())==1
    with pytest.raises(ValueError):
        await workflow.submit(cid,'alice',f'c{cid}:same-operation',{**args,'description':'被修改的预览'},confirmed=True)
    assert len(await tickets())==1
