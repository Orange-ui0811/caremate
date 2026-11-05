"""Real SQLite checkpoints and transactional SQL receipts; model/API calls stubbed."""
import json

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from sqlalchemy import select

from app.config import settings
from app.db import repository, workflow
from app.db.models import Ticket
from app.graph import nodes, runtime, tasks
from app.graph.build import build_graph


@pytest.fixture
async def data(db_session_factory, _test_engine, monkeypatch):
    async with _test_engine.begin() as conn:
        await conn.run_sync(lambda c: workflow.WorkflowRecord.__table__.drop(c, checkfirst=True))
        await conn.run_sync(lambda c: workflow.WorkflowRecord.__table__.create(c))
    monkeypatch.setattr(workflow.base, "engine", _test_engine)
    await workflow.ensure_tables()
    monkeypatch.setattr(settings, "multi_intent_enabled", True)
    async def fixture_plan(raw):
        return [{'kind':kind,'query':raw,'source':raw,'order_id':None,'order_reference':''}
                for kind in ('refund','complaint')]
    monkeypatch.setattr(tasks.task_planner,'extract',fixture_plan)
    cid = await repository.create_conversation("alice")
    msg = "我的猫砂盆漏电了，我要退款，还要投诉商家。"
    mid = await repository.append_message(cid, "user", content=msg)
    async def policy(state):
        return {"citations": [{"id": 1, "answer": "质量问题需审核"}]}
    monkeypatch.setattr(nodes, "retrieve_policy", policy)
    original_snapshot = tasks.business.order_snapshot
    monkeypatch.setattr(tasks.business, "order_snapshot", lambda oid: {**original_snapshot(oid), "status": "已签收"})
    return cid, mid, msg


async def test_safety_precedes_interrupt_and_restart_continues_all_tasks(data, tmp_path, monkeypatch):
    cid, mid, msg = data
    config = {"configurable": {"thread_id": str(cid)}}
    filename = str(tmp_path / "checkpoint.sqlite")
    async with AsyncSqliteSaver.from_conn_string(filename) as saver:
        graph = build_graph(saver)
        monkeypatch.setattr(runtime, "_graph", graph)
        output = [e async for e in runtime._stream_events(cid, runtime._graph_input("alice", msg, cid, mid, "", 0))]
        assert output[0]["type"] == "task_event"
        assert output[-1]["type"] == "interrupt" and output[-1]["kind"] == "select_order"
        state = (await graph.aget_state(config)).values
        assert {k: t["status"] for k, t in state["task_plan"].items()} == {
            "safety": "DONE", "refund": "WAITING_USER", "complaint": "PENDING"}
    # New connection and freshly compiled graph, not MemorySaver.
    async with AsyncSqliteSaver.from_conn_string(filename) as saver:
        graph = build_graph(saver)
        monkeypatch.setattr(runtime, "_graph", graph)
        snapshot = await graph.aget_state(config)
        iid = next(intr.id for task in snapshot.tasks for intr in task.interrupts)
        await runtime.validate_resume(cid, "1001", user_id="alice", interrupt_id=iid)
        out = await graph.ainvoke(Command(resume="1001"), config)
        assert all(out["task_plan"][k]["status"] == "WAITING_USER" for k in ("refund", "complaint"))
        actions = out["__interrupt__"][0].value["actions"]
        assert len(actions) == 2
        # A stale order card cannot confirm a new operation.
        with pytest.raises(runtime.ResumeRejected):
            await runtime.validate_resume(cid, "1001", user_id="alice", interrupt_id=iid)
        for action in actions:
            out = await graph.ainvoke(Command(resume={"operation_id": action["operation_id"], "confirmed": True}), config)
        assert out["task_plan"]["refund"]["status"] == "DONE"
        assert out["task_plan"]["complaint"]["status"] == "DONE"
    async with workflow.base.async_session() as session:
        assert len((await session.execute(select(Ticket))).scalars().all()) == 2
    events = await workflow.events(cid, "alice")
    assert sum(e["text"] == tasks.SAFETY_NOTICE for e in events) == 1
    await workflow.ack(cid, "alice", events[0]["event_id"])
    assert (await workflow.events(cid, "alice"))[0]["acked"] is True


async def test_cancel_refund_does_not_cancel_complaint(data, tmp_path):
    cid, mid, msg = data
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "cancel.sqlite")) as saver:
        graph = build_graph(saver)
        config = {"configurable": {"thread_id": str(cid)}}
        await graph.ainvoke(runtime._graph_input("alice", msg, cid, mid, "", 0), config)
        out = await graph.ainvoke(Command(resume={"cancelled": True}), config)
        assert out["task_plan"]["refund"]["status"] == "CANCELLED"
        assert out["task_plan"]["complaint"]["status"] == "WAITING_USER"


async def test_receipt_replay_after_success_before_checkpoint(data):
    cid, mid, _ = data
    args = {"description": "质量投诉", "ticket_type": "投诉"}
    first = await workflow.submit(cid, "alice", "fixed-op", args, confirmed=True)
    again = await workflow.submit(cid, "alice", "fixed-op", args, confirmed=True)
    assert first["ticket_no"] == again["ticket_no"]
    with pytest.raises(ValueError):
        await workflow.submit(cid, "alice", "fixed-op", {**args, "description": "changed"}, confirmed=True)
    with pytest.raises(ValueError):
        await workflow.events(cid, "bob")


async def test_new_user_message_invalidates_pending_card(data, tmp_path, monkeypatch):
    cid, mid, msg = data
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "stale.sqlite")) as saver:
        graph = build_graph(saver)
        monkeypatch.setattr(runtime, "_graph", graph)
        config = {"configurable": {"thread_id": str(cid)}}
        out = await graph.ainvoke(runtime._graph_input("alice", msg, cid, mid, "", 0), config)
        await repository.append_message(cid, "user", content="不用退款了")
        with pytest.raises(runtime.ResumeRejected):
            await runtime.validate_resume(cid, "1001", user_id="alice", interrupt_id=out["__interrupt__"][0].id)


@pytest.mark.parametrize("text", ["我的猫砂盆没有漏电，不要退款，不要投诉", "如果漏电了怎么办", "他说漏电了"])
def test_negated_and_hypothetical_safety_does_not_plan(text):
    assert tasks.explicit_tasks(text) == (False, False, False)


async def test_write_success_checkpoint_failure_is_reconciled(data, tmp_path, monkeypatch):
    cid, mid, msg = data
    config = {"configurable": {"thread_id": str(cid)}}
    path = str(tmp_path / "crash.sqlite")
    real_commit = tasks.commit_action
    async def crash_after_commit(state):
        await real_commit(state)
        raise RuntimeError("simulated checkpoint gap")
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        monkeypatch.setattr(tasks, "commit_action", crash_after_commit)
        graph = build_graph(saver)
        await graph.ainvoke(runtime._graph_input("alice", msg, cid, mid, "", 0), config)
        out = await graph.ainvoke(Command(resume="1001"), config)
        action = out["__interrupt__"][0].value["actions"][0]
        with pytest.raises(RuntimeError, match="checkpoint gap"):
            await graph.ainvoke(Command(resume={"operation_id": action["operation_id"], "confirmed": True}), config)
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        monkeypatch.setattr(tasks, "commit_action", real_commit)
        graph = build_graph(saver)
        out = await graph.ainvoke(None, config)
        assert out["task_plan"]["refund"]["status"] == "DONE"
    async with workflow.base.async_session() as session:
        assert len((await session.execute(select(Ticket))).scalars().all()) == 1
    assert sum("工单已创建" in e["text"] for e in await workflow.events(cid, "alice")) == 1


async def test_concurrent_duplicate_write_and_cancel_are_idempotent(data):
    import asyncio
    cid, _, _ = data
    args = {"ticket_type": "投诉", "description": "并发确认"}
    receipts = await asyncio.gather(*(workflow.submit(cid, "alice", "double-click", args, confirmed=True)
                                      for _ in range(2)))
    assert receipts[0]["ticket_no"] == receipts[1]["ticket_no"]
    cancelled = await workflow.submit(cid, "alice", "cancel-only", args, confirmed=False)
    assert (await workflow.submit(cid, "alice", "cancel-only", args, confirmed=True)) == cancelled


async def test_policy_failure_still_advances_complaint(data, tmp_path, monkeypatch):
    cid, mid, msg = data
    async def unavailable(state):
        raise RuntimeError("policy offline")
    monkeypatch.setattr(nodes, "retrieve_policy", unavailable)
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "failure.sqlite")) as saver:
        graph = build_graph(saver)
        config = {"configurable": {"thread_id": str(cid)}}
        await graph.ainvoke(runtime._graph_input("alice", msg, cid, mid, "", 0), config)
        out = await graph.ainvoke(Command(resume="1001"), config)
        assert out["task_plan"]["refund"]["status"] == "FAILED"
        assert out["task_plan"]["complaint"]["status"] == "WAITING_USER"


@pytest.mark.parametrize("kind", ["create-ticket", "create-refund"])
async def test_direct_form_write_receipt_and_parameter_conflict(data, kind, monkeypatch):
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    from app.api.actions import router
    monkeypatch.setattr(settings, "context_policy", "legacy")
    # This router-only fixture has no initialized checkpointer.
    from unittest.mock import AsyncMock
    from types import SimpleNamespace
    monkeypatch.setattr(runtime, 'conversation_mode', AsyncMock(return_value=True))
    monkeypatch.setattr(runtime, 'get_graph', lambda: SimpleNamespace(
        aget_state=AsyncMock(return_value=SimpleNamespace(values={'task_plan_version':1}))))
    cid, _, _ = data
    app = FastAPI()
    app.include_router(router)
    payload = {"conversation_id": cid, "user_id": "alice", "request_id": "stable-form"}
    if kind == "create-ticket":
        payload.update(description="质量投诉", ticket_type="投诉")
    else:
        payload.update(order_id="1001", reason="质量问题")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        a = await client.post('/api/actions/' + kind, json=payload)
        b = await client.post('/api/actions/' + kind, json=payload)
        assert a.status_code == b.status_code == 200
        assert a.json()["ticket_no"] == b.json()["ticket_no"]
        payload["description" if kind == "create-ticket" else "reason"] = "changed" if kind == "create-ticket" else "其他"
        assert (await client.post('/api/actions/' + kind, json=payload)).status_code == 409


async def test_third_task_runs_and_its_order_cannot_be_used_for_refund(data,tmp_path,monkeypatch):
    from app.tools import registry
    from langchain_core.tools import tool
    import app.tools.builtin.orders as order_tools
    cid,_,_=data
    raw='查2002物流，我要退款，还要投诉商家'
    mid=await repository.append_message(cid,'user',content=raw)
    async def planned(text):
        return [
            {'kind':'logistics','query':'查2002物流','source':'查2002物流','order_id':'2002','order_reference':'查2002物流'},
            {'kind':'refund','query':'我要退款','source':'我要退款','order_id':None,'order_reference':''},
            {'kind':'complaint','query':'投诉商家','source':'投诉商家','order_id':None,'order_reference':''}]
    monkeypatch.setattr(tasks.task_planner,'extract',planned)
    monkeypatch.setattr(order_tools,'order_snapshot',tasks.business.order_snapshot)
    @tool
    async def query_logistics(tracking_no:str)->dict:
        """查询物流位置。"""
        return {'current_city':'杭州','status':'运输中','tracking_no':tracking_no}
    async def specs():
        registry.scan_builtin()
        return [*registry.builtin_specs(),registry.spec_from_langchain_tool(query_logistics,source='mcp')]
    monkeypatch.setattr(registry,'get_all_specs',specs)
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'three.sqlite')) as saver:
        graph=build_graph(saver);config={'configurable':{'thread_id':str(cid)}}
        out=await graph.ainvoke(runtime._graph_input('alice',raw,cid,mid,'',0),config)
        assert out['task_plan']['logistics']['status']=='DONE'
        assert out['task_plan']['logistics']['order_id']=='2002'
        assert out['task_plan']['refund']['step']=='select_order'
        out=await graph.ainvoke(Command(resume='1001'),config)
        assert out['task_plan']['refund']['order_id']=='1001'
        assert len(out['__interrupt__'][0].value['actions'])==2


async def test_same_kind_tasks_have_distinct_ids_and_confirmations(data,tmp_path,monkeypatch):
    cid,_,_=data;raw='1001申请退款，2002也申请退款'
    mid=await repository.append_message(cid,'user',content=raw)
    async def planned(text):
        return [{'kind':'refund','query':oid+'申请退款','source':oid+'申请退款',
                 'order_id':oid,'order_reference':oid+'申请退款'} for oid in ('1001','2002')]
    monkeypatch.setattr(tasks.task_planner,'extract',planned)
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'two-refunds.sqlite')) as saver:
        graph=build_graph(saver);config={'configurable':{'thread_id':str(cid)}}
        out=await graph.ainvoke(runtime._graph_input('alice',raw,cid,mid,'',0),config)
        assert set(out['task_plan'])=={'refund','refund:2'}
        pending=out['__interrupt__'][0].value['actions']
        assert len({a['operation_id'] for a in pending})==2
        for action in pending:
            out=await graph.ainvoke(Command(resume={'operation_id':action['operation_id'],'confirmed':True}),config)
        assert all(t['status']=='DONE' for t in out['task_plan'].values())


async def test_planner_failure_never_falls_back_to_partial_business_execution(data,tmp_path,monkeypatch):
    cid,mid,raw=data
    async def broken(text):raise ValueError('planner unavailable')
    monkeypatch.setattr(tasks.task_planner,'extract',broken)
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'plan-error.sqlite')) as saver:
        out=await build_graph(saver).ainvoke(runtime._graph_input('alice',raw,cid,mid,'',0),
                                            {'configurable':{'thread_id':str(cid)}})
        assert out['task_planning_error'] is True
        assert '尚未发起' in out['answer']
    async with workflow.base.async_session() as session:
        assert not (await session.execute(select(Ticket))).scalars().all()


async def test_owned_order_from_full_source_skips_selection_without_skipping_confirmation(data,tmp_path,monkeypatch):
    cid,_,_=data;raw='订单1001商品坏了，请办理退款，并登记投诉工单'
    mid=await repository.append_message(cid,'user',content=raw)
    async def planned(text):
        return tasks.task_planner.normalize(text,[
            {'kind':'refund','query':'办理退款','source':'订单1001商品坏了，请办理退款','order_id':'1001','order_reference':'订单1001'},
            {'kind':'complaint','query':'登记投诉工单','source':'登记投诉工单'}])
    monkeypatch.setattr(tasks.task_planner,'extract',planned)
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'known-order.sqlite')) as saver:
        graph=build_graph(saver)
        out=await graph.ainvoke(runtime._graph_input('alice',raw,cid,mid,'',0),{'configurable':{'thread_id':str(cid)}})
        assert out['__interrupt__'][0].value['type']=='task_actions'
        assert out['task_plan']['refund']['order_id']=='1001'
        binding=out['task_plan']['refund']['binding']
        assert binding['status']=='resolved'
        assert all(raw[p['start']:p['end']]==p['text'] for p in binding['evidence'])
        assert '1001' in ''.join(p['text'] for p in binding['evidence'])
        assert '办理退款' in ''.join(p['text'] for p in binding['evidence'])
        assert len(out['__interrupt__'][0].value['actions'])==2
    async with workflow.base.async_session() as session:
        assert not (await session.execute(select(Ticket))).scalars().all()


async def test_missing_order_selection_is_task_scoped_and_persisted_before_confirm(data,tmp_path):
    cid,mid,raw=data
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'selected-order.sqlite')) as saver:
        graph=build_graph(saver);config={'configurable':{'thread_id':str(cid)}}
        out=await graph.ainvoke(runtime._graph_input('alice',raw,cid,mid,'',0),config)
        intr=out['__interrupt__'][0].value
        assert intr['type']=='select_order' and '退款/退货审核' in intr['selection_prompt']
        out=await graph.ainvoke(Command(resume='1001'),config)
        assert out['__interrupt__'][0].value['type']=='task_actions'
        saved=(await graph.aget_state(config)).values
        assert saved['task_plan']['refund']['order_id']=='1001'
        assert saved['task_plan']['refund']['binding']['method']=='user_selection'


async def test_browser_modes_are_isolated_and_resume_keeps_checkpoint_mode(data, tmp_path, monkeypatch):
    import asyncio
    monkeypatch.setattr(settings, 'multi_intent_enabled', False)
    async def coref(query, *args, **kwargs): return query
    async def classify(*args, **kwargs): return {'intent':'投诉','confidence':1.0}
    async def no_summary(cid): pass
    monkeypatch.setattr(nodes.coref, 'resolve', coref)
    monkeypatch.setattr(nodes.intent_mod, 'classify', classify)
    monkeypatch.setattr(runtime.summarizer, 'maybe_schedule_summary', no_summary)
    filename=str(tmp_path/'browser-modes.sqlite')
    async with AsyncSqliteSaver.from_conn_string(filename) as saver:
        monkeypatch.setattr(runtime, '_graph', build_graph(saver))
        single,multi=await asyncio.gather(
            runtime.run_turn('alice','我要退款，还要投诉商家',None,multi_intent_enabled=False),
            runtime.run_turn('alice','我要退款，还要投诉商家',None,multi_intent_enabled=True))
        assert single['state']['task_plan']=={}
        assert single['state']['multi_intent_enabled'] is False
        assert set(multi['state']['task_plan'])=={'refund','complaint'}
        assert multi['state']['multi_intent_enabled'] is True
        assert multi['interrupt']['type']=='select_order'
        assert settings.multi_intent_enabled is False
        before=len(await repository.list_dialog_messages(multi['conversation_id']))
        with pytest.raises(runtime.ResumeRejected,match='新对话'):
            await runtime.run_turn('alice','换模式',multi['conversation_id'],multi_intent_enabled=False)
        assert len(await repository.list_dialog_messages(multi['conversation_id']))==before
    # The browser preference/process default cannot change a paused conversation.
    async with AsyncSqliteSaver.from_conn_string(filename) as saver:
        monkeypatch.setattr(runtime, '_graph', build_graph(saver))
        assert await runtime.conversation_mode(multi['conversation_id']) is True
        out=await runtime.resume_turn(multi['conversation_id'],'1001',user_id='alice',interrupt_id=multi['interrupt']['interrupt_id'])
        assert out['interrupt']['type']=='task_actions'
        assert len(out['interrupt']['actions'])==2
        for _ in range(2):
            action=out['interrupt']['actions'][0]
            out=await runtime.resume_turn(multi['conversation_id'],{'operation_id':action['operation_id'],'confirmed':True},
                user_id='alice',interrupt_id=out['interrupt']['interrupt_id'])
        assert out['interrupt'] is None
        assert all(t['status']=='DONE' for t in out['state']['task_plan'].values())
    async with workflow.base.async_session() as session:
        tickets=(await session.execute(select(Ticket).where(Ticket.conversation_id==multi['conversation_id']))).scalars().all()
        assert sorted(t.ticket_type for t in tickets)==sorted(['退款','投诉'])


async def test_selected_order_survives_cancel_restart_and_three_task_followup(data,tmp_path,monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from langchain_core.tools import tool
    from app.core import llm
    from app.tools import registry
    raw='我刚买的猫砂盆漏电了，我要退款，而且我要投诉商家。'
    followup='刚刚的订单是哪个，它现在物流到哪了，可以退货吗'
    async def planned(text):
        if text==followup:
            return tasks.task_planner.normalize(text,[{'kind':k,'query':q,'source':q} for k,q in
                zip(('order','logistics','knowledge'),text.split('，'))])
        return [{'kind':k,'query':text,'source':text,'order_id':None} for k in ('refund','complaint')]
    monkeypatch.setattr(tasks.task_planner,'extract',planned)
    monkeypatch.setattr(runtime.summarizer,'maybe_schedule_summary',AsyncMock())
    registry.scan_builtin()
    @tool
    async def query_logistics(tracking_no:str)->dict:
        """合成物流测试。"""
        return {'current_city':'成都','status':'PICKED_UP'}
    logistics=registry.spec_from_langchain_tool(query_logistics,source='mcp')
    async def specs():return [*registry._BUILTIN.values(),logistics]
    monkeypatch.setattr(registry,'get_all_specs',specs)
    inputs=[]
    async def knowledge(scoped):
        inputs.append(scoped['resolved_query'])
        return {'evidence_strong':True,'evidence':'质量问题可申请售后审核[1]','citations':[{'id':1,'answer':'质量问题需审核'}]}
    monkeypatch.setattr(nodes,'retrieve_knowledge',knowledge)
    calls=[]
    async def answer(messages):
        calls.append(messages)
        return SimpleNamespace(content='订单1001是智能猫砂盆。您之前反映漏电，可以申请质量问题售后审核；是否符合条件由售后核实。')
    monkeypatch.setattr(llm,'get_chat_model',lambda:SimpleNamespace(ainvoke=answer))
    filename=str(tmp_path/'order-memory.sqlite')
    async with AsyncSqliteSaver.from_conn_string(filename) as saver:
        monkeypatch.setattr(runtime,'_graph',build_graph(saver))
        first=await runtime.run_turn('alice',raw,None,multi_intent_enabled=True)
        cid=first['conversation_id']
        selected=await runtime.resume_turn(cid,'1001',user_id='alice',interrupt_id=first['interrupt']['interrupt_id'])
        refund=next(a for a in selected['interrupt']['actions'] if a['preview']['ticket_type']=='退款')
        cancelled=await runtime.resume_turn(cid,{'operation_id':refund['operation_id'],'confirmed':False},
            user_id='alice',interrupt_id=selected['interrupt']['interrupt_id'])
        assert cancelled['state']['task_plan']['refund']['status']=='CANCELLED'
    async with AsyncSqliteSaver.from_conn_string(filename) as saver:
        monkeypatch.setattr(runtime,'_graph',build_graph(saver))
        events=[e async for e in runtime.stream_turn('alice',followup,cid)]
        assert not any(e['type']=='interrupt' for e in events)
        assert events[-1]['type']=='done'
        st=(await runtime.get_graph().aget_state({'configurable':{'thread_id':str(cid)}})).values
        assert st['task_plan']['order']['order_id']=='1001'
        assert st['task_plan']['logistics']['order_id']=='1001'
        assert st['task_plan']['knowledge']['context_order_id']=='1001'
        assert all(t['status']=='DONE' for t in st['task_plan'].values())
        assert '质量问题' in inputs[0] and '退货退款' in inputs[0]
        assert raw not in inputs[0]  # Lookup policy without treating old requests as new tasks.
        assert '漏电' in calls[0][0]['content'] and '当前问题' in calls[0][0]['content']
        assert not any('本次事项处理结果' in e.get('text','') for e in events)
    rows=await repository.list_dialog_messages(cid)
    assert not any('本次事项处理结果' in (r.content or '') for r in rows)
    async with workflow.base.async_session() as session:
        assert not (await session.execute(select(Ticket).where(Ticket.conversation_id==cid))).scalars().all()


async def test_legacy_internal_summary_is_hidden_without_deleting_audit_or_user_text(data):
    cid,_,_=data
    text='本次事项处理结果：\n订单查询 1001：已处理'
    event=await workflow.event(cid,'alice','legacy:summary',text)
    uid=await repository.append_message(cid,'user',content=text)
    ordinary=await repository.append_message(cid,'assistant',content='本次事项处理结果：这是一条正常解释，未关联内部状态事件')
    await workflow.hide_legacy_summaries()
    await workflow.hide_legacy_summaries()
    visible=await repository.list_dialog_messages(cid)
    assert event['message_id'] not in {r.id for r in visible}
    assert {uid,ordinary} <= {r.id for r in visible}
    audit=await repository.list_messages(cid)
    assert event['message_id'] in {r.id for r in audit}
    assert not any(e['event_id']=='legacy:summary' for e in await workflow.events(cid,'alice'))
