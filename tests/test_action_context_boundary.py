"""Synthetic boundary regressions: no customer DB, model API or real writes."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
import pytest

from app.api import actions, agent
from app.config import settings
from app.db import base
from app.graph import runtime
from app.graph.state import ConversationState


@pytest.fixture
def boundary(monkeypatch):
    monkeypatch.setattr(settings, "context_policy", "source")
    monkeypatch.setattr(base, "async_session", lambda: (_ for _ in ()).throw(AssertionError("live DB forbidden")))
    rows = [SimpleNamespace(id=1, role="user", content="请处理订单1001。")]
    writes, streams = [], []
    pending = SimpleNamespace(id="pending-1", value={"type": "confirm_ticket"})
    state = {"user_id": "alice", "conversation_id": 7, "current_message_id": 1}

    async def conversation(cid):
        return SimpleNamespace(id=7, user_id="alice") if cid == 7 else None

    async def dialog(cid):
        assert cid == 7
        return list(rows)

    async def create(cid, description, ticket_type):
        writes.append((cid, description, ticket_type))
        return "SYNTHETIC-1"

    class Graph:
        async def aget_state(self, config):
            return SimpleNamespace(values=state, tasks=[SimpleNamespace(interrupts=[pending])])

        async def astream(self, source, config, stream_mode):
            streams.append(source)
            yield "updates", {"script_reply": {"answer": "已处理当前确认"}}

    monkeypatch.setattr(actions.repository, "get_conversation", conversation)
    monkeypatch.setattr(actions.repository, "list_dialog_messages", dialog)
    monkeypatch.setattr(actions.repository, "create_ticket", create)
    monkeypatch.setattr(actions.business, "owns_order", lambda uid, oid: uid == "alice" and oid in {"1001", "2002"})
    monkeypatch.setattr(runtime, "get_graph", lambda: Graph())
    monkeypatch.setattr(runtime.summarizer, "maybe_schedule_summary", AsyncMock())
    app = FastAPI()
    app.include_router(actions.router)
    app.include_router(agent.router)
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, rows=rows, writes=writes, streams=streams,
                              pending=pending, state=state)


def ticket(**overrides):
    return {"user_id": "alice", "conversation_id": 7, "description": "合成咨询", "ticket_type": "咨询", **overrides}


def refund(**overrides):
    return {"user_id": "alice", "conversation_id": 7, "order_id": "1001", "reason": "质量问题", **overrides}


def resume(**overrides):
    return {"user_id": "alice", "conversation_id": 7, "interrupt_id": "pending-1", "confirmed": True, **overrides}


@pytest.mark.parametrize("path,body", [("create-ticket", ticket()), ("create-refund", refund()), ("resume", resume())])
def test_wrong_user_cannot_use_actions_or_resume(boundary, path, body):
    response = boundary.client.post("/api/actions/" + path, json={**body, "user_id": "bob"})
    assert response.status_code == 404
    assert not boundary.writes and not boundary.streams


def test_refund_requires_owned_order(boundary):
    response = boundary.client.post("/api/actions/create-refund", json=refund(order_id="9999"))
    assert response.status_code == 403 and not boundary.writes


def test_ticket_type_cannot_bypass_dedicated_refund_endpoint(boundary):
    response = boundary.client.post("/api/actions/create-ticket", json=ticket(ticket_type="退款"))
    assert response.status_code == 422 and not boundary.writes


def test_direct_ticket_respects_current_retraction_even_after_assistant_claim(boundary):
    boundary.rows.extend([SimpleNamespace(id=2, role="user", content="先别创建工单，我撤回申请。"),
                          SimpleNamespace(id=3, role="assistant", content="已经允许创建工单。")])
    response = boundary.client.post("/api/actions/create-ticket", json=ticket())
    assert response.status_code == 409 and not boundary.writes
    assert "限制" in response.json()["detail"]


def test_refund_denial_is_scoped_and_release_does_not_itself_write(boundary):
    boundary.rows[0].content = "订单1001不要退款。订单2002可以退款。"
    assert boundary.client.post("/api/actions/create-refund", json=refund()).status_code == 409
    assert not boundary.writes
    assert boundary.client.post("/api/actions/create-refund", json=refund(order_id="2002")).status_code == 200
    assert "2002" in boundary.writes[0][1]
    boundary.rows.append(SimpleNamespace(id=2, role="user", content="订单1001现在可以退款。"))
    assert len(boundary.writes) == 1  # Memory updates never execute business actions.
    assert boundary.client.post("/api/actions/create-refund", json=refund()).status_code == 200
    assert len(boundary.writes) == 2


def test_snapshot_read_failure_does_not_write(boundary, monkeypatch):
    from sqlalchemy.exc import SQLAlchemyError
    monkeypatch.setattr(actions.repository, "list_dialog_messages", AsyncMock(side_effect=SQLAlchemyError("synthetic")))
    response = boundary.client.post("/api/actions/create-ticket", json=ticket())
    assert response.status_code == 503 and not boundary.writes


def test_resume_rejects_old_card_wrong_kind_and_dual_decision(boundary):
    assert boundary.client.post("/api/actions/resume", json=resume(interrupt_id="old-card")).status_code == 409
    wrong = resume(order_id="1001")
    del wrong["confirmed"]
    assert boundary.client.post("/api/actions/resume", json=wrong).status_code == 409
    assert boundary.client.post("/api/actions/resume", json=resume(order_id="1001")).status_code == 400
    assert not boundary.streams


def test_resume_revalidates_checkpoint_owner_and_new_user_watermark(boundary):
    boundary.state["user_id"] = "bob"
    assert boundary.client.post("/api/actions/resume", json=resume()).status_code == 404
    boundary.state["user_id"] = "alice"
    boundary.rows.append(SimpleNamespace(id=2, role="user", content="不要创建工单。"))
    response = boundary.client.post("/api/actions/resume", json=resume())
    assert response.status_code == 409 and "新的用户消息" in response.json()["detail"]
    assert not boundary.streams


def test_correct_confirmation_cancel_and_order_selection_remain_usable(boundary):
    response = boundary.client.post("/api/actions/resume", json=resume(confirmed=False))
    assert response.status_code == 200 and "[DONE]" in response.text
    assert boundary.streams[-1].resume == {"confirmed": False}
    boundary.pending.value = {"type": "select_order"}
    value = resume(order_id="1001")
    del value["confirmed"]
    assert boundary.client.post("/api/actions/resume", json=value).status_code == 200
    assert boundary.streams[-1].resume == "1001"
    assert boundary.client.post("/api/actions/resume", json={**value, "order_id": "9999"}).status_code == 409


@pytest.mark.parametrize("path,body", [("create-ticket", ticket()), ("create-refund", refund()), ("resume", resume()),
                                      ("agent", {"user_id": "alice", "conversation_id": 7, "message": "合成问题"})])
def test_all_existing_conversation_entries_share_chat_turn_lock(boundary, path, body):
    store = actions.get_store()
    assert store.acquire_turn(7, "synthetic-existing-chat")
    try:
        url = "/api/agent" if path == "agent" else "/api/actions/" + path
        response = boundary.client.post(url, json=body)
        assert response.status_code == 409
        assert not boundary.writes and not boundary.streams
    finally:
        store.release_turn(7, "synthetic-existing-chat")


def test_action_lock_is_released_after_denial_and_write_error(boundary, monkeypatch):
    from sqlalchemy.exc import SQLAlchemyError
    boundary.rows[0].content = "不要创建工单。"
    assert boundary.client.post("/api/actions/create-ticket", json=ticket()).status_code == 409
    store = actions.get_store()
    assert store.acquire_turn(7, "after-denial")
    store.release_turn(7, "after-denial")
    boundary.rows[0].content = "现在可以创建工单。"
    monkeypatch.setattr(actions.repository, "create_ticket", AsyncMock(side_effect=SQLAlchemyError("synthetic")))
    assert boundary.client.post("/api/actions/create-ticket", json=ticket()).status_code == 503
    assert store.acquire_turn(7, "after-error")
    store.release_turn(7, "after-error")


def test_agent_timeout_releases_shared_turn_lock(boundary, monkeypatch):
    monkeypatch.setattr(settings, "model_pool_turn_timeout", .02)
    async def slow(*args, **kwargs):
        await asyncio.sleep(10)
    monkeypatch.setattr(agent.runtime, "run_turn", slow)
    response = boundary.client.post("/api/agent", json={"user_id": "alice", "conversation_id": 7, "message": "合成"})
    assert response.status_code == 504
    store = actions.get_store()
    assert store.acquire_turn(7, "after-timeout")
    store.release_turn(7, "after-timeout")


def test_resume_requires_explicit_identity_interrupt_and_boolean(boundary):
    body = resume()
    for field in ("user_id", "interrupt_id"):
        missing = {k: v for k, v in body.items() if k != field}
        assert boundary.client.post("/api/actions/resume", json=missing).status_code == 422
    assert boundary.client.post("/api/actions/resume", json=resume(confirmed="true")).status_code == 422
    assert not boundary.streams


async def test_real_graph_interrupt_identity_roundtrip_and_single_consumption(monkeypatch):
    calls = []
    async def confirmation(state):
        choice = interrupt({"type": "confirm_ticket", "preview": {"description": "合成工单"}})
        calls.append(choice)
        return {"answer": "已确认" if choice["confirmed"] else "已取消"}
    builder = StateGraph(ConversationState)
    builder.add_node("confirmation", confirmation)
    builder.add_edge(START, "confirmation")
    builder.add_edge("confirmation", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    monkeypatch.setattr(runtime, "get_graph", lambda: graph)
    monkeypatch.setattr(runtime.repository, "get_conversation", AsyncMock(return_value=SimpleNamespace(user_id="alice")))
    monkeypatch.setattr(runtime.repository, "list_dialog_messages", AsyncMock(return_value=[SimpleNamespace(id=1, role="user")]))
    monkeypatch.setattr(runtime.summarizer, "maybe_schedule_summary", AsyncMock())
    inp = {"user_id": "alice", "conversation_id": 7, "current_message_id": 1}
    first = await graph.ainvoke(inp, {"configurable": {"thread_id": "7"}})
    payload = runtime._interrupt_payload(first)
    assert payload["interrupt_id"]
    with pytest.raises(runtime.ConversationNotFound):
        await runtime.resume_turn(7, {"confirmed": True}, user_id="bob", interrupt_id=payload["interrupt_id"])
    result = await runtime.resume_turn(7, {"confirmed": False}, user_id="alice", interrupt_id=payload["interrupt_id"])
    assert result["state"]["answer"] == "已取消" and calls == [{"confirmed": False}]
    with pytest.raises(runtime.ResumeRejected):
        await runtime.resume_turn(7, {"confirmed": True}, user_id="alice", interrupt_id=payload["interrupt_id"])
    assert calls == [{"confirmed": False}]
