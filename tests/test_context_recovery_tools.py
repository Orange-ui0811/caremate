"""Independent integration checks for the explicitly scoped recovery tools."""
import json

import pytest
import pytest_asyncio
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from sqlalchemy import update
from sqlalchemy.dialects import mysql
from sqlalchemy.schema import CreateTable
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.core.context_policy import estimate
from app.db import base, context_repository as store
from app.db.models import Conversation, Message, ContextReceipt
from app.tools.context import get_context_specs


@pytest_asyncio.fixture
async def context_db(tmp_path, monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///" + (tmp_path / "recovery.sqlite").as_posix())
    async with engine.begin() as conn:
        await conn.run_sync(lambda c: base.Base.metadata.create_all(
            c, tables=[Conversation.__table__, Message.__table__]))
    await store.ensure_context_tables(engine)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(base, "async_session", sessions)
    async with sessions() as s:
        s.add_all([Conversation(id=1, user_id="alice"), Conversation(id=2, user_id="bob")])
        s.add_all([
            Message(id=1, conversation_id=1, role="user", content="当前处理订单610370。先别提交退款。"),
            Message(id=2, conversation_id=1, role="assistant", content="收到，还没有执行退款。"),
            Message(id=3, conversation_id=1, role="user", content="610370报错，应为610371。"),
            Message(id=99, conversation_id=2, role="user", content="订单610370属于另一个会话。"),
        ])
        await s.commit()
    yield {"engine": engine, "sessions": sessions}
    await engine.dispose()


def _specs(watermark):
    return {s.name: s for s in get_context_specs(watermark)}


def test_recovery_schemas_hide_identity_and_use_read_only_permissions():
    for spec in _specs(3).values():
        props = spec.json_schema["properties"]
        assert "conversation_id" not in props and "user_id" not in props
        assert spec.inject_conversation and spec.inject_user_id
        assert spec.permission == "read" and spec.max_retries == 0


async def test_source_tool_rejects_future_message(context_db):
    result = await _specs(1)["read_conversation_source"].tool.ainvoke(
        {"message_id": 3, "conversation_id": 1, "user_id": "alice"})
    assert "error" in result and "610371" not in json.dumps(result)


async def test_source_tool_accepts_valid_hash_with_default_page_size(context_db):
    raw = "当前处理订单610370。先别提交退款。"
    result = await _specs(3)["read_conversation_source"].tool.ainvoke(
        {"message_id": 1, "content_hash": store.content_hash(raw),
         "conversation_id": 1, "user_id": "alice"})
    assert result["text"] == raw and result["end"] == len(raw)
    assert result["next_start"] is None
    assert "error" not in result


async def test_source_tool_hash_mismatch_never_returns_text(context_db):
    try:
        result = await _specs(3)["read_conversation_source"].tool.ainvoke(
            {"message_id": 1, "content_hash": "wrong", "conversation_id": 1, "user_id": "alice"})
    except ValueError:
        return
    assert "error" in result and "text" not in result


async def test_tool_execution_overwrites_forged_identity(context_db, monkeypatch):
    from app.tools import engine
    async def audit(**kwargs):
        pass
    monkeypatch.setattr(engine.repository, "insert_tool_audit", audit)
    # Even if an upstream tool call includes hidden fields, only server identity is used.
    run = await engine.execute_tool_call({"name": "read_conversation_source", "id": "r1",
        "args": {"message_id": 1, "user_id": "bob", "conversation_id": 2}},
        1, _specs(3), user_id="alice", message_watermark=3)
    assert run.ok and "当前处理订单610370" in run.tool_message.content
    blocked = await engine.execute_tool_call({"name": "read_conversation_source", "id": "r2",
        "args": {"message_id": 99}}, 1, _specs(100), user_id="alice", message_watermark=3)
    assert not blocked.ok and "属于另一个会话" not in blocked.tool_message.content


async def test_receipt_tool_respects_turn_watermark(context_db):
    receipt = await store.record_receipt(1, "alice", tool_call_id="future", tool_name="query_order",
        arguments={}, raw_result={"next": "未来回执内容"}, formatted_result="future",
        status="成功", message_watermark=3)
    with pytest.raises(LookupError):
        await _specs(1)["read_tool_receipt"].tool.ainvoke(
            {"receipt_id": receipt["receipt_id"], "conversation_id": 1, "user_id": "alice"})
    result = await _specs(3)["read_tool_receipt"].tool.ainvoke(
        {"receipt_id": receipt["receipt_id"], "conversation_id": 1, "user_id": "alice"})
    assert "未来回执内容" in result["text"]


async def test_receipt_paging_can_reconstruct_original_without_duplication(context_db):
    raw = "第一段：完整订单结果。第二段：状态待确认。第三段：不得当作当前授权。"
    receipt = await store.record_receipt(1, "alice", tool_call_id="page", tool_name="query_order",
        arguments={}, raw_result=raw, formatted_result=raw, status="成功", message_watermark=1)
    parts, offset, flags = [], 0, []
    tool = _specs(3)["read_tool_receipt"].tool
    while offset is not None:
        page = await tool.ainvoke({"receipt_id": receipt["receipt_id"], "conversation_id": 1,
                                  "user_id": "alice", "start": offset, "length": 12})
        assert page["start"] == offset and page["end"] > offset
        parts.append(page["text"])
        flags.append(page["partial"])
        offset = page["next_start"]
    assert "".join(parts) == raw and all(flags)
    whole = await tool.ainvoke({"receipt_id": receipt["receipt_id"],
                               "conversation_id": 1, "user_id": "alice"})
    assert not whole["partial"]


async def test_source_and_search_outputs_obey_recovery_budget(context_db):
    long_text = "查询订单610370。" + "需要检查订单是否已经更正以及用户有没有取消。" * 90
    async with context_db["sessions"]() as s:
        await s.execute(update(Message).where(Message.id == 1).values(content=long_text))
        await s.commit()
    await store.save_snapshot(1, "alice", {"conversation_id": 1}, watermark=3)
    tools = _specs(3)
    largest_page = tools["read_conversation_source"].json_schema["properties"]["length"]["maximum"]
    source = await tools["read_conversation_source"].tool.ainvoke({
        "message_id": 1, "conversation_id": 1, "user_id": "alice", "length": largest_page})
    assert estimate(json.dumps(source, ensure_ascii=False)) <= 2000
    assert source["next_start"] is not None
    hits = await tools["search_conversation_history"].tool.ainvoke({
        "query": "订单 更正 取消", "conversation_id": 1, "user_id": "alice"})
    assert estimate(json.dumps(hits, ensure_ascii=False)) <= 2000
    for found in hits["sources"]:
        exact = await store.read_source(1, "alice", found["message_id"], found["start"],
                                         found["end"], found["content_hash"])
        assert exact["text"] == found["text"]


async def test_non_model_budget_recovery_is_visible_once_in_sse(monkeypatch):
    from app.graph import runtime
    from app.graph.state import ConversationState
    from app.core.context_runtime import RECOVERY_REPLY
    async def main_agent(_state):
        return {"messages": [AIMessage(RECOVERY_REPLY)], "memory_error": "budget blocked"}
    builder = StateGraph(ConversationState)
    builder.add_node("main_agent", main_agent)
    builder.add_edge(START, "main_agent")
    builder.add_edge("main_agent", END)
    graph = builder.compile()
    monkeypatch.setattr(runtime, "get_graph", lambda: graph)
    events = [e async for e in runtime._stream_events(1, {
        "messages": [HumanMessage("继续", id="db-1")], "conversation_id": 1})]
    assert [e["text"] for e in events if e["type"] == "delta"] == [RECOVERY_REPLY]


def test_mysql_receipt_ddl_can_hold_large_formatted_results():
    ddl = str(CreateTable(ContextReceipt.__table__).compile(dialect=mysql.dialect()))
    assert "formatted_result LONGTEXT" in ddl and "message_watermark BIGINT" in ddl
