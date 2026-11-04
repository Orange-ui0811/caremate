"""Context persistence tests use a private SQLite database, never production data."""
import asyncio
import json

import pytest
import pytest_asyncio
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.db import base, context_repository as store, repository
from app.db.models import (Conversation, Message, ContextSnapshot, ContextReceipt,
                           ContextMaintenanceLease, ContextSource)


@pytest_asyncio.fixture
async def context_db(tmp_path, monkeypatch):
    path = tmp_path / "context.sqlite"
    url = "sqlite+aiosqlite:///" + path.as_posix()
    engine = create_async_engine(url)
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
    yield {"engine": engine, "sessions": sessions, "url": url}
    await engine.dispose()


def _snapshot(watermark, **extra):
    return {"conversation_id": 1, "watermark": watermark, "events": [],
            "rows": [{"id": 1, "content": "do not duplicate"}], **extra}


async def test_snapshot_cas_monotonic_and_incremental_index(context_db):
    a = await store.save_snapshot(1, "alice", _snapshot(1), watermark=1)
    assert a == {"saved": True, "version": 1, "watermark": 1}
    assert "rows" not in (await store.load_snapshot(1, "alice"))["snapshot"]
    assert not (await store.save_snapshot(1, "alice", _snapshot(3), watermark=3))["saved"]
    b = await store.save_snapshot(1, "alice", _snapshot(3), watermark=3, expected_version=1)
    assert b["saved"] and b["version"] == 2
    assert not (await store.save_snapshot(1, "alice", _snapshot(1), watermark=1, expected_version=2))["saved"]
    assert (await store.load_snapshot(1, "alice"))["watermark"] == 3
    found = await store.search_sources(1, "alice", "610371", upto_msg_id=3)
    assert any(r["message_id"] == 3 and "610371" in r["text"] for r in found)


async def test_source_scope_hash_and_exact_span(context_db):
    raw = "当前处理订单610370。先别提交退款。"
    ref = await store.read_source(1, "alice", 1, 0, 13, store.content_hash(raw))
    assert ref["text"] == raw[:13]
    page = await store.read_source(1, "alice", 1, 0, 1200)
    assert page["text"] == raw and page["end"] == len(raw) and page["next_start"] is None
    assert (await store.read_source(1, "alice", 1, 0, 5))["next_start"] == 5
    with pytest.raises(PermissionError):
        await store.read_source(1, "bob", 1, 0, 2)
    with pytest.raises(LookupError):
        await store.read_source(1, "alice", 99, 0, 2)
    with pytest.raises(ValueError):
        await store.read_source(1, "alice", 1, 0, 2, "wrong")
    with pytest.raises(ValueError):
        await store.read_source(1, "alice", 1, -1, 2)
    with pytest.raises(ValueError):
        await store.save_snapshot(1, "alice", _snapshot(99), watermark=99)


async def test_scoped_search_preserves_neighbour_and_no_future(context_db):
    await store.save_snapshot(1, "alice", _snapshot(3), watermark=3)
    hits = await store.search_sources(1, "alice", "610370", upto_msg_id=1, limit=1)
    assert hits and all(r["message_id"] == 1 for r in hits)
    assert any("先别提交退款" in r["text"] for r in hits)
    assert not await store.search_sources(1, "alice", "610371", upto_msg_id=1)
    with pytest.raises(PermissionError):
        await store.search_sources(1, "bob", "", upto_msg_id=3)


async def test_index_never_returns_modified_source(context_db):
    await store.save_snapshot(1, "alice", _snapshot(3), watermark=3)
    async with context_db["sessions"]() as s:
        await s.execute(update(Message).where(Message.id == 1).values(content="被修改的旧原文"))
        await s.commit()
    assert not await store.search_sources(1, "alice", "610370", upto_msg_id=1)


async def test_source_and_receipt_recover_after_connection_restart(context_db, monkeypatch):
    raw = {"order_id": "610370", "long": "完整结果" * 1000}
    receipt = await store.record_receipt(1, "alice", tool_call_id="t1", tool_name="query_order",
        arguments={"order_id": "610370"}, raw_result=raw,
        formatted_result=json.dumps(raw, ensure_ascii=False), status="成功")
    await context_db["engine"].dispose()
    reopened = create_async_engine(context_db["url"])
    monkeypatch.setattr(base, "async_session", async_sessionmaker(reopened, expire_on_commit=False))
    try:
        recovered = await store.read_receipt(1, "alice", receipt["receipt_id"])
        assert recovered["raw_result"] == raw and recovered["historical"]
        assert (await store.read_source(1, "alice", 1, 0, 2))["text"] == "当前"
        with pytest.raises(PermissionError):
            await store.read_receipt(1, "bob", receipt["receipt_id"])
        with pytest.raises(LookupError):
            await store.read_receipt(2, "bob", receipt["receipt_id"])
    finally:
        await reopened.dispose()


async def test_receipt_hash_rejects_tampering(context_db):
    receipt = await store.record_receipt(1, "alice", tool_call_id="t1", tool_name="query_order",
        arguments={}, raw_result={"status": "待支付"}, formatted_result="待支付", status="成功")
    async with context_db["sessions"]() as s:
        await s.execute(update(ContextReceipt).values(formatted_result="已支付"))
        await s.commit()
    with pytest.raises(ValueError, match="hash"):
        await store.read_receipt(1, "alice", receipt["receipt_id"])


async def test_summary_immutable_sources_and_causal_filter(context_db):
    rows = await repository.list_dialog_messages(1)
    kwargs = dict(from_msg_id=1, upto_msg_id=2, summary="历史上暂不提交退款。",
                  source_hash=store.source_digest(rows[:2]))
    first = await store.put_summary_segment(1, "alice", **kwargs)
    second = await store.put_summary_segment(1, "alice", **{**kwargs, "summary": "不可覆盖"})
    assert second["summary"] == first["summary"] and second["id"] == first["id"]
    assert not await store.list_summary_segments(1, "alice", upto_msg_id=1)
    assert len(await store.list_summary_segments(1, "alice", upto_msg_id=2)) == 1
    for source in first["source_refs"]:
        assert await store.read_source(1, "alice", **source)
    with pytest.raises(ValueError, match="hash"):
        await store.put_summary_segment(1, "alice", **{**kwargs, "source_hash": "bad"})


async def test_deleted_conversation_blocks_all_context_reads(context_db):
    await store.save_snapshot(1, "alice", _snapshot(3), watermark=3)
    await repository.delete_conversation(1, "alice")
    with pytest.raises(PermissionError):
        await store.load_snapshot(1, "alice")
    with pytest.raises(PermissionError):
        await store.list_summary_segments(1, "alice")
    with pytest.raises(PermissionError):
        await store.search_sources(1, "alice", "订单", upto_msg_id=3)


async def test_late_legacy_summary_cannot_overwrite_newer_prefix(context_db):
    assert await repository.update_conversation_summary(1, "新摘要", 3, expected_upto_msg_id=0)
    assert not await repository.update_conversation_summary(1, "旧摘要", 2, expected_upto_msg_id=0)
    assert not await repository.update_conversation_summary(1, "旧基线生成的更长摘要", 4, expected_upto_msg_id=0)
    conv = await repository.get_conversation(1)
    assert conv.summary == "新摘要" and conv.summary_upto_msg_id == 3


async def test_summary_lease_owner_and_release(context_db):
    owner = await store.acquire_summary_lease(1, "alice")
    assert owner and await store.acquire_summary_lease(1, "alice") is None
    await store.release_summary_lease(1, "not-the-owner")
    assert await store.acquire_summary_lease(1, "alice") is None
    await store.release_summary_lease(1, owner)
    assert await store.acquire_summary_lease(1, "alice")


async def _long_history(context_db):
    async with context_db["sessions"]() as s:
        for i in range(10, 30):
            s.add(Message(id=i, conversation_id=1,
                role="user" if i % 2 == 0 else "assistant", content=f"历史{i}。" + "背景材料。" * 100))
        await s.commit()


async def test_background_segment_does_not_overwrite_current_state(context_db, monkeypatch):
    from app.core import summarizer
    await _long_history(context_db)
    monkeypatch.setattr(settings, "context_policy", "source")
    monkeypatch.setattr(settings, "context_summary_trigger_tokens", 512)
    monkeypatch.setattr(settings, "context_window_turns", 2)
    await store.save_snapshot(1, "alice", _snapshot(29), watermark=29)

    async def summarize(_dialog):
        async with context_db["sessions"]() as s:
            s.add(Message(id=30, conversation_id=1, role="user", content="撤回退款，先不要提交。"))
            await s.commit()
        await store.save_snapshot(1, "alice", _snapshot(30, cancelled=True), watermark=30, expected_version=1)
        return "这段历史仅描述此前讨论，未执行操作。"

    monkeypatch.setattr(summarizer, "summarize_source_segment", summarize)
    await summarizer.run_source_summary(1)
    snapshot = await store.load_snapshot(1, "alice")
    assert snapshot["watermark"] == 30 and snapshot["snapshot"]["cancelled"]
    segments = await store.list_summary_segments(1, "alice")
    assert len(segments) == 1 and segments[0]["upto_msg_id"] < 30
    assert (await repository.get_conversation(1)).summary is None


async def test_oversize_summary_keeps_raw_and_releases_lease(context_db, monkeypatch):
    from app.core import summarizer
    await _long_history(context_db)
    monkeypatch.setattr(settings, "context_policy", "source")
    monkeypatch.setattr(settings, "context_summary_trigger_tokens", 512)
    monkeypatch.setattr(settings, "context_summary_target_tokens", 128)
    monkeypatch.setattr(settings, "context_window_turns", 2)
    async def huge(_dialog):
        return "很长的摘要" * 500
    monkeypatch.setattr(summarizer, "summarize_source_segment", huge)
    with pytest.raises(ValueError, match="over target"):
        await summarizer.run_source_summary(1)
    assert not await store.list_summary_segments(1, "alice")
    assert len(await repository.list_dialog_messages(1)) == 23
    assert await store.acquire_summary_lease(1, "alice")


async def test_tool_receipt_keeps_preformat_raw_and_offloads_only_after_save(context_db, monkeypatch):
    from app.tools import engine
    from app.tools.registry import ToolSpec
    monkeypatch.setattr(settings, "context_policy", "source")
    monkeypatch.setattr(settings, "context_recovery_tools_enabled", True)
    async def audit(**kwargs):
        pass
    monkeypatch.setattr(engine.repository, "insert_tool_audit", audit)
    original = {"order_id": "610370", "status": "已支付", "details": "完整回执" * 1000}
    async def invoke(_args):
        return dict(original)
    def formatter(value):
        value["status"] = "支付成功"
        return value
    tool = type("Tool", (), {"ainvoke": staticmethod(invoke)})()
    spec = ToolSpec(name="query_order", description="test", json_schema={"type": "object"},
                    tool=tool, permission="read", source="builtin", format_result=formatter)
    run = await engine.execute_tool_call({"name": "query_order", "id": "tc1", "args": {}},
                                         1, {"query_order": spec}, user_id="alice")
    assert run.ok and run.receipt_persisted and run.receipt_ref
    assert "_context_receipt" in run.tool_message.content and "完整回执" not in run.tool_message.content
    full = await store.read_receipt(1, "alice", run.receipt_ref)
    assert full["raw_result"] == original and "支付成功" in full["formatted_result"]


async def test_receipt_outage_does_not_retry_successful_write(context_db, monkeypatch):
    from app.tools import engine
    from app.tools.registry import ToolSpec
    monkeypatch.setattr(settings, "context_policy", "source")
    monkeypatch.setattr(settings, "context_recovery_tools_enabled", True)
    async def audit(**kwargs):
        pass
    async def unavailable(*args, **kwargs):
        raise RuntimeError("storage unavailable")
    monkeypatch.setattr(engine.repository, "insert_tool_audit", audit)
    monkeypatch.setattr(store, "record_receipt", unavailable)
    calls = []
    async def invoke(_args):
        calls.append(1)
        return {"ticket_id": "T1", "details": "完整结果" * 1000}
    spec = ToolSpec(name="create_ticket", description="test", json_schema={"type": "object"},
        tool=type("Tool", (), {"ainvoke": staticmethod(invoke)})(), permission="write", source="builtin")
    run = await engine.execute_tool_call({"name": "create_ticket", "id": "tc1", "args": {}},
        1, {"create_ticket": spec}, user_id="alice", confirmed=True)
    assert run.ok and calls == [1] and not run.receipt_persisted and run.receipt_ref is None
    assert "完整结果" in run.tool_message.content


async def test_recovery_tool_does_not_create_recursive_receipts(context_db, monkeypatch):
    from app.tools import engine
    from app.tools.registry import ToolSpec
    monkeypatch.setattr(settings, "context_policy", "source")
    async def audit(**kwargs):
        pass
    async def unexpected(*args, **kwargs):
        pytest.fail("recovery must not create another receipt")
    async def read_page(_args):
        return {"receipt_id": "original", "text": "原始页"}
    monkeypatch.setattr(engine.repository, "insert_tool_audit", audit)
    monkeypatch.setattr(store, "record_receipt", unexpected)
    spec = ToolSpec(name="read_tool_receipt", description="test", json_schema={"type": "object"},
        tool=type("Tool", (), {"ainvoke": staticmethod(read_page)})(), permission="read", source="builtin")
    run = await engine.execute_tool_call({"name": "read_tool_receipt", "id": "t1", "args": {}},
        1, {"read_tool_receipt": spec}, user_id="alice")
    assert run.ok and not run.receipt_persisted and run.receipt_ref is None
