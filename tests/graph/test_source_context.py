"""Real LangGraph/runtime integration with fake models and isolated repositories.

These tests check wiring and state transitions, not model capability. No model
API, live database, MCP server or production business write is permitted.
"""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.config import settings
from app.core import context_runtime
from app.core.context_state import build_snapshot
from app.db import base, context_repository
from app.graph import nodes, runtime
from app.graph.build import build_graph
from app.tools.registry import ToolSpec


@pytest.fixture
def isolated_context(monkeypatch):
    for key, value in {"context_policy": "source", "context_recall": "conditional",
                       "context_recovery_tools_enabled": False,
                       "context_state_budget": 9800, "context_coref_budget": 4000,
                       "context_intent_budget": 1800, "context_input_budget": 16000,
                       "model_pool_enabled": False, "evidence_confidence_threshold": .3}.items():
        monkeypatch.setattr(settings, key, value)

    def no_database(*args, **kwargs):
        raise AssertionError("live DB access forbidden by this integration test")

    monkeypatch.setattr(base, "async_session", no_database)
    calls = SimpleNamespace(rows=[], snapshots=[], packets=[], coref=[], intent=[],
                            rewrite=[], search=[], prompts=[], low_conf=[], specs=[],
                            model_results=[], writes=[], saved_version=0)

    async def append(cid, role, content=None, **kwargs):
        mid = max((r.id for r in calls.rows), default=0) + 1
        calls.rows.append(SimpleNamespace(id=mid, role=role, content=content or "", conversation_id=cid))
        return mid

    async def list_rows(cid):
        return [r for r in calls.rows if r.conversation_id == cid]

    async def save_snapshot(cid, uid, snapshot, **kwargs):
        calls.snapshots.append((cid, uid, deepcopy(snapshot), kwargs))
        calls.saved_version += 1
        return {"version": calls.saved_version, "saved": True}

    async def coref(query, history):
        calls.coref.append((query, history))
        return query

    async def classify(query, history):
        calls.intent.append((query, history))
        return {"intent": "商品咨询", "confidence": 1.0}

    async def understand(query, context=None):
        calls.rewrite.append((query, context))
        return {"standard": query, "expanded": []}

    async def search(query, **kwargs):
        calls.search.append((query, kwargs))
        return [{"id": 1, "question": query, "answer": "断电后用软布清洁。",
                 "rerank_score": .99, "section_path": "清洁", "content_type": "faq"}]

    async def specs():
        return calls.specs

    class Model:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages, config=None):
            calls.prompts.append(deepcopy(messages))
            return calls.model_results.pop(0) if calls.model_results else AIMessage("断电后用软布清洁。[1]")

    real_packet = context_runtime.packet

    def packet(state, query, stage="answer", budget=None):
        result = real_packet(state, query, stage, budget)
        calls.packets.append((stage, deepcopy(result["manifest"]), result["history"]))
        return result

    monkeypatch.setattr(context_runtime, "packet", packet)
    monkeypatch.setattr(nodes.repository, "append_message", append)
    monkeypatch.setattr(nodes.repository, "list_dialog_messages", list_rows)
    monkeypatch.setattr(nodes.repository, "get_conversation", AsyncMock(return_value=SimpleNamespace(
        id=7, user_id="owner", summary="", summary_upto_msg_id=0)))
    monkeypatch.setattr(nodes.repository, "create_conversation", AsyncMock(return_value=7))
    monkeypatch.setattr(nodes.repository, "insert_low_confidence", AsyncMock(side_effect=lambda *a, **k: calls.low_conf.append((a, k))))
    monkeypatch.setattr(nodes.repository, "insert_tool_audit", AsyncMock())
    monkeypatch.setattr(context_repository, "load_snapshot", AsyncMock(return_value=None))
    monkeypatch.setattr(context_repository, "save_snapshot", save_snapshot)
    monkeypatch.setattr(context_repository, "list_summary_segments", AsyncMock(return_value=[]))
    monkeypatch.setattr(context_repository, "record_receipt", AsyncMock(return_value={
        "receipt_id": "test-receipt", "result_hash": "test-hash", "observed_at": "2026-10-27T00:00:00"}))
    monkeypatch.setattr(runtime.summarizer, "maybe_schedule_summary", AsyncMock())
    monkeypatch.setattr(nodes.coref, "resolve", coref)
    monkeypatch.setattr(nodes.intent_mod, "classify", classify)
    monkeypatch.setattr(nodes.query_understanding, "understand", understand)
    monkeypatch.setattr(nodes.query_understanding, "expand_queries", AsyncMock(side_effect=AssertionError("unexpected policy expansion")))
    monkeypatch.setattr(nodes.retrieval, "search_knowledge", search)
    monkeypatch.setattr(nodes.selfcheck, "check_sufficient", AsyncMock(return_value={"useful": True, "reason": ""}))
    monkeypatch.setattr(nodes.registry, "get_all_specs", specs)
    monkeypatch.setattr(nodes, "get_chat_model", lambda **kwargs: Model())
    graph = build_graph(InMemorySaver())
    monkeypatch.setattr(runtime, "get_graph", lambda: graph)
    calls.graph = graph
    return calls


def _state(*, query="这款怎么清洗？", snapshot=True):
    rows = [{"id": 101, "role": "user", "content": "我买的是 MH-W40。"},
            {"id": 102, "role": "assistant", "content": "已记录商品型号。"},
            {"id": 103, "role": "user", "content": query}]
    state = {"conversation_id": 7, "user_id": "owner", "current_message_id": 103,
             "messages": [HumanMessage(rows[0]["content"], id="db-101"),
                          AIMessage(rows[1]["content"], id="db-102"),
                          HumanMessage(query, id="db-103")], "summary": "旧摘要标记",
             "summary_upto_msg_id": 0, "memory_segments": [], "memory_error": ""}
    if snapshot:
        state["conversation_memory"] = build_snapshot(rows, 7)
    return state


def test_legacy_and_source_policies_do_not_mix_context(isolated_context, monkeypatch):
    state = _state()
    source = nodes._history_text(state)
    assert "会话状态与来源" in source and "旧摘要标记" not in source
    monkeypatch.setattr(settings, "context_policy", "legacy")
    state["conversation_memory"] = {"invalid": "source state must not be read"}
    legacy = nodes._history_text(state)
    assert "旧摘要标记" in legacy and "会话状态与来源" not in legacy


async def test_real_graph_uses_one_snapshot_and_passes_correction_to_retrieval(isolated_context, monkeypatch):
    calls = isolated_context
    state = _state(query="这款怎么清洗？")
    state["conversation_memory"] = build_snapshot([
        {"id": 101, "role": "user", "content": "我买的是 MH-W40。"},
        {"id": 102, "role": "assistant", "content": "请确认型号。"},
        {"id": 103, "role": "user", "content": "更正，型号是 MH-FD10。"},
        {"id": 104, "role": "assistant", "content": "已收到更正。"},
        {"id": 105, "role": "user", "content": "这款怎么清洗？"}], 7)
    state["messages"] = [(HumanMessage if r["role"] == "user" else AIMessage)(r["content"], id=f"db-{r['id']}")
                         for r in state["conversation_memory"]["rows"]]
    state["current_message_id"] = 105

    async def coref(query, history):
        calls.coref.append((query, history))
        assert "当前对象候选=MH-FD10" in history
        return "MH-FD10 怎么清洗？"

    monkeypatch.setattr(nodes.coref, "resolve", coref)
    out = await calls.graph.ainvoke(state, {"configurable": {"thread_id": "correction"}})
    assert out["resolved_query"] == "MH-FD10 怎么清洗？" and out["evidence_strong"]
    assert calls.rewrite[0][0] == "MH-FD10 怎么清洗？"
    assert "当前对象候选=MH-FD10" in calls.rewrite[0][1]
    assert calls.search[0][0] == "MH-FD10 怎么清洗？"
    assert {stage for stage, _, _ in calls.packets} == {"coref", "intent", "retrieval", "answer"}
    assert {manifest["snapshot_version"] for _, manifest, _ in calls.packets} == {105}
    assert all("用户:这款怎么清洗？" not in history for _, _, history in calls.packets)
    assert sum(m.content == "这款怎么清洗？" for m in calls.prompts[0]) == 1


async def test_runtime_continuous_turns_reload_actual_prior_answers(isolated_context):
    calls = isolated_context
    calls.model_results = [AIMessage("第一轮已记录 MH-W40。"), AIMessage("第二轮已更新 MH-FD10。"), AIMessage("第三轮给出清洁方式。")]
    first = await runtime.run_turn("owner", "我买的是 MH-W40。", None)
    second = await runtime.run_turn("owner", "更正，型号是 MH-FD10。", first["conversation_id"])
    third = await runtime.run_turn("owner", "这款怎么清洗？", first["conversation_id"])
    assert [r.role for r in calls.rows] == ["user", "assistant"] * 3
    assert "第一轮已记录 MH-W40。" in calls.coref[1][1]
    assert "第二轮已更新 MH-FD10。" in calls.coref[2][1]
    assert second["state"]["conversation_memory"]["candidate"]["value"] == "MH-FD10"
    assert third["state"]["conversation_memory"]["candidate"]["value"] == "MH-FD10"
    assert [s[2]["watermark"] for s in calls.snapshots] == [1, 3, 5]
    assert third["state"]["current_message_id"] == 5


async def test_prepare_turn_excludes_newer_concurrent_message_and_respects_cas(isolated_context, monkeypatch):
    calls = isolated_context
    calls.rows = [SimpleNamespace(id=1, role="user", content="当前处理订单610371。", conversation_id=7),
                  SimpleNamespace(id=2, role="assistant", content="已收到。", conversation_id=7),
                  SimpleNamespace(id=3, role="user", content="查物流。", conversation_id=7),
                  SimpleNamespace(id=4, role="user", content="改为订单999999。", conversation_id=7)]
    save = AsyncMock(return_value={"saved": False, "version": 9})
    monkeypatch.setattr(context_repository, "load_snapshot", AsyncMock(return_value={"version": 8}))
    monkeypatch.setattr(context_repository, "save_snapshot", save)
    result = await context_runtime.prepare_turn("owner", 7, 3)
    assert result["conversation_memory"]["watermark"] == 3
    assert "999999" not in str(result["conversation_memory"])
    assert not result["conversation_memory"]["store_saved"]
    assert save.await_args.kwargs == {"watermark": 3, "expected_version": 8}


async def test_missing_sources_fail_closed_without_model_or_retrieval(isolated_context):
    calls = isolated_context
    out = await calls.graph.ainvoke({"messages": [HumanMessage("这款怎么清洗？")],
                                    "conversation_id": 7, "user_id": "owner"},
                                   {"configurable": {"thread_id": "missing-source"}})
    assert out["memory_error"] and out["answer"] == context_runtime.RECOVERY_REPLY
    assert calls.coref == [] and calls.intent == [] and calls.rewrite == [] and calls.prompts == []
    assert calls.low_conf == []


async def test_essential_context_overflow_does_not_call_model(isolated_context, monkeypatch):
    monkeypatch.setattr(settings, "context_coref_budget", 30)
    calls = isolated_context
    out = await calls.graph.ainvoke(_state(), {"configurable": {"thread_id": "budget"}})
    assert out["memory_error"] and out["answer"] == context_runtime.RECOVERY_REPLY
    assert calls.coref == [] and calls.prompts == [] and calls.rewrite == []


@pytest.mark.parametrize("streaming", [False, True])
async def test_complete_payload_tool_schema_budget_blocks_before_http(isolated_context, monkeypatch, streaming):
    from app.core.llm import _CumulativeUsageChatOpenAI
    monkeypatch.setattr(settings, "context_input_budget", 512)
    sent = []

    def send(request):
        sent.append(request)
        raise AssertionError("over-budget request reached HTTP")

    async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
        model = _CumulativeUsageChatOpenAI(model="fake-test-model", api_key="not-a-real-key",
                                         base_url="https://example.invalid/v1", max_retries=0,
                                         streaming=streaming, http_async_client=client)
        schema = {"type": "function", "function": {"name": "query_order",
                  "description": "完整工具定义。" * 1000,
                  "parameters": {"type": "object", "properties": {}}}}
        with pytest.raises(context_runtime.ContextInputLimit):
            await model.ainvoke([HumanMessage("查订单。")], tools=[schema])
    assert sent == []


def test_answer_prefix_and_tool_history_stay_stable_with_room(isolated_context):
    state = _state()
    first = nodes._agent_messages(state, tool_schemas=[])
    call = AIMessage("", tool_calls=[{"id": "read-1", "name": "query_order", "args": {"order_id": "610371"}}])
    result = ToolMessage('{"status":"运输中"}', tool_call_id="read-1", name="query_order")
    continued = {**state, "messages": [*state["messages"], call, result]}
    second = nodes._agent_messages(continued, tool_schemas=[])
    assert isinstance(first[0], SystemMessage) and first[0].content == nodes.AGENT_SYSTEM
    assert sum(isinstance(m, SystemMessage) for m in second) == 1
    assert [m.model_dump() for m in second[:len(first)]] == [m.model_dump() for m in first]
    assert second[-2:] == [call, result]


async def test_new_user_retraction_blocks_resumed_old_write(isolated_context, monkeypatch):
    calls = isolated_context

    async def classify(query, history):
        return {"intent": "人工", "confidence": 1.0}

    async def execute(args):
        calls.writes.append(args)
        return {"ticket_no": "TEST-1"}

    tool = SimpleNamespace(name="create_ticket", ainvoke=execute)
    calls.specs = [ToolSpec(name="create_ticket", description="测试工单", permission="write", source="builtin",
                           tool=tool, json_schema={"type": "object", "properties": {
                               "description": {"type": "string"}, "ticket_type": {"type": "string"}},
                               "required": ["description"]})]
    calls.model_results = [AIMessage("", tool_calls=[{"id": "ticket-1", "name": "create_ticket", "args": {
        "description": "商品清洁咨询", "ticket_type": "咨询"}}]), AIMessage("本次没有提交工单。")]
    monkeypatch.setattr(nodes.intent_mod, "classify", classify)
    first = await runtime.run_turn("owner", "请建立咨询工单。", None)
    assert first["interrupt"]["type"] == "confirm_ticket" and calls.writes == []
    # Another request arrives while this execution is interrupted. It is stored
    # but the old graph state still carries the earlier message watermark.
    await nodes.repository.append_message(7, "user", content="先别创建工单，我撤回申请。")
    with pytest.raises(runtime.ResumeRejected, match="新的用户消息"):
        await runtime.resume_turn(7, {"confirmed": True}, user_id="owner",
                                  interrupt_id=first["interrupt"]["interrupt_id"])
    assert calls.writes == []


async def test_current_restriction_blocks_even_attempt_to_create_preview(isolated_context, monkeypatch):
    calls = isolated_context
    calls.rows = [SimpleNamespace(id=1, role="user", content="当前处理订单610371。不要创建工单。", conversation_id=7)]
    state = {"conversation_id": 7, "user_id": "owner", "current_message_id": 1,
             "conversation_memory": build_snapshot([{"id": 1, "role": "user", "content": calls.rows[0].content}], 7),
             "messages": [AIMessage("", tool_calls=[{"id": "blocked", "name": "create_ticket", "args": {"description": "问题"}}])]}
    result = await nodes.agent_tools(state)
    assert result["messages"][0].status == "error"
    assert "有效的撤回或限制" in result["messages"][0].content
    assert calls.writes == []
