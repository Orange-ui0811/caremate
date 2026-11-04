import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from app.graph import business_evidence as be
from app.graph.runtime import _graph_input


def state(query="查询订单710001的最新物流"):
    return {"intent": "物流", "resolved_query": query, "messages": [HumanMessage(query)],
            "user_id": "owner", "conversation_id": 7, "current_message_id": 11}


@pytest.fixture
def calls(monkeypatch):
    calls = []
    monkeypatch.setattr(be.settings, "context_fresh_business_evidence", True)
    monkeypatch.setattr(be.business, "owns_order", lambda uid, oid: uid == "owner" and oid == "710001")
    async def specs():
        return [SimpleNamespace(name=n, permission="read") for n in ("query_order", "query_logistics")]
    async def execute(call, cid, specs, **kwargs):
        calls.append({**call, "user_id": kwargs["user_id"], "watermark": kwargs["message_watermark"]})
        value = {"tracking_no": "SF71000188"} if call["name"] == "query_order" else {"status": "派送中"}
        return SimpleNamespace(ok=True, tool_message=ToolMessage(json.dumps(value), tool_call_id=call["id"]))
    monkeypatch.setattr(be.registry, "get_all_specs", specs)
    monkeypatch.setattr(be.engine, "execute_tool_call", execute)
    return calls


async def test_current_receipts_follow_owned_order_then_tracking(calls):
    result = await be.prepare_business_evidence(state())
    assert [c["name"] for c in calls] == ["query_order", "query_logistics"]
    assert calls[1]["args"] == {"tracking_no": "SF71000188"}
    assert all(c["user_id"] == "owner" and c["watermark"] == 11 for c in calls)
    assert result["fresh_business_evidence"]["status"] == "fresh"
    assert len(result["messages"]) == 4
    assert "无需重复查询" in be.evidence_instruction({**state(), **result})


@pytest.mark.parametrize("query", ["查询订单710001和710002", "查那笔物流", "查订单999999", "订单710001不要再查物流",
                                   "订单710001上次的轨迹再说一遍", "不要查询订单710001的物流"])
async def test_no_automatic_call_with_ambiguity_foreign_or_negation(calls, query):
    await be.prepare_business_evidence(state(query))
    assert calls == []


async def test_failed_order_does_not_invent_tracking(calls, monkeypatch):
    async def fail(call, cid, specs, **kwargs):
        calls.append(call)
        return SimpleNamespace(ok=False, tool_message=ToolMessage("超时", tool_call_id=call["id"]))
    monkeypatch.setattr(be.engine, "execute_tool_call", fail)
    result = await be.prepare_business_evidence(state())
    assert result["fresh_business_evidence"]["status"] == "order_lookup_failed"
    assert len(calls) == 1
    assert "不得用旧答复" in be.evidence_instruction({**state(), **result})


async def test_archived_order_preview_uses_tracking_without_repeat(calls, monkeypatch):
    original = be.engine.execute_tool_call
    async def preview(call, cid, specs, **kwargs):
        result = await original(call, cid, specs, **kwargs)
        if call["name"] == "query_order":
            result.tool_message.content = json.dumps({"_context_receipt": "owned-receipt", "partial": True,
                                                       "fields": {"tracking_no": "SF71000188"}})
        return result
    monkeypatch.setattr(be.engine, "execute_tool_call", preview)
    result = await be.prepare_business_evidence(state())
    assert result["fresh_business_evidence"]["status"] == "fresh"
    assert len(calls) == 2


async def test_disabled_preserves_live_execution(calls, monkeypatch):
    monkeypatch.setattr(be.settings, "context_fresh_business_evidence", False)
    assert await be.prepare_business_evidence(state()) == {"fresh_business_evidence": {}}
    assert not calls


def test_freshness_is_scoped_to_current_message_and_reset():
    stale = {**state(), "fresh_business_evidence": {"message_id": 10, "status": "fresh"}}
    assert be.evidence_instruction(stale) == ""
    assert _graph_input("owner", "hi", 7, 12, "", 0)["fresh_business_evidence"] == {}
