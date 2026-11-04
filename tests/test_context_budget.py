"""Full-input budget failures must remain visible and must not be retried."""
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

from app.core import coref, intent, query_understanding, selfcheck, llm
from app.core.context_runtime import ContextInputLimit
from app.graph import nodes


@pytest.mark.parametrize("stage", ["coref", "intent", "rewrite", "expand", "check"])
async def test_budget_exception_is_not_relabelled_as_model_or_evidence_failure(monkeypatch, stage):
    async def blocked(_):
        raise ContextInputLimit("test full request budget")
    fake = RunnableLambda(blocked)
    monkeypatch.setattr(coref, "get_chat_model", lambda **kw: fake)
    monkeypatch.setattr(llm, "structured", lambda *args, **kw: fake)
    functions = {"coref": lambda: coref.resolve("问题", "背景"),
                 "intent": lambda: intent.classify("问题", "背景"),
                 "rewrite": lambda: query_understanding.understand("问题", context="来源"),
                 "expand": lambda: query_understanding.expand_queries("问题"),
                 "check": lambda: selfcheck.check_sufficient("问题", ["证据"])}
    with pytest.raises(ContextInputLimit):
        await functions[stage]()


@pytest.mark.parametrize("stage", ["resolve_reference", "classify_intent"])
async def test_nodes_route_complete_budget_failures_to_explicit_recovery(monkeypatch, stage):
    blocked = AsyncMock(side_effect=ContextInputLimit("complete request too large"))
    monkeypatch.setattr(coref, "resolve", blocked)
    monkeypatch.setattr(intent, "classify", blocked)
    state = {"messages": [HumanMessage("问题", id="db-1")]}
    result = await getattr(nodes, stage)(state)
    assert result["memory_error"] == "complete request too large"
    assert "complete_input_budget" in result["trace"]["context_recovery"]
    blocked.assert_awaited_once()


async def test_structured_does_not_retry_pre_request_budget_failure(monkeypatch):
    calls = []
    class Model:
        def with_structured_output(self, *args, **kwargs):
            async def blocked(_):
                calls.append(1)
                raise ContextInputLimit("no request sent")
            return RunnableLambda(blocked)
    monkeypatch.setattr(llm, "get_chat_model", lambda **kw: Model())
    with pytest.raises(ContextInputLimit):
        await llm.structured(intent._Intent).ainvoke("问题")
    assert len(calls) == 1
