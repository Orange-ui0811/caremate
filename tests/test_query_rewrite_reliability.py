"""Rewrite success, shared deadline and cancellation through real HTTP adapters."""
import asyncio
import json
import time

import httpx
import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from app.config import settings
from app.core import llm, model_pool, model_transport, query_understanding as qu
from app.core.context_runtime import ContextInputLimit


async def test_success_reports_actual_model_fields(monkeypatch):
    async def invoke(_):
        return qu._Rewrite(standard="  猫砂盆清洁方法  ", expanded=["清洗", "保养"])
    monkeypatch.setattr(qu.llm, "structured", lambda *_a, **_k: RunnableLambda(invoke))
    result = await qu.understand("怎么洗")
    assert result["standard"] == "猫砂盆清洁方法"
    assert result["expanded"] == ["清洗", "保养"]
    assert result["rewrite"]["status"] == "succeeded"


@pytest.mark.parametrize("value", ["", "  ", "\n\t"])
def test_blank_standard_is_not_a_success(value):
    with pytest.raises(ValidationError):
        qu._Rewrite(standard=value)


async def test_parse_failure_is_explicitly_distinct_from_success(monkeypatch):
    async def invoke(_):
        raise OutputParserException("missing standard")
    monkeypatch.setattr(qu.llm, "structured", lambda *_a, **_k: RunnableLambda(invoke))
    result = await qu.understand("原问题")
    assert (result["standard"], result["expanded"]) == ("原问题", [])
    assert result["rewrite"]["status"] == "failed"
    assert result["rewrite"]["error_type"] == "OutputParserException"


async def test_two_attempts_share_one_rewrite_deadline(monkeypatch):
    calls, cancelled = [], asyncio.Event()

    class Model:
        def with_structured_output(self, *_a, **_k):
            async def invoke(_):
                calls.append(time.monotonic())
                if len(calls) == 1:
                    await asyncio.sleep(.05)
                    raise ValueError("empty arguments")
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            return RunnableLambda(invoke)

    monkeypatch.setattr(llm, "get_chat_model", lambda **_k: Model())
    monkeypatch.setattr(settings, "query_rewrite_timeout", .12)
    started = time.monotonic()
    result = await qu.understand("原问题")
    assert len(calls) == 2 and cancelled.is_set()
    assert result["rewrite"]["status"] == "timeout"
    assert time.monotonic() - started < .5
    # The second attempt gets only what remains, not a fresh .12s budget.
    assert result["rewrite"]["elapsed_ms"] < 160


@pytest.mark.parametrize("exc", [asyncio.CancelledError(), ContextInputLimit("budget")])
async def test_caller_cancellation_and_context_limit_are_not_hidden(monkeypatch, exc):
    async def invoke(_):
        raise exc
    monkeypatch.setattr(qu.llm, "structured", lambda *_a, **_k: RunnableLambda(invoke))
    with pytest.raises(type(exc)):
        await qu.understand("原问题")


def sse(delta, finish=None, usage=None):
    return ("data: " + json.dumps({"id": "rewrite-test", "object": "chat.completion.chunk", "created": 1,
        "model": "DeepSeek-V4-Flash", "choices": [{"index": 0, "delta": delta,
        "finish_reason": finish}], "usage": usage}, ensure_ascii=False) + "\n\n").encode()


async def test_real_sdk_and_pool_use_provider_contract_and_parse_sse(monkeypatch):
    requests = []
    args = {"standard": "猫砂盆有哪些型号及清洁方法", "expanded": ["款式", "清洗"]}

    async def upstream(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body["chat_template_kwargs"] == {"thinking": False}
        assert "thinking" not in body
        assert body["tool_choice"]["function"]["name"] == "_Rewrite"
        assert body["parallel_tool_calls"] is False
        wire = sse({"role": "assistant", "tool_calls": [{"index": 0, "id": "call-ok",
            "type": "function", "function": {"name": "_Rewrite", "arguments": ""}}]})
        for part in [json.dumps(args, ensure_ascii=False)[:18], json.dumps(args, ensure_ascii=False)[18:]]:
            wire += sse({"tool_calls": [{"index": 0, "function": {"arguments": part}}]})
        wire += sse({}, "tool_calls", {"prompt_tokens": 25, "completion_tokens": 15, "total_tokens": 40})
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=wire + b"data: [DONE]\n\n")

    monkeypatch.setattr(settings, "chat_base_url", "https://ai.gitee.com/v1")
    monkeypatch.setattr(settings, "chat_model", "DeepSeek-V4-Flash")
    monkeypatch.setattr(settings, "chat_thinking", "disabled")
    monkeypatch.setattr(settings, "model_pool_chat_keys", "test-only-key")
    monkeypatch.setattr(settings, "model_pool_fallback_base_url", "")
    monkeypatch.setattr(settings, "model_pool_enabled", True)
    async with httpx.AsyncClient(transport=model_transport.AsyncPoolTransport(inner=httpx.MockTransport(upstream))) as ac:
        with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(500))) as sc:
            monkeypatch.setattr(model_transport, "clients", lambda *_: (sc, ac))
            result = await qu.understand("猫砂盆怎么选和洗")
    assert len(requests) == 1
    assert result["standard"] == args["standard"] and result["expanded"] == args["expanded"]
    assert result["rewrite"]["status"] == "succeeded"
    assert not model_pool.get_store().snapshot()["running"]


async def test_rewrite_timeout_closes_stream_and_releases_pool_slot(monkeypatch):
    closed = asyncio.Event()

    class Stalled(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield sse({"role": "assistant", "content": ""})
            await asyncio.Event().wait()

        async def aclose(self):
            closed.set()

    async def upstream(_):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Stalled())

    monkeypatch.setattr(settings, "model_pool_enabled", True)
    monkeypatch.setattr(settings, "model_pool_chat_keys", "test-only-key")
    monkeypatch.setattr(settings, "model_pool_fallback_base_url", "")
    monkeypatch.setattr(settings, "query_rewrite_timeout", .2)
    store = model_pool.get_store()
    store.set_policy(model_pool.Policy(call_timeout=90, read_timeout=60))
    async with httpx.AsyncClient(transport=model_transport.AsyncPoolTransport(inner=httpx.MockTransport(upstream))) as ac:
        with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(500))) as sc:
            monkeypatch.setattr(model_transport, "clients", lambda *_: (sc, ac))
            result = await qu.understand("原问题")
    assert result["rewrite"]["status"] == "timeout"
    assert closed.is_set()
    snapshot = store.snapshot()
    assert not snapshot["running"] and snapshot["waiting"] == 0
    assert len(snapshot["recent_calls"]) == 1


@pytest.mark.parametrize("score,useful", [(0., True), (1., False), (1., True)])
async def test_retrieval_trace_preserves_rewrite_outcome_on_each_exit(monkeypatch, score, useful):
    from app.graph import nodes
    from langchain_core.messages import HumanMessage
    monkeypatch.setattr(nodes.context_runtime, "enabled", lambda: False)

    async def understand(_):
        return {"standard": "标准问题", "expanded": ["同义词"],
                "rewrite": {"status": "timeout", "elapsed_ms": 12000, "error_type": "TimeoutError"}}

    async def search(*_a, **_k):
        return [{"id": 1, "question": "问题", "answer": "答案", "section_path": "手册",
                 "content_type": "faq", "rerank_score": score}]

    async def check(*_a):
        return {"useful": useful, "reason": "检查结果"}

    monkeypatch.setattr(nodes.query_understanding, "understand", understand)
    monkeypatch.setattr(nodes.retrieval, "search_knowledge", search)
    monkeypatch.setattr(nodes.selfcheck, "check_sufficient", check)
    result = await nodes.retrieve_knowledge({"messages": [HumanMessage("原问题")]})
    assert result["trace"]["query_rewrite"]["status"] == "timeout"
    assert result["trace"]["query_rewrite"]["standard"] == "标准问题"
    assert result["evidence_strong"] is (score == 1. and useful)
