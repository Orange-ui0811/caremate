"""Exercise query handoff through the real graph without external model/DB calls."""
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.config import settings
from app.graph import nodes, runtime
from app.graph.build import build_graph


@pytest.fixture
def handoff(monkeypatch):
    calls = SimpleNamespace(coref=[], intent=[], rewrite=[], retrieval=[], check=[],
                            prompts=[], saved=[], low_confidence=[])

    async def classify(query, history):
        calls.intent.append((query, history))
        return {"intent": "商品咨询", "confidence": 1.0}

    async def understand(query):
        calls.rewrite.append(query)
        return {"standard": query, "expanded": ["清洁方法"]}

    async def search(query, **kwargs):
        calls.retrieval.append((query, kwargs))
        return [{"id": 1, "question": query, "answer": "断电后使用软布清洁。",
                 "rerank_score": .95, "section_path": "商品/清洁", "content_type": "faq"}]

    async def check(query, evidence):
        calls.check.append((query, evidence))
        return {"useful": True, "reason": ""}

    async def specs():
        return []

    async def append(cid, role, content=None, **kwargs):
        calls.saved.append((cid, role, content))
        return 100 + len(calls.saved)

    async def low_confidence(*args, **kwargs):
        calls.low_confidence.append((args, kwargs))

    class Model:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages, config=None):
            calls.prompts.append(messages)
            return AIMessage("断电后使用软布清洁。[1]")

    monkeypatch.setattr(settings, "evidence_confidence_threshold", .3)
    monkeypatch.setattr(nodes.intent_mod, "classify", classify)
    monkeypatch.setattr(nodes.query_understanding, "understand", understand)
    monkeypatch.setattr(nodes.retrieval, "search_knowledge", search)
    monkeypatch.setattr(nodes.selfcheck, "check_sufficient", check)
    monkeypatch.setattr(nodes.registry, "get_all_specs", specs)
    monkeypatch.setattr(nodes, "get_chat_model", lambda **kwargs: Model())
    monkeypatch.setattr(nodes.repository, "append_message", append)
    monkeypatch.setattr(nodes.repository, "insert_low_confidence", low_confidence)
    return calls


async def test_graph_passes_resolved_product_to_retrieval_and_selfcheck(monkeypatch, handoff):
    original = "这款怎么清洗？"
    resolved = "MH-W40 饮水机怎么清洗？"

    async def coref(query, history):
        handoff.coref.append((query, history))
        assert "MH-W40" in history
        return resolved

    monkeypatch.setattr(nodes.coref, "resolve", coref)
    messages = [HumanMessage("我买的是 MH-W40 饮水机。", id="db-1"),
                AIMessage("请告诉我需要咨询的问题。", id="db-2"),
                HumanMessage(original, id="db-3")]
    out = await build_graph().ainvoke({"messages": messages, "conversation_id": 7})

    assert out["route"] == "knowledge"
    assert out["trace"]["confidence"] == "strong"
    assert out["resolved_query"] == resolved
    assert handoff.coref[0][0] == original
    assert handoff.intent[0][0] == resolved
    assert handoff.rewrite == [resolved]
    assert handoff.retrieval == [(resolved, {"strategy": "hybrid_rerank",
                                          "bm25_text": resolved + " 清洁方法"})]
    assert handoff.check[0][0] == resolved
    assert resolved in out["evidence"]
    assert any(resolved in str(m.content) for m in handoff.prompts[0])
    assert next(m for m in out["messages"] if m.id == "db-3").content == original
    assert messages[-1].content == original
    assert handoff.low_confidence == []


@pytest.mark.parametrize("resolved", [None, "", " \n\t "])
async def test_knowledge_query_falls_back_to_original_when_resolution_is_empty(handoff, resolved):
    original = "MH-W40 怎么清洗？"
    state = {"messages": [HumanMessage(original)]}
    if resolved is not None:
        state["resolved_query"] = resolved
    out = await nodes.retrieve_knowledge(state)
    assert out["evidence_strong"]
    assert handoff.rewrite == [original]
    assert handoff.retrieval[0][0] == original
    assert handoff.check[0][0] == original


@pytest.mark.parametrize("standard", ["", " \n "])
async def test_empty_search_rewrite_keeps_resolved_object(monkeypatch, handoff, standard):
    resolved = "MH-W40 饮水机怎么清洗？"

    async def understand(query):
        handoff.rewrite.append(query)
        return {"standard": standard, "expanded": []}

    monkeypatch.setattr(nodes.query_understanding, "understand", understand)
    out = await nodes.retrieve_knowledge({"messages": [HumanMessage("这款怎么清洗？")],
                                          "resolved_query": "  " + resolved + "  "})
    assert out["evidence_strong"]
    assert handoff.rewrite == [resolved]
    assert handoff.retrieval[0][0] == resolved
    assert handoff.check[0][0] == resolved


async def test_checkpoint_new_turn_does_not_reuse_previous_resolved_task(monkeypatch, handoff):
    queries = []

    async def coref(query, history):
        queries.append(query)
        return "订单 1001 到哪里了？" if len(queries) == 1 else query

    async def classify(query, history):
        handoff.intent.append((query, history))
        return {"intent": "物流" if len(queries) == 1 else "商品咨询", "confidence": 1.0}

    monkeypatch.setattr(nodes.coref, "resolve", coref)
    monkeypatch.setattr(nodes.intent_mod, "classify", classify)
    graph = build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "context-handoff-turns"}}
    first = await graph.ainvoke(runtime._graph_input("test", "那笔单到哪里了？", 7, 1, "", 0), config)
    assert first["route"] == "business"
    assert first["resolved_query"] == "订单 1001 到哪里了？"
    assert handoff.rewrite == []

    current = "换个问题，MH-FD10 怎么清洗？"
    second = await graph.ainvoke(runtime._graph_input("test", current, 7, 3, "", 0), config)
    assert second["route"] == "knowledge"
    assert second["resolved_query"] == current
    assert handoff.intent[-1][0] == current
    assert handoff.rewrite == [current]
    assert handoff.check[0][0] == current
    assert next(m for m in second["messages"] if m.id == "db-3").content == current


async def test_graph_empty_resolution_keeps_current_request_and_not_old_query(monkeypatch, handoff):
    original = "MH-FD10 怎么清洗？"

    async def coref(query, history):
        return " \n "

    monkeypatch.setattr(nodes.coref, "resolve", coref)
    out = await build_graph().ainvoke({"messages": [HumanMessage(original)],
                                      "resolved_query": "订单 1001 到哪里了？"})
    assert out["resolved_query"] == original
    assert out["trace"]["coref"] == "passthrough"
    assert handoff.intent[0][0] == original
    assert handoff.rewrite == [original]
    assert handoff.check[0][0] == original


@pytest.mark.parametrize("gate", ["retrieval_low_conf", "self_check"])
async def test_resolved_query_keeps_existing_rejection_gates(monkeypatch, handoff, gate):
    original, resolved = "这款怎么清洗？", "MH-W40 饮水机怎么清洗？"

    async def coref(query, history):
        return resolved

    async def empty_search(query, **kwargs):
        handoff.retrieval.append((query, kwargs))
        return []

    async def insufficient(query, evidence):
        handoff.check.append((query, evidence))
        return {"useful": False, "reason": "缺少具体型号的清洁依据"}

    monkeypatch.setattr(nodes.coref, "resolve", coref)
    if gate == "retrieval_low_conf":
        monkeypatch.setattr(nodes.retrieval, "search_knowledge", empty_search)
    else:
        monkeypatch.setattr(nodes.selfcheck, "check_sufficient", insufficient)
    out = await build_graph().ainvoke({"messages": [HumanMessage(original)], "conversation_id": 7})
    assert handoff.rewrite == [resolved]
    assert handoff.retrieval[0][0] == resolved
    assert out["fallback_source"] == gate
    assert out["trace"]["confidence"] == "weak"
    assert handoff.prompts == []
    assert handoff.low_confidence[0][0][1] == original
    assert handoff.low_confidence[0][0][2] == gate
