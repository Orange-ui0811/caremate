import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.graph import nodes


def test_resolve_answer_prefers_explicit():
    assert nodes.resolve_answer({"answer": "固定话术", "messages": []}) == "固定话术"


def test_resolve_answer_falls_back_to_last_ai():
    state = {"messages": [HumanMessage("hi"), AIMessage("模型答复")]}
    assert nodes.resolve_answer(state) == "模型答复"


@pytest.mark.asyncio
async def test_log_node_persists_assistant(monkeypatch):
    saved = {}

    async def fake_append(cid, role, content=None, **k):
        saved.update(cid=cid, role=role, content=content, **k)
        return 1

    monkeypatch.setattr(nodes.repository, "append_message", fake_append)
    await nodes.log_node({"conversation_id": 8, "intent": "订单", "route": "business",
                          "messages": [AIMessage("订单已发货")], "trace": {"forced_rag": False}})
    assert saved["cid"] == 8
    assert saved["role"] == "assistant"
    assert saved["content"] == "订单已发货"
    assert saved["activities"] == []


async def test_log_node_persists_low_confidence_retrieval_activity(monkeypatch):
    saved = {}
    async def fake_append(cid, role, **kwargs):
        saved.update(kwargs)
    monkeypatch.setattr(nodes.repository, "append_message", fake_append)
    await nodes.log_node({"conversation_id": 1, "answer": "暂未查到",
                          "trace": {"forced_rag": True, "confidence": "weak"}})
    assert saved["activities"] == [{"kind": "retrieval", "name": "retrieve_knowledge"}]


async def test_log_node_retains_exact_reply_citation_snapshot(monkeypatch):
    saved={}
    async def append(*args,**kw):saved.update(kw)
    monkeypatch.setattr(nodes.repository,'append_message',append)
    citations=[{'n':6,'id':20,'answer':'质量问题退货运费由平台承担','section_path':'退换货运费'}]
    await nodes.log_node({'conversation_id':136,'answer':'运费由平台承担[6]。','citations':citations})
    assert saved['activities']==[{'kind':'citations','items':citations}]
