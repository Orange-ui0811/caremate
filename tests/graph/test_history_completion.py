"""真实图与 checkpoint 回归;模型/数据库用本地替身,不发送会话到外部服务。"""
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages

from app.graph import nodes, runtime
from app.graph.build import build_graph


@pytest.mark.parametrize("outlet", ["闲聊", "其他", "投诉", "retrieval_low_conf", "self_check"])
@pytest.mark.parametrize("stream_next", [False, True])
async def test_fixed_reply_is_in_next_model_history(monkeypatch, outlet, stream_next):
    rows, prompts, pooled = [], [], []
    initial = "我想看看玩具" if outlet in {"retrieval_low_conf", "self_check"} else "刚才的问题"

    async def append(cid, role, content=None, **kwargs):
        row = SimpleNamespace(id=len(rows) + 1, role=role, content=content)
        rows.append(row)
        return row.id

    async def create(uid):
        return 1

    async def get(cid):
        return SimpleNamespace(summary="", summary_upto_msg_id=0)

    async def dialog(cid):
        return rows[:]

    async def noop(*args, **kwargs):
        return None

    async def coref(query, history):
        return query

    async def intent(query, history):
        label = outlet if query.startswith(initial) and outlet in {"闲聊", "其他", "投诉"} else "商品咨询"
        return {"intent": label, "confidence": 1.0}

    async def understand(query):
        return {"standard": query, "expanded": []}

    async def search(query, **kwargs):
        if query.startswith(initial) and outlet == "retrieval_low_conf":
            return []
        return [{"id": 1, "question": "退货政策", "answer": "签收后7天无理由退货",
                 "rerank_score": 0.95, "section_path": "政策/退货", "content_type": "policy"}]

    async def check(query, evidence):
        return {"useful": not query.startswith(initial), "reason": "缺少商品依据"}

    async def pool(*args, **kwargs):
        pooled.append(args)

    async def specs():
        return []

    class Model:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages, config=None):
            prompts.append(messages)
            return AIMessage("支持签收后7天无理由退货[1]。", id="model-answer")

    monkeypatch.setattr(runtime.repository, "append_message", append)
    monkeypatch.setattr(runtime.repository, "create_conversation", create)
    monkeypatch.setattr(runtime.repository, "get_conversation", get)
    monkeypatch.setattr(runtime.repository, "list_dialog_messages", dialog)
    monkeypatch.setattr(runtime.summarizer, "maybe_schedule_summary", noop)
    monkeypatch.setattr(nodes.repository, "insert_low_confidence", pool)
    monkeypatch.setattr(nodes.coref, "resolve", coref)
    monkeypatch.setattr(nodes.intent_mod, "classify", intent)
    monkeypatch.setattr(nodes.query_understanding, "understand", understand)
    monkeypatch.setattr(nodes.retrieval, "search_knowledge", search)
    monkeypatch.setattr(nodes.selfcheck, "check_sufficient", check)
    monkeypatch.setattr(nodes.registry, "get_all_specs", specs)
    monkeypatch.setattr(nodes, "get_chat_model", lambda **kwargs: Model())
    graph = build_graph(checkpointer=InMemorySaver())
    monkeypatch.setattr(runtime, "get_graph", lambda: graph)

    # 第一轮走 SSE:固定回复只能发一次,log 补入历史不能再发一遍。
    events = [event async for event in runtime.stream_turn("test", initial, None)]
    reply = rows[-1].content
    assert [e["text"] for e in events if e["type"] == "delta"] == [reply]
    first = (await graph.aget_state({"configurable": {"thread_id": "1"}})).values
    assert [(m.type, m.content) for m in first["messages"]] == [("human", initial), ("ai", reply)]
    assert not prompts
    assert len(pooled) == int(outlet in {"retrieval_low_conf", "self_check"})

    # 连续两次固定回复都必须入历史,重复文案也不能遗漏。
    followup = initial + "，再确认一下"
    events = [event async for event in runtime.stream_turn("test", followup, 1)]
    second_reply = rows[-1].content
    assert [e["text"] for e in events if e["type"] == "delta"] == [second_reply]
    assert not prompts
    assert len(pooled) == 2 * int(outlet in {"retrieval_low_conf", "self_check"})

    # 下一轮分别验证流式和非流式入口,模型看见前两轮完整问答。
    if stream_next:
        events = [event async for event in runtime.stream_turn("test", "退货政策是什么", 1)]
        assert events[-1] == {"type": "done", "conversation_id": 1}
        state = (await graph.aget_state({"configurable": {"thread_id": "1"}})).values
    else:
        out = await runtime.run_turn("test", "退货政策是什么", 1)
        state = out["state"]
    assert len(prompts) == 1
    dialog_messages = [m for m in prompts[0] if m.type != "system" and m.id != nodes.TURN_CTX_ID]
    assert [(m.type, m.content) for m in dialog_messages] == [
        ("human", initial), ("ai", reply), ("human", followup), ("ai", second_reply),
        ("human", "退货政策是什么")]
    context = next(m.content for m in prompts[0] if m.id == nodes.TURN_CTX_ID)
    assert '"退货政策是什么"' in context and "玩具" not in context
    assert "不要补答历史问题" in context
    final = state["messages"]
    assert [m.type for m in final] == ["human", "ai", "human", "ai", "human", "ai"]
    assert final[-1].content == rows[-1].content
    assert state["fallback_source"] == ""
    assert state["suggested_actions"] == []


async def test_existing_broken_checkpoint_restored_from_displayed_dialog(monkeypatch):
    rows = [SimpleNamespace(id=1, role="user", content="玩具多少钱"),
            SimpleNamespace(id=2, role="assistant", content="没有可靠信息，暂时无法确认。"),
            SimpleNamespace(id=3, role="user", content="有哪些商品"),
            SimpleNamespace(id=4, role="assistant", content="暂时未查到，请联系人工。"),
            SimpleNamespace(id=5, role="user", content="退货政策是什么")]

    async def dialog(cid):
        return rows

    monkeypatch.setattr(runtime.repository, "list_dialog_messages", dialog)
    inp = await runtime._new_turn_input("test", rows[-1].content, 1, 5, "", 0, restore_history=True)
    broken = [HumanMessage(rows[0].content, id="db-1"),
              AIMessage("", tool_calls=[{"id": "old", "name": "query_product", "args": {}}]),
              ToolMessage('{"price":203}', tool_call_id="old"),
              HumanMessage(rows[2].content, id="db-3")]
    restored = add_messages(broken, inp["messages"])
    assert [(m.type, m.content) for m in restored] == [
        ("human" if r.role == "user" else "ai", r.content) for r in rows]
    assert [m.id for m in restored] == [f"db-{r.id}" for r in rows]
    assert not any(isinstance(m, ToolMessage) for m in restored)


@pytest.mark.parametrize("content", ["正常回答", [{"type": "text", "text": "正常回答"}]])
async def test_model_answer_not_duplicated_in_history(monkeypatch, content):
    async def append(*args, **kwargs):
        return 2
    monkeypatch.setattr(nodes.repository, "append_message", append)
    out = await nodes.log_node({"conversation_id": 1,
                               "messages": [HumanMessage("问题"), AIMessage(content)]})
    assert out == {}


async def test_identical_previous_fallback_does_not_hide_current_reply(monkeypatch):
    async def append(*args, **kwargs):
        return 4
    monkeypatch.setattr(nodes.repository, "append_message", append)
    out = await nodes.log_node({"conversation_id": 1, "answer": "未查到",
                               "messages": [HumanMessage("旧问题"), AIMessage("未查到"),
                                            HumanMessage("新问题")]})
    assert out["messages"][0].content == "未查到"
    assert out["messages"][0].id == "db-4"


def test_no_current_reply_never_reuses_previous_answer():
    assert nodes.resolve_answer({"messages": [HumanMessage("旧问题"), AIMessage("旧答案"),
                                              HumanMessage("新问题")]}) == ""
