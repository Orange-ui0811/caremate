r"""两组同功能对照：原生 HTTP / LangChain，普通 Python / LangGraph。

运行：.venv\Scripts\python.exe examples/framework_comparison.py
模型示例使用内存中的固定响应，只验证调用和解析，不访问外网、不读取密钥。
图示例使用固定知识条目，不调用模型、不连接业务数据库。
"""

import asyncio
import json
from typing import Literal

import httpx
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class AfterSale(BaseModel):
    """从用户描述中提取售后信息。"""

    order_id: str | None = Field(default=None, description="原文明示的订单号；没有则为空")
    request_type: Literal["退款", "换货", "维修", "投诉", "其他"]


MESSAGES = [
    {"role": "system", "content": "提取售后信息，只使用原文明示的信息，不编造订单号。"},
    {"role": "user", "content": "订单1001的猫粮还没发货，我想退款。"},
]


async def extract_without_langchain(client: httpx.AsyncClient) -> AfterSale:
    """自己组装模型协议、读取 tool_calls，再交给 Pydantic 校验。"""
    response = await client.post(
        "https://example.invalid/v1/chat/completions",
        headers={"Authorization": "Bearer demo-not-a-real-key"},
        json={
            "model": "demo-model",
            "temperature": 0,
            "messages": MESSAGES,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "AfterSale",
                    "description": "从用户描述中提取售后信息。",
                    "parameters": AfterSale.model_json_schema(),
                },
            }],
            "tool_choice": {"type": "function", "function": {"name": "AfterSale"}},
        },
    )
    response.raise_for_status()
    calls = response.json()["choices"][0]["message"].get("tool_calls") or []
    call = next((c for c in calls if c["function"]["name"] == "AfterSale"), None)
    if call is None:
        raise ValueError("模型没有返回要求的结构化结果")
    return AfterSale.model_validate_json(call["function"]["arguments"])


async def extract_with_langchain(client: httpx.AsyncClient) -> AfterSale:
    """LangChain 负责工具定义转换、响应解析，以及调用 Pydantic 校验。"""
    model = ChatOpenAI(
        model="demo-model",
        base_url="https://example.invalid/v1",
        api_key="demo-not-a-real-key",
        temperature=0,
        max_retries=0,
        http_async_client=client,
    )
    extractor = model.with_structured_output(AfterSale, method="function_calling")
    return await extractor.ainvoke(MESSAGES)


class ChatState(TypedDict, total=False):
    question: str
    route: str
    evidence: str
    answer: str


# 以下是两种编排方式共用的业务函数。框架不会替你编写这些业务规则。
def classify(state: ChatState) -> dict:
    question = state["question"]
    route = "knowledge" if any(w in question for w in ("退款", "会员")) else "other"
    return {"route": route}


def retrieve(state: ChatState) -> dict:
    evidence = "退款原路退回，到账时间以平台售后规则为准。" if "退款" in state["question"] else ""
    return {"evidence": evidence}


def answer(state: ChatState) -> dict:
    return {"answer": "查到的资料：" + state["evidence"]}


def fallback(state: ChatState) -> dict:
    return {"answer": "暂时没有可用资料，请联系人工客服。"}


def run_without_langgraph(question: str) -> ChatState:
    state: ChatState = {"question": question}
    state.update(classify(state))
    if state["route"] != "knowledge":
        state.update(fallback(state))
        return state
    state.update(retrieve(state))
    if state["evidence"]:
        state.update(answer(state))
    else:
        state.update(fallback(state))
    return state


def make_graph():
    builder = StateGraph(ChatState)
    builder.add_node("classify", classify)
    builder.add_node("retrieve", retrieve)
    builder.add_node("answer", answer)
    builder.add_node("fallback", fallback)
    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify", lambda state: state["route"],
        {"knowledge": "retrieve", "other": "fallback"},
    )
    builder.add_conditional_edges(
        "retrieve", lambda state: "enough" if state["evidence"] else "missing",
        {"enough": "answer", "missing": "fallback"},
    )
    builder.add_edge("answer", END)
    builder.add_edge("fallback", END)
    return builder.compile()


async def main():
    requests = []

    def mock_provider(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert str(request.url) == "https://example.invalid/v1/chat/completions"
        assert body["model"] == "demo-model" and body["messages"] == MESSAGES
        function = body["tools"][0]["function"]
        assert function["name"] == "AfterSale"
        assert set(function["parameters"]["properties"]) == {"order_id", "request_type"}
        requests.append(body)
        return httpx.Response(200, json={
            "id": "demo-response", "object": "chat.completion", "created": 0,
            "model": "demo-model",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "demo-call", "type": "function", "function": {
                    "name": "AfterSale",
                    "arguments": json.dumps({"order_id": "1001", "request_type": "退款"}, ensure_ascii=False),
                }}],
            }}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(mock_provider)) as client:
        direct = await extract_without_langchain(client)
        wrapped = await extract_with_langchain(client)
        assert direct == wrapped and len(requests) == 2
        print("模型调用对照（固定模拟响应，无外网）：")
        print("  不用 LangChain：", direct.model_dump_json())
        print("  使用 LangChain：", wrapped.model_dump_json())

    graph = make_graph()
    print("流程对照（三条分支）：")
    for question in ("退款政策是什么？", "会员可以取消吗？", "今天天气怎么样？"):
        direct = run_without_langgraph(question)
        wrapped = graph.invoke({"question": question})
        assert direct == wrapped
        print(f"  {question} → {wrapped['answer']}（两种实现一致）")


if __name__ == "__main__":
    asyncio.run(main())
