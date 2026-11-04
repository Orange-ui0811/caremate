from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.core.activities import turn_activities


def test_activities_do_not_reuse_tools_from_previous_turn():
    state = {"trace": {"forced_rag": True}, "messages": [
        HumanMessage("旧问题"), ToolMessage("{}", name="query_product", tool_call_id="old"),
        AIMessage("旧回答"), HumanMessage("退货政策"), AIMessage("政策回答")]
    }
    assert turn_activities(state) == [{"kind": "retrieval", "name": "retrieve_knowledge"}]


def test_actual_tools_and_policy_are_recorded_without_synthetic_refund():
    state = {"trace": {"retrieve_policy": {"hits": 0}}, "messages": [
        HumanMessage("退款"), ToolMessage("{}", name="query_order", tool_call_id="a"),
        ToolMessage("{}", name="query_order", tool_call_id="b"),
        ToolMessage("{}", name="submit_refund", tool_call_id="c")]
    }
    assert turn_activities(state) == [
        {"kind": "retrieval", "name": "retrieve_policy"}, {"kind": "tool", "name": "query_order"}]


def test_plain_reply_does_not_claim_retrieval():
    assert turn_activities({"messages": [HumanMessage("你好"), AIMessage("你好")]}) == []
