import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from scripts.eval_context_engineering import contexts, make_cases, present, score_rewrite, tracked
from scripts.audit_context_eval import audit_rewrite


def test_fixed_corpus_has_balanced_reference_and_correction_cases():
    fixture = make_cases()
    assert fixture == make_cases()
    assert len(fixture["cases"]) == 24
    for kind in {c["kind"] for c in fixture["cases"]}:
        assert sum(c["kind"] == kind for c in fixture["cases"]) == 6
    for case in fixture["cases"]:
        assert len(case["history"]) == 2 * case["history_turns"]
        assert all(any(present(entity, row["content"]) for row in case["history"])
                   for entity in case["historical_entities"])


def test_scoring_rejects_ambiguous_rewrite_and_conflicting_orders():
    case = make_cases()["cases"][0]
    correct = "查询订单" + case["required_entity_groups"][0][0] + "的物流"
    assert score_rewrite(case, correct)
    assert not score_rewrite(case, case["query"])
    assert not score_rewrite(case, correct + "或订单" + case["forbidden_entities"][0])
    assert not score_rewrite(case, "查询订单" + case["required_entity_groups"][0][0] + "的退款金额")
    assert not present("710001", "订单1710001")
    assert present("MH-LP100", "型号 mh lp100")


def test_summary_is_not_leaked_into_full_or_window_only_baselines():
    case = make_cases()["cases"][0]
    histories, agent = contexts(case, "摘要专用事实MARKER", 28)
    assert "MARKER" in histories["summary_window"]
    for name in ["full_history", "window_only"]:
        assert "MARKER" not in histories[name]
        assert all("MARKER" not in str(m.content) for m in agent[name])
    assert len(agent["summary_window"]) < len(agent["full_history"])


async def test_real_langchain_callback_path_captures_provider_usage():
    model = FakeMessagesListChatModel(responses=[AIMessage("订单710001物流", usage_metadata={
        "input_tokens": 123, "output_tokens": 4, "total_tokens": 127})])
    async def call():
        result = await model.ainvoke("test")
        return result.content
    result = await tracked(call, 5)
    assert result["text"] == "订单710001物流"
    assert result["calls"][0]["input_tokens"] == 123
    assert not result["errors"]


def test_audit_rejects_correct_sku_paired_with_invented_order_or_wrong_product():
    case = next(c for c in make_cases()["cases"] if c["id"] == "ctx-09")
    assert audit_rewrite(case, "自动喂食器基础版（型号MH-FD10）应该怎么清洗？")["passed"]
    invented = audit_rewrite(case, "订单1001的智能猫砂盆（型号MH-FD10）应该怎么清洗？")
    assert not invented["passed"]
    assert "fabricated_numeric_entity:1001" in invented["reasons"]
    assert "wrong_product_family:猫砂盆" in invented["reasons"]
