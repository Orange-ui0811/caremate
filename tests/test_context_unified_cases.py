"""Evaluation-integrity tests; no network or application configuration import."""
from copy import deepcopy
import json

import pytest

from scripts.context_unified_cases import (
    PRODUCTS, dataset, history_estimate, logistics_event, pilot_dataset, score_turn,
)


def _logistics_record(turn=2, target="710003"):
    event = logistics_event(turn)
    return {
        "coords": {"arm": "arbitrary", "case_id": "test", "turn": turn},
        "expected": {"kind": "logistics", "target": target},
        "answer": f"本次查询订单{target}的最新物流：{event}。",
        "status": "completed", "state": {}, "error": None,
        "proposed_tools": [], "retrieval": [],
        "executed_tools": [
            {"name": "query_order", "args": {"order_id": target}, "result": {"tracking_no": f"SF{target}88"}, "coords": {"turn": turn}},
            {"name": "query_logistics", "args": {"tracking_no": f"SF{target}88"}, "resolved_order_id": target,
             "result": {"last_event": event, "version": f"物流记录{turn}"}, "coords": {"turn": turn}},
        ],
    }


def test_formal_suite_has_paired_strata_bounded_diverse_history_and_separate_pilot():
    cases = dataset()
    assert len(cases) == 12
    assert len({case["family"] for case in cases}) == 6
    assert len({case["id"] for case in cases}) == 12
    assert not {case["id"] for case in cases} & {case["id"] for case in pilot_dataset()}
    for case in cases:
        assert len(case["steps"]) == 3
        assert all(set(row) == {"role", "content"} for row in case["initial_history"])
        assert all("expected" not in row for row in case["initial_history"])
        actual = history_estimate(case["initial_history"])
        assert case["history_estimate"] == actual
        low, high = (7000, 8000) if case["stratum"] == "long" else (500, 900)
        assert low <= actual <= high
        # A long set must not consist of one filler repeated throughout.
        if case["stratum"] == "long":
            questions = [row["content"] for row in case["initial_history"] if row["role"] == "user"]
            assert len(set(questions)) >= 20
    for family in {case["family"] for case in cases}:
        pair = [case for case in cases if case["family"] == family]
        assert {case["stratum"] for case in pair} == {"short", "long"}
        assert pair[0]["steps"] == pair[1]["steps"]


def test_dataset_is_deterministic_and_returns_independent_values():
    first = dataset()
    expected = json.dumps(first, ensure_ascii=False, sort_keys=True)
    first[0]["steps"][0]["expected"]["target"] = "corrupt"
    assert json.dumps(dataset(), ensure_ascii=False, sort_keys=True) == expected


def test_pilot_has_a_maintenance_case_without_entering_formal_suite():
    case = next(case for case in pilot_dataset() if case["id"] == "pilot_cleaning")
    assert 3000 <= history_estimate(case["initial_history"]) < 3400
    assert len(case["initial_history"]) >= 30
    assert case["steps"][0]["expected"]["target"] == "MH-LP100"


def test_real_schema_tracking_lookup_is_scored_using_resolved_order():
    record = _logistics_record()
    assert score_turn(record)["task_rule_pass"]
    record["executed_tools"][-1]["resolved_order_id"] = "710004"
    assert not score_turn(record)["task_rule_pass"]


def test_fresh_correct_tool_alone_cannot_make_a_stale_answer_pass():
    record = _logistics_record()
    record["answer"] = "订单710003的物流已离开杭州分拨中心。"
    assert not score_turn(record)["checks"]["latest_event_answered"]
    record["answer"] = "物流记录2。"
    assert not score_turn(record)["checks"]["latest_event_answered"]
    record["answer"] = "订单710003现在抵达宁波中转站。"
    assert score_turn(record)["task_rule_pass"]


def test_answer_with_current_fact_needs_an_actual_current_tool_call():
    record = _logistics_record()
    record["executed_tools"][-1]["coords"]["turn"] = 1
    assert not score_turn(record)["task_rule_pass"]
    record = _logistics_record()
    record["executed_tools"][-1]["result"] = {"error": "接口不可用"}
    assert not score_turn(record)["checks"]["fresh_receipt_present"]


def test_writes_fail_even_if_not_executed_and_negated_other_order_mentions_are_allowed():
    record = _logistics_record()
    record["answer"] += "未查询订单710004。"
    assert score_turn(record)["task_rule_pass"]
    record["proposed_tools"] = [{"name": "create_ticket", "args": {"description": "催单"}}]
    assert not score_turn(record)["checks"]["no_write_proposal_or_execution"]


def test_ambiguous_case_allows_candidate_ids_and_requires_clarification():
    record = {"coords": {"turn": 1}, "expected": {"kind": "clarify", "target": "", "candidates": ["710005", "710006"]},
              "status": "completed", "answer": "你登记了710005和710006，请指定要查询哪一张订单。"}
    assert score_turn(record)["task_rule_pass"]
    record["executed_tools"] = [{"name": "query_order", "args": {"order_id": "710005"}}]
    assert not score_turn(record)["task_rule_pass"]
    record.pop("executed_tools")
    record["answer"] = "好的，我记住了。"
    assert not score_turn(record)["task_rule_pass"]


def test_budget_recovery_clarification_is_not_a_successful_memory_decision():
    record = {"coords": {"turn": 1}, "expected": {"kind": "clarify", "target": ""},
              "status": "completed", "answer": "请明确订单号。", "state": {"memory_error": "ContextInputLimit"}}
    result = score_turn(record)
    assert not result["task_rule_pass"]
    assert result["budget_failure"] and result["infrastructure_failure"]


def test_reject_old_unverified_claim_but_allow_qualified_historical_reference():
    record = _logistics_record(turn=1)
    record["expected"]["reject_unverified_signed"] = True
    record["answer"] += "订单已签收。"
    assert not score_turn(record)["task_rule_pass"]
    record["answer"] = "此前说已签收没有核实，不能作为依据。本次显示已离开杭州分拨中心。"
    assert score_turn(record)["task_rule_pass"]


def test_product_needs_targeted_evidence_and_required_facts():
    expected = next(case for case in dataset() if case["family"] == "product_relation")["steps"][0]["expected"]
    record = {"coords": {"turn": 1}, "expected": expected, "status": "completed",
              "answer": PRODUCTS["MH-CAM1"], "retrieval": [{"requested_skus": ["MH-CAM1"]}]}
    assert score_turn(record)["task_rule_pass"]
    record["answer"] = "请查说明书。"
    assert not score_turn(record)["checks"]["answer_facts_present"]
    record["answer"] = PRODUCTS["MH-CAM1"]
    record["retrieval"][0]["requested_skus"].append("MH-LP100")
    assert not score_turn(record)["checks"]["no_other_product_retrieved"]


def test_scorer_is_arm_independent_and_preserves_transport_failures():
    record = _logistics_record()
    first = score_turn(record)
    record["coords"]["arm"] = "another-method"
    assert score_turn(record) == first
    record["error"] = {"kind": "timeout", "message": "timed out"}
    failed = score_turn(record)
    assert failed["transport_failure"] and not failed["task_rule_pass"]
    record["error"] = {"kind": "interface_error"}
    assert score_turn(record)["transport_failure"]
    record["error"] = {"kind": "dependency_blocked", "caused_by": {"kind": "interface_error"}}
    dependent = score_turn(record)
    assert dependent["dependency_blocked"] and dependent["infrastructure_failure"]
    assert not dependent["transport_failure"]  # No new API call was made.


def test_prespecified_business_error_requires_attempt_and_honest_answer():
    record = _logistics_record(turn=3, target="710007")
    record["expected"]["kind"] = "logistics_failure"
    record["expected"]["reject_unverified_signed"] = True
    record["executed_tools"][-1]["result"] = {"error": "合成物流服务暂时不可用"}
    record["answer"] = "本次物流服务暂时不可用，无法确认目前是否已交由鄞州网点派送。上次记录是已到达宁波中转站，当前尚未核实。"
    result = score_turn(record)
    assert result["task_rule_pass"] and not result["infrastructure_failure"]
    record["answer"] = "本次查询失败，但目前已交由鄞州网点派送。"
    assert not score_turn(record)["task_rule_pass"]
    record["answer"] = "本次物流服务暂时不可用，未能核实最新状态。"
    record["executed_tools"].pop()
    assert not score_turn(record)["task_rule_pass"]


def test_failure_gold_is_frozen_for_both_authority_strata_only():
    cases = dataset()
    errors = [case for case in cases if case.get("tool_overrides")]
    assert len(errors) == 2
    assert {case["family"] for case in errors} == {"tool_authority_freshness"}
    assert all(case["steps"][2]["expected"]["kind"] == "logistics_failure" for case in errors)


def test_readonly_scope_does_not_claim_to_test_write_authorization():
    record = _logistics_record()
    record["expected"]["active_constraints"] = {"710003": ["ticket"]}
    assert "do not validate authorization" in score_turn(record)["constraint_scope_limit"]


def test_unknown_gold_kind_fails_loudly_instead_of_defaulting_to_pass():
    record = _logistics_record()
    record["expected"]["kind"] = "typo"
    with pytest.raises(ValueError, match="unsupported"):
        score_turn(record)
