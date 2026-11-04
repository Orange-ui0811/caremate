"""Dataset causality, source grounding and frozen scorer compatibility checks."""
import copy
import hashlib
import json

import pytest

from scripts.context_benchmark_cases import KINDS, make_suite
from scripts.context_matched_cases import dataset, export
from scripts.eval_context_comparison import score_probe, tool_result


@pytest.mark.parametrize("length_stratified", [False, True])
def test_shape_visible_sources_and_unmodified_regressions(length_stratified):
    data = dataset(length_stratified)
    assert len(data["cases"]) == 16
    originals = {c["id"]: c for c in make_suite()["cases"]}
    for cohort in ("regression", "transfer"):
        cases = [c for c in data["cases"] if c["cohort"] == cohort]
        assert {c["kind"] for c in cases} == set(KINDS)
        for case in cases:
            assert len(case["history"]) == case["turns"] * 2
            assert len(case["probes"]) == 2
            for i, row in enumerate(case["history"]):
                assert row["role"] == ("user" if i % 2 == 0 else "assistant")
            for probe in case["probes"]:
                assert probe["gold_evidence"] and probe["rationale"]
                for e in probe["gold_evidence"]:
                    assert 1 <= e["seq"] <= 2 * probe["at_turn"]
                    row = case["history"][e["seq"] - 1]
                    assert row["role"] == e["role"]
                    assert row["content"][e["start"]:e["end"]] == e["text"]
                assert probe["expected_snapshot"]["valid_at_turn"] == probe["at_turn"]
            if cohort == "regression":
                old = originals[case["id"]]
                assert case["history"] == old["history"] and case["tool_data"] == old["tool_data"]
                for new_probe, old_probe in zip(case["probes"], old["probes"]):
                    assert {key: new_probe[key] for key in old_probe} == old_probe


@pytest.mark.parametrize("length_stratified", [False, True])
def test_later_changes_are_not_visible_at_first_probe(length_stratified):
    for case in dataset(length_stratified)["cases"]:
        p1, p2 = case["probes"]
        if case["kind"] in {"correction", "topic_switch", "temporal_update", "negation"}:
            assert p1["expected"] != p2["expected"]
            assert any(e["seq"] > 2 * p1["at_turn"] for e in p2["gold_evidence"])
        if case["kind"] == "correction":
            assert p2["expected"]["order_id"] not in "\n".join(r["content"] for r in case["history"][:2 * p1["at_turn"]])


@pytest.mark.parametrize("length_stratified", [False, True])
def test_gold_reference_answers_pass_unchanged_rule_scorer(length_stratified):
    for case in dataset(length_stratified)["cases"]:
        for probe in case["probes"]:
            expected = probe["expected"]
            plan = {key: expected[key] for key in ("action", "order_id", "sku")}
            plan["answer"] = ""
            value = tool_result(case, plan)
            if expected["action"] == "logistics":
                assert value["order_id"] == expected["order_id"]
                text = f"订单{value['order_id']}，运单{value['tracking_no']}，{value['status']}。"
            elif expected["action"] == "care":
                assert value["sku"] == expected["sku"] and value["care"]
                text = value["sku"] + "：" + value["care"]
            elif expected["action"] == "recall":
                text = "承诺到货日期为" + expected["date"]
            else:
                text = "请确认具体要查哪一笔订单。"
            call = {"ok": True, "calls": [{"input_tokens": 1, "output_tokens": 1}]}
            result = {"coref": {**call, "value": text}, "plan": {**call, "value": plan}, "reply": {**call, "value": text}, "context_text": text}
            score = score_probe(case, probe, result)
            assert score["task_pass"] and score["coref_pass"], (case["id"], probe["id"], score)


def test_export_is_deterministic_frozen_and_self_describing(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    digest = export(a)
    assert export(b) == digest
    assert hashlib.sha256((a / "dataset.json").read_bytes()).hexdigest() == digest
    assert json.loads((a / "dataset.json").read_text(encoding="utf-8")) == dataset()
    assert json.loads((a / "README.json").read_text(encoding="utf-8"))["probe_count"] == 32
    with pytest.raises(ValueError, match="never overwritten"):
        export(a)


def test_v2_selection_is_fixed_and_v1_content_is_unchanged(tmp_path):
    original_hash = "69297651e76563777889bf2bb38ac6da29147f4f249afdc95a56de0866b7def5"
    assert export(tmp_path / "v1") == original_hash
    old, new = dataset(), dataset(length_stratified=True)
    assert old["cases"][8:] == new["cases"][8:]
    regressions = new["cases"][:8]
    assert [c["kind"] for c in regressions] == list(KINDS)
    assert [c["turns"] for c in regressions] == [24, 64, 24, 64, 24, 64, 24, 64]
    assert [c["id"] for c in regressions] == ["svc-01", "svc-09", "svc-13", "svc-21", "svc-25", "svc-33", "svc-37", "svc-45"]
    export(tmp_path / "v2", length_stratified=True)
    frozen = json.loads((tmp_path / "v2" / "dataset.json").read_text(encoding="utf-8"))
    assert frozen == new and frozen["version"] == "context-matched-cases-length-stratified-v2"
