"""Guard evaluation labels, causal inputs, scoring and state/retrieval isolation."""
import json

from langchain_core.messages import AIMessage, HumanMessage

from scripts.context_benchmark_cases import KINDS, make_suite
from scripts.context_benchmark_strategies import ContextAdapter, retrieve_history
from scripts.eval_context_comparison import score_probe, tool_result, bootstrap_difference
from scripts.audit_context_semantics import audited_score


def test_balanced_dataset_has_two_causal_queries_and_distinct_tool_values():
    suite = make_suite()
    assert suite == make_suite()
    assert len(suite["cases"]) == 48
    for kind in KINDS:
        assert sum(c["kind"] == kind for c in suite["cases"]) == 6
    for case in suite["cases"]:
        assert len(case["history"]) == case["turns"] * 2
        assert [p["at_turn"] for p in case["probes"]] == [18, case["turns"]]
        assert len({o["tracking_no"] for o in case["tool_data"]["orders"].values()}) == 4
        for probe in case["probes"]:
            expected = probe["expected"]
            prefix = json.dumps(case["history"][:2 * probe["at_turn"]], ensure_ascii=False)
            for key in ("order_id", "sku", "date"):
                assert not expected[key] or expected[key] in prefix


def test_corrections_and_temporal_updates_have_different_checkpoint_answers():
    for case in make_suite()["cases"]:
        if case["kind"] == "correction":
            a, b = [p["expected"]["order_id"] for p in case["probes"]]
            assert a != b
            assert b not in json.dumps(case["history"][:36])
        if case["kind"] == "temporal_update":
            assert case["probes"][0]["expected"]["date"] != case["probes"][1]["expected"]["date"]


async def test_state_builder_only_receives_visible_prefix_and_original_ids():
    seen = []
    async def call(stage, fn, prompt=None):
        seen.append("\n".join(str(m.content) for m in prompt))
        return {"ok": True, "value": {"facts": [{"text": "订单关系", "source_ids": [1], "valid": True}]},
                "stage": stage, "calls": [], "errors": [], "seconds": 0}
    case = next(c for c in make_suite()["cases"] if c["kind"] == "correction")
    adapter = ContextAdapter("D_state", {}, call, "shared system")
    await adapter.advance(case["history"][:36])
    assert adapter.boundary == 14
    assert len(seen) == 1
    assert case["probes"][0]["query"] not in seen[0]
    assert case["probes"][1]["expected"]["order_id"] not in seen[0]
    assert "SF" not in seen[0]
    assert "[1]" in seen[0]


async def test_state_rejects_future_citation_without_advancing_boundary():
    async def call(stage, fn, prompt=None):
        return {"ok": True, "value": {"facts": [{"text": "future", "source_ids": [999], "valid": True}]},
                "stage": stage, "calls": [], "errors": [], "seconds": 0}
    adapter = ContextAdapter("D_state", {}, call, "system")
    await adapter.advance(make_suite()["cases"][0]["history"][:30])
    assert adapter.boundary == 0
    assert adapter.state == {"facts": []}
    assert not adapter.maintenance[0]["ok"]


def test_retrieval_returns_original_messages_and_never_reads_future_rows():
    case = make_suite()["cases"][0]
    rows = case["history"][:36]
    result = retrieve_history(rows, case["probes"][0]["query"], {"facts": []})
    assert len(result) <= 4
    assert all(max(c["source_ids"]) <= 36 for c in result)
    assert any(1 in c["source_ids"] for c in result)


def test_wrong_tool_target_cannot_pass_even_with_correct_words_in_answer():
    case = make_suite()["cases"][0]
    probe = case["probes"][0]
    target = probe["expected"]["order_id"]
    result = {"coref": {"ok": True, "value": "订单" + target + "物流", "calls": [{}]},
              "plan": {"ok": True, "value": {"action": "logistics", "order_id": "999999", "sku": ""}},
              "reply": {"ok": True, "value": "订单" + target + "运单SF" + target + "88"}}
    score = score_probe(case, probe, result)
    assert score["answer_rule_pass"]
    assert not score["target_pass"]
    assert not score["task_pass"]
    assert "error" in tool_result(case, result["plan"]["value"])


def test_semantic_audit_accepts_related_order_but_rejects_another_order_in_care_reply():
    case = next(c for c in make_suite()["cases"] if c["kind"] == "early_product")
    probe = case["probes"][0]
    sku = probe["expected"]["sku"]
    related, other = case["all_orders"][:2]
    result = {"coref": {"ok": True, "value": sku + "怎么清洁", "calls": [{}]},
              "plan": {"ok": True, "value": {"action": "care", "order_id": "", "sku": sku}},
              "reply": {"ok": True, "value": f"订单{related}的{sku}，清洁前断电，主机用软布。"}}
    assert score_probe(case, probe, result)["task_pass"]
    assert audited_score(case, probe, result)["task_pass"]
    result["reply"]["value"] += "另一个订单" + other
    assert not audited_score(case, probe, result)["task_pass"]


def test_bootstrap_preserves_paired_conversation_unit():
    assert bootstrap_difference([0, 0, 0]) == [0, 0]
    assert bootstrap_difference([.5, .5, .5]) == [.5, .5]


async def test_harness_pressure_keeps_recent_messages_and_merges_checkpoint():
    calls = []
    async def call(stage, fn, prompt=None):
        calls.append(prompt)
        return {"ok": True, "value": "## Primary Request and Intent\n- order610010",
                "stage": stage, "calls": [], "errors": [], "seconds": 0}
    harness = {"instruction": "EXACT_ORIGINAL_PROMPT", "preamble": "CHECKPOINT"}
    adapter = ContextAdapter("H_original", harness, call, "system", capacity=600)
    case = make_suite()["cases"][0]
    await adapter.advance(case["history"][:36])
    assert calls
    assert calls[0][-1].content == "EXACT_ORIGINAL_PROMPT"
    assert any("<compacted-summary>" in str(m.content) for m in adapter.surface)
    assert adapter.surface[-1].content == case["history"][35]["content"]
    assert sum(len(s["calls"]) for s in adapter.maintenance) == 0


def test_full_and_window_baselines_do_not_use_state():
    case = make_suite()["cases"][0]
    for name in ("A_full", "B_recent6", "B_window3000"):
        adapter = ContextAdapter(name, {}, None, "system")
        adapter.summary = "MUST_NOT_LEAK"
        result = adapter.contexts(case["history"][:36], case["probes"][0]["query"])
        assert "MUST_NOT_LEAK" not in result["coref_history"]
        assert all("MUST_NOT_LEAK" not in str(m.content) for m in result["body"])
