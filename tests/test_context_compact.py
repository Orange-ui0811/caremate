"""The cost experiment changes budgets/ablation, not hidden selection rules."""
from copy import deepcopy
import hashlib

import pytest

from app.core.context_compact import STAGE_BUDGETS, VERSION, assemble_compact_context
from app.core.context_policy import estimate
from app.core.context_state import assemble_context, build_snapshot, resolve_source


def snapshot(*texts):
    rows = []
    for text in texts:
        rows.extend([
            {"id": len(rows) + 1, "role": "user", "content": text},
            {"id": len(rows) + 2, "role": "assistant", "content": "收到，尚未执行业务操作。"},
        ])
    return build_snapshot(rows, "compact-test")


@pytest.mark.parametrize("stage", STAGE_BUDGETS)
def test_source_is_exact_original_assembly_at_the_declared_stage_budget(stage):
    state = snapshot("当前处理订单610371。先别提交退款。", "610371报错，应为610370。",
                     *("普通包装背景。" * 60 for _ in range(12)))
    before = deepcopy(state)
    query = "那单的物流到哪里了？"
    original = assemble_context(state, query, stage="answer" if stage == "plan" else stage,
                                budget=STAGE_BUDGETS[stage], recall="conditional")
    compact = assemble_compact_context(state, query, stage=stage)
    for key in ("messages", "history", "state_text", "quote_text", "summary_text", "blocked", "reason"):
        assert compact[key] == original[key]
    for key, value in original["manifest"].items():
        if key != "stage":
            assert compact["manifest"][key] == value
    assert compact["manifest"]["stage"] == stage
    assert compact["manifest"]["assembly_policy_version"] == VERSION
    assert compact["manifest"]["approx_tokens"] <= STAGE_BUDGETS[stage]
    assert state == before


def test_window_ablation_removes_state_and_recall_but_keeps_original_turn_selection():
    state = snapshot("当前处理订单610371。先别提交退款。", *("普通背景。" * 80 for _ in range(20)))
    query = "最早的订单是哪一笔？"
    stripped = {**state, "candidate": None, "active_constraints": [], "events": []}
    expected = assemble_context(stripped, query, stage="coref", budget=1800, recall="off")
    actual = assemble_compact_context(state, query, stage="coref", policy="compact_window")
    assert actual["messages"] == expected["messages"]
    assert actual["quote_text"] == ""
    assert not actual["manifest"]["protected_refs"]
    assert not actual["manifest"]["recall_triggered"]
    assert "当前对象候选=" not in actual["state_text"]
    assert state["candidate"]["value"] == "610371"
    assert state["active_constraints"]


@pytest.mark.parametrize("policy", ["compact_window", "compact_source"])
def test_caller_cap_only_reduces_budget_and_current_query_counts_once(policy):
    state = snapshot("现在查询订单610371。")
    query = "它现在到哪里了？"
    state = build_snapshot([*state["rows"], {"id": 3, "role": "user", "content": query}], "compact-test")
    packet = assemble_compact_context(state, query, stage="answer", policy=policy, budget=900)
    assert packet["manifest"]["budget"] == 900
    assert packet["manifest"]["approx_tokens"] == sum(estimate(m.content) for m in packet["messages"])
    assert sum(m.content == query for m in packet["messages"]) == 1
    assert query not in packet["history"]
    expanded = assemble_compact_context(state, query, stage="coref", policy=policy, budget=9000)
    assert expanded["manifest"]["budget"] == 1800


def test_source_budget_exhaustion_blocks_instead_of_dropping_correction_or_restriction():
    state = snapshot("当前处理订单610371。先别提交退款。", "610371报错，应为610370。")
    packet = assemble_compact_context(state, "提交这单退款。", stage="plan", budget=50)
    assert packet["blocked"] and packet["messages"] == []
    assert packet["manifest"]["protected_refs"]
    enough = assemble_compact_context(state, "提交这单退款。", stage="plan")
    assert not enough["blocked"]
    assert "先别提交退款。" in enough["quote_text"]
    assert "610371报错，应为610370。" in enough["quote_text"]
    for ref in enough["manifest"]["protected_refs"]:
        source = resolve_source(state, ref)
        assert source["text"] in enough["quote_text"]


def test_retracted_restriction_is_not_reintroduced_by_compact_source():
    state = snapshot("当前处理订单610371。先别提交退款。", "现在可以退订单610371。",
                     "现在处理订单610372。先别提交退款。")
    packet = assemble_compact_context(state, "查询订单610371的物流。", stage="business")
    assert not packet["blocked"]
    assert not state["active_constraints"] or all(c["entity"] != "order:610371" for c in state["active_constraints"])
    assert "有效限制：对象=order:610371" not in packet["state_text"]


@pytest.mark.parametrize("policy", ["compact_source", "compact_window"])
def test_existing_summary_quarter_budget_rule_is_visible_and_unchanged(policy):
    state = snapshot(*("普通包装背景。" * 100 for _ in range(14)))
    first = state["rows"][0]
    segment = {"id": "valid-summary", "conversation_id": "compact-test", "from_msg_id": 1,
               "upto_msg_id": 20, "summary": "之前一直讨论包装，未进行业务核验。" * 26,
               "source_refs": [{"message_id": 1,
                                "content_hash": hashlib.sha256(first["content"].encode()).hexdigest()}]}
    coref = assemble_compact_context(state, "后来的包装呢？", stage="coref", policy=policy, summaries=[segment])
    plan = assemble_compact_context(state, "后来的包装呢？", stage="plan", policy=policy, summaries=[segment])
    assert 450 < estimate(segment["summary"]) < 900
    assert coref["manifest"]["summary_refs"] == []
    assert plan["manifest"]["summary_refs"] == ["valid-summary"]
    assert "effective_budget//4" in coref["manifest"]["summary_admission"]
    bad = {**segment, "conversation_id": "foreign"}
    assert not assemble_compact_context(state, "包装呢？", summaries=[bad], policy=policy)["manifest"]["summary_refs"]


@pytest.mark.parametrize("options", [
    {"stage": "unknown"}, {"policy": "unknown"}, {"budget": 0},
    {"budget": -1}, {"budget": 3.5}, {"budget": True},
])
def test_invalid_policy_stage_or_budget_fails_before_assembly(options):
    with pytest.raises(ValueError):
        assemble_compact_context(snapshot("普通背景。"), "你好", **options)
