from copy import deepcopy
import hashlib

import pytest

from app.core.context_policy import estimate
from app.core.context_state import build_snapshot
from app.core.context_task import assemble_task_context


def snapshot(dialog):
    rows = [{"id": i + 1, "role": role, "content": text} for i, (role, text) in enumerate(dialog)]
    return build_snapshot(rows, 7)


def test_current_query_counted_once_and_long_query_blocks():
    query = "请查询订单710001。"
    snap = snapshot([("user", "你好"), ("assistant", "您好"), ("user", query)])
    p = assemble_task_context(snap, query)
    assert sum(m.content == query for m in p["messages"]) == 1
    assert query not in p["history"]
    assert p["manifest"]["approx_tokens"] == max(sum(estimate(m.content) for m in p["messages"]),
                                                   estimate(p["history"]) + estimate(query))
    blocked = assemble_task_context(snap, "长" * 3000, stage="intent")
    assert blocked["blocked"] and blocked["messages"] == []


def test_exact_provenance_offsets_and_no_mutation():
    snap = snapshot([("user", "订单710001不要退款，也不要创建工单。"), ("assistant", "我之前说已经退款，但这不代表实际执行。")])
    old = deepcopy(snap)
    p = assemble_task_context(snap, "订单710001查物流")
    assert snap == old
    for ref in p["manifest"]["source_refs"]:
        row = next(r for r in snap["rows"] if r["id"] == ref["message_id"])
        assert ref["start"] == 0 and ref["end"] == len(row["content"])
        assert ref["content_hash"] == hashlib.sha256(row["content"].encode()).hexdigest()
        assert row["content"] in p["history"]
    assert "未经本轮核验" in p["history"]
    assert "不要退款，也不要创建工单" in p["history"]


def test_lexical_retrieval_without_trigger_and_relationship_expansion():
    dialog = [("user", "办公室的设备由小林保管，订单710002。"), ("assistant", "收到。"),
              ("user", "订单710002型号是MH-W34，另一订单710001型号是MH-CAM6。"), ("assistant", "了解。")]
    for i in range(8):
        dialog += [("user", f"无关背景{i}：今天准备整理柜子。"), ("assistant", "好的。")]
    snap = snapshot(dialog)
    p = assemble_task_context(snap, "小林保管的设备如何清洁？", stage="coref")
    assert not p["blocked"]
    assert "小林保管" in p["history"] and "MH-W34" in p["history"]
    assert p["manifest"]["retrieval_enabled"]
    assert {1, 2, 3, 4}.issubset(p["manifest"]["selected_message_ids"])
    assert p["manifest"]["selected_message_ids"] == sorted(p["manifest"]["selected_message_ids"])


def test_stale_candidate_and_partial_retraction_are_only_original_evidence():
    snap = snapshot([("user", "本次只查订单710001。"), ("assistant", "收到。"),
                     ("user", "接下来咨询的配送对象换成710002。只撤回710002工单限制；710001不要退款。"),
                     ("assistant", "好的。")])
    assert snap["candidate"]["value"] == "710001"
    p = assemble_task_context(snap, "现在这个包裹在哪里？")
    assert "当前对象候选" not in p["history"] and "有效限制：" not in p["history"]
    assert "配送对象换成710002" in p["history"] and "只撤回710002工单限制" in p["history"]


def test_full_does_not_silently_fall_back_and_window_keeps_whole_units():
    snap = snapshot([("user", "老" * 600), ("assistant", "旧答复" * 100),
                     ("user", "近期问题"), ("assistant", "近期答复")])
    full = assemble_task_context(snap, "继续", policy="full", budget=400)
    window = assemble_task_context(snap, "继续", policy="window", budget=400)
    assert full["blocked"]
    assert not window["blocked"]
    assert window["manifest"]["selected_message_ids"] == [3, 4]


def test_strongest_evidence_is_not_partially_truncated_to_fit():
    snap = snapshot([("user", "订单710001是办公室那件。" + "附加描述" * 400), ("assistant", "收到"),
                     ("user", "其他问题"), ("assistant", "其他答复")])
    p = assemble_task_context(snap, "办公室订单710001在哪里？", budget=500)
    assert p["blocked"] and "strongest" in p["reason"] and not p["messages"]


def test_summary_whole_admission_at_intent_budget_and_task_does_not_use_it():
    snap = snapshot([("user", "旧业务背景" * 250), ("assistant", "旧答复"),
                     ("user", "当前问题"), ("assistant", "当前回复")])
    text = "历史订单分别属于两个地点，未经核验。" * 14
    assert 250 < estimate(text) <= 450
    summaries = [{"id": "sum-1", "from_msg_id": 1, "upto_msg_id": 2, "summary": text}]
    p = assemble_task_context(snap, "继续", stage="intent", policy="window_summary", summaries=summaries)
    assert not p["blocked"] and p["manifest"]["summary_refs"] == ["sum-1"]
    assert text in p["summary_text"]
    task = assemble_task_context(snap, "继续", stage="intent", summaries=summaries)
    assert task["summary_text"] == "" and task["manifest"]["summary_refs"] == []


def test_summary_invalid_scope_hash_or_partial_unit_is_rejected():
    snap = snapshot([("user", "旧问"), ("assistant", "旧答"), ("user", "新问"), ("assistant", "新答")])
    for fields in ({"conversation_id": 8}, {"source_hash": "wrong"}, {"upto_msg_id": 1}):
        seg = {"from_msg_id": 1, "upto_msg_id": 2, "summary": "背景", **fields}
        p = assemble_task_context(snap, "继续", policy="window_summary", summaries=[seg])
        assert p["manifest"]["summary_refs"] == []


def test_retrieval_and_source_labels_are_independent_ablations():
    dialog = [("user", "遥远设备由小林保管。"), ("assistant", "收到。")]
    for _ in range(5):
        dialog += [("user", "别的事情。"), ("assistant", "好的。")]
    snap = snapshot(dialog)
    off = assemble_task_context(snap, "小林的设备呢？", retrieval=False)
    assert "小林保管" not in off["history"]
    no_labels = assemble_task_context(snap, "小林的设备呢？", source_labels=False)
    assert "小林保管" in no_labels["history"] and "c7/m" not in no_labels["history"]
    assert no_labels["manifest"]["source_refs"]


def test_scope_validation_and_budget_never_expands_stage_cap():
    snap = snapshot([("user", "你好"), ("assistant", "你好")])
    p = assemble_task_context(snap, "继续", stage="intent", budget=8000)
    assert p["manifest"]["budget"] == 1000
    snap["rows"][0]["conversation_id"] = 8
    with pytest.raises(ValueError, match="conversation mismatch"):
        assemble_task_context(snap, "继续")


def test_full_is_exempt_from_small_stage_caps_but_still_bounded():
    snap = snapshot([("user", "远期背景" * 300), ("assistant", "收到。")])
    p = assemble_task_context(snap, "继续", stage="intent", policy="full")
    assert not p["blocked"] and p["manifest"]["approx_tokens"] > 1000
    assert p["manifest"]["budget"] == 16000
    assert p["manifest"]["full_stage_cap_exemption"]
    assert assemble_task_context(snap, "继续", stage="intent", policy="full", budget=1000)["blocked"]


def test_summary_target_is_not_a_truncation_or_rejection_threshold():
    snap = snapshot([("user", "旧业务背景" * 250), ("assistant", "旧答复"),
                     ("user", "当前问题"), ("assistant", "当前回复")])
    text = "历史背景及关系必须保留。" * 35
    assert estimate(text) > 450
    seg = {"from_msg_id": 1, "upto_msg_id": 2, "summary": text}
    p = assemble_task_context(snap, "继续", policy="window_summary", summaries=[seg])
    assert text in p["summary_text"] and not p["blocked"]


def test_rewritten_query_does_not_turn_current_original_into_history():
    raw = "它现在在哪里？"
    snap = snapshot([("user", "订单710001。"), ("assistant", "收到。"), ("user", raw)])
    p = assemble_task_context(snap, "查询订单710001的当前物流", stage="retrieval")
    assert raw not in p["history"] and 3 not in p["manifest"]["selected_message_ids"]
    assert p["manifest"]["current_message_id"] == 3 and p["manifest"]["current_query_is_rewrite"]
    assert p["messages"][-1].id == "task-current-query"
    assert p["manifest"]["approx_tokens"] >= estimate(p["history"]) + estimate(p["messages"][-1].content)
    with pytest.raises(ValueError, match="final visible user"):
        assemble_task_context(snap, "查询订单710001", current_message_id=1)
