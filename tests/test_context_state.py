"""Behavioral checks for replay, update scope, provenance and bounded projections."""
from copy import deepcopy
import hashlib
import json

import pytest

from app.core.context_state import assemble_context, build_snapshot, resolve_source, write_constraint
from app.core.context_policy import estimate


def rows(*texts, start=101):
    result = []
    for text in texts:
        result += [{"id": start + len(result), "role": "user", "content": text},
                   {"id": start + len(result) + 1, "role": "assistant", "content": "已收到。"}]
    return result


def test_stable_db_source_survives_roundtrip_and_detects_changes():
    data = rows("当前处理订单610371。先别提交退款。")
    state = json.loads(json.dumps(build_snapshot(data, 7), ensure_ascii=False))
    ref = state["candidate"]["source_ref"]
    source = resolve_source(state, ref)
    assert source["message_id"] == 101 and source["content_hash"] == hashlib.sha256(data[0]["content"].encode()).hexdigest()
    assert data[0]["content"][source["start"]:source["end"]] == source["text"]
    assert ref.startswith("c7/m101:")
    state["rows"][0]["content"] = "changed"
    with pytest.raises(ValueError, match="changed"):
        resolve_source(state, ref)


def test_incremental_replay_is_identical_and_history_is_immutable():
    data = rows("当前处理订单610371。", "610371报错，应为610370。")
    before = build_snapshot(data[:2], 3)
    replay = build_snapshot(data, 3, previous=before)
    assert replay == build_snapshot(data, 3)
    assert replay["candidate"]["value"] == "610370"
    assert before["candidate"]["value"] == "610371"
    changed = deepcopy(data)
    changed[0]["content"] += "改动"
    with pytest.raises(ValueError, match="immutable"):
        build_snapshot(changed, 3, previous=before)
    with pytest.raises(ValueError, match="scope"):
        build_snapshot(data, 4, previous=before)


def test_restrictions_survive_other_tasks_and_are_not_last_two_only():
    data = rows("当前处理订单610371。先别提交退款。", "现在处理订单610372。不要创建工单。",
                "现在处理订单610373。不要修改地址。", "现在处理订单610374。不要提交退款。")
    state = build_snapshot(data, 3)
    assert len(state["active_constraints"]) == 4
    assert write_constraint(state, "submit_refund", {"order_id": "610371"})
    assert write_constraint(state, "submit_refund", {"order_id": "610372"}) is None
    assert write_constraint(state, "query_order", {"order_id": "610371"}) is None
    packet = assemble_context(state, "查订单610371的退款申请。", stage="business")
    assert "先别提交退款" in packet["quote_text"]
    assert "不要修改地址" not in packet["quote_text"]


def test_correction_moves_same_task_restriction_without_global_alias():
    state = build_snapshot(rows("当前处理订单610371。先别提交退款。", "610371报错，应为610370。"), 3)
    assert write_constraint(state, "submit_refund", {"order_id": "610370"})
    assert state["active_constraints"][0]["original_entity"] == "order:610371"
    assert state["active_constraints"][0]["updated_by"] == state["candidate"]["source_ref"]


@pytest.mark.parametrize("user_text", [
    "回到订单710005，继续查物流，仍然不要创建工单。",
    "回到订单710005，只查询物流，不要创建工单。",
])
def test_return_to_explicit_order_binds_new_restriction_without_moving_old_tasks(user_text):
    data = rows("现在处理订单710005。先别提交退款，只查一下物流。",
                "换一笔，这次只查订单710006的物流，不要提交任何申请。", user_text)
    before = build_snapshot(data[:4], 7)
    after = build_snapshot(data, 7, previous=before)
    assert after["policy_version"] == "source-state-v3"
    assert after["candidate"]["value"] == "710005"
    assert before["candidate"]["value"] == "710006"
    assert {(c["entity"], c["action"]) for c in after["active_constraints"]} == {
        ("order:710005", "refund"), ("order:710006", "write"), ("order:710005", "ticket")}
    for original in before["active_constraints"]:
        current = next(c for c in after["active_constraints"] if c["event_id"] == original["event_id"])
        assert current == original
    latest = next(c for c in after["active_constraints"] if c["action"] == "ticket")
    assert latest["task_id"] == after["candidate"]["task_id"]
    assert resolve_source(after, latest["source_ref"])["text"] == user_text
    assert after == build_snapshot(data, 7)


@pytest.mark.parametrize("user_text", [
    "不要回到订单710005。", "现在不要回到订单710005。",
    "如果回到订单710005，不要创建工单。", "假设回到订单710005。",
    "客服说“回到订单710005”。",
])
def test_negative_hypothetical_or_quoted_return_does_not_select_order(user_text):
    state = build_snapshot(rows("现在处理订单710006。不要提交退款。", user_text), 7)
    assert state["candidate"]["value"] == "710006"
    assert {(c["entity"], c["action"]) for c in state["active_constraints"]} == {("order:710006", "refund")}


def test_release_is_object_scoped_and_never_grants_authorization():
    data = rows("先别退订单A。", "先别退订单B。", "现在可以退订单A。")
    state = build_snapshot(data, 8)
    assert write_constraint(state, "submit_refund", {"order_id": "A"}) is None
    assert write_constraint(state, "submit_refund", {"order_id": "B"})
    assert any(e["kind"] == "restriction_retracted" for e in state["events"])
    assert "confirmed" not in json.dumps(state)


def test_two_objects_in_one_sentence_keep_distinct_restriction_scope():
    state = build_snapshot(rows("不要退订单A，先查订单B。"), 8)
    assert state["candidate"]["value"] == "B"
    assert write_constraint(state, "submit_refund", {"order_id": "A"})
    assert write_constraint(state, "submit_refund", {"order_id": "B"}) is None


def test_polite_question_does_not_erase_an_explicit_stop_request():
    state = build_snapshot(rows("当前处理订单610371。", "先别提交退款，好吗？"), 8)
    assert write_constraint(state, "submit_refund", {"order_id": "610371"})


def test_hypotheticals_and_assistant_claims_do_not_update_state():
    data = rows("当前处理订单610371。先别提交退款。", "如果610371报错，应为610370。", "是否可以取消退款限制？")
    data[-1]["content"] = "现在处理订单999999。现在可以退款。退款已经批准。"
    state = build_snapshot(data, 3)
    assert state["candidate"]["value"] == "610371"
    assert write_constraint(state, "submit_refund", {"order_id": "610371"})
    packet = assemble_context(state, "该订单怎么样？", recall="always")
    assert "历史客服表述，未作业务验证" in packet["quote_text"]


def test_unresolved_and_product_candidates_have_source_identity():
    state = build_snapshot(rows("当前处理订单610371。", "目前还没有确定选哪笔订单，别默认。"), 3)
    assert state["candidate"]["value"] is None
    packet = assemble_context(state, "查那单。")
    assert "未确定，需澄清" in packet["state_text"]
    product = build_snapshot(rows("我买的是 MH-CAM1。"), 3)
    assert product["candidate"]["entity"] == "product:MH-CAM1"


def test_all_original_sentences_are_searchable_and_recall_can_be_disabled():
    data = rows("最早买的商品是 MH-CAM1。放在办公室。", *("只是普通包装背景。" * 30 for _ in range(20)))
    state = build_snapshot(data, 9)
    # The relation sentence has no identifier, but remains indexed.
    relation = next(s for s in state["sources"] if s["text"] == "放在办公室。")
    assert resolve_source(state, relation["ref"])["text"] == "放在办公室。"
    off = assemble_context(state, "最早放办公室的那款是什么？", budget=1000, recall="off")
    on = assemble_context(state, "最早放办公室的那款是什么？", budget=1000, recall="conditional")
    assert not off["manifest"]["recall_triggered"] and on["manifest"]["recall_triggered"]
    assert "MH-CAM1" in on["quote_text"]
    assert on["manifest"]["approx_tokens"] <= 1000


def test_stage_views_share_snapshot_and_do_not_duplicate_current_message():
    data = rows(*(f"普通背景{i}。" for i in range(8)))
    data.append({"id": 200, "role": "user", "content": "现在查询订单610371。"})
    state = build_snapshot(data, 1)
    coref = assemble_context(state, data[-1]["content"], stage="coref")
    intent = assemble_context(state, data[-1]["content"], stage="intent")
    assert coref["manifest"]["snapshot_version"] == intent["manifest"]["snapshot_version"] == 200
    assert len(intent["manifest"]["selected_message_ids"]) < len(coref["manifest"]["selected_message_ids"])
    assert sum(m.content == data[-1]["content"] for m in coref["messages"]) == 1
    assert data[-1]["content"] not in coref["history"]


def test_essential_evidence_cannot_be_silently_truncated():
    state = build_snapshot(rows("当前处理订单610371。先别提交退款。"), 1)
    blocked = assemble_context(state, "退款。", budget=30)
    assert blocked["blocked"] and not blocked["messages"]
    assert blocked["manifest"]["protected_refs"]
    data = rows("当前处理订单610371。" + "先别提交退款" + "长句" * 2000)
    long = assemble_context(build_snapshot(data, 1), "处理退款。", budget=1000)
    assert long["blocked"]


def test_summary_compression_keeps_recovery_and_rejects_future_or_foreign_segments():
    data = rows(*(f"第{i}轮背景。" + "背景描述。" * 60 for i in range(12)))
    state = build_snapshot(data, 3)
    segment = {"id": 7, "conversation_id": 3, "from_msg_id": data[0]["id"],
               "upto_msg_id": data[15]["id"], "summary": "前八轮讨论包装背景。",
               "source_refs": [{"message_id": data[0]["id"], "content_hash": hashlib.sha256(data[0]["content"].encode()).hexdigest()}]}
    packet = assemble_context(state, "后来还有什么？", budget=1200, summaries=[segment])
    assert packet["manifest"]["summary_refs"] == ["7"]
    assert packet["manifest"]["summary_covered_ids"] == list(range(101, 117))
    assert resolve_source(state, state["sources"][0]["ref"])["text"] == "第0轮背景。"
    foreign = {**segment, "conversation_id": 8}
    future = {**segment, "upto_msg_id": 9999}
    bad_hash = {**segment, "source_refs": [{"message_id": 101, "content_hash": "wrong"}]}
    for bad in (foreign, future, bad_hash):
        result = assemble_context(state, "后来还有什么？", budget=1200, summaries=[bad])
        assert not result["manifest"]["summary_refs"]


def test_budget_counts_every_packet_component():
    state = build_snapshot(rows(*(f"当前处理订单{610370+i}。" for i in range(20))), 1)
    for stage in ("coref", "intent", "retrieval", "business", "answer"):
        packet = assemble_context(state, "最早的订单是什么？", stage=stage, budget=1600, recall="always")
        assert not packet["blocked"]
        assert sum(estimate(m.content) for m in packet["messages"]) == packet["manifest"]["approx_tokens"] <= 1600
        for ref in packet["manifest"]["selected_refs"]:
            resolve_source(state, ref)
