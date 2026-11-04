"""Offline regression cases discovered in the independent context review.

Only synthetic dialogue and pure state functions are used. No database, model,
network request or real customer record is accessed.
"""
import pytest

from app.core.context_state import build_snapshot, write_constraint


def state_after(message):
    return build_snapshot([
        {"id": 1, "role": "user", "content": "当前处理订单810201。先别退款。"},
        {"id": 2, "role": "assistant", "content": "收到，尚未执行。"},
        {"id": 3, "role": "user", "content": message},
    ], 901)


@pytest.mark.parametrize("message", [
    "不同意退款。", "我不同意退款。", "不允许退款。", "现在不可以退款。",
    "我没有同意退款。", "我尚未同意退款。", "不是不同意退款。",
    "不能允许退款。", "不许继续退款。", "我不想继续退款。",
    "客服说现在可以退款。", "我说的是‘现在可以退款’这句话。",
    "“现在可以退款”。", '"允许退款"。', "`同意退款`。",
    "请复述：现在可以退款。", "是否可以退款？", "现在可以退款吗？",
    "现在可以退款，是这个意思吗？", "现在可以退款只是模板措辞。",
    "同意退款不代表允许执行。", "我已经同意退款。",
    "如果我同意退款就执行。", "可以退款吗，先不要操作。",
    "```\n现在可以退款。\n```", "~~~\n现在可以退款。\n~~~",
    "> 现在可以退款。",
])
def test_non_affirmative_statement_does_not_retract_existing_constraint(message):
    snapshot = state_after(message)
    assert write_constraint(snapshot, "submit_refund", {"order_id": "810201"})
    assert not any(event["kind"] == "restriction_retracted" for event in snapshot["events"])


@pytest.mark.parametrize("message", [
    "现在可以退款。", "我现在同意退款。", "允许退款。", "请继续退款。",
    "解除退款限制。", "撤销退款限制。", "取消禁止退款。",
    "订单810201现在可以退款。", "现在可以退订单810201。",
    "```\n这只是引用背景。\n```\n现在可以退款。",
])
def test_direct_affirmative_release_only_removes_memory_denial(message):
    snapshot = state_after(message)
    assert write_constraint(snapshot, "submit_refund", {"order_id": "810201"}) is None
    assert any(event["kind"] == "restriction_retracted" for event in snapshot["events"])
    # No event grants actual tool permission or replaces the confirmation gate.
    assert not any(event["kind"] in ("authorization", "confirmation") for event in snapshot["events"])


@pytest.mark.parametrize("message", ["我不同意退款。", "不允许退款。", "现在不可以退款。"])
def test_explicit_negative_permission_creates_constraint_without_previous_ban(message):
    snapshot = build_snapshot([
        {"id": 1, "role": "user", "content": "当前处理订单810201。"},
        {"id": 2, "role": "user", "content": message},
    ], 901)
    assert write_constraint(snapshot, "submit_refund", {"order_id": "810201"})
