"""No live model calls: verify explicit length bounds and failure admission."""
from types import SimpleNamespace

from langchain_core.runnables import RunnableLambda
import pytest

from app.core.context_policy import estimate
from scripts import context_unified_summary as summary


def plan(start=1, end=2, digest="source-a"):
    return {"from_msg_id": start, "upto_msg_id": end, "source_hash": digest}


def test_explicit_character_limit_keeps_original_or_rejects():
    accepted = "中" * 300
    assert summary.validate_summary(" " + accepted + " ") == accepted
    assert estimate(accepted) <= summary.MAX_ESTIMATED_TOKENS
    with pytest.raises(ValueError, match="300 characters"):
        summary.validate_summary("中" * 301)
    with pytest.raises(ValueError, match="empty"):
        summary.validate_summary(" \n ")


async def test_overlong_response_uses_one_model_invocation_without_repair(monkeypatch):
    seen = []
    async def respond(value):
        seen.append(value.to_string())
        return SimpleNamespace(summary="中" * 301)
    monkeypatch.setattr(summary.llm, "structured", lambda *args, **kwargs: RunnableLambda(respond))
    with pytest.raises(ValueError, match="300 characters"):
        await summary.summarize_source_segment("用户：订单710001需要查询。")
    assert len(seen) == 1 and "最多300个字符" in seen[0]


async def test_bounded_response_preserves_relations_and_negation_verbatim(monkeypatch):
    text = "用户称家里对应710001；更正为710002。只查物流，不提交退款。历史未查询。"
    monkeypatch.setattr(summary.llm, "structured", lambda *args, **kwargs:
                        RunnableLambda(lambda value: SimpleNamespace(summary=text)))
    assert await summary.summarize_source_segment("synthetic history") == text


def test_failed_prefix_does_not_count_already_visible_tail_as_new_material():
    gate = summary.SummaryCooldown()
    rows = [{"id": i, "content": "旧背景" * 150} for i in range(1, 5)]
    gate.begin(plan(end=2), rows)
    gate.failed(rows)
    result = gate.decision(plan(end=3, digest="source-b"), rows)
    assert result["reason"] == "failure_cooldown"
    assert result["new_source_estimated_tokens"] == 0
    assert result["last_failure_watermark"] == 4
    rows.append({"id": 5, "content": "新背景"})
    assert not gate.decision(plan(end=3, digest="source-b"), rows)["allowed"]


def test_cooldown_releases_only_new_range_after_threshold_then_success_resets():
    gate = summary.SummaryCooldown()
    rows = [SimpleNamespace(id=1, content="旧用户消息"), SimpleNamespace(id=2, content="旧回复")]
    gate.begin(plan(), rows)
    gate.failed(rows)
    rows.append(SimpleNamespace(id=3, content="新" * 2000))
    assert gate.decision(plan(), rows)["reason"] == "same_range_already_attempted"
    expanded = plan(end=3, digest="source-b")
    assert gate.begin(expanded, rows)["new_source_estimated_tokens"] >= 2400
    gate.succeeded()
    assert gate.last_failure_watermark is None
    assert not gate.decision(expanded, rows)["allowed"]
    assert gate.decision(plan(start=4, end=5, digest="source-c"), rows)["allowed"]


def test_no_plan_never_creates_an_attempt():
    gate = summary.SummaryCooldown()
    assert gate.decision(None, []) == {
        "allowed": False, "reason": "no_summary_plan", "visible_watermark": 0,
        "last_failure_watermark": None, "new_source_estimated_tokens": 0,
        "required_new_source_estimated_tokens": 2400, "range": None}
    assert not gate.attempted_ranges


async def test_workflow_seed_failure_is_recorded_once_then_cooled_down(tmp_path, monkeypatch):
    from scripts import eval_context_unified as runner
    monkeypatch.setattr(runner.base.settings, "model_pool_enabled", False)
    seen = []
    async def respond(value):
        seen.append(value.to_string())
        return SimpleNamespace(summary="超" * 301)
    monkeypatch.setattr(summary.llm, "structured", lambda *args, **kwargs: RunnableLambda(respond))
    history = [row for _ in range(24) for row in (
        {"role": "user", "content": "旧" * 100}, {"role": "assistant", "content": "已记录。"})]
    fixture = runner.Fixture({"id": "offline-summary-failure", "initial_history": history, "steps": []})
    recorder = runner.base.Recorder(tmp_path)
    original = runner.base.summarizer.summarize_source_segment
    with runner.isolated_workflow(fixture, "window_summary", recorder) as (_, maintenance):
        await runner.seed_fixture(fixture, "window_summary", recorder)
        assert len(seen) == 1 and not fixture.segments
        assert sum(m["triggered"] for m in maintenance) == 1
        assert any(m["summary_admission"]["reason"] == "failure_cooldown" for m in maintenance)
        assert any(m["error_type"] == "ValueError" for m in maintenance)
        await fixture.append(fixture.cid, "user", "新" * 2000)
        await fixture.append(fixture.cid, "assistant", "已记录。")
        await runner.base.summarizer.maybe_schedule_summary(fixture.cid)
        assert len(seen) == 2
        await runner.base.summarizer.close_tasks()
    assert runner.base.summarizer.summarize_source_segment is original
