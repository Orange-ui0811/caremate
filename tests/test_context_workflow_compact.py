"""Offline contract and actual-SDK safeguards for the compact workflow trial."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest

from app.core.context_policy import estimate
from app.core.context_state import build_snapshot
from scripts import verify_context_workflow as original
from scripts import verify_context_workflow_compact as compact


def test_dataset_preserves_old_steps_and_declares_new_pressure():
    data = compact.dataset()
    assert len(data["cases"]) == 9
    by_id = {case["id"]: case for case in data["cases"]}
    for case in original.dataset()["cases"]:
        assert by_id[case["id"]]["steps"] == case["steps"]
        assert by_id[case["id"]]["initial_history"] == []
    for case in data["cases"][6:]:
        assert 6500 <= sum(estimate(r["content"]) for r in case["initial_history"]) < 7000
        assert len(case["steps"]) == 3
        assert [r["role"] for r in case["initial_history"]] == ["user", "assistant"] * (len(case["initial_history"]) // 2)


def test_prepare_is_frozen_private_and_model_free(tmp_path, monkeypatch):
    monkeypatch.setattr(compact._impl.llm, "get_chat_model", lambda **kwargs: pytest.fail("model created during prepare"))
    original_fixture, original_isolated = original.Fixture, original.isolated_workflow
    manifest = compact.prepare(tmp_path, case_ids=["pressure_order_correction"])
    assert manifest["arms"] == ["source", "compact_source"]
    assert manifest["compact_caps"]["coref"] == 1800
    assert original.Fixture is original_fixture and original.isolated_workflow is original_isolated
    stored = compact._impl.read(tmp_path / "dataset.json")
    planning = stored["cases"][0]["predeclared_summary_check"]
    assert planning["with_three_queries_and_1200_tokens_per_answer"] < planning["trigger"]
    assert "chat_api_key" not in (tmp_path / "manifest.json").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="immutable"):
        compact.prepare(tmp_path)
    report = compact.audit(tmp_path)
    assert report["audit_passed"] and not report["complete"] and len(report["missing"]) == 6


def test_audit_rejects_active_writer(tmp_path):
    compact.prepare(tmp_path, case_ids=["product_update"])
    (tmp_path / "writer.lock").write_text("active", encoding="utf-8")
    with pytest.raises(ValueError, match="writer is active"):
        compact.audit(tmp_path)


async def test_seeded_fixture_scopes_sources_and_summary_store(tmp_path):
    case = compact.dataset()["cases"][6]
    fixture = compact.Fixture(case)
    assert len(fixture.rows) == len(case["initial_history"])
    original_packet = compact._impl.context_runtime.packet
    with compact.isolated_workflow(fixture, "compact_source", compact._impl.Recorder(tmp_path)):
        assert compact._impl.settings.context_policy == "source"
        state = {"conversation_id": fixture.cid,
                 "conversation_memory": build_snapshot(fixture.rows, fixture.cid), "memory_segments": []}
        packet = compact._impl.context_runtime.packet(state, "查询订单710002的物流", "coref")
        assert packet["manifest"]["stage_budget"] == 1800
        assert not packet["blocked"]
        assert "710002" in packet["history"]
        denied = compact._impl.context_runtime.packet(state, "查询订单710002的物流", "answer", budget=0)
        assert denied["blocked"]
        with pytest.raises(PermissionError):
            await compact._impl.context_repository.read_source(fixture.cid, "foreign", 1, 0, 100)
        lease = await compact._impl.context_repository.acquire_summary_lease(fixture.cid, fixture.uid)
        assert lease and await fixture.acquire_summary_lease(fixture.cid, fixture.uid) is None
        rows = fixture.rows[:2]
        row_hash = compact._impl.context_repository.source_digest([SimpleNamespace(**r) for r in rows])
        stored = await compact._impl.context_repository.put_summary_segment(fixture.cid, fixture.uid,
            from_msg_id=1, upto_msg_id=2, summary="用户提及订单710001，尚未查询。",
            source_hash=row_hash, model="fake", prompt_version="offline")
        assert stored["source_refs"][0]["content_hash"] == hashlib.sha256(rows[0]["content"].encode()).hexdigest()
        assert await fixture.list_segments(fixture.cid, fixture.uid, upto_msg_id=1) == []
        assert len(await fixture.list_segments(fixture.cid, fixture.uid, upto_msg_id=2)) == 1
        await fixture.release_summary_lease(fixture.cid, lease)
        assert fixture.lease is None
    assert compact._impl.context_runtime.packet is original_packet


async def test_mock_http_runs_same_real_workflow_for_both_arms(tmp_path, monkeypatch):
    impl = compact._impl
    monkeypatch.setattr(impl.settings, "model_pool_enabled", False)
    monkeypatch.setattr(impl.settings, "chat_base_url", "https://ai.gitee.com/v1")
    monkeypatch.setattr(impl.settings, "chat_model", "fake-offline-only")
    requests = []
    def reply(request):
        payload = json.loads(request.content)
        requests.append(payload)
        purpose = impl.PURPOSE.get()
        turn = impl.COORDS.get()["turn"]
        sku = "MH-W40" if turn == 1 else "MH-FD10"
        message = {"role": "assistant", "content": sku + " 请断电后清洁。"}
        arguments = {"intent_classification": {"intent": "商品咨询", "confidence": 1.0},
                     "query_rewrite": {"standard": sku + "清洁方法", "expanded": []},
                     "evidence_check": {"useful": True, "reason": "合成夹具足够"}}.get(purpose)
        if arguments:
            message = {"role": "assistant", "content": "", "tool_calls": [{
                "id": "offline-call", "type": "function", "function": {
                    "name": payload["tools"][0]["function"]["name"],
                    "arguments": json.dumps(arguments, ensure_ascii=False)}}]}
        usage = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        result = {"id": "offline-completion", "object": "chat.completion", "created": 1,
                  "model": "fake-offline-only", "choices": [{"index": 0, "message": message,
                  "finish_reason": "tool_calls" if arguments else "stop"}], "usage": usage}
        if not payload.get("stream"):
            return httpx.Response(200, json=result)
        delta = deepcopy(message)
        if arguments:
            delta["tool_calls"][0]["index"] = 0
        chunk = {"id": "offline-completion", "object": "chat.completion.chunk", "created": 1,
                 "model": "fake-offline-only", "choices": [{"index": 0, "delta": delta,
                 "finish_reason": "tool_calls" if arguments else "stop"}], "usage": usage}
        return httpx.Response(200, content=("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\ndata: [DONE]\n\n").encode(),
                              headers={"content-type": "text/event-stream"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        def factory(**kwargs):
            return impl.llm._CumulativeUsageChatOpenAI(model="fake-offline-only", base_url="https://ai.gitee.com/v1",
                api_key="offline-placeholder", http_async_client=client, max_retries=0,
                streaming=kwargs.get("streaming", False), stream_usage=True)
        monkeypatch.setattr(impl.llm, "get_chat_model", factory)
        compact.prepare(tmp_path, case_ids=["product_update"])
        report = await compact.run(tmp_path)
    assert report["complete"] and report["audit_passed"], report["issues"]
    for arm in compact.ARMS:
        assert report["groups"][arm]["task_rule_pass"] == 3
        assert report["groups"][arm]["maintenance_model_calls"] == 0
    calls = [impl.read(p) for p in (tmp_path / "calls").glob("*.json")]
    assert len(calls) == len(requests) == 30
    assert all(call["status"] == "ok" and call["usage"]["input_tokens"] == 100 for call in calls)
    third = impl.read(tmp_path / "turns" / "compact_source-product_update-3.json")
    assert third["actual_dialog_rows"][3]["content"] == "MH-FD10 请断电后清洁。"
    projections = impl.read(tmp_path / "projections" / "compact_source-product_update.json")
    assert {r["manifest"].get("stage_budget") for r in projections} == {1000, 1800, 4200}
