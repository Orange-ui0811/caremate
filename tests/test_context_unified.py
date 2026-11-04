"""Offline contract checks, including a mocked HTTP run of the actual Workflow."""
from copy import deepcopy
import json
from types import SimpleNamespace

import httpx
import pytest

from app.core.context_state import build_snapshot
from scripts import eval_context_unified as unified


def test_freeze_is_model_free_and_immutable(tmp_path, monkeypatch):
    monkeypatch.setattr(unified.base.llm, "get_chat_model", lambda **kw: pytest.fail("model created while freezing"))
    manifest = unified.prepare(tmp_path)
    assert manifest["replicates"] == 1
    assert manifest["arms"] == list(unified.ARMS)
    assert len(unified.read(tmp_path / "dataset.json")["cases"]) == 12
    assert manifest["config"]["overrides"]["context_fresh_business_evidence"] is True
    assert "chat_api_key" not in (tmp_path / "manifest.json").read_text(encoding="utf-8")
    report = unified.audit(tmp_path)
    assert report["audit_passed"] and not report["complete"]
    assert len(report["missing"]) == 180
    with pytest.raises(ValueError, match="immutable"):
        unified.prepare(tmp_path)
    (tmp_path / "writer.lock").write_text("active", encoding="utf-8")
    with pytest.raises(ValueError, match="writer is active"):
        unified.audit(tmp_path)


async def test_dynamic_tools_follow_turn_not_call_count_and_sources_are_scoped(tmp_path):
    case = {"id": "offline", "initial_history": [{"role": "user", "content": "订单710001属于我。"}], "steps": []}
    fixture = unified.Fixture(case)
    token = unified.base.COORDS.set({"arm": "task", "case_id": "offline", "turn": 2})
    try:
        first, oid = fixture.logistics("SF71000188")
        second, _ = fixture.logistics("SF71000188")
        assert first == second and oid == "710001"
        assert first["observed_version"] == 2 and first["version"] == "物流记录2"
        assert "error" in fixture.logistics("invalid")[0]
        fixture.rows = [{"id": 1, "conversation_id": fixture.cid, **case["initial_history"][0]},
                        {"id": 2, "conversation_id": fixture.cid, "role": "assistant", "content": "已记录。"},
                        {"id": 3, "conversation_id": fixture.cid, "role": "user", "content": "查询订单710001物流"}]
        with pytest.raises(PermissionError):
            await fixture.source(fixture.cid, "foreign", 1, 0, 20)
        with unified.isolated_workflow(fixture, "task", unified.base.Recorder(tmp_path)):
            state = {"conversation_id": fixture.cid, "conversation_memory": build_snapshot(fixture.rows, fixture.cid)}
            packet = unified.base.context_runtime.packet(state, "查询订单710001物流", "coref")
            assert not packet["blocked"] and "710001" in packet["history"]
            assert unified.base.settings.context_fresh_business_evidence
    finally:
        unified.base.COORDS.reset(token)


async def test_actual_workflow_all_five_arms_with_mock_http(tmp_path, monkeypatch):
    impl = unified.base
    monkeypatch.setattr(impl.settings, "model_pool_enabled", False)
    monkeypatch.setattr(impl.settings, "chat_base_url", "https://ai.gitee.com/v1")
    monkeypatch.setattr(impl.settings, "chat_model", "fake-offline-only")
    case = {"id": "offline-live-path", "family": "fixture", "stratum": "short",
            "initial_history": [{"role": "user", "content": "我有订单710001，稍后查。"},
                                {"role": "assistant", "content": "请在需要时明确查询。"}],
            "steps": [{"query": "请重新查询订单710001最新物流。",
                       "expected": {"kind": "logistics", "target": "710001"}} for _ in range(3)]}
    monkeypatch.setattr(unified, "dataset", lambda pilot=False: {"version": "offline", "cases": [case], "pilot": True})
    requests = []

    def reply(request):
        payload = json.loads(request.content)
        requests.append(payload)
        purpose, turn = impl.PURPOSE.get(), impl.COORDS.get()["turn"]
        event = unified.cases_module().logistics_event(turn)
        if isinstance(event, dict):
            event = event.get("last_event") or event.get("status") or str(event)
        text = ("请查询订单710001最新物流。" if purpose == "coreference" else
                f"订单710001，物流记录{turn}：{event}。")
        message = {"role": "assistant", "content": text}
        arguments = {"intent_classification": {"intent": "物流", "confidence": 1.0},
                     "conversation_summary": {"summary": "用户有订单710001，尚未核实当前物流。"}}.get(purpose)
        if arguments:
            message = {"role": "assistant", "content": "", "tool_calls": [{"id": "offline-call", "type": "function",
                "function": {"name": payload["tools"][0]["function"]["name"],
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
        unified.prepare(tmp_path, pilot=True)
        report = await unified.run(tmp_path)
    assert report["complete"] and report["audit_passed"], report["issues"]
    for arm in unified.ARMS:
        assert report["groups"][arm]["task_rule_pass"] == 3
        assert report["groups"][arm]["unknown_usage_calls"] == 0
        turn = unified.read(tmp_path / "turns" / f"{arm}-offline-live-path-3.json")
        logistics = [t for t in turn["executed_tools"] if t["name"] == "query_logistics"]
        assert logistics and logistics[0]["args"] == {"tracking_no": "SF71000188"}
        assert logistics[0]["result"]["observed_version"] == 3
        assert "物流记录1" in turn["actual_dialog_rows"][3]["content"]
    assert sum(g["model_calls"] for g in report["groups"].values()) == len(requests)
    with pytest.raises(ValueError, match="existing live output"):
        await unified.run(tmp_path)
