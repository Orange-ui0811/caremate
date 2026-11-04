"""Offline safeguards for the real-model workflow verification harness."""
from copy import deepcopy
import json

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from scripts import verify_context_workflow as workflow


def test_prepare_never_calls_model_and_empty_audit_is_incomplete(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("prepare must not construct or call a model")
    monkeypatch.setattr(workflow.llm, "get_chat_model", forbidden)
    manifest = workflow.prepare(tmp_path / "run", arms=["source"], case_ids=["order_update"])
    assert manifest["dataset_sha256"]
    assert len(workflow.read(tmp_path / "run" / "dataset.json")["cases"]) == 1
    audit = workflow.audit(tmp_path / "run")
    assert audit["audit_passed"] and not audit["complete"] and len(audit["missing"]) == 3
    text = (tmp_path / "run" / "manifest.json").read_text(encoding="utf-8")
    assert "chat_api_key" not in text and "embed_api_key" not in text
    with pytest.raises(ValueError, match="immutable"):
        workflow.prepare(tmp_path / "run")


def test_score_uses_tool_arguments_not_mentioning_an_old_id():
    record = {"expected": {"kind": "logistics", "target": "710002"}, "status": "completed",
              "answer": "已查询710002，没有查询旧订单710001。", "proposed_tools": [],
              "executed_tools": [{"name": "query_logistics", "args": {"order_id": "710002"}}]}
    assert workflow.score_turn(record)["task_rule_pass"]
    wrong = deepcopy(record)
    wrong["executed_tools"][0]["args"]["order_id"] = "1001"
    assert not workflow.score_turn(wrong)["task_rule_pass"]
    write = deepcopy(record)
    write["proposed_tools"] = [{"name": "create_ticket", "args": {"description": "test"}}]
    assert not workflow.score_turn(write)["checks"]["no_write_proposal_or_execution"]


async def test_callback_captures_usage_and_safe_error_type(tmp_path):
    recorder = workflow.Recorder(tmp_path)
    token = workflow.COORDS.set({"arm": "source", "case_id": "test", "turn": 1})
    try:
        await recorder.on_chat_model_start({}, [[HumanMessage("合成测试问题")]], run_id="one",
                                           invocation_params={"model": "fake", "api_key": "must-not-store"})
        recorder.record_wire_payload({"model": "fake", "messages": [{"role": "user", "content": "合成测试问题"}],
                                      "api_key": "must-not-store"})
        response = LLMResult(generations=[[ChatGeneration(message=AIMessage("合成回答", usage_metadata={
            "input_tokens": 9, "output_tokens": 2, "total_tokens": 11}))]])
        await recorder.on_llm_end(response, run_id="one")
        row = workflow.read(tmp_path / "calls" / "one.json")
        assert row["usage"]["total_tokens"] == 11 and row["status"] == "ok"
        assert "must-not-store" not in json.dumps(row)
        await recorder.on_chat_model_start({}, [[HumanMessage("另一个合成问题")]], run_id="two")
        await recorder.on_llm_error(RuntimeError("must-not-store"), run_id="two")
        error = workflow.read(tmp_path / "calls" / "two.json")
        assert error["status"] == "runtime_error" and error["error_type"] == "RuntimeError"
        assert "must-not-store" not in json.dumps(error)
    finally:
        workflow.COORDS.reset(token)


async def test_fixture_is_scope_checked_and_real_repositories_are_restored(tmp_path):
    fixture = workflow.Fixture()
    original = workflow.repository.append_message
    with workflow.isolated_workflow(fixture, "source", workflow.Recorder(tmp_path)):
        assert workflow.repository.append_message != original
        mid = await workflow.repository.append_message(fixture.cid, "user", content="纯合成消息")
        assert mid == 1 and fixture.rows[0]["content"] == "纯合成消息"
        recovered = await workflow.context_repository.read_source(fixture.cid, fixture.uid, mid, 0, 100)
        assert recovered["text"] == "纯合成消息"
        with pytest.raises(PermissionError):
            await workflow.context_repository.read_source(fixture.cid, "other-user", mid, 0, 100)
        with pytest.raises(AssertionError, match="live database"):
            workflow.db.async_session()
    assert workflow.repository.append_message == original


def test_audit_detects_frozen_dataset_mutation(tmp_path):
    workflow.prepare(tmp_path, arms=["legacy"], case_ids=["product_update"])
    changed = workflow.read(tmp_path / "dataset.json")
    changed["cases"][0]["steps"][0]["expected"]["target"] = "CHANGED"
    workflow.write(tmp_path / "dataset.json", changed)
    report = workflow.audit(tmp_path)
    assert not report["audit_passed"] and "dataset hash mismatch" in report["issues"]


async def test_mock_http_drives_real_graph_and_records_all_stages(tmp_path, monkeypatch):
    """No external network: actual SDK, callbacks, parsers, graph and SSE are used."""
    monkeypatch.setattr(workflow.settings, "model_pool_enabled", False)
    monkeypatch.setattr(workflow.settings, "chat_base_url", "https://ai.gitee.com/v1")
    monkeypatch.setattr(workflow.settings, "chat_model", "fake-offline-only")
    requests = []

    def reply(request):
        payload = json.loads(request.content)
        requests.append(payload)
        purpose = workflow.PURPOSE.get()
        turn = workflow.COORDS.get()["turn"]
        sku = "MH-W40" if turn == 1 else "MH-FD10"
        message = {"role": "assistant", "content": sku + " 请断电后清洁。"}
        arguments = {"intent_classification": {"intent": "商品咨询", "confidence": 1.0},
                     "query_rewrite": {"standard": sku + "清洁方法", "expanded": []},
                     "evidence_check": {"useful": True, "reason": "夹具足够"}}.get(purpose)
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
        body = "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\ndata: [DONE]\n\n"
        return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(reply)
    async with httpx.AsyncClient(transport=transport) as client:
        def factory(**kwargs):
            return workflow.llm._CumulativeUsageChatOpenAI(
                model="fake-offline-only", base_url="https://ai.gitee.com/v1", api_key="offline-placeholder",
                http_async_client=client, max_retries=0, streaming=kwargs.get("streaming", False), stream_usage=True)
        monkeypatch.setattr(workflow.llm, "get_chat_model", factory)
        out = tmp_path / "mock-http"
        workflow.prepare(out, arms=["legacy", "source"], case_ids=["product_update"])
        report = await workflow.run(out)
    assert report["complete"] and report["audit_passed"]
    assert report["groups"]["source"]["task_rule_pass"] == 3
    assert report["groups"]["legacy"]["task_rule_pass"] == 3
    calls = [workflow.read(p) for p in (out / "calls").glob("*.json")]
    assert {c["purpose"] for c in calls} == {"coreference", "intent_classification", "query_rewrite", "evidence_check", "answer_generation"}
    assert len(calls) == len(requests) == 30
    assert all(c["status"] == "ok" and c["usage"]["input_tokens"] == 100 and c.get("wire_payload") for c in calls)
    third = workflow.read(out / "turns" / "source-product_update-3.json")
    assert len(third["actual_dialog_rows"]) == 6
    assert third["actual_dialog_rows"][3]["content"] == "MH-FD10 请断电后清洁。"
    assert third["first_visible_sse_delta_seconds"] is not None
