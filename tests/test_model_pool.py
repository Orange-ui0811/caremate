"""No paid calls: exercise admission/streaming/replay through real HTTP adapters."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import sys
import time
import uuid

import httpx
import pytest
from langchain_core.messages import AIMessageChunk

from app.config import settings
from app.core import model_pool as pool
from app.core.model_transport import AsyncPoolTransport, SyncPoolTransport


@pytest.fixture(autouse=True)
def isolated_pool(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "model_pool_db_path", str(tmp_path / "pool.sqlite3"))
    monkeypatch.setattr(settings, "model_pool_enabled", True)
    monkeypatch.setattr(settings, "model_pool_chat_keys", "fake-key-1,fake-key-2")
    monkeypatch.setattr(settings, "chat_base_url", "https://fake.test/v1")
    monkeypatch.setattr(settings, "model_pool_fallback_api_key", "")
    pool.get_store().set_policy(pool.Policy(global_limit=4, key_limit=2, max_queue=128,
                                          queue_timeout=10, call_timeout=10, read_timeout=5))


async def test_80_arrivals_obey_global_and_per_key_limits_without_dropping_requests(tmp_path):
    active = peak = 0
    by_key, key_peaks = {}, {}
    async def upstream(request):
        nonlocal active, peak
        key = request.headers["authorization"]
        active += 1
        peak = max(peak, active)
        by_key[key] = by_key.get(key, 0) + 1
        key_peaks[key] = max(key_peaks.get(key, 0), by_key[key])
        await asyncio.sleep(.02)
        active -= 1
        by_key[key] -= 1
        return httpx.Response(200, json={"ok": True})
    async with httpx.AsyncClient(transport=AsyncPoolTransport(inner=httpx.MockTransport(upstream))) as client:
        responses = await asyncio.gather(*(client.post("https://fake.test/v1/chat/completions",
                                                       json={"model": "same-model", "messages": []}) for _ in range(80)))
    assert len(responses) == 80 and all(r.status_code == 200 for r in responses)
    assert peak <= 4 and max(key_peaks.values()) <= 2 and len(key_peaks) == 2
    snapshot = pool.get_store().snapshot()
    assert not snapshot["running"] and snapshot["waiting"] == 0
    assert all(r["outcome"] == "SUCCEEDED" for r in snapshot["recent_calls"])
    (tmp_path / "load_metrics.json").write_text(json.dumps({"scope": "Mock upstream, not provider capacity",
        "arrivals": 80, "successes": 80, "configured_global_limit": 4, "configured_key_limit": 2,
        "observed_upstream_peak": peak, "observed_per_key_peak": max(key_peaks.values()),
        "remaining_leases": len(snapshot["running"]), "remaining_waiters": snapshot["waiting"]}), encoding="utf-8")


async def test_two_store_instances_claim_atomically_and_old_owner_cannot_release_new_lease(tmp_path):
    path = tmp_path / "shared.sqlite3"
    a, b = pool.Store(path), pool.Store(path)
    policy = pool.Policy(global_limit=2, key_limit=1)
    choices = [pool.Provider("https://fake.test/v1", "secret")]
    def claim(store):
        token = uuid.uuid4().hex
        store.enqueue(token, "foreground", policy)
        value = store.acquire(token, choices, policy, "chat", "m", "foreground", "r", 1, time.time())
        return token, value
    with ThreadPoolExecutor(2) as executor:
        results = list(executor.map(claim, [a, b]))
    assert sum(spec is not None for _, spec in results) == 1
    winner = next(token for token, spec in results if spec)
    loser = next(token for token, spec in results if not spec)
    with a.connect() as c:
        c.execute("UPDATE leases SET expires=0 WHERE id=?", (winner,))
    spec = b.acquire(loser, choices, policy, "chat", "m", "foreground", "r2", 1, time.time())
    assert spec
    a.finish(winner, "CANCELLED")
    assert len(b.snapshot()["running"]) == 1
    b.finish(loser, "SUCCEEDED")


class SlowStream(httpx.AsyncByteStream):
    def __init__(self, gate, fail=False):
        self.gate, self.fail = gate, fail
    async def __aiter__(self):
        yield b'data: {"content":"hello"}\n\n'
        await self.gate.wait()
        if self.fail:
            raise httpx.ReadError("interrupted")
        yield b'data: [DONE]\n\n'
    async def aclose(self):
        pass


async def test_stream_holds_slot_until_close_and_partial_stream_is_not_retried():
    gate, calls = asyncio.Event(), []
    async def upstream(request):
        calls.append(1)
        return httpx.Response(200, stream=SlowStream(gate, fail=True))
    async with httpx.AsyncClient(transport=AsyncPoolTransport(inner=httpx.MockTransport(upstream))) as client:
        async with client.stream("POST", "https://fake.test/v1/chat/completions", json={"model": "m", "stream": True}) as response:
            iterator = response.aiter_bytes()
            await anext(iterator)
            assert len(pool.get_store().snapshot()["running"]) == 1
            gate.set()
            with pytest.raises(httpx.ReadError):
                await anext(iterator)
    assert len(calls) == 1 and not pool.get_store().snapshot()["running"]


async def test_cancel_and_total_stream_timeout_release_capacity():
    store = pool.get_store()
    store.set_policy(pool.Policy(call_timeout=.15, read_timeout=.1, queue_timeout=.1))
    async def upstream(request):
        return httpx.Response(200, stream=SlowStream(asyncio.Event()))
    async with httpx.AsyncClient(transport=AsyncPoolTransport(inner=httpx.MockTransport(upstream))) as client:
        with pytest.raises(httpx.ReadTimeout):
            await client.post("https://fake.test/v1/chat/completions", json={"model": "m"})
    assert not store.snapshot()["running"]
    assert store.snapshot()["recent_calls"][0]["outcome"] == "TIMEOUT"
    async def waiting(request):
        await asyncio.Event().wait()
    async with httpx.AsyncClient(transport=AsyncPoolTransport(inner=httpx.MockTransport(waiting))) as client:
        task = asyncio.create_task(client.post("https://fake.test/v1/chat/completions", json={"model": "m"}))
        for _ in range(100):
            if store.snapshot()["running"]:
                break
            await asyncio.sleep(.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not store.snapshot()["running"] and store.snapshot()["waiting"] == 0


@pytest.mark.parametrize("status", [401, 403, 429, 503])
async def test_bad_key_or_throttled_key_rotates_with_finite_attempts(status):
    calls = []
    async def upstream(request):
        calls.append(request.headers["authorization"])
        return httpx.Response(status if len(calls) == 1 else 200, headers={"retry-after": "60"}, json={"ok": len(calls) > 1})
    async with httpx.AsyncClient(transport=AsyncPoolTransport(inner=httpx.MockTransport(upstream))) as client:
        result = await client.post("https://fake.test/v1/chat/completions", json={"model": "m"})
    assert result.status_code == 200 and len(calls) == 2 and calls[0] != calls[1]
    assert not pool.get_store().snapshot()["running"]
    serialized = json.dumps(pool.get_store().snapshot())
    assert "fake-key" not in serialized and "Bearer" not in serialized


async def test_background_does_not_take_reserved_foreground_capacity():
    store = pool.get_store()
    policy = pool.Policy(global_limit=2, key_limit=2, background_limit=1)
    store.set_policy(policy)
    choices = pool.providers("chat", "m")
    def claim(token, priority):
        store.enqueue(token, priority, policy)
        return store.acquire(token, choices, policy, "chat", "m", priority, token, 1, time.time())
    assert claim("bg1", "background")
    assert claim("bg2", "background") is None
    assert claim("fg", "foreground")
    store.finish("bg1", "SUCCEEDED")
    store.finish("fg", "SUCCEEDED")
    store.dequeue("bg2")


def test_sync_adapter_uses_the_same_capacity_and_records_completion():
    with httpx.Client(transport=SyncPoolTransport(inner=httpx.MockTransport(lambda r: httpx.Response(200, json={"ok": True})))) as client:
        assert client.post("https://fake.test/v1/chat/completions", json={"model": "m"}).status_code == 200
    assert pool.get_store().snapshot()["recent_calls"][0]["outcome"] == "SUCCEEDED"


async def test_shared_model_usage_is_isolated_per_concurrent_task():
    from app.core.llm import get_chat_model
    model = get_chat_model()
    async def stream(prompt):
        result = None
        for i in range(3):
            generation = model._convert_chunk_to_generation_chunk({"choices": [{"index": 0, "delta": {"content": "x"}}],
                "usage": {"prompt_tokens": prompt, "completion_tokens": i, "total_tokens": prompt+i}}, AIMessageChunk, None)
            result = generation.message if result is None else result + generation.message
            await asyncio.sleep(0)
        return result.usage_metadata
    a, b = await asyncio.gather(stream(10), stream(100))
    assert a["input_tokens"] == 10 and b["input_tokens"] == 100
    assert a["output_tokens"] == b["output_tokens"] == 2


async def test_chat_result_replay_payload_conflict_and_user_isolation(monkeypatch):
    from app.main import app
    from app.graph import runtime
    calls = []
    async def fake(user, message, cid):
        calls.append(user)
        yield {"type": "delta", "text": "result"}
        yield {"type": "done", "conversation_id": 7}
    monkeypatch.setattr(runtime, "stream_turn", fake)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        body = {"user_id": "u1", "message": "q", "request_id": "same"}
        first = await client.post("/api/chat", json=body)
        second = await client.post("/api/chat", json=body)
        assert second.text == first.text and second.headers["x-result-replay"] == "true"
        assert calls == ["u1"]
        assert (await client.post("/api/chat", json=body | {"message": "different"})).status_code == 409
        assert (await client.post("/api/chat", json=body | {"user_id": "u2"})).status_code == 200
    assert calls == ["u1", "u2"]


async def test_inflight_duplicate_and_interrupted_turn_are_not_reexecuted(monkeypatch):
    from app.main import app
    from app.graph import runtime
    gate, entered, calls = asyncio.Event(), asyncio.Event(), []
    async def fake(*args):
        calls.append(1)
        entered.set()
        await gate.wait()
        yield {"type": "done", "conversation_id": 7}
    monkeypatch.setattr(runtime, "stream_turn", fake)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        body = {"user_id": "u1", "message": "q", "request_id": "pending"}
        original = asyncio.create_task(client.post("/api/chat", json=body))
        await entered.wait()
        assert (await client.post("/api/chat", json=body)).status_code == 409
        original.cancel()
        with pytest.raises(asyncio.CancelledError):
            await original
        assert (await client.post("/api/chat", json=body)).status_code == 409
    assert len(calls) == 1


async def test_admin_api_persists_policy_and_never_exposes_credentials():
    from app.main import app
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        data = (await client.get("/api/model-pool/runtime")).json()
        assert "fake-key" not in json.dumps(data)
        policy = data["policy"] | {"global_limit": 3}
        assert (await client.put("/api/model-pool/policy", json=policy)).status_code == 200
        assert pool.Store(pool.get_store().path).policy().global_limit == 3
        assert (await client.put("/api/model-pool/policy", json=policy | {"background_limit": 3})).status_code == 422


def test_actual_independent_processes_share_global_and_key_limits(tmp_path):
    path = tmp_path / "processes.sqlite3"
    store = pool.Store(path)
    code = r'''
import sys,time,uuid
from app.core.model_pool import Store,Policy,Provider
s=Store(sys.argv[1]); p=Policy(global_limit=2,key_limit=1,queue_timeout=10,call_timeout=10,read_timeout=5)
choices=[Provider('https://fake.test/v1','a'),Provider('https://fake.test/v1','b')]
for i in range(6):
 t=uuid.uuid4().hex; started=time.time(); s.enqueue(t,'foreground',p)
 while not s.acquire(t,choices,p,'chat','m','foreground',t,1,started):
  if time.time()-started>10: raise RuntimeError('admission timeout')
  time.sleep(.01)
 time.sleep(.04); s.finish(t,'SUCCEEDED')
'''
    children = [subprocess.Popen([sys.executable, "-c", code, str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(3)]
    peak, key_peak = 0, 0
    deadline = time.monotonic() + 25
    try:
        while any(c.poll() is None for c in children):
            running = store.snapshot()["running"]
            peak = max(peak, len(running))
            for key in {r["resource"] for r in running}:
                key_peak = max(key_peak, sum(r["resource"] == key for r in running))
            assert time.monotonic() < deadline
            time.sleep(.01)
        for child in children:
            _, errors = child.communicate()
            assert child.returncode == 0, errors.decode(errors="replace")
        assert 0 < peak <= 2 and key_peak == 1
        assert len(store.snapshot()["recent_calls"]) == 18
        assert not store.snapshot()["running"]
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=5)


async def test_queue_overflow_and_queued_cancellation_do_not_leak_waiters():
    store = pool.get_store()
    policy = pool.Policy(global_limit=1, key_limit=1, background_limit=0, max_queue=1, queue_timeout=.1)
    store.set_policy(policy)
    store.enqueue("a", "foreground", policy)
    with pytest.raises(pool.PoolBusy):
        store.enqueue("b", "foreground", policy)
    store.dequeue("a")
    entered = asyncio.Event()
    async def blocked(request):
        entered.set()
        await asyncio.Event().wait()
    async with httpx.AsyncClient(transport=AsyncPoolTransport(inner=httpx.MockTransport(blocked))) as client:
        one = asyncio.create_task(client.post("https://fake.test/v1/chat/completions", json={"model": "m"}))
        await entered.wait()
        two = asyncio.create_task(client.post("https://fake.test/v1/chat/completions", json={"model": "m"}))
        for _ in range(100):
            if store.snapshot()["waiting"]:
                break
            await asyncio.sleep(.001)
        two.cancel()
        with pytest.raises(asyncio.CancelledError):
            await two
        assert store.snapshot()["waiting"] == 0
        assert len(store.snapshot()["running"]) == 1
        one.cancel()
        with pytest.raises(asyncio.CancelledError):
            await one
    assert not store.snapshot()["running"]


async def test_langchain_function_calling_stream_and_usage_are_preserved(monkeypatch):
    from pydantic import BaseModel
    from app.core import llm, model_transport
    class Probe(BaseModel):
        order_id: str
    async def upstream(request):
        payload = json.loads(request.content)
        assert payload["tools"][0]["function"]["name"] == "Probe"
        chunks = [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [{"index": 0, "id": "call1", "type": "function", "function": {"name": "Probe", "arguments": '{"order_id":"1001"}'}}]}, "finish_reason": None}],
             "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
             "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}},
        ]
        content = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, text=content, headers={"content-type": "text/event-stream"})
    async with httpx.AsyncClient(transport=AsyncPoolTransport(inner=httpx.MockTransport(upstream))) as asynchronous:
        with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as sync:
            monkeypatch.setattr(model_transport, "clients", lambda channel="chat": (sync, asynchronous))
            result = await llm.structured(Probe).ainvoke("order 1001")
            assert result.order_id == "1001"
    assert pool.get_store().snapshot()["recent_calls"][0]["outcome"] == "SUCCEEDED"


async def test_same_conversation_parallel_turns_are_rejected_then_retriable(monkeypatch):
    from app.main import app
    from app.graph import runtime
    entered, gate, calls = asyncio.Event(), asyncio.Event(), []
    async def fake(user, message, cid):
        calls.append(message)
        entered.set()
        await gate.wait()
        yield {"type": "done", "conversation_id": cid}
    monkeypatch.setattr(runtime, "stream_turn", fake)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        a = {"user_id": "u", "message": "a", "conversation_id": 7, "request_id": "a"}
        first = asyncio.create_task(client.post("/api/chat", json=a))
        await entered.wait()
        b = a | {"request_id": "b", "message": "b"}
        assert (await client.post("/api/chat", json=b)).status_code == 409
        gate.set()
        assert (await first).status_code == 200
        assert (await client.post("/api/chat", json=b)).status_code == 200
    assert calls == ["a", "b"]


async def test_fallback_requires_explicit_configuration_and_keeps_custom_models_separate(monkeypatch):
    monkeypatch.setattr(settings, "model_pool_chat_keys", "primary")
    monkeypatch.setattr(settings, "chat_model", "original-model")
    monkeypatch.setattr(settings, "model_pool_fallback_base_url", "https://backup.test/v1")
    monkeypatch.setattr(settings, "model_pool_fallback_api_key", "backup")
    monkeypatch.setattr(settings, "model_pool_fallback_model", "explicit-backup-model")
    assert len(pool.providers("chat", "custom-intent-model")) == 1
    seen = []
    async def upstream(request):
        seen.append((request.url.host, json.loads(request.content)["model"]))
        return httpx.Response(401 if request.url.host == "fake.test" else 200, json={"ok": True})
    async with httpx.AsyncClient(transport=AsyncPoolTransport(inner=httpx.MockTransport(upstream))) as client:
        assert (await client.post("https://fake.test/v1/chat/completions", json={"model": "original-model"})).status_code == 200
    assert seen == [("fake.test", "original-model"), ("backup.test", "explicit-backup-model")]


def test_policy_is_shared_even_when_worker_environment_differs(tmp_path, monkeypatch):
    path = tmp_path / "shared-policy.sqlite3"
    a, b = pool.Store(path), pool.Store(path)
    monkeypatch.setattr(settings, "model_pool_global_limit", 3)
    assert a.policy().global_limit == 3
    monkeypatch.setattr(settings, "model_pool_global_limit", 8)
    assert b.policy().global_limit == 3
