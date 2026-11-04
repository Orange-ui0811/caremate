import json
import time

import httpx
import pytest
from langchain_core.runnables import RunnableLambda

from app.config import settings
from app.core import model_pool as pool, model_transport as transport, intent


def test_terminal_stream_frame_is_detected_across_byte_boundaries():
    tail, done = transport.stream_completion(b"", b'data: {"choices":[{"finish_reas')
    assert not done
    tail, done = transport.stream_completion(tail, b'on":"tool_calls"}]}\n\n')
    assert done
    assert not transport.stream_completion(b"", b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n')[1]


async def test_gzipped_completed_stream_is_not_recorded_as_cancelled():
    import gzip
    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield gzip.compress(b'data: {"choices":[{"finish_reason":"tool_calls"}]}\n\n')
        async def aclose(self):
            pass
    class Store:
        outcome = None
        def finish(self, _token, outcome, _status):
            self.outcome = outcome
    store = Store()
    response = httpx.Response(200, headers={"content-encoding":"gzip"}, stream=Body())
    body = transport.AsyncBody(response, store, "test", time.monotonic()+2)
    iterator = body.__aiter__()
    await anext(iterator)
    await body.aclose()
    await iterator.aclose()
    assert store.outcome == "SUCCEEDED"


async def test_final_auth_error_body_is_preserved_without_fallback(monkeypatch):
    monkeypatch.setattr(settings, "model_pool_fallback_api_key", "")
    async def upstream(_):
        return httpx.Response(403, json={"error":{"code":"channel_auth_failed"}})
    async with httpx.AsyncClient(transport=transport.AsyncPoolTransport(inner=httpx.MockTransport(upstream))) as c:
        # One attempt: the next request will fail fast because this route is open.
        pool.get_store().set_policy(pool.Policy(max_attempts=1))
        r = await c.post("https://primary.test/v1/chat/completions", json={"model":"main"})
        assert r.json()["error"]["code"] == "channel_auth_failed"
    assert not pool.get_store().snapshot()["running"]


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    monkeypatch.setattr(settings, "chat_base_url", "https://primary.test/v1")
    monkeypatch.setattr(settings, "chat_model", "main")
    monkeypatch.setattr(settings, "model_pool_chat_keys", "key-a,key-b")
    monkeypatch.setattr(settings, "model_pool_fallback_base_url", "https://api.deepseek.com/v1")
    monkeypatch.setattr(settings, "model_pool_fallback_api_key", "backup-test")
    monkeypatch.setattr(settings, "model_pool_fallback_model", "deepseek-flash")


@pytest.mark.parametrize("sync", [False, True])
async def test_channel_error_uses_backup_and_preserves_key_and_other_routes(sync):
    seen = []
    def upstream(request):
        body = json.loads(request.content)
        seen.append((request.url.host, body))
        if request.url.host == "primary.test":
            return httpx.Response(403, json={"error": {"code": "channel_auth_failed"}})
        assert body["thinking"] == {"type": "disabled"}
        assert "chat_template_kwargs" not in body
        assert "x-failover-enabled" not in request.headers
        return httpx.Response(200, json={"ok": True})
    payload = {"model": "main", "messages": [], "chat_template_kwargs": {"thinking": False}}
    if sync:
        with httpx.Client(transport=transport.SyncPoolTransport(inner=httpx.MockTransport(upstream))) as c:
            assert c.post("https://primary.test/v1/chat/completions", json=payload).status_code == 200
            assert c.post("https://primary.test/v1/chat/completions", json=payload).status_code == 200
    else:
        async with httpx.AsyncClient(transport=transport.AsyncPoolTransport(inner=httpx.MockTransport(upstream))) as c:
            assert (await c.post("https://primary.test/v1/chat/completions", json=payload)).status_code == 200
            assert (await c.post("https://primary.test/v1/chat/completions", json=payload)).status_code == 200
    assert [s[0] for s in seen] == ["primary.test", "api.deepseek.com", "api.deepseek.com"]
    store = pool.get_store()
    snapshot = store.snapshot()
    assert not any(s["invalid"] for s in snapshot["slots"])
    assert not snapshot["running"] and snapshot["waiting"] == 0
    assert len(snapshot["channel_circuits"]) == 1
    primary = pool.providers("chat", "main")[0]
    store.enqueue("other-model", "foreground", store.policy())
    assert store.acquire("other-model", [pool.Provider(primary.base_url, primary.api_key, "other")],
                         store.policy(), "chat", "other", "foreground", "test", 1, time.time())
    store.finish("other-model", "SUCCEEDED")


async def test_all_routes_blocked_fail_fast_instead_of_queueing(monkeypatch):
    monkeypatch.setattr(settings, "model_pool_fallback_api_key", "")
    store = pool.get_store()
    for spec in pool.providers("chat", "main"):
        store.mark_channel(spec, "chat", "main")
    async with httpx.AsyncClient(transport=transport.AsyncPoolTransport(inner=httpx.MockTransport(
            lambda _: pytest.fail("blocked route reached network")))) as c:
        start = time.monotonic()
        with pytest.raises(pool.PoolBusy):
            await c.post("https://primary.test/v1/chat/completions", json={"model": "main"})
    assert time.monotonic() - start < 1
    assert not store.snapshot()["running"] and store.snapshot()["waiting"] == 0


def test_real_credential_failure_still_disables_key():
    store = pool.get_store()
    p = pool.providers("chat", "main")[0]
    with store.transaction() as c:
        c.execute("INSERT INTO slots(resource) VALUES(?)", (p.fingerprint,))
    response = httpx.Response(401, json={"error": {"code": "invalid_api_key"}})
    transport.mark_failure(store, p, response, "chat", "main")
    assert store.snapshot()["slots"][0]["invalid"] == 1


async def test_unavailable_intent_does_not_claim_user_is_unclear(monkeypatch):
    from app.graph import nodes
    async def broken(_):
        raise httpx.ConnectError("test outage")
    monkeypatch.setattr(intent.llm, "structured", lambda *_a, **_k: RunnableLambda(broken))
    result = await intent.classify("有什么商品")
    assert result["classification_error"] == "ConnectError"
    reply = await nodes.script_reply({"intent":"其他", "trace":{"intent_status":"unavailable"}})
    assert "模型服务暂时不可用" in reply["answer"]
    assert "不太确定您的意思" not in reply["answer"]
    normal = await nodes.script_reply({"intent":"其他"})
    assert "不太确定您的意思" in normal["answer"]
