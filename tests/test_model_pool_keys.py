"""Credential lifecycle: persistence, routing, redaction and local admin boundaries."""
import json

from fastapi import FastAPI
import httpx
import pytest

from app.api.model_pool import router
from app.config import settings
from app.core import model_pool as pool


@pytest.fixture(autouse=True)
def config(monkeypatch):
    for channel in ("chat", "embed", "rerank"):
        monkeypatch.setattr(settings, channel + "_base_url", "https://fake.test/v1")
        monkeypatch.setattr(settings, channel + "_api_key", "environment-key")
    monkeypatch.setattr(settings, "model_pool_chat_keys", "")
    monkeypatch.setattr(settings, "model_pool_fallback_api_key", "")


def test_keys_encrypted_and_visible_in_new_store_and_all_channels():
    store = pool.get_store()
    secret = "new-secret-never-return-this"
    store.set_credentials([{"api_key": secret, "channels": ["chat", "embed", "rerank"]}])
    other = pool.Store(store.path)
    for channel in ("chat", "embed", "rerank"):
        assert other.credential_providers(channel, "m")[0].api_key == secret
        assert pool.providers(channel, "m")[0].api_key == secret
    with store.connect() as c:
        assert secret not in c.execute("SELECT secret FROM credentials").fetchone()[0]
    assert secret not in json.dumps(store.snapshot())
    assert secret not in repr(pool.providers("chat"))


async def test_key_admin_add_delete_and_does_not_resurrect_environment():
    app = FastAPI(); app.include_router(router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        secret = "test-secret-123"
        response = await client.post("/api/model-pool/keys", json={"keys": [secret], "channels": ["chat", "embed", "rerank"]})
        assert response.status_code == 200
        assert secret not in response.text
        assert {p.api_key for p in pool.providers("chat")} == {secret, "environment-key"}
        data = (await client.get("/api/model-pool/runtime")).json()
        assert len(data["slots"]) == 2
        assert all(s["managed"] for s in data["slots"])
        assert secret not in json.dumps(data) and "environment-key" not in json.dumps(data)
        for slot in data["slots"]:
            assert (await client.delete("/api/model-pool/keys/" + slot["resource"])).status_code == 200
        assert pool.providers("chat") == []
        assert pool.providers("embed") == []


async def test_bad_secret_input_is_never_echoed_and_origin_is_checked():
    app = FastAPI(); app.include_router(router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        body = {"keys": ["secret has whitespace"], "channels": ["chat"]}
        bad = await client.post("/api/model-pool/keys", json=body)
        assert bad.status_code == 400 and "secret" not in bad.text
        assert (await client.post("/api/model-pool/keys", json=body, headers={"origin": "https://evil.test"})).status_code == 403
        assert (await client.delete("/api/model-pool/keys/anything", headers={"origin": "null"})).status_code == 403
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=("192.168.1.10", 123)), base_url="http://test") as client:
        assert (await client.post("/api/model-pool/keys", json=body)).status_code == 403


def test_mixed_provider_configuration_rolls_back(monkeypatch):
    monkeypatch.setattr(settings, "embed_base_url", "https://another.test/v1")
    with pytest.raises(ValueError):
        pool.get_store().add_credentials(["invalid-provider-key"], ["chat", "embed"])
    assert pool.get_store().credential_metadata() == {}
    assert pool.providers("chat")[0].api_key == "environment-key"


def test_expanded_limits_and_timeout_validation():
    assert pool.Policy(global_limit=320, key_limit=32, background_limit=160,
                       queue_timeout=30, call_timeout=120, read_timeout=60).global_limit == 320
    with pytest.raises(ValueError):
        pool.Policy(call_timeout=30, read_timeout=60)
