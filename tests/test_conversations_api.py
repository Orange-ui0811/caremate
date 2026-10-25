"""ch07 验收配套:会话列表 / 历史消息两个只读接口(路由层,repository 打桩;
真库行为在 test_repository.py::test_list_conversations_* 覆盖)。"""
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture()
def conv_client():
    from app.api.conversations import router
    test_app = FastAPI()
    test_app.include_router(router)
    return TestClient(test_app)


def test_list_conversations_passes_user_id(conv_client, monkeypatch):
    from app.api import conversations as api
    seen = {}

    async def fake_list(user_id):
        seen["user_id"] = user_id
        return [{"id": 2, "status": "进行中", "preview": "猫粮怎么选",
                 "has_summary": False, "updated_at": None},
                {"id": 1, "status": "进行中", "preview": "订单1001到哪了",
                 "has_summary": True, "updated_at": None}]
    monkeypatch.setattr(api.repository, "list_conversations", fake_list)

    r = conv_client.get("/api/conversations", params={"user_id": "u1"})
    assert r.status_code == 200 and seen["user_id"] == "u1"
    items = r.json()["items"]
    assert [it["id"] for it in items] == [2, 1]
    assert items[1]["has_summary"] is True


def test_list_messages_serializes_dialog(conv_client, monkeypatch):
    from app.api import conversations as api

    async def fake_get(cid):
        return object()

    async def fake_msgs(cid):
        return [SimpleNamespace(role="user", content="订单1001到哪了", created_at=None),
                SimpleNamespace(role="assistant", content="在路上", created_at=None,
                                activities=[{"kind": "tool", "name": "query_logistics"}])]
    monkeypatch.setattr(api.repository, "get_conversation", fake_get)
    monkeypatch.setattr(api.repository, "list_dialog_messages", fake_msgs)

    r = conv_client.get("/api/conversations/7/messages")
    assert r.status_code == 200
    assert [(m["role"], m["content"]) for m in r.json()["items"]] == [
        ("user", "订单1001到哪了"), ("assistant", "在路上")]
    assert r.json()["items"][1]["activities"] == [{"kind": "tool", "name": "query_logistics"}]


def test_list_messages_404_when_missing(conv_client, monkeypatch):
    from app.api import conversations as api

    async def fake_get(cid):
        return None
    monkeypatch.setattr(api.repository, "get_conversation", fake_get)
    assert conv_client.get("/api/conversations/999999/messages").status_code == 404


def test_delete_conversation_scoped_and_repeatable(conv_client, monkeypatch):
    from app.api import conversations as api
    calls = []

    async def fake_delete(cid, user_id):
        calls.append((cid, user_id))
        return cid == 7 and user_id == "owner"

    monkeypatch.setattr(api.repository, "delete_conversation", fake_delete)
    for _ in range(2):
        assert conv_client.delete("/api/conversations/7?user_id=owner").json() == {"ok": True}
    assert conv_client.delete("/api/conversations/7?user_id=other").status_code == 404
    assert conv_client.delete("/api/conversations/999?user_id=owner").status_code == 404
    assert conv_client.delete("/api/conversations/7").status_code == 422
    assert conv_client.delete("/api/conversations/7?user_id=").status_code == 422
    assert calls == [(7, "owner"), (7, "owner"), (7, "other"), (999, "owner")]
