from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.fixture(autouse=True)
def action_scope(monkeypatch):
    from app.api import actions
    monkeypatch.setattr(actions.context_runtime, "enabled", lambda: False)
    monkeypatch.setattr(actions.runtime, "conversation_mode", AsyncMock(return_value=False))
    monkeypatch.setattr(actions.repository, "get_conversation", AsyncMock(return_value=SimpleNamespace(user_id="u1")))


def test_create_ticket_action_writes(client, monkeypatch):
    async def fake_create(cid, desc, ttype):
        assert cid == 5 and ttype == "投诉"
        return "T20260715001"

    from app.api import actions
    monkeypatch.setattr(actions.repository, "create_ticket", fake_create)
    r = client.post("/api/actions/create-ticket", json={
        "user_id": "u1", "conversation_id": 5, "description": "东西坏了", "ticket_type": "投诉"})
    assert r.status_code == 200
    assert r.json()["ticket_no"] == "T20260715001"


def test_create_ticket_action_rejects_bad_type(client):
    r = client.post("/api/actions/create-ticket", json={
        "user_id": "u1", "conversation_id": 5, "description": "x", "ticket_type": "乱填"})
    assert r.status_code == 422
