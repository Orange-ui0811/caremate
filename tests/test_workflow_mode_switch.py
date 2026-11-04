from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from app.graph import runtime
from app.main import app


@pytest.mark.parametrize('enabled', [False, True])
async def test_browser_choice_reaches_streaming_runtime(monkeypatch, enabled):
    seen=[]
    async def stream(uid, message, cid, *, multi_intent_enabled):
        seen.append(multi_intent_enabled)
        yield {'type':'done','conversation_id':17}
    monkeypatch.setattr(runtime, 'stream_turn', stream)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app),base_url='http://t') as client:
        r=await client.post('/api/chat',json={'user_id':'mode-test','message':'合成测试','multi_intent_enabled':enabled})
    assert r.status_code==200 and seen==[enabled]


@pytest.mark.parametrize('value', ['true', 'false', 0, 1, {}])
async def test_mode_rejects_coerced_boolean(value):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app),base_url='http://t') as client:
        r=await client.post('/api/chat',json={'user_id':'mode-test','message':'合成测试','multi_intent_enabled':value})
    assert r.status_code==422


async def test_restoring_mode_requires_conversation_ownership(monkeypatch):
    from app.api import conversations
    monkeypatch.setattr(conversations.repository,'get_conversation',AsyncMock(return_value=SimpleNamespace(user_id='owner')))
    mode=AsyncMock(return_value=True);monkeypatch.setattr(runtime,'conversation_mode',mode)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app),base_url='http://t') as client:
        denied=await client.get('/api/conversations/7/workflow-mode?user_id=other')
        assert denied.status_code==404 and mode.await_count==0
        owned=await client.get('/api/conversations/7/workflow-mode?user_id=owner')
    assert owned.json()=={'multi_intent_enabled':True}
