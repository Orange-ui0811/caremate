"""Conversation ownership must be checked before recording a new message."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.graph import runtime


@pytest.mark.parametrize("streaming", [False, True])
async def test_other_conversation_cannot_be_mutated_before_context_check(monkeypatch, streaming):
    monkeypatch.setattr(runtime.repository, "get_conversation", AsyncMock(return_value=
        SimpleNamespace(user_id="owner", summary="", summary_upto_msg_id=0)))
    append = AsyncMock(side_effect=AssertionError("must not write foreign conversation"))
    monkeypatch.setattr(runtime.repository, "append_message", append)
    with pytest.raises(runtime.ConversationNotFound):
        if streaming:
            _ = [e async for e in runtime.stream_turn("other", "新请求", 17)]
        else:
            await runtime.run_turn("other", "新请求", 17)
    append.assert_not_awaited()
