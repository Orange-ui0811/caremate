import json
import logging
import asyncio
import hashlib
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.exc import SQLAlchemyError

from app.graph import runtime
from app.schemas.chat import ChatRequest
from app.config import settings
from app.core.model_pool import call_context, get_store

logger = logging.getLogger(__name__)
router = APIRouter()


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sse_error(message: str) -> AsyncIterator[str]:
    yield "event: error\n"
    yield _sse({"message": message})


@router.post("/api/chat")
async def chat(req: ChatRequest):
    trace = req.request_id or uuid.uuid4().hex
    replay_key = None
    store = None
    if req.request_id:
        store = get_store()
        replay_key = hashlib.sha256((req.user_id + "\0" + req.request_id).encode()).hexdigest()
        digest = hashlib.sha256(req.model_dump_json(exclude={"request_id"}).encode()).hexdigest()
        status, saved = await asyncio.to_thread(store.begin_replay, replay_key, digest)
        if status == "SUCCEEDED":
            return StreamingResponse(iter([saved]), media_type="text/event-stream",
                                     headers={"X-Request-ID": trace, "X-Result-Replay": "true"})
        if status != "NEW":
            detail = {"CONFLICT": "同一请求ID对应的内容不一致，请创建新请求",
                      "RUNNING": "本次请求仍在执行，请稍后取回结果",
                      "INTERRUPTED": "本次请求已中断，请先查看会话和操作记录，避免重复执行",
                      "FAILED": "本次请求未完成，请先查看会话和操作记录，再发起新问题"}
            raise HTTPException(status_code=409, detail=detail.get(status, "本次结果不可回放，请查看会话记录"))

    turn_store, turn_owner = get_store(), uuid.uuid4().hex
    if req.conversation_id is not None:
        acquired = await asyncio.to_thread(turn_store.acquire_turn, req.conversation_id, turn_owner)
        if not acquired:
            if store:
                await asyncio.to_thread(store.finish_replay, replay_key, "REJECTED")
            raise HTTPException(409, "当前会话还有问题正在处理，请等待该轮结束后再发送")

    async def event_stream() -> AsyncIterator[str]:
        try:
            async for ev in runtime.stream_turn(req.user_id, req.message, req.conversation_id):
                if ev["type"] == "tool":
                    yield _sse({"event": "tool", "name": ev["name"]})
                elif ev["type"] == "activity":
                    yield _sse({"event": "activity", "kind": ev["kind"], "name": ev["name"]})
                elif ev["type"] == "delta":
                    yield _sse({"delta": ev["text"]})
                elif ev["type"] == "citations":
                    yield _sse({"event": "citations", "items": ev["items"]})
                elif ev["type"] == "interrupt":
                    # ch08:负载透传(select_order 带 orders / confirm_ticket 带 preview),前端按 kind 分发
                    yield _sse({"event": "interrupt",
                                **{k: v for k, v in ev.items() if k != "type"}})
                elif ev["type"] == "actions":
                    yield _sse({"event": "actions", "items": ev["items"]})
                elif ev["type"] == "done":
                    yield _sse({"event": "done", "conversation_id": ev["conversation_id"]})
        except runtime.ConversationNotFound:
            for f in _sse_error("会话不存在"):
                yield f
            return
        except SQLAlchemyError:
            logger.exception("数据库错误 user_id=%s", req.user_id)
            for f in _sse_error("数据库暂时不可用,请稍后重试"):
                yield f
            return
        except Exception:
            logger.exception("图编排失败 user_id=%s", req.user_id)
            for f in _sse_error("上游模型暂时不可用,请稍后重试"):
                yield f
            return
        yield "data: [DONE]\n\n"

    async def governed_stream():
        pieces, size, completed = [], 0, False
        try:
            with call_context(trace):
                async with asyncio.timeout(settings.model_pool_turn_timeout):
                    async for part in event_stream():
                        size += len(part.encode("utf-8"))
                        if size <= 1024 * 1024:
                            pieces.append(part)
                        if part == "data: [DONE]\n\n":
                            if store:
                                await asyncio.to_thread(store.finish_replay, replay_key,
                                                        "SUCCEEDED" if size <= 1024 * 1024 else "UNAVAILABLE",
                                                        "".join(pieces) if size <= 1024 * 1024 else None)
                            completed = True
                        yield part
        except TimeoutError:
            for part in _sse_error("本轮处理已超时，请查看会话记录后重试，或联系人工客服"):
                yield part
        finally:
            if req.conversation_id is not None:
                await asyncio.shield(asyncio.to_thread(turn_store.release_turn, req.conversation_id, turn_owner))
            if store and not completed:
                # Never blindly rerun an interrupted turn: tools may already have executed.
                await asyncio.shield(asyncio.to_thread(store.finish_replay, replay_key, "INTERRUPTED"))

    return StreamingResponse(governed_stream(), media_type="text/event-stream",
                             headers={"X-Request-ID": trace, "X-Result-Replay": "false",
                                      "Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
