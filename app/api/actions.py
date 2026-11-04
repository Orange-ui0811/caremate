import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.exc import SQLAlchemyError

from app.config import settings
from app.core import context_runtime
from app.core.model_pool import get_store
from app.db import repository
from app.graph import runtime
from app.tools import business
from app.schemas.actions import (
    CreateRefundRequest,
    CreateRefundResponse,
    CreateTicketRequest,
    CreateTicketResponse,
    ResumeRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _check_action(conversation_id: int, user_id: str, tool_name: str, args: dict) -> None:
    """Buttons confirm a requested action; memory only supplies current denials.

    Identity matches /api/chat's caller-supplied user_id. This demo does not yet
    authenticate that identity using a server login/session.
    """
    conv = await repository.get_conversation(conversation_id)
    if conv is None or conv.user_id != user_id:
        raise HTTPException(status_code=404, detail="会话不存在")
    if tool_name == "submit_refund" and not business.owns_order(user_id, args.get("order_id", "")):
        raise HTTPException(status_code=403, detail="没有找到属于当前用户的这笔订单")
    if not context_runtime.enabled():
        return
    from app.core.context_state import build_snapshot
    rows = await repository.list_dialog_messages(conversation_id)
    snapshot = build_snapshot([{"id": row.id, "role": row.role, "content": row.content or ""}
                               for row in rows], conversation_id)
    state = {"conversation_id": conversation_id, "user_id": user_id,
             "current_message_id": max((row.id for row in rows if row.role == "user"), default=0),
             "conversation_memory": snapshot}
    denial = await context_runtime.write_denial(state, tool_name, args)
    if denial:
        raise HTTPException(status_code=409, detail=denial)


async def _acquire_action_turn(conversation_id: int):
    store, owner = get_store(), uuid.uuid4().hex
    if not await asyncio.to_thread(store.acquire_turn, conversation_id, owner):
        raise HTTPException(status_code=409, detail="当前会话还有请求正在处理，请等待后重新确认")
    return store, owner


@asynccontextmanager
async def _action_turn(conversation_id: int):
    store, owner = await _acquire_action_turn(conversation_id)
    try:
        async with asyncio.timeout(settings.model_pool_turn_timeout):
            yield
    except TimeoutError:
        raise HTTPException(status_code=504, detail="本次操作已超时，请先核实是否已提交再重试")
    finally:
        await asyncio.shield(asyncio.to_thread(store.release_turn, conversation_id, owner))


@router.post("/api/actions/create-ticket", response_model=CreateTicketResponse)
async def create_ticket_action(req: CreateTicketRequest) -> CreateTicketResponse:
    """建工单按钮:用户点了才写 tickets 表(复用 ch02 工单能力)。转人工是另一件事,纯前端。"""
    try:
        async with _action_turn(req.conversation_id):
            await _check_action(req.conversation_id, req.user_id, "create_ticket",
                                {"description": req.description, "ticket_type": req.ticket_type})
            ticket_no = await _idempotent_ticket(req, req.description, req.ticket_type)
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    except SQLAlchemyError:
        logger.exception("建工单失败 conv=%s", req.conversation_id)
        raise HTTPException(status_code=503, detail="工单系统暂时不可用,请稍后重试")
    return CreateTicketResponse(ticket_no=ticket_no)


@router.post("/api/actions/create-refund", response_model=CreateRefundResponse)
async def create_refund_action(req: CreateRefundRequest) -> CreateRefundResponse:
    """退款表单提交:写 tickets(ticket_type='退款'),描述带订单号 + 固定类目原因。复用 ch02 工单能力。"""
    desc = f"退款申请 订单号={req.order_id} 原因={req.reason}"
    try:
        async with _action_turn(req.conversation_id):
            await _check_action(req.conversation_id, req.user_id, "submit_refund",
                                {"order_id": req.order_id, "reason": req.reason})
            ticket_no = await _idempotent_ticket(req, desc, "退款")
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    except SQLAlchemyError:
        logger.exception("退款单创建失败 conv=%s", req.conversation_id)
        raise HTTPException(status_code=503, detail="退款系统暂时不可用,请稍后重试")
    return CreateRefundResponse(ticket_no=ticket_no)


async def _idempotent_ticket(req, description, ticket_type):
    from app.core import workflow_mode
    enabled = await runtime.conversation_mode(req.conversation_id)
    if not enabled:
        with workflow_mode.use(False):
            return await repository.create_ticket(req.conversation_id, description, ticket_type)
    if not req.request_id:
        raise ValueError("操作卡版本已过期，请重新打开表单确认")
    from app.db.workflow import submit
    receipt = await submit(req.conversation_id, req.user_id,
                           f"c{req.conversation_id}:form:{req.request_id}",
                           {"description": description, "ticket_type": ticket_type}, confirmed=True)
    return receipt["ticket_no"]


@router.post("/api/actions/resume")
async def resume_action(req: ResumeRequest):
    """中断续跑(SSE,事件与 /api/chat 同构):订单选择器点选回 order_id;
    工单预览卡回 confirmed(ch08,resume 值为 {"confirmed": bool},agent_tools 按此放行/拒绝)。"""
    if req.cancelled:
        if req.order_id is not None or req.confirmed is not None or req.operation_id is not None:
            raise HTTPException(status_code=400, detail="取消选单不能携带其他操作参数")
        resume_value = {"cancelled": True}
    elif req.operation_id:
        if req.order_id is not None or req.confirmed is None:
            raise HTTPException(status_code=400, detail="任务确认需要明确布尔值，不能同时选单")
        resume_value = {"operation_id": req.operation_id, "confirmed": req.confirmed}
    elif (req.order_id is None) == (req.confirmed is None):
        raise HTTPException(status_code=400, detail="order_id 与 confirmed 必须且只能传一个")
    else:
        resume_value = req.order_id if req.order_id is not None else {"confirmed": bool(req.confirmed)}
    try:
        await runtime.validate_resume(req.conversation_id, resume_value,
                                      user_id=req.user_id, interrupt_id=req.interrupt_id)
    except runtime.ConversationNotFound:
        raise HTTPException(status_code=404, detail="会话不存在")
    except runtime.ResumeRejected as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except SQLAlchemyError:
        raise HTTPException(status_code=503, detail="当前无法核实会话，请稍后重试")
    turn_store, turn_owner = await _acquire_action_turn(req.conversation_id)

    async def bounded_resume():
        async with asyncio.timeout(settings.model_pool_turn_timeout):
            async for event in runtime.stream_resume(req.conversation_id, resume_value,
                    user_id=req.user_id, interrupt_id=req.interrupt_id):
                yield event

    async def event_stream() -> AsyncIterator[str]:
        try:
            async for ev in bounded_resume():
                if ev["type"] == "task_event":
                    yield _sse({"event": "task_event", **{k: v for k, v in ev.items() if k != "type"}})
                elif ev["type"] == "tool":
                    yield _sse({"event": "tool", "name": ev["name"]})
                elif ev["type"] == "activity":
                    yield _sse({"event": "activity", "kind": ev["kind"], "name": ev["name"]})
                elif ev["type"] == "delta":
                    yield _sse({"delta": ev["text"]})
                elif ev["type"] == "citations":
                    yield _sse({"event": "citations", "items": ev["items"]})
                elif ev["type"] == "actions":
                    yield _sse({"event": "actions", "items": ev["items"]})
                elif ev["type"] == "interrupt":
                    yield _sse({"event": "interrupt",
                                **{k: v for k, v in ev.items() if k != "type"}})
                elif ev["type"] == "done":
                    yield _sse({"event": "done", "conversation_id": ev["conversation_id"]})
        except runtime.ConversationNotFound:
            yield "event: error\n"
            yield _sse({"message": "会话不存在"})
            return
        except runtime.ResumeRejected as exc:
            yield "event: error\n"
            yield _sse({"message": str(exc)})
            return
        except TimeoutError:
            yield "event: error\n"
            yield _sse({"message": "续跑已超时，请先查看会话记录再重试"})
            return
        except Exception:
            logger.exception("续跑失败 conv=%s", req.conversation_id)
            yield "event: error\n"
            yield _sse({"message": "上游暂时不可用,请稍后重试"})
            return
        finally:
            await asyncio.shield(asyncio.to_thread(turn_store.release_turn, req.conversation_id, turn_owner))
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
