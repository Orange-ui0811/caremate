"""会话列表、历史消息回载与当前用户的会话删除。"""
from fastapi import APIRouter, HTTPException, Query

from app.db import repository
from app.db import workflow
from app.graph import runtime

router = APIRouter()


async def _owned_mode(conversation_id: int, user_id: str) -> bool:
    conv = await repository.get_conversation(conversation_id)
    if conv is None or conv.user_id != user_id:
        raise HTTPException(404, '会话不存在')
    return await runtime.conversation_mode(conversation_id)


@router.get('/api/conversations/{conversation_id}/workflow-mode')
async def workflow_mode(conversation_id: int, user_id: str):
    return {'multi_intent_enabled': await _owned_mode(conversation_id, user_id)}


@router.get("/api/conversations/{conversation_id}/task-events")
async def task_events(conversation_id: int, user_id: str, after: int = Query(default=0, ge=0)):
    if not await _owned_mode(conversation_id, user_id):
        return {"items": [], "interrupt": None}
    try:
        items = await workflow.events(conversation_id, user_id, after)
        snapshot = await runtime.get_graph().aget_state({"configurable": {"thread_id": str(conversation_id)}})
        pending = [intr for task in snapshot.tasks for intr in task.interrupts]
        card = None
        if snapshot.values.get("task_plan") and len(pending) == 1:
            rows = await repository.list_dialog_messages(conversation_id)
            latest = max((r.id for r in rows if r.role == "user"), default=0)
            if latest == snapshot.values.get("current_message_id"):
                card = {**pending[0].value, "kind": pending[0].value["type"],
                        "conversation_id": conversation_id, "interrupt_id": pending[0].id}
        return {"items": items, "interrupt": card}
    except ValueError:
        raise HTTPException(404, "会话不存在")


@router.post("/api/conversations/{conversation_id}/task-events/{event_id}/ack")
async def ack_task_event(conversation_id: int, event_id: str, user_id: str):
    try:
        await workflow.ack(conversation_id, user_id, event_id)
    except ValueError:
        raise HTTPException(404, "事件不存在")
    return {"ok": True}


@router.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: int, user_id: str = Query(min_length=1, max_length=64)):
    if not await repository.delete_conversation(conversation_id, user_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"ok": True}


@router.get("/api/conversations")
async def list_conversations(user_id: str):
    items = await repository.list_conversations(user_id)
    return {"items": items}


@router.get("/api/conversations/{conversation_id}/messages")
async def list_messages(conversation_id: int):
    if await repository.get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    msgs = await repository.list_dialog_messages(conversation_id)
    return {"items": [
        {"id": m.id, "role": m.role, "content": m.content or "",
         "activities": getattr(m, "activities", None) or [],
         "created_at": m.created_at.isoformat() if m.created_at else None}
        for m in msgs
    ]}
