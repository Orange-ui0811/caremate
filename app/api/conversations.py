"""会话列表、历史消息回载与当前用户的会话删除。"""
from fastapi import APIRouter, HTTPException, Query

from app.db import repository

router = APIRouter()


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
        {"role": m.role, "content": m.content or "",
         "activities": getattr(m, "activities", None) or [],
         "created_at": m.created_at.isoformat() if m.created_at else None}
        for m in msgs
    ]}
