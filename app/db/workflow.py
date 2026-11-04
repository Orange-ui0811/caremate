"""Transactional workflow messages and operation receipts (MySQL in the app)."""
import hashlib
import json
import uuid
from contextvars import ContextVar
from contextlib import contextmanager

from sqlalchemy import JSON, BigInteger, String, select, inspect, text, Enum
from sqlalchemy.orm import Mapped, mapped_column

from app.db import base
from app.db.models import Conversation, Message, Ticket

operation_context = ContextVar("operation_context", default=None)
_internal_events = ContextVar('internal_workflow_events', default=False)


@contextmanager
def internal_events():
    token = _internal_events.set(True)
    try:
        yield
    finally:
        _internal_events.reset(token)


class WorkflowRecord(base.Base):
    __tablename__ = "workflow_records"
    key: Mapped[str] = mapped_column(String(192), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(BigInteger, index=True)
    payload: Mapped[dict] = mapped_column(JSON)


async def ensure_tables():
    async with base.engine.begin() as conn:
        def ticket_enum(c):
            inspector = inspect(c)
            if c.dialect.name != "mysql" or not inspector.has_table("tickets"):
                return None
            column = next(x for x in inspector.get_columns("tickets") if x["name"] == "ticket_type")
            values = getattr(column["type"], "enums", [])
            if values and "退款" not in values:
                return Enum(*values, "退款").compile(dialect=c.dialect)
        enum_type = await conn.run_sync(ticket_enum)
        if enum_type:
            await conn.execute(text(f"ALTER TABLE tickets MODIFY COLUMN ticket_type {enum_type} NOT NULL"))
        await conn.run_sync(lambda c: WorkflowRecord.__table__.create(c, checkfirst=True))


async def save_plan(cid, uid, plan_id, origin, tasks):
    async with base.async_session() as session:
        await _owned(session, cid, uid)
        key = "plan:" + plan_id
        row = await session.get(WorkflowRecord, key)
        payload = {"origin": origin, "tasks": tasks}
        if row:
            row.payload = payload
        else:
            session.add(WorkflowRecord(key=key, conversation_id=cid, payload=payload))
        await session.commit()


async def supersede(cid, uid, origin):
    async with base.async_session() as session:
        await _owned(session, cid, uid)
        rows = (await session.execute(select(WorkflowRecord).where(
            WorkflowRecord.conversation_id == cid))).scalars()
        for row in rows:
            if row.key.startswith("plan:") and row.payload["origin"] < origin:
                row.payload = {**row.payload, "tasks": {k: {**t, "status": "SUPERSEDED"}
                    if t["status"] in {"PENDING", "RUNNING", "WAITING_USER", "WAITING_DEPENDENCY"} else t
                    for k, t in row.payload["tasks"].items()}}
        await session.commit()


async def _owned(session, cid, uid):
    conv = (await session.execute(select(Conversation).where(
        Conversation.id == cid).with_for_update())).scalar_one_or_none()
    if conv is None or conv.deleted_at is not None or conv.user_id != uid:
        raise ValueError("会话不存在")
    return conv


async def event(cid, uid, key, text, *, task_id="", actions=None, citations=None):
    internal = _internal_events.get()
    if internal:
        key = 'internal:' + key
    async with base.async_session() as session:
        await _owned(session, cid, uid)
        old = await session.get(WorkflowRecord, key)
        if old:
            if old.conversation_id != cid:
                raise ValueError("事件归属不匹配")
            return old.payload
        if internal:
            payload={'event_id':key,'text':text,'task_id':task_id,'actions':actions or [],'internal':True}
            session.add(WorkflowRecord(key=key,conversation_id=cid,payload=payload))
            await session.commit()
            return payload
        msg = Message(conversation_id=cid, role="assistant", content=text)
        session.add(msg)
        await session.flush()
        payload = {"event_id": key, "message_id": msg.id, "sequence": msg.id,
                   "text": text, "task_id": task_id, "actions": actions or [], "acked": False}
        if citations:
            payload['citations'] = citations
        session.add(WorkflowRecord(key=key, conversation_id=cid, payload=payload))
        await session.commit()
        return payload


async def events(cid, uid, after=0):
    async with base.async_session() as session:
        await _owned(session, cid, uid)
        records = (await session.execute(select(WorkflowRecord).where(
            WorkflowRecord.conversation_id == cid))).scalars()
        return sorted([r.payload for r in records if "sequence" in r.payload and not r.key.endswith(':summary')
                       and r.payload["sequence"] > after], key=lambda p: p["sequence"])


async def plans(cid, uid, before):
    async with base.async_session() as session:
        await _owned(session, cid, uid)
        records = (await session.execute(select(WorkflowRecord).where(
            WorkflowRecord.conversation_id == cid, WorkflowRecord.key.like('plan:%')))).scalars()
        return sorted([r.payload for r in records if r.payload.get('origin', before) < before],
                      key=lambda p: p['origin'], reverse=True)[:20]


async def hide_legacy_summaries():
    """Retain original audit rows; hide only messages linked to old summary events."""
    async with base.async_session() as session:
        records = (await session.execute(select(WorkflowRecord).where(WorkflowRecord.key.like('%:summary')))).scalars()
        for row in records:
            msg = await session.get(Message, row.payload.get('message_id'))
            if (msg is not None and msg.conversation_id == row.conversation_id and msg.role == 'assistant'
                    and (msg.content or '').startswith('本次事项处理结果：')):
                activities = msg.activities or []
                if not any(a.get('kind') == 'internal_task_summary' for a in activities):
                    msg.activities = [*activities, {'kind':'internal_task_summary'}]
        await session.commit()


async def ack(cid, uid, key):
    async with base.async_session() as session:
        await _owned(session, cid, uid)
        row = await session.get(WorkflowRecord, key)
        if row is None or row.conversation_id != cid or "sequence" not in row.payload:
            raise ValueError("事件不存在")
        row.payload = {**row.payload, "acked": True}
        await session.commit()


async def submit(cid, uid, operation_id, args, *, confirmed):
    """One local ticket + receipt per operation, even if the graph checkpoint lags."""
    digest = hashlib.sha256(json.dumps(args, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    key = "op:" + operation_id
    async with base.async_session() as session:
        conv = await _owned(session, cid, uid)
        old = await session.get(WorkflowRecord, key)
        if old:
            if old.conversation_id != cid or old.payload["digest"] != digest:
                raise ValueError("同一操作的参数不一致")
            return old.payload
        receipt = {"digest": digest, "status": "CANCELLED"}
        if confirmed:
            ticket_no = "T" + uuid.uuid4().hex[:24]
            session.add(Ticket(ticket_no=ticket_no, conversation_id=cid,
                               ticket_type=args["ticket_type"], description=args["description"]))
            conv.status = "已转人工"
            receipt = {"digest": digest, "status": "DONE", "ticket_no": ticket_no}
        session.add(WorkflowRecord(key=key, conversation_id=cid, payload=receipt))
        await session.commit()
        return receipt
