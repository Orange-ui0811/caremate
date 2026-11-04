"""Owned, recoverable context storage. No model calls or implicit authorization.

All reads and writes are scoped to an undeleted conversation owned by user_id.
Snapshots use CAS and increasing watermarks; summary segments never overwrite
live task state. The raw dialog continues to live in the existing messages table.
"""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import re
import uuid

from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError

from app.db import base as db
from app.db.models import (Conversation, Message, ContextSnapshot,
                           ContextReceipt, ContextSummarySegment, ContextMaintenanceLease,
                           ContextSource, ContextSourceTerm)


def _now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _digest(value) -> str:
    return content_hash(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":"), allow_nan=False))


def source_digest(rows) -> str:
    def field(row, key):
        return row[key] if isinstance(row, dict) else getattr(row, key)
    return _digest([{"id": int(field(r, "id")), "role": field(r, "role"),
                     "content": field(r, "content") or ""} for r in rows])


async def ensure_context_tables(engine=None) -> None:
    engine = engine if engine is not None else db.engine
    tables = [ContextSnapshot.__table__, ContextReceipt.__table__,
              ContextSummarySegment.__table__, ContextMaintenanceLease.__table__,
              ContextSource.__table__, ContextSourceTerm.__table__]
    async with engine.begin() as conn:
        await conn.run_sync(lambda sync: db.Base.metadata.create_all(sync, tables=tables))


async def _owned(session, conversation_id: int, user_id: str, *, lock=False):
    query = select(Conversation).where(Conversation.id == conversation_id,
                                      Conversation.user_id == user_id,
                                      Conversation.deleted_at.is_(None))
    if lock:
        query = query.with_for_update()
    conv = (await session.execute(query)).scalar_one_or_none()
    if conv is None:
        raise PermissionError("conversation unavailable")
    return conv


def _snapshot(row):
    return {"version": row.version, "watermark": row.watermark, "snapshot": row.payload}


async def load_snapshot(conversation_id: int, user_id: str) -> dict | None:
    async with db.async_session() as s:
        await _owned(s, conversation_id, user_id)
        row = await s.get(ContextSnapshot, conversation_id)
        return _snapshot(row) if row else None


async def save_snapshot(conversation_id: int, user_id: str, snapshot: dict, *,
                        watermark: int, expected_version: int = 0) -> dict:
    if int(snapshot.get("conversation_id", conversation_id)) != conversation_id:
        raise ValueError("snapshot conversation mismatch")
    if watermark < 0 or expected_version < 0:
        raise ValueError("invalid snapshot version")
    # Never duplicate full chat history into the derived state table.
    payload = json.loads(json.dumps({k: v for k, v in snapshot.items() if k != "rows"},
                                    ensure_ascii=False, allow_nan=False))
    digest = _digest(payload)
    async with db.async_session() as s:
        await _owned(s, conversation_id, user_id, lock=True)
        if watermark:
            exists = (await s.execute(select(Message.id).where(
                Message.id == watermark, Message.conversation_id == conversation_id))).scalar_one_or_none()
            if exists is None:
                raise ValueError("watermark is not a message in this conversation")
        row = await s.get(ContextSnapshot, conversation_id)
        if row is not None:
            result = {"saved": False, "version": row.version, "watermark": row.watermark}
            if row.version != expected_version or watermark < row.watermark:
                return result
            if watermark == row.watermark:
                return {**result, "saved": row.payload_hash == digest}
            previous_watermark = row.watermark
            changed = await s.execute(update(ContextSnapshot).where(
                ContextSnapshot.conversation_id == conversation_id,
                ContextSnapshot.version == expected_version,
                ContextSnapshot.watermark < watermark,
            ).values(version=expected_version + 1, watermark=watermark,
                     payload=payload, payload_hash=digest, updated_at=_now()))
            if changed.rowcount == 1:
                await _index_sources(s, conversation_id, previous_watermark, watermark)
            await s.commit()
            if changed.rowcount == 1:
                return {"saved": True, "version": expected_version + 1, "watermark": watermark}
            await s.refresh(row)
            return {"saved": False, "version": row.version, "watermark": row.watermark}
        if expected_version != 0:
            return {"saved": False, "version": 0, "watermark": 0}
        s.add(ContextSnapshot(conversation_id=conversation_id, version=1,
                              watermark=watermark, payload=payload, payload_hash=digest,
                              updated_at=_now()))
        try:
            await _index_sources(s, conversation_id, 0, watermark)
            await s.commit()
            return {"saved": True, "version": 1, "watermark": watermark}
        except IntegrityError:
            # Another worker created the snapshot after our read (e.g. SQLite).
            await s.rollback()
            row = await s.get(ContextSnapshot, conversation_id)
            if row is None:
                raise
            return {"saved": False, "version": row.version, "watermark": row.watermark}


async def read_source(conversation_id: int, user_id: str, message_id: int,
                      start: int, end: int, content_hash: str | None = None) -> dict:
    async with db.async_session() as s:
        await _owned(s, conversation_id, user_id)
        row = (await s.execute(select(Message).where(Message.id == message_id,
                Message.conversation_id == conversation_id))).scalar_one_or_none()
        if row is None:
            raise LookupError("source message unavailable")
        raw = row.content or ""
        actual_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        if content_hash is not None and content_hash != actual_hash:
            raise ValueError("source content hash changed")
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end and start < len(raw)):
            raise ValueError("invalid source span")
        if end > len(raw) and content_hash is not None:
            raise ValueError("invalid exact source span")
        # Plain paging callers need not know the message length in advance.
        end = min(end, len(raw))
        return {"conversation_id": conversation_id, "message_id": message_id,
                "start": start, "end": end, "content_hash": actual_hash,
                "role": row.role, "text": raw[start:end], "total_chars": len(raw),
                "next_start": end if end < len(raw) else None}


async def record_receipt(conversation_id: int, user_id: str, *, tool_call_id: str,
                         tool_name: str, arguments: dict, raw_result,
                         formatted_result: str, status: str,
                         message_watermark: int | None = None) -> dict:
    payload = json.loads(json.dumps(raw_result, ensure_ascii=False, default=str, allow_nan=False))
    arguments = json.loads(json.dumps(arguments, ensure_ascii=False, default=str, allow_nan=False))
    receipt_id = uuid.uuid4().hex
    observed = _now()
    async with db.async_session() as s:
        await _owned(s, conversation_id, user_id)
        if message_watermark is None:
            message_watermark = (await s.execute(select(func.max(Message.id)).where(
                Message.conversation_id == conversation_id, Message.role == "user"))).scalar_one_or_none()
        elif (await s.execute(select(Message.id).where(
            Message.id == message_watermark, Message.conversation_id == conversation_id,
            Message.role == "user"))).scalar_one_or_none() is None:
            raise ValueError("receipt watermark is not a user message in this conversation")
        result_hash = _digest({"raw": payload, "formatted": formatted_result,
            "arguments": arguments, "tool_name": tool_name, "status": status,
            "message_watermark": message_watermark})
        s.add(ContextReceipt(id=receipt_id, conversation_id=conversation_id,
                             message_watermark=message_watermark,
                             tool_call_id=tool_call_id[:128], tool_name=tool_name[:128],
                             arguments=arguments, raw_result=payload,
                             formatted_result=formatted_result, status=status,
                             result_hash=result_hash, created_at=observed))
        await s.commit()
    return {"receipt_id": receipt_id, "result_hash": result_hash,
            "message_watermark": message_watermark, "observed_at": observed.isoformat() + "Z"}


async def read_receipt(conversation_id: int, user_id: str, receipt_id: str, *,
                        upto_msg_id: int | None = None) -> dict:
    async with db.async_session() as s:
        await _owned(s, conversation_id, user_id)
        row = (await s.execute(select(ContextReceipt).where(
            ContextReceipt.id == receipt_id,
            ContextReceipt.conversation_id == conversation_id))).scalar_one_or_none()
        if row is None:
            raise LookupError("receipt unavailable")
        if upto_msg_id is not None and (row.message_watermark is None or row.message_watermark > upto_msg_id):
            raise LookupError("receipt not visible at current message watermark")
        digest = _digest({"raw": row.raw_result, "formatted": row.formatted_result,
                          "arguments": row.arguments, "tool_name": row.tool_name,
                          "status": row.status, "message_watermark": row.message_watermark})
        if digest != row.result_hash:
            raise ValueError("receipt content hash changed")
        return {"receipt_id": row.id, "conversation_id": row.conversation_id,
                "message_watermark": row.message_watermark,
                "tool_call_id": row.tool_call_id, "tool_name": row.tool_name,
                "arguments": row.arguments, "raw_result": row.raw_result,
                "formatted_result": row.formatted_result, "status": row.status,
                "result_hash": row.result_hash, "observed_at": row.created_at.isoformat() + "Z",
                "historical": True}


def _segment(row):
    return {"id": row.id, "conversation_id": row.conversation_id,
            "from_msg_id": row.from_msg_id, "upto_msg_id": row.upto_msg_id,
            "summary": row.summary, "source_hash": row.source_hash,
            "source_refs": row.source_refs, "model": row.model,
            "prompt_version": row.prompt_version, "created_at": row.created_at.isoformat() + "Z"}


async def list_summary_segments(conversation_id: int, user_id: str, *,
                                upto_msg_id: int | None = None) -> list[dict]:
    async with db.async_session() as s:
        await _owned(s, conversation_id, user_id)
        query = select(ContextSummarySegment).where(ContextSummarySegment.conversation_id == conversation_id)
        if upto_msg_id is not None:
            query = query.where(ContextSummarySegment.upto_msg_id <= upto_msg_id)
        rows = (await s.execute(query.order_by(ContextSummarySegment.from_msg_id,
                                               ContextSummarySegment.upto_msg_id))).scalars()
        return [_segment(row) for row in rows]


async def put_summary_segment(conversation_id: int, user_id: str, *, from_msg_id: int,
                               upto_msg_id: int, summary: str, source_hash: str,
                               model: str = "", prompt_version: str = "source-v1") -> dict:
    if not summary.strip() or from_msg_id <= 0 or upto_msg_id < from_msg_id:
        raise ValueError("invalid summary segment")
    async with db.async_session() as s:
        await _owned(s, conversation_id, user_id, lock=True)
        rows = list((await s.execute(select(Message).where(
            Message.conversation_id == conversation_id,
            Message.id >= from_msg_id, Message.id <= upto_msg_id,
            Message.role.in_(("user", "assistant"))).order_by(Message.id))).scalars())
        if not rows or rows[0].id != from_msg_id or rows[-1].id != upto_msg_id:
            raise ValueError("summary source interval unavailable")
        if source_digest(rows) != source_hash:
            raise ValueError("summary source hash changed")
        key = _digest({"conversation_id": conversation_id, "from": from_msg_id,
                       "upto": upto_msg_id, "source_hash": source_hash,
                       "model": model, "prompt_version": prompt_version})
        old = await s.get(ContextSummarySegment, key)
        if old:
            return _segment(old)  # One immutable result per frozen source/prompt.
        refs = [{"message_id": r.id, "start": 0, "end": len(r.content or ""),
                 "content_hash": content_hash(r.content or "")} for r in rows]
        row = ContextSummarySegment(id=key, conversation_id=conversation_id,
            from_msg_id=from_msg_id, upto_msg_id=upto_msg_id, summary=summary.strip(),
            source_hash=source_hash, source_refs=refs, model=model,
            prompt_version=prompt_version, created_at=_now())
        s.add(row)
        try:
            await s.commit()
        except IntegrityError:
            await s.rollback()
            row = await s.get(ContextSummarySegment, key)
            if row is None:
                raise
        return _segment(row)


async def acquire_summary_lease(conversation_id: int, user_id: str, *, seconds=300) -> str | None:
    """Cross-process duplicate suppression; immutable segments remain the write guard."""
    owner, now = uuid.uuid4().hex, _now()
    if seconds <= 0:
        raise ValueError("lease duration must be positive")
    async with db.async_session() as s:
        await _owned(s, conversation_id, user_id, lock=True)
        row = await s.get(ContextMaintenanceLease, conversation_id)
        if row is not None:
            if row.expires_at > now:
                return None
            changed = await s.execute(update(ContextMaintenanceLease).where(
                ContextMaintenanceLease.conversation_id == conversation_id,
                ContextMaintenanceLease.expires_at <= now).values(
                    owner=owner, expires_at=now + timedelta(seconds=seconds)))
            await s.commit()
            return owner if changed.rowcount == 1 else None
        s.add(ContextMaintenanceLease(conversation_id=conversation_id, owner=owner,
                                      expires_at=now + timedelta(seconds=seconds)))
        try:
            await s.commit()
            return owner
        except IntegrityError:
            await s.rollback()
            return None


async def release_summary_lease(conversation_id: int, owner: str) -> None:
    async with db.async_session() as s:
        await s.execute(delete(ContextMaintenanceLease).where(
            ContextMaintenanceLease.conversation_id == conversation_id,
            ContextMaintenanceLease.owner == owner))
        await s.commit()


def _terms(text):
    from app.core.context_policy import terms
    return {word if len(word) <= 128 else "#" + content_hash(word) for word in terms(text)}


async def _index_sources(session, conversation_id: int, after_id: int, upto_id: int) -> None:
    """Incremental indexing of actual messages, never arbitrary snapshot quotations."""
    rows = (await session.execute(select(Message).where(
        Message.conversation_id == conversation_id, Message.id > after_id,
        Message.id <= upto_id, Message.role.in_(("user", "assistant"))).order_by(Message.id))).scalars()
    for row in rows:
        raw = row.content or ""
        digest = content_hash(raw)
        for match in re.finditer(r"[^。！？\n]+[。！？\n]*", raw):
            start, end = match.span()
            key = _digest([conversation_id, row.id, start, end, digest])
            session.add(ContextSource(id=key, conversation_id=conversation_id,
                message_id=row.id, start=start, end=end, role=row.role,
                text=raw[start:end], content_hash=digest))
            for term in sorted(_terms(raw[start:end])):
                session.add(ContextSourceTerm(conversation_id=conversation_id, term=term, source_id=key))


async def search_sources(conversation_id: int, user_id: str, query: str, *,
                          upto_msg_id: int, limit: int = 5) -> list[dict]:
    """Scoped lexical retrieval plus same-message neighbours; no semantic model.

    Retrieval never searches other conversations or messages after the caller's
    snapshot. A long sentence is returned as an explicitly incomplete source
    span with full_end so the caller can page the complete original message.
    """
    limit = max(1, min(int(limit), 20))
    terms = sorted(_terms(query))
    if not terms or upto_msg_id <= 0:
        # Still enforce ownership even for empty searches.
        async with db.async_session() as s:
            await _owned(s, conversation_id, user_id)
        return []
    async with db.async_session() as s:
        await _owned(s, conversation_id, user_id)
        early = bool(re.search(r"最早|最开始|第一", query))
        order = ContextSource.message_id.asc() if early else ContextSource.message_id.desc()
        score = func.count(ContextSourceTerm.term)
        hits = list((await s.execute(select(ContextSource, score.label("score")).join(
            ContextSourceTerm, ContextSourceTerm.source_id == ContextSource.id).where(
                ContextSource.conversation_id == conversation_id,
                ContextSourceTerm.conversation_id == conversation_id,
                ContextSource.message_id <= upto_msg_id,
                ContextSourceTerm.term.in_(terms)).group_by(ContextSource.id).order_by(
                    score.desc(), order, ContextSource.start).limit(limit))).all())
        selected = {}
        for hit, points in hits:
            selected[hit.id] = (hit, int(points), "match")
            for direction in ("before", "after"):
                predicate = ContextSource.start < hit.start if direction == "before" else ContextSource.start > hit.start
                sorting = ContextSource.start.desc() if direction == "before" else ContextSource.start.asc()
                neighbour = (await s.execute(select(ContextSource).where(
                    ContextSource.conversation_id == conversation_id,
                    ContextSource.message_id == hit.message_id, predicate).order_by(sorting).limit(1))).scalar_one_or_none()
                if neighbour and neighbour.id not in selected:
                    selected[neighbour.id] = (neighbour, 0, "neighbour")
        output = []
        for src, points, reason in sorted(selected.values(), key=lambda x: (x[0].message_id, x[0].start)):
            message = await s.get(Message, src.message_id)
            if message is None or message.conversation_id != conversation_id:
                continue
            raw = message.content or ""
            if content_hash(raw) != src.content_hash or raw[src.start:src.end] != src.text:
                continue
            lo, hi = src.start, src.end
            if hi - lo > 700:
                # Preserve literal positions and flag fragments; never silently call them complete sentences.
                anchors = [src.text.lower().find(t) for t in terms if not t.startswith("#")]
                anchors = [x for x in anchors if x >= 0]
                offset = max(0, (min(anchors) if anchors else 0) - 120)
                lo, hi = src.start + offset, min(src.end, src.start + offset + 700)
            output.append({"conversation_id": conversation_id, "message_id": src.message_id,
                "start": lo, "end": hi, "full_start": src.start, "full_end": src.end,
                "complete": lo == src.start and hi == src.end, "text": raw[lo:hi],
                "content_hash": src.content_hash, "role": src.role,
                "score": points, "selection": reason})
        return output
