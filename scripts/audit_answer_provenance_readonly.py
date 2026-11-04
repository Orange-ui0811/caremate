"""Read existing conversation evidence only; no model calls or service writes."""
import asyncio
import json
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from sqlalchemy import select

from app.config import settings
from app.core.runtime_settings import get_confidence_settings
from app.db.base import async_session, engine
from app.db.models import Message, LowConfidenceQuestion, KnowledgeChunk, Faq

OUTPUT = Path("data/backend-investigation-20261011/answer-provenance-readonly.json")
CIDS = [4, 5, 12, 13, 14, 15, 18]


def row_dict(row):
    return {c.name: getattr(row, c.name) for c in row.__table__.columns}


def message_dict(m):
    return {"role": m.type, "id": m.id, "content": m.content,
            "tool_calls": getattr(m, "tool_calls", []),
            "name": getattr(m, "name", None)}


async def main():
    report = {"read_only": True, "upstream_model_calls": 0,
              "current_confidence_settings": get_confidence_settings(), "conversations": {}}
    try:
        async with async_session() as session:
            for cid in CIDS:
                msgs = (await session.scalars(select(Message).where(
                    Message.conversation_id == cid).order_by(Message.id))).all()
                lows = (await session.scalars(select(LowConfidenceQuestion).where(
                    LowConfidenceQuestion.conversation_id == cid))).all()
                report["conversations"][str(cid)] = {
                    "display_messages": [row_dict(r) for r in msgs],
                    "low_confidence_records": [row_dict(r) for r in lows], "checkpoints": []}
            report["knowledge_chunks"] = [row_dict(r) for r in
                (await session.scalars(select(KnowledgeChunk).order_by(KnowledgeChunk.id))).all()]
            report["faq"] = [row_dict(r) for r in
                (await session.scalars(select(Faq).order_by(Faq.id))).all()]
        uri = Path(settings.checkpointer_db_path).resolve().as_uri() + "?mode=ro"
        async with aiosqlite.connect(uri, uri=True) as conn:
            saver = AsyncSqliteSaver(conn)
            # Existing schema only: suppress the saver setup DDL in this read-only connection.
            saver.is_setup = True
            for cid in CIDS:
                async for cp in saver.alist({"configurable": {"thread_id": str(cid)}}):
                    state = cp.checkpoint["channel_values"]
                    selected = {k: state.get(k) for k in (
                        "intent", "route", "resolved_query", "trace", "answer", "evidence",
                        "citations", "evidence_strong", "evidence_confidence", "fallback_source",
                        "retrieved_snapshot", "steps", "summary", "summary_upto_msg_id")}
                    selected["messages"] = [message_dict(m) for m in state.get("messages", [])]
                    report["conversations"][str(cid)]["checkpoints"].append({
                        "id": cp.checkpoint["id"], "timestamp": cp.checkpoint["ts"],
                        "metadata": cp.metadata, "state": selected})
        OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        cur = report["conversations"]["18"]
        print(json.dumps({"output": str(OUTPUT), "settings": report["current_confidence_settings"],
            "messages": cur["display_messages"], "low_confidence": cur["low_confidence_records"],
            "latest_state": cur["checkpoints"][0],
            "checkpoint_counts": {cid: len(data["checkpoints"]) for cid, data in report["conversations"].items()},
            "knowledge_count": len(report["knowledge_chunks"])}, ensure_ascii=False, default=str))
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
