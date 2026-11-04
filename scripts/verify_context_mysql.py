"""Verify context storage on a strictly isolated MySQL test schema.

Never drops tables or resets test data. A unique synthetic conversation is the
only data namespace written or removed. URLs and database credentials are never
included in output or error reports.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import uuid

from sqlalchemy import delete, inspect, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.core.context_state import build_snapshot
from app.db import base, context_repository as store, repository
from app.db.models import (Conversation, Message, ContextSnapshot, ContextReceipt,
                           ContextSummarySegment, ContextMaintenanceLease,
                           ContextSource, ContextSourceTerm)

ROOT = Path(__file__).resolve().parents[1]
CONTEXT_MODELS = (ContextSourceTerm, ContextSource, ContextMaintenanceLease,
                  ContextSummarySegment, ContextReceipt, ContextSnapshot)
TABLES = [Conversation.__table__, Message.__table__,
          *(model.__table__ for model in CONTEXT_MODELS)]


class VerificationFailed(Exception):
    pass


class UnsafeTestTarget(Exception):
    pass


def checked_test_url():
    candidate = make_url(settings.test_database_url)
    production = make_url(settings.database_url)
    schema = candidate.database or ""
    if candidate.get_backend_name() != "mysql":
        raise UnsafeTestTarget("not mysql")
    if not re.fullmatch(r"[A-Za-z0-9_]+_test", schema):
        raise UnsafeTestTarget("schema name must end in _test")
    if schema.casefold() == (production.database or "").casefold():
        raise UnsafeTestTarget("test and production schema names must differ")
    return candidate


def safe_error(exc):
    original = getattr(exc, "orig", exc)
    args = getattr(original, "args", ())
    code = args[0] if args and isinstance(args[0], int) else None
    reasons = {2003: "test_database_unreachable", 2005: "test_database_host_unresolved",
               1045: "test_database_authentication_failed", 1044: "test_database_permission_denied",
               1049: "test_schema_missing", 1054: "test_schema_columns_missing",
               1146: "test_schema_table_missing", 1406: "stored_value_exceeds_column",
               1055: "mysql_group_by_incompatible"}
    return {"type": type(exc).__name__, "database_code": code,
            "reason": reasons.get(code, "verification_failed_without_sensitive_details")}


def checked(report, name, condition):
    report["checks"].append({"name": name, "passed": bool(condition)})
    if not condition:
        raise VerificationFailed(name)


async def expect_failure(report, name, exception_type, operation):
    try:
        await operation()
    except exception_type:
        checked(report, name, True)
    else:
        checked(report, name, False)


async def verify(report):
    url = checked_test_url()  # Guard before any connection or DDL.
    report["target_guard"] = {"mysql": True, "test_suffix": True,
                              "different_from_production": True,
                              "schema_name_sha256": hashlib.sha256(url.database.encode()).hexdigest()}
    engine = create_async_engine(url, pool_pre_ping=True, echo=False,
                                  connect_args={"connect_timeout": 10})
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    prior_sessions = base.async_session
    synthetic_user = "context_mysql_verify_" + uuid.uuid4().hex
    cid = None
    try:
        report["stage"] = "check_tables"
        async with engine.begin() as conn:
            report["server_version"] = str((await conn.execute(text("SELECT VERSION()"))).scalar_one())

            def prepare(sync):
                existing = set(inspect(sync).get_table_names())
                missing = [t.name for t in TABLES if t.name not in existing]
                # create_all(checkfirst=True) creates absent tables only; no ALTER/DROP.
                base.Base.metadata.create_all(sync, tables=TABLES, checkfirst=True)
                inspector = inspect(sync)
                gaps = {table.name: sorted(set(table.columns.keys()) -
                        {c["name"] for c in inspector.get_columns(table.name)}) for table in TABLES}
                return missing, {name: columns for name, columns in gaps.items() if columns}

            created, gaps = await conn.run_sync(prepare)
            report["created_tables"] = created
            report["missing_columns"] = gaps
            checked(report, "schema_columns_available_without_altering_existing_tables", not gaps)
        base.async_session = sessions
        report["stage"] = "synthetic_conversation"
        cid = await repository.create_conversation(synthetic_user)
        report["synthetic_conversation_id"] = cid
        first = "当前处理订单810201。先别提交退款。"
        mid1 = await repository.append_message(cid, "user", content=first)
        mid2 = await repository.append_message(cid, "assistant", content="收到，尚未执行退款。")
        mid3 = await repository.append_message(cid, "user", content="810201报错，应为810202。")
        rows = [{"id": mid1, "role": "user", "content": first},
                {"id": mid2, "role": "assistant", "content": "收到，尚未执行退款。"},
                {"id": mid3, "role": "user", "content": "810201报错，应为810202。"}]
        early = build_snapshot(rows[:1], cid)
        current = build_snapshot(rows, cid)
        report["stage"] = "snapshot_and_index"
        one = await store.save_snapshot(cid, synthetic_user, early, watermark=mid1)
        checked(report, "initial_snapshot_saved", one["saved"] and one["version"] == 1)
        two = await store.save_snapshot(cid, synthetic_user, current, watermark=mid3, expected_version=1)
        checked(report, "snapshot_cas_advanced", two["saved"] and two["version"] == 2)
        stale = await store.save_snapshot(cid, synthetic_user, early, watermark=mid1, expected_version=2)
        conflict = await store.save_snapshot(cid, synthetic_user, current, watermark=mid3, expected_version=1)
        same = await store.save_snapshot(cid, synthetic_user, current, watermark=mid3, expected_version=2)
        checked(report, "older_watermark_rejected", not stale["saved"])
        checked(report, "stale_version_rejected", not conflict["saved"])
        checked(report, "same_snapshot_idempotent", same["saved"] and same["version"] == 2)
        loaded = await store.load_snapshot(cid, synthetic_user)
        checked(report, "snapshot_rows_not_duplicated", loaded["watermark"] == mid3 and "rows" not in loaded["snapshot"])
        hits = await store.search_sources(cid, synthetic_user, "810202", upto_msg_id=mid3, limit=5)
        checked(report, "mysql_incremental_scoped_term_index", any(r["message_id"] == mid3 and "810202" in r["text"] for r in hits))
        past = await store.search_sources(cid, synthetic_user, "810202", upto_msg_id=mid1, limit=5)
        checked(report, "search_excludes_future_messages", not past)
        original = await store.read_source(cid, synthetic_user, mid1, 0, len(first), store.content_hash(first))
        checked(report, "exact_source_round_trip", original["text"] == first)
        await expect_failure(report, "source_hash_checked", ValueError,
            lambda: store.read_source(cid, synthetic_user, mid1, 0, 2, "wrong"))
        await expect_failure(report, "owner_scope_checked", PermissionError,
            lambda: store.load_snapshot(cid, synthetic_user + "_other"))

        report["stage"] = "receipts_and_segments"
        large_text = "合成回执" * 6000
        checked(report, "receipt_exercises_more_than_64kb", len(large_text.encode("utf-8")) > 65535)
        receipt = await store.record_receipt(cid, synthetic_user, tool_call_id="synthetic-mysql-verification",
            tool_name="synthetic_query_order", arguments={"order_id": "810202"},
            raw_result={"order_id": "810202", "details": large_text}, formatted_result=large_text,
            status="成功", message_watermark=mid3)
        recovered = await store.read_receipt(cid, synthetic_user, receipt["receipt_id"], upto_msg_id=mid3)
        checked(report, "mysql_longtext_receipt_round_trip", recovered["formatted_result"] == large_text and
                recovered["raw_result"]["details"] == large_text)
        await expect_failure(report, "receipt_future_watermark_rejected", LookupError,
            lambda: store.read_receipt(cid, synthetic_user, receipt["receipt_id"], upto_msg_id=mid1))
        segment_args = {"from_msg_id": mid1, "upto_msg_id": mid2,
            "summary": "用户此前暂不提交退款；客服未执行。", "source_hash": store.source_digest(rows[:2]),
            "model": "synthetic-no-model", "prompt_version": "mysql-storage-verification-v1"}
        segment = await store.put_summary_segment(cid, synthetic_user, **segment_args)
        again = await store.put_summary_segment(cid, synthetic_user, **{**segment_args, "summary": "不得覆盖已有段"})
        checked(report, "summary_segment_immutable", segment["id"] == again["id"] and segment["summary"] == again["summary"])
        checked(report, "summary_segment_no_future_prefix", not await store.list_summary_segments(cid, synthetic_user, upto_msg_id=mid1))
        checked(report, "summary_preserves_current_state", (await store.load_snapshot(cid, synthetic_user))["watermark"] == mid3)
        lease = await store.acquire_summary_lease(cid, synthetic_user)
        checked(report, "mysql_maintenance_lease", bool(lease) and await store.acquire_summary_lease(cid, synthetic_user) is None)
        await store.release_summary_lease(cid, "wrong-owner")
        checked(report, "lease_owner_checked", await store.acquire_summary_lease(cid, synthetic_user) is None)
        await store.release_summary_lease(cid, lease)
        checked(report, "legacy_summary_cas", await repository.update_conversation_summary(
            cid, "合成新摘要", mid3, expected_upto_msg_id=0))
        checked(report, "legacy_late_summary_rejected", not await repository.update_conversation_summary(
            cid, "合成旧摘要", mid2, expected_upto_msg_id=0))

        report["stage"] = "concurrent_cas"
        mid4 = await repository.append_message(cid, "assistant", content="已经记录更正，但仍未执行退款。")
        mid5 = await repository.append_message(cid, "user", content="只查订单810202物流。")
        rows4 = rows + [{"id": mid4, "role": "assistant", "content": "已经记录更正，但仍未执行退款。"}]
        rows5 = rows4 + [{"id": mid5, "role": "user", "content": "只查订单810202物流。"}]
        concurrent = await asyncio.gather(
            store.save_snapshot(cid, synthetic_user, build_snapshot(rows4, cid), watermark=mid4, expected_version=2),
            store.save_snapshot(cid, synthetic_user, build_snapshot(rows5, cid), watermark=mid5, expected_version=2))
        checked(report, "exactly_one_concurrent_cas_winner", sum(x["saved"] for x in concurrent) == 1)
        final = await store.load_snapshot(cid, synthetic_user)
        if final["watermark"] < mid5:
            advanced = await store.save_snapshot(cid, synthetic_user, build_snapshot(rows5, cid),
                watermark=mid5, expected_version=final["version"])
            checked(report, "cas_conflict_can_retry_latest_prefix", advanced["saved"])
        checked(report, "final_watermark_current", (await store.load_snapshot(cid, synthetic_user))["watermark"] == mid5)
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed" if isinstance(exc, VerificationFailed) else "blocked"
        report["error"] = safe_error(exc)
    finally:
        report["cleanup"] = {"status": "not_needed" if cid is None else "pending", "tables_retained": True}
        if cid is not None:
            try:
                if not re.fullmatch(r"context_mysql_verify_[0-9a-f]{32}", synthetic_user):
                    raise UnsafeTestTarget("synthetic namespace invalid")
                async with sessions() as session:
                    conv = (await session.execute(select(Conversation).where(
                        Conversation.id == cid).with_for_update())).scalar_one_or_none()
                    if conv is None or conv.user_id != synthetic_user:
                        raise UnsafeTestTarget("conversation namespace changed; cleanup refused")
                    counts = {}
                    for model in CONTEXT_MODELS:
                        result = await session.execute(delete(model).where(model.conversation_id == cid))
                        counts[model.__tablename__] = result.rowcount
                    counts["messages"] = (await session.execute(delete(Message).where(Message.conversation_id == cid))).rowcount
                    counts["conversations"] = (await session.execute(delete(Conversation).where(
                        Conversation.id == cid, Conversation.user_id == synthetic_user))).rowcount
                    await session.commit()
                    report["cleanup"] = {"status": "passed", "tables_retained": True,
                                         "only_synthetic_conversation": True, "deleted_row_counts": counts}
            except Exception as exc:
                report["cleanup"] = {"status": "blocked", "tables_retained": True,
                                      "error": safe_error(exc)}
                report["status"] = "blocked"
        base.async_session = prior_sessions
        await engine.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/verification/context-implementation-20261027/mysql-report.json")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT):
        raise SystemExit("Report output must stay in the project workspace")
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "status": "pending",
              "stage": "target_guard", "checks": [], "real_model_calls": 0,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        asyncio.run(verify(report))
    except Exception as exc:
        report["status"], report["error"] = "blocked", safe_error(exc)
    report["passed_checks"] = sum(c["passed"] for c in report["checks"])
    report["total_checks"] = len(report["checks"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "passed_checks": report["passed_checks"],
                      "total_checks": report["total_checks"], "stage": report["stage"],
                      "cleanup": report.get("cleanup", {}).get("status"),
                      "error": report.get("error"), "report": str(output)}, ensure_ascii=False))
    return 0 if report["status"] == "passed" and report.get("cleanup", {}).get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
