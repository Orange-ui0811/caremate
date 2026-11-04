"""Frozen five-arm evaluation through the actual customer-service Workflow.

Preparing and auditing are offline. Only --run makes model calls. All customer,
knowledge and business data are isolated synthetic fixtures. Historical scripts
and results are imported read-only; no historical module globals are changed.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import statistics
import time
from unittest.mock import patch
from urllib.parse import urlsplit

from langchain_core.messages import message_to_dict
from langchain_core.tools import tool

from scripts import verify_context_workflow_compact as old

base = old._impl
ROOT = base.ROOT
ARMS = ("legacy", "full", "window", "window_summary", "task")
CAPS = {"coref": 2400, "intent": 1000, "retrieval": 2400, "answer": 4200}
SETTINGS = {"context_input_budget": 16000, "context_output_budget": 8192,
            "context_coref_budget": 2400, "context_intent_budget": 1000,
            "context_state_budget": 4200, "context_summary_trigger_tokens": 2400,
            "context_summary_target_tokens": 450, "context_window_turns": 8,
            "summary_trigger_messages": 30, "context_fresh_business_evidence": True}
FULL_OVERRIDES = {"context_state_budget": 16000, "context_coref_budget": 16000,
                  "context_intent_budget": 16000}
read, write, digest = base.read, base.write, base.digest


def cases_module():
    from scripts import context_unified_cases
    return context_unified_cases


def dataset(pilot=False):
    module = cases_module()
    value = module.pilot_dataset() if pilot else module.dataset()
    cases = value["cases"] if isinstance(value, dict) else value
    return {"version": "context-unified-20261027-v1", "synthetic": True,
            "pilot": pilot, "cases": deepcopy(cases)}


def effective_config():
    from scripts.context_unified_summary import CONTRACT
    return {"environment": base.config(), "overrides": SETTINGS, "full_overrides": FULL_OVERRIDES, "stage_caps": CAPS,
            "thinking": base.OFF, "failover": False,
            "window_summary_contract": CONTRACT,
            "legacy_note": "Original legacy history algorithm/window estimator; shared execution fix and hard input ceiling."}


class Fixture(old.Fixture):
    def __init__(self, case):
        super().__init__({**case, "initial_history": []})
        self.case = case
        self.seed = deepcopy(case.get("initial_history", []))

    async def update_summary(self, cid, summary, upto, *, expected_upto_msg_id):
        self.check(cid)
        if self.summary_upto != expected_upto_msg_id:
            return False
        self.summary, self.summary_upto = summary, upto
        return True

    def order(self, order_id):
        order_id = str(order_id)
        if order_id not in base.ORDER_IDS:
            return {"error": "合成订单不存在"}
        mapping = getattr(cases_module(), "ORDER_PRODUCTS", {})
        return {"order_id": order_id, "status": "已发货", "tracking_no": "SF" + order_id + "88",
                "product": mapping.get(order_id, "合成测试商品"), "amount": 199,
                "synthetic": True}

    def logistics(self, tracking_no):
        order_id = next((oid for oid in base.ORDER_IDS
                         if self.order(oid)["tracking_no"] == tracking_no), "")
        if not order_id:
            return {"error": "合成运单不存在"}, ""
        turn = (base.COORDS.get() or {}).get("turn", 0)
        events = {1: "已离开杭州分拨中心", 2: "已到达宁波中转站", 3: "已交由鄞州网点派送"}
        event = getattr(cases_module(), "logistics_event", lambda n: events.get(n, events[1]))(turn)
        if isinstance(event, dict):
            event = event.get("last_event") or event.get("status") or str(event)
        result = {"order_id": order_id, "tracking_no": tracking_no, "status": event,
                  "last_event": f"物流记录{turn}：{event}",
                  "version": f"物流记录{turn}",
                  "observed_version": turn, "observed_at": f"2026-10-27T{9 + turn:02d}:00:00+08:00",
                  "synthetic": True}
        overrides = self.case.get("tool_overrides", {}).get(str(turn), {})
        result.update(deepcopy(overrides))
        if result.get("error"):
            result = {k: result[k] for k in ("error", "order_id", "tracking_no", "observed_at", "synthetic")}
        return result, order_id

    def specs(self):
        fixture = self

        @tool
        async def query_order(order_id: str) -> dict:
            """按订单号查询当前用户的订单详情及运单号。"""
            result = fixture.order(order_id)
            fixture.tool_calls.append({"name": "query_order", "args": {"order_id": order_id},
                "resolved_order_id": str(order_id), "result": deepcopy(result), "coords": deepcopy(base.COORDS.get())})
            return result

        @tool
        async def query_logistics(tracking_no: str) -> dict:
            """按运单号查询当前物流状态；须先取得正确订单的运单号。"""
            result, order_id = fixture.logistics(tracking_no)
            fixture.tool_calls.append({"name": "query_logistics", "args": {"tracking_no": tracking_no},
                "resolved_order_id": order_id, "result": deepcopy(result), "coords": deepcopy(base.COORDS.get())})
            return result

        @tool
        async def query_faq(query: str) -> dict:
            """查询合成商品知识。"""
            result = {"hits": await fixture.search(query)}
            fixture.tool_calls.append({"name": "query_faq", "args": {"query": query},
                                      "result": result, "coords": deepcopy(base.COORDS.get())})
            return result

        @tool
        async def create_ticket(description: str, ticket_type: str = "咨询") -> dict:
            """创建客服工单，属于写操作并需要用户明确确认。"""
            result = {"ticket_id": "SYNTHETIC-ONLY", "synthetic": True}
            fixture.tool_calls.append({"name": "create_ticket", "args": {"description": description,
                "ticket_type": ticket_type}, "result": result, "coords": deepcopy(base.COORDS.get())})
            return result

        return [base.registry.spec_from_langchain_tool(t, source="builtin", max_retries=0)
                for t in (query_order, query_logistics, query_faq, create_ticket)]

    async def search(self, query, **kwargs):
        products = getattr(cases_module(), "PRODUCTS", base.PRODUCTS)
        requested = list(dict.fromkeys(re.findall(r"MH-[A-Za-z0-9]+", query)))
        hits = [{"id": list(products).index(sku) + 1, "question": f"{sku}怎么清洁？",
                 "answer": products[sku], "rerank_score": .99,
                 "section_path": f"合成夹具/{sku}", "content_type": "faq"}
                for sku in requested if sku in products]
        if not requested and re.search(r"退货|退款|售后", query):
            hits = [{"id": 99, "question": "合成订单退款政策",
                     "answer": "退款须核验订单归属并经用户确认，禁止将查询视为提交。",
                     "rerank_score": .99, "section_path": "合成夹具/政策", "content_type": "policy"}]
        self.retrievals.append({"query": query, "requested_skus": requested,
                               "returned_ids": [h["id"] for h in hits], "hits": deepcopy(hits),
                               "coords": deepcopy(base.COORDS.get())})
        return hits


@contextmanager
def isolated_workflow(fixture, arm, recorder):
    from app.core.context_task import assemble_task_context
    from scripts import context_unified_summary as bounded_summary
    projections, maintenance = [], []
    summary_cooldown = bounded_summary.SummaryCooldown()
    with ExitStack() as stack:
        for name, value in {**SETTINGS, **(FULL_OVERRIDES if arm == "full" else {})}.items():
            stack.enter_context(patch.object(base.settings, name, value))
        graph = stack.enter_context(old._base_isolated(
            fixture, "legacy" if arm == "legacy" else "source", recorder))
        stack.enter_context(patch.object(base.repository, "update_conversation_summary", fixture.update_summary))
        for name in ("acquire_summary_lease", "release_summary_lease", "put_summary_segment"):
            stack.enter_context(patch.object(base.context_repository, name, getattr(fixture, name)))
        if arm == "window_summary":
            stack.enter_context(patch.object(base.summarizer, "summarize_source_segment",
                                             bounded_summary.summarize_source_segment))

            async def store_bounded_segment(cid, uid, **kwargs):
                kwargs["prompt_version"] = bounded_summary.PROMPT_VERSION
                return await fixture.put_summary_segment(cid, uid, **kwargs)
            stack.enter_context(patch.object(base.context_repository, "put_summary_segment", store_bounded_segment))

        async def deny_write(state, tool_name, args):
            fixture.check(state.get("conversation_id"), state.get("user_id"))
            latest = max((row["id"] for row in fixture.rows if row["role"] == "user"), default=0)
            if latest > state.get("current_message_id", 0):
                return "当前请求已过期，本次操作未执行。"
            return "本轮隔离评测只允许查询，不授予写操作确认，本次操作未执行。"
        stack.enter_context(patch.object(base.context_runtime, "write_denial", deny_write))
        if arm != "legacy":
            def packet(state, query, stage="answer", budget=None):
                started = time.monotonic()
                inputs = None
                if state.get("memory_error"):
                    result = {"messages": [], "history": "", "blocked": True,
                              "reason": state["memory_error"], "manifest": {"stage": stage}}
                else:
                    inputs = {"snapshot": base.context_runtime.snapshot_from_state(state), "query": query,
                        "stage": stage, "policy": arm, "budget": budget,
                        "summaries": state.get("memory_segments") or []}
                    result = assemble_task_context(**inputs)
                saved_packet = deepcopy(result)
                if "messages" in saved_packet:
                    saved_packet["messages"] = [message_to_dict(m) for m in saved_packet["messages"]]
                projections.append({"coords": deepcopy(base.COORDS.get()), "stage": stage,
                    "packet": saved_packet, "manifest": deepcopy(result.get("manifest", {})),
                    "assembler_input": deepcopy(inputs), "assembly_seconds": time.monotonic() - started})
                write(recorder.out / "projections" / f"{arm}-{fixture.case['id']}.json", projections)
                return result
            stack.enter_context(patch.object(base.context_runtime, "packet", packet))
        else:
            original_history, original_agent = base.nodes._history_text, base.nodes._agent_messages

            def save_legacy(state, stage, history, messages, started, *, function, max_turns=6, tool_schemas=None):
                summary = state.get("summary") or ""
                manifest = {"stage": stage, "policy": "legacy",
                    "legacy_summary_visible": bool(summary and summary in history),
                    "legacy_summary_upto": state.get("summary_upto_msg_id", 0)}
                projections.append({"coords": deepcopy(base.COORDS.get()), "stage": stage,
                    "function": function,
                    "assembler_input": {"state": {k: ([message_to_dict(m) for m in v] if k == "messages" else deepcopy(v))
                                                   for k, v in state.items()},
                        "max_turns": max_turns, "context_window_max_tokens": base.settings.context_window_max_tokens,
                        "context_policy": "legacy", "tool_schemas": deepcopy(tool_schemas)},
                    "manifest": manifest, "packet": {"history": history,
                    "messages": [message_to_dict(m) for m in messages], "blocked": False, "manifest": manifest},
                    "assembly_seconds": time.monotonic() - started})
                write(recorder.out / "projections" / f"{arm}-{fixture.case['id']}.json", projections)

            def legacy_history(state, max_turns=6, *, stage="coref"):
                started = time.monotonic()
                value = original_history(state, max_turns, stage=stage)
                save_legacy(state, stage, value, [], started, function="history", max_turns=max_turns)
                return value

            def legacy_agent(state, *, tool_schemas=None):
                started = time.monotonic()
                value = original_agent(state, tool_schemas=tool_schemas)
                history = "\n".join(str(m.content) for m in value)
                save_legacy(state, "answer", history, value, started, function="answer", tool_schemas=tool_schemas)
                return value
            stack.enter_context(patch.object(base.nodes, "_history_text", legacy_history))
            stack.enter_context(patch.object(base.nodes, "_agent_messages", legacy_agent))
        original_schedule = base.summarizer.maybe_schedule_summary

        async def schedule(cid):
            started, before, call_before = time.monotonic(), len(fixture.segments), set(recorder.calls)
            error, task, admission = None, None, None
            allowed = arm == "legacy"
            if arm == "window_summary":
                plan = await base.summarizer.source_summary_plan(cid, fixture.uid)
                admission = summary_cooldown.decision(plan, fixture.rows)
                allowed = admission["allowed"]
                if allowed:
                    summary_cooldown.begin(plan, fixture.rows)
            if allowed:
                await original_schedule(cid)
                task = base.summarizer._running.get(cid)
                if task is not None:
                    try:
                        await task
                    except Exception as exc:
                        error = type(exc).__name__
                if arm == "window_summary":
                    if error or len(fixture.segments) == before:
                        error = error or "SummaryNotStored"
                        summary_cooldown.failed(fixture.rows)
                    else:
                        summary_cooldown.succeeded()
            maintenance.append({"coords": deepcopy(base.COORDS.get()), "triggered": task is not None,
                "error_type": error, "seconds": time.monotonic() - started,
                "summary_admission": admission,
                "summary_contract": bounded_summary.CONTRACT if arm == "window_summary" else None,
                "new_segments": deepcopy(fixture.segments[before:]),
                "legacy_summary": fixture.summary, "legacy_summary_upto": fixture.summary_upto,
                "call_ids": sorted(set(recorder.calls) - call_before),
                "policy": arm, "row_watermark": max((r["id"] for r in fixture.rows), default=0)})
            write(recorder.out / "maintenance" / f"{arm}-{fixture.case['id']}.json", maintenance)
        stack.enter_context(patch.object(base.summarizer, "maybe_schedule_summary", schedule))
        yield graph, maintenance


def full_preflight(case):
    from app.core.context_policy import estimate
    from app.core.context_state import build_snapshot
    from app.core.context_task import assemble_task_context
    fixture = Fixture(case)
    schema_tokens = estimate(json.dumps([spec.json_schema for spec in fixture.specs()], ensure_ascii=False))
    rows = [{"id": i, **r} for i, r in enumerate(case.get("initial_history", []), 1)]
    query = case["steps"][0]["query"] if case.get("steps") else "检查上下文"
    rows.append({"id": len(rows) + 1, "role": "user", "content": query})
    packet = assemble_task_context(build_snapshot(rows, fixture.cid), query, policy="full", stage="answer")
    history_tokens = estimate(packet["history"])
    allowance = 3500
    return {"history_estimate": history_tokens, "tool_argument_schema_estimate": schema_tokens,
            "system_and_three_turn_growth_allowance": allowance,
            "planning_estimate": history_tokens + schema_tokens + allowance,
            "within_planning_limit": history_tokens + schema_tokens + allowance <= SETTINGS["context_input_budget"],
            "packet_blocked": packet["blocked"],
            "note": "Full assembler history including source labels plus conservative overhead/growth allowance; not exact provider tokens. Every actual payload is checked and full history is never clipped."}


def prepare(out, arms=ARMS, case_ids=None, pilot=False, turn_timeout=180):
    out = Path(out).resolve()
    if (out / "manifest.json").exists():
        raise ValueError("existing manifest is immutable; use a new output directory")
    if not arms or any(arm not in ARMS for arm in arms):
        raise ValueError("unknown arm")
    data = dataset(pilot)
    if case_ids:
        chosen = set(case_ids)
        data["cases"] = [case for case in data["cases"] if case["id"] in chosen]
        if {case["id"] for case in data["cases"]} != chosen:
            raise ValueError("unknown case selection")
    names = set(base.SOURCES + ["scripts/verify_context_workflow_compact.py", "scripts/eval_context_unified.py",
        "scripts/context_unified_cases.py", "scripts/context_unified_summary.py",
        "app/core/context_task.py", "tests/test_context_unified.py"])
    names.update(str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "app").rglob("*.py"))
    names.update(str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "docs/experiments/context-redesign-20261027").glob("*.md"))
    names.update(str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "tests").glob("test_context_unified*.py"))
    for extra in ("scripts/analyze_context_unified.py", "scripts/audit_context_unified.py",
                  "tests/test_context_task.py", "tests/test_business_evidence.py"):
        if (ROOT / extra).exists():
            names.add(extra)
    sources = {}
    for name in sorted(names):
        source, target = ROOT / name, out / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        sources[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    write(out / "dataset.json", data)
    manifest = {"created_at": base.stamp(), "version": "context-unified-workflow-v1", "arms": list(arms),
        "dataset_sha256": digest(data), "sources": sources, "config": effective_config(),
        "turn_timeout_seconds": turn_timeout, "replicates": 1, "pilot": pilot,
        "group_order": "case-major; arm order rotated by case index; arms sequential (global hooks)",
        "full_preflight": {c["id"]: full_preflight(c) for c in data["cases"]},
        "scope": "real model and LangGraph/SSE; synthetic isolated DB, retrieval and dynamic business tools",
        "common_execution_fix": "fresh owned-order logistics evidence enabled for every arm",
        "common_write_policy": "all arms use identical owner/watermark checks and deny writes in this read-only test; attempted write proposals remain scored; no positive write authorization claim",
        "token_scope": "seed memory preparation plus three actual follow-ups and all maintenance/retries; excludes generation of scripted seed history and real RAG/business service costs",
        "initialization": "append seed dialog in chronological order; run the production maintenance check after every assistant message; wait for each triggered job",
        "summary": {"legacy": "original rolling summary, threshold 30 messages, retain 8 user turns",
                    "window_summary": "source segments; trigger 2400 estimated tokens, retain 8 user turns; explicit 300-character / 450-estimated-token limit; after failure wait for 2400 estimated tokens of newly arrived source; never semantically retry the same range",
                    "other_arms": "no model maintenance"},
        "recovery_tools": False, "retry_policy": "existing SDK/structured-output retry only; all attempts retained; no semantic reruns",
        "persistence_exception": "shared model pool may write technical SQLite metadata; customer and business databases are isolated",
        "frozen": True}
    write(out / "manifest.json", manifest)
    return manifest


async def seed_fixture(fixture, arm, recorder):
    token = base.COORDS.set({"arm": arm, "case_id": fixture.case["id"], "turn": 0})
    requests = []
    try:
        for row in fixture.seed:
            await fixture.append(fixture.cid, row["role"], row["content"])
            if row["role"] == "assistant":
                request_id = f"unified-{recorder.out.name}-{arm}-{fixture.case['id']}-seed-{len(fixture.rows)}"
                with base.call_context(request_id=request_id, priority="background"):
                    await base.summarizer.maybe_schedule_summary(fixture.cid)
                requests.append({"request_id": request_id, "pool_attempts": base._pool_attempts(request_id)})
        write(recorder.out / "seeds" / f"{arm}-{fixture.case['id']}.json", {
            "coords": deepcopy(base.COORDS.get()), "rows": fixture.rows, "summary": fixture.summary,
            "summary_upto": fixture.summary_upto, "segments": fixture.segments,
            "requests": requests,
            "call_ids": sorted(k for k, c in recorder.calls.items() if c["coords"] == base.COORDS.get())})
    finally:
        base.COORDS.reset(token)


def _record(path, value):
    write(path, {**value, "record_sha256": digest(value)})


async def run(out):
    out = Path(out).resolve()
    manifest, data = read(out / "manifest.json"), read(out / "dataset.json")
    if effective_config() != manifest["config"] or digest(data) != manifest["dataset_sha256"]:
        raise ValueError("configuration or frozen dataset changed")
    if urlsplit(base.settings.chat_base_url).hostname != "ai.gitee.com":
        raise ValueError("this experiment requires the configured Gitee endpoint")
    for name, sha in manifest["sources"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha:
            raise ValueError("source changed after freeze: " + name)
    if (out / "calls").exists() or (out / "turns").exists() or (out / "seeds").exists():
        raise ValueError("existing live output cannot be reused or silently rerun")
    lock = out / "writer.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(base.stamp())
    recorder = base.Recorder(out)
    try:
        for index, case in enumerate(data["cases"]):
            arms = manifest["arms"]
            offset = index % len(arms)
            for arm in arms[offset:] + arms[:offset]:
                fixture = Fixture(case)
                with isolated_workflow(fixture, arm, recorder) as (graph, maintenance):
                    await seed_fixture(fixture, arm, recorder)
                    prior_failure = None
                    for number, step in enumerate(case["steps"], 1):
                        coords = {"arm": arm, "case_id": case["id"], "turn": number}
                        token = base.COORDS.set(coords)
                        before_calls, before_tools = set(recorder.calls), len(fixture.tool_calls)
                        before_retrieval, before_maintenance = len(fixture.retrievals), len(maintenance)
                        events, first_delta, problem, interrupt = [], None, None, None
                        started, foreground_complete = time.monotonic(), None
                        request_id = f"unified-{out.name}-{arm}-{case['id']}-{number}"

                        async def consume():
                            nonlocal first_delta, interrupt, foreground_complete
                            async for event in base.runtime.stream_turn(fixture.uid, step["query"], fixture.cid if fixture.rows else None):
                                events.append(event)
                                if event.get("type") == "delta" and event.get("text") and first_delta is None:
                                    first_delta = time.monotonic() - started
                                if event.get("type") == "interrupt":
                                    interrupt = event
                                if event.get("type") in {"done", "final"}:
                                    foreground_complete = time.monotonic() - started
                            if interrupt and (interrupt.get("type") == "confirm_ticket" or interrupt.get("kind") == "confirm_ticket"):
                                async for event in base.runtime.stream_resume(fixture.cid, {"confirmed": False},
                                    user_id=fixture.uid, interrupt_id=interrupt["interrupt_id"]):
                                    events.append(event)
                            if foreground_complete is None:
                                foreground_complete = time.monotonic() - started

                        try:
                            if prior_failure:
                                problem = {"kind": "dependency_blocked", "caused_by": prior_failure}
                            else:
                                with base.call_context(request_id=request_id, priority="background"):
                                    await asyncio.wait_for(consume(), timeout=manifest["turn_timeout_seconds"])
                        except Exception as exc:
                            problem = {"kind": base._error_kind(exc), "type": type(exc).__name__}
                            prior_failure = {"turn": number, **problem}
                        try:
                            state = (await graph.aget_state({"configurable": {"thread_id": str(fixture.cid)}})).values
                            proposed, seen_current = [], False
                            for message in state.get("messages", []):
                                if str(getattr(message, "id", "")) == f"db-{state.get('current_message_id', 0)}":
                                    seen_current = True
                                if seen_current:
                                    proposed.extend(deepcopy(getattr(message, "tool_calls", [])))
                            serial_state = {k: ([message_to_dict(m) for m in v] if k == "messages" else v)
                                            for k, v in state.items()}
                            record = {"coords": coords, "query": step["query"], "expected": step["expected"],
                                "status": "failed" if problem else "completed", "error": problem,
                                "answer": base.nodes.resolve_answer(state) if not problem else "",
                                "interrupt": interrupt, "sse_events": events, "state": serial_state,
                                "actual_dialog_rows": deepcopy(fixture.rows), "proposed_tools": proposed,
                                "executed_tools": deepcopy(fixture.tool_calls[before_tools:]),
                                "retrieval": deepcopy(fixture.retrievals[before_retrieval:]),
                                "call_ids": sorted(set(recorder.calls) - before_calls), "request_id": request_id,
                                "pool_attempts": base._pool_attempts(request_id),
                                "first_visible_sse_delta_seconds": first_delta,
                                "foreground_complete_seconds": foreground_complete,
                                "end_to_end_seconds": time.monotonic() - started,
                                "maintenance_seconds": sum(m["seconds"] for m in maintenance[before_maintenance:]),
                                "summary": fixture.summary, "summary_upto": fixture.summary_upto,
                                "segments": deepcopy(fixture.segments), "receipts": deepcopy(fixture.receipts)}
                            record["score"] = cases_module().score_turn(record)
                            _record(out / "turns" / f"{arm}-{case['id']}-{number}.json", record)
                            print(json.dumps({**coords, "status": record["status"],
                                "pass": record["score"].get("task_rule_pass", False),
                                "seconds": round(record["end_to_end_seconds"], 2)}, ensure_ascii=False), flush=True)
                        finally:
                            base.COORDS.reset(token)
                    await base.summarizer.close_tasks()
    finally:
        lock.unlink(missing_ok=True)
        await base.summarizer.close_tasks()
        from app.core.model_transport import close_clients
        await close_clients()
    return audit(out)


def audit(out):
    out = Path(out).resolve()
    if (out / "writer.lock").exists():
        raise ValueError("writer is active; audit only after it exits")
    manifest, data = read(out / "manifest.json"), read(out / "dataset.json")
    score_path = out / "source_snapshot/scripts/context_unified_cases.py"
    spec = importlib.util.spec_from_file_location("_frozen_context_unified_cases", score_path)
    frozen_cases = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen_cases)
    issues = []
    if digest(data) != manifest["dataset_sha256"]:
        issues.append("dataset hash mismatch")
    for name, sha in manifest["sources"].items():
        path = out / "source_snapshot" / name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            issues.append("source snapshot mismatch: " + name)
    records = {p.stem: read(p) for p in (out / "turns").glob("*.json")}
    calls = {p.stem: read(p) for p in (out / "calls").glob("*.json")}
    seeds = {p.stem: read(p) for p in (out / "seeds").glob("*.json")}
    expected = {f"{arm}-{case['id']}-{i}" for arm in manifest["arms"] for case in data["cases"]
                for i in range(1, len(case["steps"]) + 1)}
    for key, value in [*records.items(), *calls.items()]:
        if value.get("record_sha256") != digest({k: v for k, v in value.items() if k != "record_sha256"}):
            issues.append("record hash mismatch: " + key)
    for key, record in records.items():
        if frozen_cases.score_turn(record) != record["score"]:
            issues.append("score mismatch: " + key)
        coords = record["coords"]
        case = next(c for c in data["cases"] if c["id"] == coords["case_id"])
        step = case["steps"][coords["turn"] - 1]
        if step["query"] != record["query"] or step["expected"] != record["expected"]:
            issues.append("frozen step mismatch: " + key)
        for call_id in record["call_ids"]:
            if call_id not in calls or calls[call_id]["coords"] != coords:
                issues.append("missing or cross-turn call: " + key)
        previous = records.get(f"{coords['arm']}-{coords['case_id']}-{coords['turn'] - 1}")
        rows = record.get("actual_dialog_rows", [])
        if previous and rows[:len(previous["actual_dialog_rows"])] != previous["actual_dialog_rows"]:
            issues.append("history continuity mismatch: " + key)
    assigned = [c for r in [*records.values(), *seeds.values()] for c in r["call_ids"]]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(calls):
        issues.append("model call assignment mismatch (seed, follow-up, maintenance)")
    for key, call in calls.items():
        if digest(call["input"]) != call["input_sha256"]:
            issues.append("input hash mismatch: " + key)
        if call.get("wire_payload") is not None and digest(call["wire_payload"]) != call.get("wire_sha256"):
            issues.append("wire hash mismatch: " + key)
        if call.get("status") == "ok" and base.structured_response_check(call) != call.get("structured_response_check"):
            issues.append("protocol diagnosis mismatch: " + key)
    groups = {}
    for arm in manifest["arms"]:
        rs = [r for r in records.values() if r["coords"]["arm"] == arm]
        cs = [c for c in calls.values() if c["coords"]["arm"] == arm]
        used = lambda part, field: sum((c.get("usage") or {}).get(field, 0) for c in part)
        ms = [c for c in cs if c.get("purpose") == "conversation_summary"]
        groups[arm] = {"turns": len(rs), "planned_turns": sum(len(c["steps"]) for c in data["cases"]),
            "task_rule_pass": sum(bool(r["score"].get("task_rule_pass")) for r in rs),
            "input_tokens": used(cs, "input_tokens"), "output_tokens": used(cs, "output_tokens"),
            "total_tokens": used(cs, "input_tokens") + used(cs, "output_tokens"),
            "maintenance_model_calls": len(ms), "maintenance_input_tokens": used(ms, "input_tokens"),
            "maintenance_output_tokens": used(ms, "output_tokens"), "model_calls": len(cs),
            "unknown_usage_calls": sum(not c.get("usage") for c in cs),
            "interface_error_calls": sum(c.get("status") == "interface_error" for c in cs),
            "protocol_invalid_calls": sum(c.get("structured_response_check", {}).get("status") == "invalid" for c in cs),
            "unfinished_calls": sum(c.get("status") == "running" for c in cs),
            "failed_turns": sum(r["status"] != "completed" for r in rs),
            "end_to_end_p50_seconds": statistics.median([r["end_to_end_seconds"] for r in rs]) if rs else None}
    missing = sorted(expected - set(records))
    expected_seed = {f"{arm}-{c['id']}" for arm in manifest["arms"] for c in data["cases"]}
    missing_seed = sorted(expected_seed - set(seeds))
    missing_maintenance = sorted(key for key in expected_seed if not (out / "maintenance" / f"{key}.json").exists())
    if set(records) - expected:
        issues.append("unexpected turn records")
    complete = not (missing or missing_seed or missing_maintenance or issues or any(g["unfinished_calls"] for g in groups.values()))
    result = {"audited_at": base.stamp(), "complete": complete, "audit_passed": not issues,
        "missing": missing, "missing_seed": missing_seed, "missing_maintenance": missing_maintenance,
        "issues": issues, "groups": groups, "planned_turns": len(expected),
        "scope": manifest["scope"], "token_scope": manifest["token_scope"],
        "limits": ["Single replicate, internally authored synthetic challenge set; not independent external blind testing.",
            "Same-family short/long trajectories are correlated. Rule scores are not human-labelled overall accuracy.",
            "All seed maintenance and follow-up model costs included; scripted seed generation and real RAG costs excluded.",
            "Unknown usage is unknown cost. All failures remain in the planned denominator.",
            "Legacy uses its original memory estimator; window/window_summary/task share stage ceilings.",
            "Fresh business evidence is a shared execution fix, not a task-memory-only advantage."]}
    write(out / "audited_report.json", result)
    lines = ["# 统一客服上下文真实 Workflow 对照", "", f"记录完整：{complete}；离线复算：{not issues}。", "",
        "真实模型与 Workflow，合成历史、知识和动态工具夹具；非线上流量准确率。", "",
        "|组别|任务通过/计划回合|输入 Token|输出 Token|总 Token|摘要调用|未知 usage 调用|运行失败回合|",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, g in groups.items():
        lines.append(f"|{arm}|{g['task_rule_pass']}/{g['planned_turns']}|{g['input_tokens']}|{g['output_tokens']}|{g['total_tokens']}|{g['maintenance_model_calls']}|{g['unknown_usage_calls']}|{g['failed_turns']}|")
    lines += ["", "成本含预置历史的记忆维护、三次追问及重试；不含脚本原始历史生成和真实 RAG 服务费用。",
        "五组共用实时业务查询修复。旧策略是系统锚点，同预算组件比较以 window/window_summary/task 为准。",
        "单次、内部合成小样本；持平只表示本轮未观察到新增失败，不证明普遍非劣。",
        "", f"缺失回合：{len(missing)}；缺失种子：{len(missing_seed)}；缺失维护记录：{len(missing_maintenance)}；审计问题：{issues}"]
    (out / "AUDITED_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--turn-timeout", type=int, default=180)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    result = audit(args.output) if args.audit else (asyncio.run(run(args.output)) if args.run else
        prepare(args.output, tuple(args.arms), args.cases, args.pilot, args.turn_timeout))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
