"""Frozen live-model workflow verification with isolated synthetic business state.

Default invocation only prepares files. --run calls the configured Gitee model;
all conversations, retrieval, tools and DB writes are in-memory fixtures. This
is not a production RAG accuracy benchmark. --audit is entirely offline.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import statistics
import time
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlsplit

import jsonschema
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import message_to_dict
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from app.config import settings
from app.core import llm, coref, intent, query_understanding, selfcheck, context_runtime, summarizer
from app.core.model_pool import PURPOSE, call_context
from app.db import base as db, repository, context_repository
from app.graph import nodes, runtime
from app.graph.build import build_graph
from app.tools import business, registry

ROOT = Path(__file__).resolve().parents[1]
COORDS = ContextVar("context_workflow_coordinates", default=None)
OFF = {"chat_template_kwargs": {"thinking": False, "enable_thinking": False}}
SOURCES = ["scripts/verify_context_workflow.py", "app/config.py", "app/core/context_state.py",
           "app/core/context_runtime.py", "app/core/context_policy.py", "app/core/memory.py",
           "app/core/summarizer.py", "app/core/coref.py", "app/core/intent.py",
           "app/core/query_understanding.py", "app/core/selfcheck.py", "app/core/llm.py",
           "app/core/prompts.py", "app/core/model_pool.py", "app/core/model_transport.py",
           "app/graph/nodes.py", "app/graph/runtime.py", "app/graph/state.py",
           "app/graph/build.py", "app/graph/routing.py", "app/tools/engine.py",
           "app/tools/context.py", "app/core/confidence.py", "app/tools/registry.py"]
PRODUCTS = {
    "MH-W40": "MH-W40饮水机清洁前先断电，用软布清洁外壳，电机不得浸水。",
    "MH-FD10": "MH-FD10喂食器先断电，拆下食盆清洗并完全晾干，机身用软布擦拭。",
    "MH-CAM1": "MH-CAM1摄像头先断电，用镜头布轻擦镜头，避免液体进入接口。",
    "MH-LP100": "MH-LP100猫砂盆先断电，清空砂仓并拆洗可拆卸部件，主机不得浸水。",
}
ORDER_IDS = tuple(str(x) for x in range(710001, 710009))


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), default=str).encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dataset():
    def step(query, kind, target=""):
        return {"query": query, "expected": {"kind": kind, "target": target}}
    return {"version": "workflow-synthetic-v1", "synthetic": True, "cases": [
        {"id": "product_update", "steps": [
            step("我买的是 MH-W40 饮水机，请告诉我清洁方法。", "care", "MH-W40"),
            step("更正，型号是 MH-FD10 喂食器，请查它的清洁方法。", "care", "MH-FD10"),
            step("刚更正的这款，清洗时机身能怎么处理？", "care", "MH-FD10")]},
        {"id": "order_update", "steps": [
            step("请查询订单710001的物流。", "logistics", "710001"),
            step("710001报错，应为710002，请重新查正确订单的物流。", "logistics", "710002"),
            step("按最后更正的那笔再查一下物流。", "logistics", "710002")]},
        {"id": "unresolved_order", "steps": [
            step("我有710003和710004两笔订单，目前还没有确定选哪笔，先不要查询。", "no_action"),
            step("查那笔订单的物流。我仍未指定哪一笔，你先问清楚。", "clarify"),
            step("现在只查订单710004的物流。", "logistics", "710004")]},
        {"id": "no_write_scope", "steps": [
            step("现在处理订单710005。先别提交退款，只查一下物流。", "logistics", "710005"),
            step("换一笔，这次只查订单710006的物流，不要提交任何申请。", "logistics", "710006"),
            step("回到订单710005，继续查物流，仍然不要创建工单。", "logistics", "710005")]},
        {"id": "earlier_product", "steps": [
            step("办公室买的是 MH-CAM1 摄像头，请查它怎么清洁。", "care", "MH-CAM1"),
            step("家里另外买了 MH-LP100 猫砂盆，现在查猫砂盆清洁方法。", "care", "MH-LP100"),
            step("回到最开始办公室的那款，它的镜头怎么清洁？", "care", "MH-CAM1")]},
        {"id": "missing_knowledge", "steps": [
            step("尚未上市的 MH-NEW99 应该怎么清洗？没有资料就明确告诉我。", "refuse", "MH-NEW99"),
            step("那先查有资料的 MH-W40 饮水机清洁方法。", "care", "MH-W40"),
            step("之前尚未上市、没有资料的那款，可以确认清洁方法了吗？", "refuse", "MH-NEW99")]},
    ]}


def config():
    # Explicit allow-list: never serialize Settings, keys, headers or credentials.
    return {k: getattr(settings, k) for k in (
        "chat_model", "chat_base_url", "context_state_budget", "context_input_budget",
        "context_output_budget", "context_coref_budget", "context_intent_budget",
        "context_recall", "context_window_max_tokens", "context_window_turns",
        "summary_trigger_messages", "context_summary_trigger_tokens", "model_pool_enabled",
        "max_agent_steps")}


def prepare(out, arms=("legacy", "source"), case_ids=None, turn_timeout=180):
    if (out / "manifest.json").exists():
        raise ValueError("existing manifest is immutable; use a new output directory")
    data = dataset()
    if case_ids:
        chosen = set(case_ids)
        data["cases"] = [c for c in data["cases"] if c["id"] in chosen]
        if {c["id"] for c in data["cases"]} != chosen:
            raise ValueError("unknown case selection")
    if not arms or any(arm not in ("legacy", "source") for arm in arms):
        raise ValueError("unknown arm")
    out.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name in SOURCES:
        source, target = ROOT / name, out / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        sources[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    write(out / "dataset.json", data)
    manifest = {"created_at": stamp(), "version": "workflow-verification-v1", "arms": list(arms),
                "dataset_sha256": digest(data), "sources": sources, "config": config(),
                "thinking": OFF, "turn_timeout_seconds": turn_timeout,
                "scope": "real LangGraph and Gitee model; synthetic in-memory DB, business and retrieval",
                "persistence_exception": "the existing shared model-call pool may write technical SQLite admission/attempt metadata; customer DB and business writes are isolated",
                "background_summary": "real scheduler; three-turn trajectories normally do not trigger maintenance",
                "recovery_tools": False, "group_order": "sequential", "frozen": True}
    write(out / "manifest.json", manifest)
    return manifest


def _error_kind(error):
    name = type(error).__name__
    if name == "ContextInputLimit":
        return "budget_blocked"
    if "Parser" in name or name in {"ValidationError", "JSONDecodeError"}:
        return "protocol_error"
    if any(word in name for word in ("Timeout", "Connection", "HTTP", "API", "PoolBusy", "RateLimit")):
        return "interface_error"
    return "runtime_error"


def structured_response_check(item):
    """Diagnose forced function output without hiding successful transport calls."""
    payload = item.get("wire_payload") or item["input"]["parameters"]
    choice = payload.get("tool_choice")
    name = ((choice.get("function") or {}).get("name") if isinstance(choice, dict) else None)
    if not name:
        return {"status": "not_applicable"}
    schema = next((t["function"].get("parameters") for t in payload.get("tools", [])
                   if t.get("function", {}).get("name") == name), None)
    candidates = [t for out in item.get("outputs", []) for t in out.get("data", {}).get("tool_calls", [])
                  if t.get("name") == name]
    if not candidates:
        return {"status": "invalid", "reason": "missing_expected_tool_call", "tool": name}
    try:
        if schema:
            jsonschema.validate(candidates[0].get("args"), schema)
    except jsonschema.ValidationError:
        return {"status": "invalid", "reason": "schema_mismatch", "tool": name}
    return {"status": "valid", "tool": name}


class Recorder(AsyncCallbackHandler):
    """Capture all LangChain model starts/ends, including production retries."""
    raise_error = True
    run_inline = True

    def __init__(self, out):
        self.out, self.calls, self.errors = out, {}, []

    def _save(self, key):
        item = self.calls[key]
        write(self.out / "calls" / f"{key}.json", {**item, "record_sha256": digest(item)})

    async def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        key = str(run_id)
        params = kwargs.get("invocation_params") or {}
        allowed = {k: params[k] for k in ("model_name", "model", "temperature", "max_tokens",
                    "streaming", "tools", "response_format", "tool_choice", "extra_body", "reasoning_effort") if k in params}
        inp = {"messages": [[message_to_dict(m) for m in batch] for batch in messages], "parameters": allowed}
        self.calls[key] = {"id": key, "coords": deepcopy(COORDS.get()), "purpose": PURPOSE.get(),
                           "input": inp, "input_sha256": digest(inp), "status": "running",
                           "started_at": stamp(), "started_clock": time.monotonic()}
        self._save(key)

    def record_wire_payload(self, payload):
        active = [c for c in self.calls.values() if c["status"] == "running" and c["coords"] == COORDS.get()]
        if active:
            item = active[-1]
            # OpenAI request payload has no access token; still whitelist fields.
            wire = {k: payload[k] for k in ("model", "messages", "tools", "tool_choice", "response_format",
                    "temperature", "max_tokens", "max_completion_tokens", "stream", "stream_options",
                    "extra_body", "reasoning_effort") if k in payload}
            item["wire_payload"] = deepcopy(wire)
            item["wire_sha256"] = digest(wire)
            self._save(item["id"])

    async def on_llm_end(self, response, *, run_id, **kwargs):
        item = self.calls[str(run_id)]
        outputs, usage = [], None
        for batch in response.generations:
            for generation in batch:
                message = getattr(generation, "message", None)
                outputs.append(message_to_dict(message) if message is not None else {"text": generation.text})
                if message is not None and getattr(message, "usage_metadata", None):
                    usage = deepcopy(message.usage_metadata)
        item.update(status="ok", outputs=outputs, usage=usage,
                    seconds=time.monotonic() - item.pop("started_clock"), finished_at=stamp())
        item["structured_response_check"] = structured_response_check(item)
        self._save(str(run_id))

    async def on_llm_error(self, error, *, run_id, **kwargs):
        item = self.calls.get(str(run_id))
        if item:
            item.update(status=_error_kind(error), error_type=type(error).__name__,
                        seconds=time.monotonic() - item.pop("started_clock"), finished_at=stamp())
            self._save(str(run_id))

    async def on_chain_error(self, error, *, run_id, **kwargs):
        kind = _error_kind(error)
        if kind in ("protocol_error", "budget_blocked"):
            record = {"coords": deepcopy(COORDS.get()), "purpose": PURPOSE.get(),
                      "kind": kind, "error_type": type(error).__name__, "run_id": str(run_id)}
            self.errors.append(record)
            write(self.out / "chain_errors.json", self.errors)


class Fixture:
    """Only data from these pure synthetic fixtures can reach the real model."""
    def __init__(self):
        self.rows, self.retrievals, self.tool_calls, self.audits = [], [], [], []
        self.snapshots, self.receipts, self.segments = {}, {}, []
        self.cid, self.uid = 900001, "synthetic-workflow-user"
        self.summary, self.summary_upto = "", 0

    def check(self, cid, uid=None):
        if cid != self.cid or (uid is not None and uid != self.uid):
            raise PermissionError("synthetic conversation scope mismatch")

    async def create(self, uid):
        self.check(self.cid, uid)
        return self.cid

    async def conversation(self, cid):
        self.check(cid)
        return SimpleNamespace(id=cid, user_id=self.uid, summary=self.summary, summary_upto_msg_id=self.summary_upto)

    async def append(self, cid, role, content=None, **kwargs):
        self.check(cid)
        mid = len(self.rows) + 1
        self.rows.append({"id": mid, "conversation_id": cid, "role": role, "content": content or ""})
        return mid

    async def dialog(self, cid):
        self.check(cid)
        return [SimpleNamespace(**row) for row in self.rows]

    async def count(self, cid, after):
        self.check(cid)
        return sum(row["id"] > (after or 0) for row in self.rows)

    async def load(self, cid, uid):
        self.check(cid, uid)
        return deepcopy(self.snapshots.get(cid))

    async def save(self, cid, uid, snapshot, *, watermark, expected_version):
        self.check(cid, uid)
        prior = self.snapshots.get(cid)
        version = prior["version"] if prior else 0
        if version != expected_version or (prior and prior["watermark"] >= watermark):
            return {"saved": False, "version": version}
        self.snapshots[cid] = {"version": version + 1, "watermark": watermark, "snapshot": deepcopy(snapshot)}
        return {"saved": True, "version": version + 1}

    async def list_segments(self, cid, uid, **kwargs):
        self.check(cid, uid)
        return deepcopy(self.segments)

    async def source(self, cid, uid, message_id, start, end):
        self.check(cid, uid)
        row = next(r for r in self.rows if r["id"] == message_id)
        text = row["content"]
        if start < 0 or start > len(text) or end < start:
            raise ValueError("invalid source range")
        return {"conversation_id": cid, "message_id": message_id, "start": start, "end": min(end, len(text)),
                "total_chars": len(text), "text": text[start:end], "role": row["role"],
                "content_hash": hashlib.sha256(text.encode()).hexdigest()}

    async def search_sources(self, cid, uid, query, *, upto_msg_id=None, limit=5):
        self.check(cid, uid)
        from app.core.context_policy import terms
        scored = sorted(((len(terms(query) & terms(r["content"])), r) for r in self.rows
                         if upto_msg_id is None or r["id"] <= upto_msg_id), key=lambda x: (-x[0], -x[1]["id"]))
        return [await self.source(cid, uid, r["id"], 0, len(r["content"])) for score, r in scored[:limit] if score]

    async def receipt(self, cid, uid, **kwargs):
        self.check(cid, uid)
        rid = "fixture-receipt-" + str(len(self.receipts) + 1)
        item = {"receipt_id": rid, **deepcopy(kwargs), "observed_at": stamp(),
                "result_hash": digest(kwargs.get("raw_result"))}
        self.receipts[rid] = item
        return item

    async def read_receipt(self, cid, uid, receipt_id, *, upto_msg_id=None):
        self.check(cid, uid)
        item = self.receipts[receipt_id]
        if upto_msg_id is not None and (item.get("message_watermark") or 0) > upto_msg_id:
            raise PermissionError("receipt is from future turn")
        return deepcopy(item)

    async def search(self, query, **kwargs):
        requested = list(dict.fromkeys(re.findall(r"MH-[A-Za-z0-9]+", query)))
        hits = [{"id": list(PRODUCTS).index(sku) + 1, "question": f"{sku}怎么清洁？",
                 "answer": PRODUCTS[sku], "rerank_score": .99,
                 "section_path": f"合成夹具/{sku}", "content_type": "faq"}
                for sku in requested if sku in PRODUCTS]
        if not requested and re.search(r"退货|退款|售后", query):
            hits = [{"id": 99, "question": "合成订单退款政策", "answer": "退款须核验订单归属并经用户确认，禁止将查询视为提交。",
                     "rerank_score": .99, "section_path": "合成夹具/政策", "content_type": "policy"}]
        self.retrievals.append({"query": query, "requested_skus": requested,
                                "returned_ids": [h["id"] for h in hits], "coords": deepcopy(COORDS.get())})
        return hits

    async def audit(self, **kwargs):
        self.audits.append(deepcopy(kwargs))

    async def low_confidence(self, *args, **kwargs):
        return None

    def order(self, order_id):
        if str(order_id) not in ORDER_IDS:
            return {"error": "合成订单不存在"}
        return {"order_id": str(order_id), "status": "已发货", "tracking_no": "SF" + str(order_id) + "88",
                "last_event": "2026-10-27 09:00 到达杭州分拨中心", "product": "合成测试商品", "amount": 199}

    def specs(self):
        fixture = self
        @tool
        async def query_logistics(order_id: str) -> dict:
            """查询指定订单的物流，必须先确定订单号。"""
            fixture.tool_calls.append({"name": "query_logistics", "args": {"order_id": order_id}, "coords": deepcopy(COORDS.get())})
            return fixture.order(order_id)
        @tool
        async def query_order(order_id: str) -> dict:
            """查询指定订单详情，必须先确定订单号。"""
            fixture.tool_calls.append({"name": "query_order", "args": {"order_id": order_id}, "coords": deepcopy(COORDS.get())})
            return fixture.order(order_id)
        @tool
        async def query_faq(query: str) -> dict:
            """按具体型号和问题查询商品知识，未查到就明确说明。"""
            fixture.tool_calls.append({"name": "query_faq", "args": {"query": query}, "coords": deepcopy(COORDS.get())})
            return {"hits": await fixture.search(query)}
        @tool
        async def create_ticket(description: str, ticket_type: str = "咨询") -> dict:
            """创建客服工单，必须先展示预览并等待用户确认。"""
            fixture.tool_calls.append({"name": "create_ticket", "args": {"description": description, "ticket_type": ticket_type}, "coords": deepcopy(COORDS.get())})
            return {"ticket_no": "SYNTHETIC-ONLY", "synthetic": True}
        return [registry.spec_from_langchain_tool(t, source="builtin", max_retries=0)
                for t in (query_logistics, query_order, query_faq, create_ticket)]


@contextmanager
def isolated_workflow(fixture, arm, recorder):
    """Patch only this CLI process; never the already-running application."""
    with ExitStack() as stack:
        def replace(obj, name, value):
            stack.enter_context(patch.object(obj, name, value))
        def no_db(*args, **kwargs):
            raise AssertionError("live database access is forbidden in workflow verification")
        replace(db, "async_session", no_db)
        for name, value in {"context_policy": arm, "context_recovery_tools_enabled": False,
                            "chat_failover_enabled": False, "intent_model": "", "summary_model": "",
                            "langfuse_public_key": ""}.items():
            replace(settings, name, value)
        for name, function in {"get_conversation": fixture.conversation, "create_conversation": fixture.create,
                "append_message": fixture.append, "list_dialog_messages": fixture.dialog,
                "count_messages_after": fixture.count, "insert_tool_audit": fixture.audit,
                "insert_low_confidence": fixture.low_confidence}.items():
            replace(repository, name, function)
        for name, function in {"load_snapshot": fixture.load, "save_snapshot": fixture.save,
                "list_summary_segments": fixture.list_segments, "read_source": fixture.source,
                "search_sources": fixture.search_sources, "record_receipt": fixture.receipt,
                "read_receipt": fixture.read_receipt}.items():
            replace(context_repository, name, function)
        replace(nodes.retrieval, "search_knowledge", fixture.search)
        replace(nodes, "get_evidence_confidence_threshold", lambda: .26)
        replace(business, "order_snapshot", fixture.order)
        replace(business, "owns_order", lambda uid, oid: uid == fixture.uid and str(oid) in ORDER_IDS)
        replace(business, "list_user_orders", lambda uid: [fixture.order(oid) for oid in ORDER_IDS] if uid == fixture.uid else [])
        specs = fixture.specs()
        async def get_specs():
            return specs
        replace(registry, "get_all_specs", get_specs)
        graph = build_graph(InMemorySaver())
        replace(runtime, "get_graph", lambda: graph)
        factory = llm.get_chat_model
        def model(**kwargs):
            instance = factory(**kwargs)
            return instance.model_copy(update={"extra_body": deepcopy(OFF), "reasoning_effort": None,
                                               "max_tokens": settings.context_output_budget,
                                               "callbacks": [recorder]})
        for module in (llm, coref, intent, query_understanding, selfcheck, nodes, summarizer):
            if hasattr(module, "get_chat_model"):
                replace(module, "get_chat_model", model)
        original_payload = llm._CumulativeUsageChatOpenAI._get_request_payload
        def payload(instance, *args, **kwargs):
            value = original_payload(instance, *args, **kwargs)
            recorder.record_wire_payload(value)
            # Shared full-request ceiling also applies to the legacy control.
            context_runtime.check_payload(value)
            return value
        replace(llm._CumulativeUsageChatOpenAI, "_get_request_payload", payload)
        from app.core import model_transport
        original_outbound = model_transport.outbound
        def outbound(request, provider, channel, body, policy, remaining):
            if channel != "chat" or urlsplit(provider.base_url).hostname != "ai.gitee.com":
                raise ValueError("workflow verification requires the Gitee chat endpoint")
            if provider.model and provider.model != settings.chat_model:
                raise ValueError("workflow verification requires the frozen chat model")
            return original_outbound(request, provider, channel, body, policy, remaining)
        replace(model_transport, "outbound", outbound)
        yield graph


def score_turn(record):
    expected = record["expected"]
    invocations = record.get("executed_tools", [])
    proposals = record.get("proposed_tools", [])
    queries = record.get("retrieval", [])
    no_write = not any(t["name"] in {"create_ticket", "submit_refund"} for t in proposals + invocations)
    kind, target = expected["kind"], expected["target"]
    if kind == "logistics":
        checks = {"targeted_tool": any(t["name"] == "query_logistics" and str(t["args"].get("order_id")) == target for t in invocations),
                  "no_wrong_order_tool": not any(t["name"] in {"query_logistics", "query_order"} and str(t["args"].get("order_id")) != target for t in invocations)}
    elif kind == "care":
        checks = {"retrieval_target_rule": any(target in q.get("requested_skus", []) for q in queries),
                  "evidence_gate_passed": bool(record.get("state", {}).get("evidence_strong")) or bool(invocations),
                  "response_present": bool(record.get("answer"))}
    elif kind == "refuse":
        state = record.get("state", {})
        checks = {"retrieval_target_rule": any(target in q.get("requested_skus", []) for q in queries),
                  "evidence_rejected": not state.get("evidence_strong", False) and state.get("fallback_source") in ("retrieval_low_conf", "self_check")}
    else:
        checks = {"no_order_query": not any(t["name"] in {"query_order", "query_logistics"} for t in invocations),
                  "response_present": bool(record.get("answer"))}
        if kind == "clarify":
            checks["clarification_rule"] = bool(re.search(r"哪[一笔个]|订单号|明确|具体|选择|指定|确认.*订单", record.get("answer", "")))
    checks["no_write_proposal_or_execution"] = no_write
    checks["run_completed"] = record.get("status") == "completed" and not record.get("interrupt")
    return {"checks": checks, "task_rule_pass": all(checks.values()),
            "scope": "tool-argument and retrieval-target rules; not human semantic accuracy or real RAG recall"}


def _pool_attempts(request_id):
    path = Path(settings.model_pool_db_path).resolve()
    if not settings.model_pool_enabled or not path.exists():
        return []
    try:
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute(
                "SELECT id,request_id,channel,model,priority,attempt,started,finished,queue_ms,outcome,http_status FROM attempts WHERE request_id=? ORDER BY started", (request_id,))]
    except sqlite3.Error:
        return None


def turn_failure_kinds(record, calls):
    kinds = set()
    if record.get("error"):
        kinds.add(record["error"]["kind"])
    if record.get("state", {}).get("memory_error"):
        kinds.add("context_blocked")
    for cid in record.get("call_ids", []):
        call = calls.get(cid, {})
        if call.get("status") != "ok":
            kinds.add(call.get("status", "missing_call"))
        if call.get("structured_response_check", {}).get("status") == "invalid":
            kinds.add("protocol_error")
    return sorted(kinds)


def percentile95(values):
    if not values:
        return None
    ordered = sorted(values)
    position = .95 * (len(ordered) - 1)
    left = int(position)
    return ordered[left] + (ordered[min(left + 1, len(ordered) - 1)] - ordered[left]) * (position - left)


async def run(out):
    manifest, data = read(out / "manifest.json"), read(out / "dataset.json")
    if config() != manifest["config"] or digest(data) != manifest["dataset_sha256"]:
        raise ValueError("configuration or frozen dataset changed")
    if urlsplit(settings.chat_base_url).hostname != "ai.gitee.com":
        raise ValueError("this prepared verification is restricted to the configured Gitee endpoint")
    for name, sha in manifest["sources"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha:
            raise ValueError("source changed after prepare: " + name)
    if (out / "calls").exists() or (out / "turns").exists():
        raise ValueError("existing live attempt is preserved; audit it or prepare a new run, never silently rerun")
    lock = out / "writer.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(stamp())
    recorder = Recorder(out)
    try:
        for arm in manifest["arms"]:
            for case in data["cases"]:
                fixture = Fixture()
                with isolated_workflow(fixture, arm, recorder) as graph:
                    for number, step in enumerate(case["steps"], 1):
                        coords = {"arm": arm, "case_id": case["id"], "turn": number}
                        token = COORDS.set(coords)
                        request_id = f"workflow-{out.name}-{arm}-{case['id']}-{number}"
                        before_tools, before_retrieval = len(fixture.tool_calls), len(fixture.retrievals)
                        call_before = set(recorder.calls)
                        events, first_delta, problem, interrupt = [], None, None, None
                        started = time.monotonic()
                        async def consume():
                            nonlocal first_delta, interrupt
                            async for event in runtime.stream_turn(fixture.uid, step["query"], fixture.cid if fixture.rows else None):
                                events.append(event)
                                if event.get("type") == "delta" and event.get("text") and first_delta is None:
                                    first_delta = time.monotonic() - started
                                if event.get("type") == "interrupt":
                                    interrupt = event
                            if interrupt and interrupt.get("kind") == "confirm_ticket":
                                # Never grant consent, even to the isolated write fixture.
                                async for event in runtime.stream_resume(fixture.cid, {"confirmed": False},
                                        user_id=fixture.uid, interrupt_id=interrupt["interrupt_id"]):
                                    events.append(event)
                                    if event.get("type") == "delta" and event.get("text") and first_delta is None:
                                        first_delta = time.monotonic() - started
                        try:
                            with call_context(request_id=request_id, priority="background"):
                                await asyncio.wait_for(consume(), timeout=manifest["turn_timeout_seconds"])
                        except Exception as error:
                            problem = {"kind": _error_kind(error), "type": type(error).__name__}
                        elapsed = time.monotonic() - started
                        state = (await graph.aget_state({"configurable": {"thread_id": str(fixture.cid)}})).values
                        proposed = []
                        current = state.get("current_message_id", 0)
                        seen_current = False
                        for message in state.get("messages", []):
                            if str(getattr(message, "id", "")) == f"db-{current}":
                                seen_current = True
                            if seen_current:
                                proposed.extend(deepcopy(getattr(message, "tool_calls", [])))
                        answer = nodes.resolve_answer(state)
                        serial_state = {k: ([message_to_dict(m) for m in value] if k == "messages" else value)
                                        for k, value in state.items()}
                        record = {"coords": coords, "query": step["query"], "expected": step["expected"],
                                  "status": "failed" if problem else "completed", "error": problem,
                                  "answer": answer, "interrupt": interrupt, "sse_events": events,
                                  "first_visible_sse_delta_seconds": first_delta, "end_to_end_seconds": elapsed,
                                  "call_ids": sorted(set(recorder.calls) - call_before), "request_id": request_id,
                                  "pool_attempts": _pool_attempts(request_id), "state": serial_state,
                                  "actual_dialog_rows": deepcopy(fixture.rows), "proposed_tools": proposed,
                                  "executed_tools": deepcopy(fixture.tool_calls[before_tools:]),
                                  "retrieval": deepcopy(fixture.retrievals[before_retrieval:])}
                        record["score"] = score_turn(record)
                        write(out / "turns" / f"{arm}-{case['id']}-{number}.json", {**record, "record_sha256": digest(record)})
                        COORDS.reset(token)
                        print(json.dumps({**coords, "status": record["status"], "rule_pass": record["score"]["task_rule_pass"],
                                          "seconds": round(elapsed, 2)}, ensure_ascii=False), flush=True)
                        if problem:
                            # Do not continue a broken interactive history or hide it by retries.
                            break
                    await summarizer.close_tasks()
    finally:
        lock.unlink(missing_ok=True)
        from app.core.model_transport import close_clients
        await close_clients()
    return audit(out)


def audit(out):
    manifest, data = read(out / "manifest.json"), read(out / "dataset.json")
    issues = []
    if digest(data) != manifest["dataset_sha256"]:
        issues.append("dataset hash mismatch")
    for name, sha in manifest["sources"].items():
        path = out / "source_snapshot" / name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            issues.append("source snapshot mismatch: " + name)
    expected = {f"{arm}-{case['id']}-{i}" for arm in manifest["arms"] for case in data["cases"]
                for i in range(1, len(case["steps"]) + 1)}
    records = {p.stem: read(p) for p in sorted((out / "turns").glob("*.json"))}
    calls = {p.stem: read(p) for p in sorted((out / "calls").glob("*.json"))}
    for key, value in [*records.items(), *calls.items()]:
        raw = {k: v for k, v in value.items() if k != "record_sha256"}
        if value.get("record_sha256") != digest(raw):
            issues.append("record hash mismatch: " + key)
    for key, record in records.items():
        if score_turn(record) != record["score"]:
            issues.append("score mismatch: " + key)
        if any(cid not in calls for cid in record["call_ids"]):
            issues.append("missing call record: " + key)
        for cid in record["call_ids"]:
            if cid in calls and calls[cid]["coords"] != record["coords"]:
                issues.append("cross-turn call: " + key)
        coordinates = record["coords"]
        case = next((c for c in data["cases"] if c["id"] == coordinates["case_id"]), None)
        if case:
            expected_step = case["steps"][coordinates["turn"] - 1]
            if record["query"] != expected_step["query"] or record["expected"] != expected_step["expected"]:
                issues.append("frozen step mismatch: " + key)
        rows = record.get("actual_dialog_rows", [])
        current = record.get("state", {}).get("current_message_id")
        if record["status"] == "completed" and not record.get("interrupt"):
            if len(rows) < 2 or rows[-2]["role"] != "user" or rows[-2]["content"] != record["query"] or rows[-1]["role"] != "assistant" or rows[-1]["content"] != record["answer"]:
                issues.append("actual answer feedback mismatch: " + key)
        previous_key = f"{coordinates['arm']}-{coordinates['case_id']}-{coordinates['turn'] - 1}"
        if previous_key in records:
            previous_rows = records[previous_key]["actual_dialog_rows"]
            if rows[:len(previous_rows)] != previous_rows:
                issues.append("history continuity mismatch: " + key)
        snapshot = record.get("state", {}).get("conversation_memory")
        if snapshot and current:
            expected_rows = [{"id": r["id"], "role": r["role"], "content": r["content"]}
                             for r in rows if r["id"] <= current]
            if snapshot["rows"] != expected_rows or snapshot["source_hash"] != digest(expected_rows):
                issues.append("source snapshot prefix mismatch: " + key)
    assigned = [cid for record in records.values() for cid in record["call_ids"]]
    if len(assigned) != len(set(assigned)):
        issues.append("call assigned to multiple turns")
    if set(calls) - set(assigned):
        issues.append("unassigned model calls (including late background work)")
    for key, call in calls.items():
        if digest(call["input"]) != call["input_sha256"]:
            issues.append("input hash mismatch: " + key)
        if call.get("wire_payload") is not None and digest(call["wire_payload"]) != call["wire_sha256"]:
            issues.append("wire hash mismatch: " + key)
        if call.get("status") == "ok" and structured_response_check(call) != call.get("structured_response_check"):
            issues.append("structured-response check mismatch: " + key)
    groups = {}
    for arm in manifest["arms"]:
        arm_records = [r for r in records.values() if r["coords"]["arm"] == arm]
        arm_calls = [c for c in calls.values() if c["coords"]["arm"] == arm]
        usages = [c["usage"] for c in arm_calls if c.get("usage")]
        elapsed = [r["end_to_end_seconds"] for r in arm_records]
        first = [r["first_visible_sse_delta_seconds"] for r in arm_records if r["first_visible_sse_delta_seconds"] is not None]
        clean = [r for r in arm_records if not turn_failure_kinds(r, calls)]
        groups[arm] = {"turns": len(arm_records), "task_rule_pass": sum(r["score"]["task_rule_pass"] for r in arm_records),
                       "model_calls": len(arm_calls), "interface_error_calls": sum(c["status"] == "interface_error" for c in arm_calls),
                       "budget_blocked_calls": sum(c["status"] == "budget_blocked" for c in arm_calls),
                       "unfinished_calls": sum(c["status"] == "running" for c in arm_calls),
                       "unknown_usage_calls": sum(not c.get("usage") for c in arm_calls),
                       "protocol_invalid_calls": sum(c.get("structured_response_check", {}).get("status") == "invalid" for c in arm_calls),
                       "failure_affected_turns": len(arm_records) - len(clean),
                       "clean_turns": len(clean),
                       "clean_task_rule_pass": sum(r["score"]["task_rule_pass"] for r in clean),
                       "input_tokens": sum(u.get("input_tokens", 0) for u in usages),
                       "output_tokens": sum(u.get("output_tokens", 0) for u in usages),
                       "cache_read_tokens": sum((u.get("input_token_details") or {}).get("cache_read", 0) for u in usages),
                       "reasoning_tokens": sum((u.get("output_token_details") or {}).get("reasoning", 0) for u in usages),
                       "end_to_end_p50_seconds": statistics.median(elapsed) if elapsed else None,
                       "end_to_end_p95_seconds": percentile95(elapsed),
                       "first_visible_delta_p50_seconds": statistics.median(first) if first else None,
                       "first_visible_delta_p95_seconds": percentile95(first)}
    missing, unexpected = sorted(expected - set(records)), sorted(set(records) - expected)
    if unexpected:
        issues.append("unexpected turn records")
    complete = not missing and not issues and not any(c["status"] == "running" for c in calls.values())
    result = {"audited_at": stamp(), "complete": complete, "audit_passed": not issues,
              "missing": missing, "issues": issues, "groups": groups,
              "turn_failure_kinds": {key: turn_failure_kinds(record, calls) for key, record in records.items()},
              "scope": f"{len(data['cases'])} synthetic short interactive trajectories; real model/workflow; fake retrieval and business tools",
              "limits": ["Rule pass is not human semantic accuracy or real RAG recall.",
                         "Actual model answers are fed into the next turn; all generated turns are charged.",
                         "Short trajectories do not validate long-context compression or production DB latency.",
                         "First visible SSE delta may be a fixed fallback, not model first-token latency.",
                         "Unknown usage is unknown cost, not zero. Protocol failures and pool attempts remain in raw records."]}
    write(out / "audited_report.json", result)
    lines = ["# 真实 Workflow 合成轨迹验证", "", f"记录完整：{complete}；离线复算：{not issues}。", "",
             "使用真实模型与LangGraph，数据库、知识检索和业务工具为隔离合成夹具。不是实际知识库召回评测。", "",
             "|组别|规则通过/已记录轮次|无调用故障子集通过|模型调用|接口失败|输入Token|输出Token|端到端P50秒|首次可见SSE P50秒|",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, g in groups.items():
        lines.append(f"|{arm}|{g['task_rule_pass']}/{g['turns']}|{g['clean_task_rule_pass']}/{g['clean_turns']}|{g['model_calls']}|{g['interface_error_calls']}|{g['input_tokens']}|{g['output_tokens']}|{g['end_to_end_p50_seconds']}|{g['first_visible_delta_p50_seconds']}|")
    lines += ["", "实际模型回复连续写回后续历史；费用覆盖本次生成的全部轮次。首次SSE可能是固定回复，不能称为模型首Token。",
              "短轨迹不验证长会话压缩；假存储不代表生产数据库延迟。语义正确性仍需独立复核。",
              "回读工具固定关闭；本次仅验证主动装配历史。模型调用池保留共享治理及其技术SQLite记录，客户数据库和业务写入完全隔离。",
              "无调用故障子集排除了接口/协议/运行错误和上下文阻断；两组子集可能不同，不能直接当作配对能力差异。",
              "", "缺失记录：" + str(len(missing)), "审计问题：" + str(issues)]
    (out / "AUDITED_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=("legacy", "source"), default=["legacy", "source"])
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--turn-timeout", type=float, default=180)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.run:
        result = asyncio.run(run(args.output))
    elif args.audit:
        result = audit(args.output)
    else:
        result = prepare(args.output, args.arms, args.cases, args.turn_timeout)
        print("PREPARED: no model API was called", flush=True)
    print(json.dumps({k: result[k] for k in ("complete", "audit_passed", "issues", "missing") if k in result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
