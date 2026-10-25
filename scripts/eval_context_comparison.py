"""Auditable context-policy comparison on isolated customer-service replay.

No production DB, Milvus, HTTP chat endpoint, or external Harness executable is used.
All tested arms share the production coreference prompt and a common service reader.
The read-only tool fixtures and expected answers are never passed to memory builders.
"""
import argparse
import asyncio
import hashlib
import json
import math
import random
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

from app.config import settings
from app.core import coref, llm
from app.core.prompts import AGENT_SYSTEM
from scripts.context_benchmark_cases import make_suite
from scripts.context_benchmark_strategies import ContextAdapter, LABELS, STRATEGIES, read_harness_prompt
from scripts.eval_context_engineering import Usage

ROOT = Path(__file__).resolve().parents[1]
FILES = ["scripts/context_benchmark_cases.py", "scripts/context_benchmark_strategies.py",
         "scripts/eval_context_comparison.py", "scripts/audit_context_comparison.py", "scripts/audit_context_semantics.py",
         "app/config.py", "app/core/coref.py", "app/core/memory.py",
         "app/core/summarizer.py", "app/core/prompts.py", "app/core/llm.py", "app/graph/nodes.py"]
HARNESS_FILES = ["packages/compaction/compaction-basic/src/" + f for f in
                 ("config.ts", "index.ts", "region.ts", "summarizer.ts", "types.ts")]
HARNESS_FILES += ["packages/llm/llm-deepseek/src/common/defaults.ts", "docs/subsystems/compaction.md",
                  "packages/compaction/compaction-tool-result-pruner/src/config.ts"]
READER_SYSTEM = AGENT_SYSTEM + """

## 离线只读客服测试接口
本次可选行动：logistics=查询指定订单物流；care=按型号查清洁说明；recall=复述对话中已明确的事实；clarify=询问缺失信息。
规划时只选一个行动。订单号、型号必须来自当前用户或历史证据。涉及更正，以最后确认的值为准；失效值不能用于工具参数。
“最早/最开始”按原始提及顺序，“当前”按用户最后的有效指定；不要自行按消息邻近程度选择有歧义的订单。
历史不能唯一确定目标时选择clarify，不要猜。查物流提供order_id，查商品清洁说明提供sku；其他行动的这两个字段留空。
仅仅回顾专员在对话里说过的承诺日期属于recall，不需要查询订单，不得把承诺说成已经完成。
answer仅在recall/clarify时填写；logistics/care等待工具结果后再回答。"""
REPLY_SYSTEM = AGENT_SYSTEM + """
## 本次回答要求
根据本轮只读工具结果或明确历史证据回答当前问题。查询物流时写明订单号和运单号；商品清洁说明写明型号。
如果工具查不到或目标未确定，应说明并澄清，不得编造。用户要求日期格式时使用YYYY-MM-DD。
回答应简短，不扩展无关订单，不把旧承诺与更新后的承诺混用。"""


class ServicePlan(BaseModel):
    action: Literal["logistics", "care", "recall", "clarify"]
    order_id: str = Field(default="", description="物流查询的订单号；其他行动留空")
    sku: str = Field(default="", description="查清洁说明的商品型号；其他行动留空")
    answer: str = Field(default="", description="回顾已知事实或澄清问题；需要工具时留空")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def serialize_messages(messages):
    return [m.model_dump(mode="json") for m in messages]


class Calls:
    def __init__(self, timeout, concurrency):
        self.timeout, self.semaphore = timeout, asyncio.Semaphore(concurrency)

    async def __call__(self, stage, fn, prompt=None):
        usage = Usage()
        started = time.monotonic()
        async with self.semaphore:
            queue = time.monotonic() - started
            actual = time.monotonic()
            async def invoke(_):
                return await fn()
            try:
                value = await asyncio.wait_for(RunnableLambda(invoke).ainvoke(
                    {}, config={"callbacks": [usage]}), self.timeout)
                result = {"ok": value is not None, "value": value}
            except Exception as exc:
                result = {"ok": False, "value": None, "error_type": type(exc).__name__}
        return result | {"stage": stage, "calls": usage.calls, "errors": usage.errors,
                         "seconds": round(time.monotonic() - actual, 4), "queue_seconds": round(queue, 4),
                         "prompt": serialize_messages(prompt) if prompt is not None else None}


def tool_result(case, plan):
    """Deterministic read-only backend: no database writes or hidden target-selection help."""
    if plan["action"] == "logistics":
        return case["tool_data"]["orders"].get(plan["order_id"], {"error": "订单号不存在，请确认"})
    if plan["action"] == "care":
        text = case["tool_data"]["manuals"].get(plan["sku"])
        return {"sku": plan["sku"], "care": text} if text else {"error": "该型号无说明，请确认"}
    return None


def score_probe(case, probe, result):
    """Score only frozen fields and literal safety conditions; no model judge."""
    expected = probe["expected"]
    plan = result.get("plan", {}).get("value") or {}
    reply = result.get("reply", {}).get("value") or ""
    rewrite = result.get("coref", {}).get("value") or ""
    action = expected["action"]
    action_ok = plan.get("action") == action
    target_ok = action_ok
    if action == "logistics":
        target_ok = action_ok and plan.get("order_id") == expected["order_id"]
        answer_ok = expected["order_id"] in reply and "SF" + expected["order_id"] + "88" in reply
        coref_ok = expected["order_id"] in rewrite
        required = [expected["order_id"]]
    elif action == "care":
        target_ok = action_ok and plan.get("sku", "").upper() == expected["sku"]
        answer_ok = expected["sku"].upper() in reply.upper() and "断电" in reply and "软布" in reply
        product = next((p["content"] for p in case["history"][:2] if p["role"] == "user"), "")
        # A product name OR exact SKU makes the original half-sentence self-contained.
        name = re.search(r"商品是(.+?)，型号", product)
        coref_ok = expected["sku"].upper() in rewrite.upper() or bool(name and name.group(1) in rewrite)
        required = [expected["sku"]]
    elif action == "recall":
        answer_ok = expected["date"] in reply and all(
            date == expected["date"] for date in re.findall(r"2026-\d{2}-\d{2}", reply))
        coref_ok = bool(rewrite) and any(w in rewrite for w in ("日期", "哪天", "到货", "承诺"))
        required = [expected["date"]]
    else:
        answer_ok = any(w in reply for w in ("哪", "订单号", "确认", "具体")) and not plan.get("order_id")
        coref_ok = not re.search(r"(?<!\d)\d{6}(?!\d)", rewrite)
        required = []
    # Cross-order contamination in the actual tool parameters is always a failure.
    invalid_numbers = re.findall(r"(?<!\d)\d{6}(?!\d)", rewrite)
    fabricated = [x for x in invalid_numbers if x not in case["all_orders"]]
    if fabricated:
        coref_ok = False
    if action in ("logistics", "care"):
        allowed_orders = {expected["order_id"]} if action == "logistics" else {
            oid for row in case["history"][:2 * probe["at_turn"]] if expected["sku"] in row["content"]
            for oid in case["all_orders"] if oid in row["content"]}
        unwanted = [oid for oid in case["all_orders"] if oid not in allowed_orders and oid in reply]
        if unwanted:
            answer_ok = False
    stages_ok = all(result.get(k, {}).get("ok", False) for k in ("coref", "plan", "reply"))
    context = result.get("context_text", "")
    retained = all(x in context for x in required)
    return {"coref_pass": bool(coref_ok and result.get("coref", {}).get("calls")),
            "action_pass": bool(action_ok), "target_pass": bool(target_ok),
            "answer_rule_pass": bool(answer_ok), "task_pass": bool(stages_ok and target_ok and answer_ok),
            "required_fact_literal_present": retained, "fabricated_rewrite_orders": fabricated,
            "api_success": stages_ok}


async def run_probe(case, probe, adapter, caller):
    rows = case["history"][:2 * probe["at_turn"]]
    context = adapter.contexts(rows, probe["query"])
    rewrite = await caller("coref", lambda: coref.resolve(probe["query"], context["coref_history"]))
    # Raw inputs sufficient to recreate the production prompt, without storing credentials.
    rewrite["input"] = {"query": probe["query"], "history": context["coref_history"]}
    effective_query = rewrite["value"] or probe["query"]
    prompt = [SystemMessage(READER_SYSTEM), *context["body"], HumanMessage(
        "本轮原话：" + probe["query"] + "\n指代改写结果（需用历史核实）：" + effective_query)]
    async def plan_call():
        return (await llm.structured(ServicePlan, temperature=0).ainvoke(prompt)).model_dump()
    plan = await caller("plan", plan_call, prompt)
    data = tool_result(case, plan["value"]) if plan["ok"] else None
    reply_prompt = [SystemMessage(REPLY_SYSTEM), *context["body"], HumanMessage(
        "本轮原话：" + probe["query"] + "\n指代改写：" + effective_query)]
    if plan["ok"]:
        p = plan["value"]
        if data is not None:
            name = "query_logistics" if p["action"] == "logistics" else "query_faq"
            args = {"order_id": p["order_id"]} if p["action"] == "logistics" else {"query": p["sku"] + " 清洁方法"}
            reply_prompt.extend([AIMessage("", tool_calls=[{"name": name, "args": args, "id": "eval-readonly", "type": "tool_call"}]),
                                 ToolMessage(json.dumps(data, ensure_ascii=False), tool_call_id="eval-readonly")])
        else:
            reply_prompt.append(HumanMessage("本轮计划：" + json.dumps(p, ensure_ascii=False) + "\n请核实历史后作答。"))
    async def reply_call():
        return (await llm.get_chat_model(temperature=0).bind(max_tokens=768).ainvoke(reply_prompt)).content
    reply = await caller("reply", reply_call, reply_prompt)
    result = {"probe_id": probe["id"], "at_turn": probe["at_turn"], "coref": rewrite, "plan": plan,
              "tool_result": data, "reply": reply, "retrieved": context["retrieved"],
              "context_text": "\n".join(str(m.content) for m in context["body"]),
              "approx_context_tokens": context["approx_context_tokens"]}
    result["score"] = score_probe(case, probe, result)
    return result


async def evaluate_case(case, output, harness, caller, capacity):
    folder = output / "cases" / case["id"]
    # D/E share exactly the same query-independent state extraction to isolate retrieval's effect.
    d_snapshots = {}
    state_file = folder / "shared_state.json"
    if state_file.exists():
        d_snapshots = json.loads(state_file.read_text(encoding="utf-8"))
    builder = ContextAdapter("D_state", harness, caller, READER_SYSTEM, capacity)
    for probe in case["probes"]:
        if probe["id"] in d_snapshots:
            builder.restore(d_snapshots[probe["id"]])
        else:
            await builder.advance(case["history"][:2 * probe["at_turn"]])
            d_snapshots[probe["id"]] = builder.snapshot()
            write_json(state_file, d_snapshots)
    order = list(STRATEGIES)
    random.Random(1729 + int(case["id"].split("-")[1])).shuffle(order)
    for name in order:
        path = folder / (name + ".json")
        result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"strategy": name, "probes": [], "snapshots": {}}
        adapter = ContextAdapter(name, harness, caller, READER_SYSTEM, capacity)
        for probe in case["probes"]:
            if probe["id"] in result["snapshots"]:
                adapter.restore(result["snapshots"][probe["id"]])
            elif name in ("D_state", "E_state_retrieval"):
                adapter.restore(d_snapshots[probe["id"]])
            else:
                await adapter.advance(case["history"][:2 * probe["at_turn"]])
            result["snapshots"][probe["id"]] = adapter.snapshot()
            if not any(r["probe_id"] == probe["id"] for r in result["probes"]):
                result["probes"].append(await run_probe(case, probe, adapter, caller))
                write_json(path, result)
        result["maintenance"] = adapter.maintenance
        write_json(path, result)
        passed = sum(p["score"]["task_pass"] for p in result["probes"])
        print(f"{case['id']} {case['kind']} {name} task={passed}/2 maintenance={len(adapter.maintenance)}", flush=True)


def percentile(values, q):
    if not values:
        return None
    data = sorted(values)
    return data[min(len(data) - 1, max(0, math.ceil(q * len(data)) - 1))]


def bootstrap_difference(pairs, seed=8331):
    """Paired bootstrap at whole-conversation level, keeping the two probes together."""
    rng = random.Random(seed)
    if not pairs:
        return None
    draws = [sum(rng.choice(pairs) for _ in pairs) / len(pairs) for _ in range(4000)]
    return [percentile(draws, .025), percentile(draws, .975)]


def aggregate(output):
    fixture = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    metrics, by_case = {}, {}
    failures = []
    for name in STRATEGIES:
        entries, maintenance = [], []
        case_rates = {}
        for case in fixture["cases"]:
            path = output / "cases" / case["id"] / (name + ".json")
            if not path.exists():
                continue
            row = json.loads(path.read_text(encoding="utf-8"))
            scores = []
            for result in row["probes"]:
                probe = next(p for p in case["probes"] if p["id"] == result["probe_id"])
                score = score_probe(case, probe, result)
                result["score"] = score
                entries.append((case, probe, result))
                scores.append(int(score["task_pass"]))
                if not score["task_pass"]:
                    failures.append({"case": case["id"], "kind": case["kind"], "strategy": name,
                                     "probe": probe["id"], "expected": probe["expected"],
                                     "coref": result["coref"]["value"], "plan": result["plan"]["value"],
                                     "reply": result["reply"]["value"], "score": score})
            if len(scores) == 2:
                case_rates[case["id"]] = statistics.mean(scores)
            # On an interrupted run, take the latest persisted snapshot if the final key isn't written yet.
            maintenance.extend(row.get("maintenance", list(row["snapshots"].values())[-1]["maintenance"]))
        all_probe_records = [r[stage] for _, _, r in entries for stage in ("coref", "plan", "reply")]
        all_records = maintenance + all_probe_records
        calls = [c for r in all_records for c in r["calls"]]
        tokens = {key: sum(c.get(key) or 0 for c in calls) for key in ("input_tokens", "output_tokens", "cache_read")}
        missing = sum(c.get("input_tokens") is None or c.get("output_tokens") is None for c in calls)
        latencies = [sum(r[stage]["seconds"] for stage in ("coref", "plan", "reply")) for _, _, r in entries]
        categories = {}
        for kind in sorted({c["kind"] for c, _, _ in entries}):
            group = [r for c, _, r in entries if c["kind"] == kind]
            categories[kind] = {"passed": sum(r["score"]["task_pass"] for r in group), "n": len(group)}
        memory_input = sum(c.get("input_tokens") or 0 for r in maintenance for c in r["calls"])
        stage_usage = {}
        for stage in ("coref", "plan", "reply"):
            stage_calls = [c for _, _, r in entries for c in r[stage]["calls"]]
            stage_usage[stage] = {key: sum(c.get(key) or 0 for c in stage_calls)
                                  for key in ("input_tokens", "output_tokens", "cache_read")}
        n = len(entries)
        metrics[name] = {"label": LABELS[name], "n": n,
                         **{key: sum(r["score"][key] for _, _, r in entries) for key in
                            ("coref_pass", "action_pass", "target_pass", "answer_rule_pass", "task_pass", "api_success")},
                         "task_rate": sum(r["score"]["task_pass"] for _, _, r in entries) / n if n else None,
                         "usage": tokens, "maintenance_input_tokens": memory_input,
                         "stage_usage": stage_usage,
                         "calls": len(calls), "maintenance_calls": sum(len(r["calls"]) for r in maintenance),
                         "maintenance_failures": sum(not r["ok"] for r in maintenance),
                         "provider_errors": sum(len(r["errors"]) for r in all_records), "usage_missing_calls": missing,
                         "p50_probe_seconds": percentile(latencies, .5), "p95_probe_seconds": percentile(latencies, .95),
                         "maintenance_seconds_sum": round(sum(r["seconds"] for r in maintenance), 3),
                         "context_tokens_mean_approx": statistics.mean(r["approx_context_tokens"] for _, _, r in entries) if n else None,
                         "by_kind": categories}
        by_case[name] = case_rates
    comparisons = {}
    for name in STRATEGIES:
        pairs = [by_case[name][cid] - by_case["C_current"][cid]
                 for cid in sorted(by_case[name].keys() & by_case["C_current"].keys())]
        comparisons[name] = {"task_rate_difference_vs_current": statistics.mean(pairs) if pairs else None,
                             "paired_conversation_bootstrap_95ci": bootstrap_difference(pairs), "n_conversations": len(pairs)}
    complete = all(m["n"] == len(fixture["cases"]) * 2 for m in metrics.values())
    # D/E are independent deployment comparisons, so shared preprocessing is charged to both.
    # Physical API usage counts the shared extraction only once.
    shared = []
    for case in fixture["cases"]:
        path = output / "cases" / case["id"] / "shared_state.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            shared.extend(list(payload.values())[-1]["maintenance"])
    shared_calls = [c for r in shared for c in r["calls"]]
    physical_usage = {key: sum(m["usage"][key] for m in metrics.values()) - sum(c.get(key) or 0 for c in shared_calls)
                      for key in ("input_tokens", "output_tokens", "cache_read")}
    report = {"manifest": manifest, "complete": complete, "strategies": metrics, "comparisons": comparisons,
              "physical_usage": physical_usage,
              "physical_successful_api_calls": sum(m["calls"] for m in metrics.values()) - len(shared_calls),
              "limitations": [
                  "Fixed synthetic histories, 8 scenario families x 6 variants; two correlated probes per conversation.",
                  "Production coref and C summary/context assemblers; common isolated reader and deterministic read-only backend, not the complete production graph or real Milvus retrieval.",
                  "Costs include every maintenance and three LLM stages for both probes. Historical filler replies are supplied; this is not full-generation cost for all 24-64 historical turns.",
                  "D/E share identical state extraction; its cost is attributed to each standalone arm but physically executed once.",
                  "Harness is a Python text-policy migration using frozen original prompt/config, not a launched TS runtime; token meter is project approximate, no image/tool-result pruning or overflow recovery exercised.",
                  "Harness default 1M context / 0.8 threshold does not compact these histories; A is its text-context equivalent, not an independent Harness run.",
                  "H migrations use 3750 approximate capacity (threshold3000, retain600); this does not enforce identical provider token lengths across arms.",
                  "One sampled run per probe/arm; temperature0 for shared reader, temperature0.3 for coref and all memory policies.",
                  "Latency excludes evaluator queue wait; C/D summaries awaited at replay boundaries, not production asynchronous latency. Sum of calls is not wall-clock duration.",
                  "Rule task pass requires correct action/target AND answer anchors; it is not a human semantic correctness or production acceptance metric.",
                  "Bootstrap is clustered by conversation, not by scenario family, and does not establish out-of-template generalization."]}
    write_json(output / "report.json", report)
    write_json(output / "failures.json", failures)
    lines = ["# 客服上下文工程对照评测", "", f"完成：{complete}；{len(fixture['cases'])}条固定合成对话，每条2次追问。", "",
             "|方案|任务规则通过|指代通过|回放输入Token(含维护)|输出Token|维护输入Token|P50/P95秒|", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, m in metrics.items():
        if not m["n"]:
            continue
        lines.append(f"|{m['label']}|{m['task_pass']}/{m['n']} ({m['task_rate']:.1%})|{m['coref_pass']}/{m['n']}|"
                     f"{m['usage']['input_tokens']:,}|{m['usage']['output_tokens']:,}|{m['maintenance_input_tokens']:,}|"
                     f"{m['p50_probe_seconds']:.2f}/{m['p95_probe_seconds']:.2f}|")
    lines += ["", "成本为这次固定历史回放中全部维护与两次追问的实报用量，不是全部历史轮次生成成本。", "",
              "Harness默认80万Token触发点在本集不触发；原提示词/客服适配两组是压缩阈值约3000Token的Python迁移实验。", "",
              "## 各场景任务通过", "", "|场景|" + "|".join(STRATEGIES) + "|", "|---|" + "---:|" * len(STRATEGIES)]
    for kind in sorted({c["kind"] for c in fixture["cases"]}):
        lines.append("|" + kind + "|" + "|".join(
            f"{m['by_kind'].get(kind, {}).get('passed', 0)}/{m['by_kind'].get(kind, {}).get('n', 0)}" for m in metrics.values()) + "|")
    lines += ["", "## 与当前方案的配对差值", ""]
    for name, c in comparisons.items():
        if c["n_conversations"]:
            low, high = c["paired_conversation_bootstrap_95ci"]
            lines.append(f"- {LABELS[name]}：{c['task_rate_difference_vs_current']:+.1%}；对话级bootstrap 95%区间 [{low:+.1%}, {high:+.1%}]。")
    lines += ["", "## 口径与限制", ""] + ["- " + item for item in report["limitations"]]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def prepare(output, args, fixture):
    signature = {"dataset_sha256": sha(fixture), "source_sha256": {}, "harness_sha256": {},
                 "model": settings.chat_model, "summary_model": settings.summary_model or settings.chat_model,
                 "thinking": settings.chat_thinking, "capacity_approx": args.capacity,
                 "failover_enabled": settings.chat_failover_enabled,
                 "window_tokens": settings.context_window_max_tokens, "keep_turns": settings.context_window_turns,
                 "summary_trigger_messages": settings.summary_trigger_messages}
    for file in FILES:
        signature["source_sha256"][file] = hashlib.sha256((ROOT / file).read_bytes()).hexdigest()
    for file in HARNESS_FILES:
        signature["harness_sha256"][file] = hashlib.sha256((Path(args.harness) / file).read_bytes()).hexdigest()
    manifest_path = output / "manifest.json"
    manifest = signature | {"created_at": datetime.now(timezone.utc).isoformat(), "concurrency": args.concurrency,
                            "strategies": list(STRATEGIES), "harness_root": str(Path(args.harness).resolve())}
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(saved.get(k) != v for k, v in signature.items()):
            raise RuntimeError("Frozen source/config/data differs; select a new output directory")
        manifest = saved
    write_json(manifest_path, manifest)
    write_json(output / "dataset.json", fixture)
    for files, root, prefix in ((FILES, ROOT, "mewhelp"), (HARNESS_FILES, Path(args.harness), "harness")):
        for file in files:
            dest = output / "source_snapshot" / prefix / file
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((root / file).read_bytes())
    harness = read_harness_prompt(args.harness)
    write_json(output / "harness_prompt.json", harness)
    return harness


async def main(args):
    output = Path(args.output)
    if args.recompute:
        report = aggregate(output)
    else:
        fixture = make_suite()
        if args.limit:
            fixture["cases"] = fixture["cases"][:args.limit]
        harness = prepare(output, args, fixture)
        if args.prepare:
            print(f"Prepared {len(fixture['cases'])} conversations at {output}")
            return
        caller = Calls(args.timeout, args.concurrency)
        workers = asyncio.Semaphore(args.case_workers)
        async def run(case):
            async with workers:
                await evaluate_case(case, output, harness, caller, args.capacity)
        await asyncio.gather(*(run(c) for c in fixture["cases"]))
        report = aggregate(output)
    print(json.dumps({"complete": report["complete"], "report": str(output / "REPORT.md"),
                      "strategies": report["strategies"]}, ensure_ascii=False, indent=2), flush=True)
    if not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", default="D:/harness/deepseek-harness")
    parser.add_argument("--output", default="data/ch07/context-comparison/20261016-v2")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--case-workers", type=int, default=6)
    parser.add_argument("--capacity", type=int, default=3750)
    parser.add_argument("--timeout", type=float, default=100)
    args = parser.parse_args()
    if min(args.concurrency, args.case_workers, args.capacity, args.timeout) <= 0:
        parser.error("numeric options must be positive")
    asyncio.run(main(args))
