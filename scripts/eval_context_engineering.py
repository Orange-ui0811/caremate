"""Reproducible synthetic context benchmark; never reads/writes customer DBs.

Prepare fixed cases, then run the real summarizer/coref and context assemblers:
  python -m scripts.eval_context_engineering --prepare
  python -m scripts.eval_context_engineering --output data/ch07/context-eval/run-01
  python -m scripts.eval_context_engineering --output data/ch07/context-eval/run-01 --recompute

Results are cached per call. Re-running resumes the same inputs; --recompute is
entirely offline. No prompt tuning is performed on these evaluation cases.
"""
import argparse
import asyncio
import hashlib
import json
import random
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableLambda

from app.config import settings
from app.core import coref, summarizer
from app.core.prompts import AGENT_SYSTEM
from app.graph.nodes import _agent_messages, _history_text

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "tests/data/context_engineering_v1.json"
STRATEGIES = ("full_history", "window_only", "summary_window")
SOURCE_FILES = ("app/core/memory.py", "app/core/summarizer.py", "app/core/coref.py",
                "app/core/prompts.py", "app/core/llm.py", "app/graph/nodes.py",
                "scripts/eval_context_engineering.py")
PRODUCTS = [("智能猫砂盆Pro", "MH-LP100"), ("自动饮水机Plus", "MH-W40"),
            ("自动喂食器基础版", "MH-FD10"), ("大型猫爬架", "MH-CT30"),
            ("四季保暖猫窝", "MH-NEST20"), ("全景看护摄像头", "MH-CAM1")]
FILLERS = [
    ("我还想了解购物前一般需要核对哪些信息，暂时只是了解平台规则，不是在咨询前面那两笔订单。请说明商品页面、配送范围、规格说明和售后条件应该怎么看。",
     "购物前可以核对商品详情中的规格、配送范围、付款方式和售后条件。不同商品的说明可能不同，具体以相应页面为准。这段只是在解释通用购物步骤，没有变更此前订单或工单，也没有产生新的执行请求。"),
    ("顺便解释一下发票抬头和电子发票的区别，我现在只想了解概念，不需要为任何订单申请开票，也不需要修改已有业务资料。",
     "发票抬头是票据上的购买方名称，电子发票则是一种票据形式。申请时应核对购买方信息与页面要求；这里仅解释概念，没有提交开票申请，也没有改变之前的订单与售后诉求。具体办理渠道以平台规则为准。"),
    ("浏览商品的时候，尺寸、材质、包装清单这些信息应该按什么顺序核对？这是通用咨询，没有指定新商品，也不是要替换我们已经讨论过的商品。",
     "可以先核对尺寸是否适合使用空间，再看材质与安装条件，最后检查包装清单和使用说明。购买前还应查看对应页面的售后要求。以上属于通用核对建议，没有新增商品选择、订单号或工单处理决定。"),
    ("我继续问一个通用问题：收货前通常应当准备什么，包装检查和配件核对有什么区别？不需要查询具体订单，只解释检查步骤就可以。",
     "收货时可以先检查外包装，再按商品说明核对配件与外观。如果发现异常，可以保留图片并通过平台售后渠道反馈。此处只说明一般检查流程，没有替任何订单确认签收，也没有修改前面登记的业务信息。"),
]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def make_cases():
    cases = []
    kinds = ("early_order", "early_product", "corrected_order", "recent_reference")
    for group, kind in enumerate(kinds):
        for variant, turns in enumerate((24, 32, 40, 48, 56, 64)):
            index = len(cases) + 1
            order, other, corrected = str(710000 + index), str(820000 + index), str(930000 + index)
            product, sku = PRODUCTS[variant]
            other_product, other_sku = PRODUCTS[(variant + 1) % len(PRODUCTS)]
            ticket = f"CTX-T{index:04d}"
            history = [
                {"role": "user", "content": f"我最早咨询的订单是{order}，买的是{product}，型号{sku}，请帮我保留这笔订单的咨询记录。"},
                {"role": "assistant", "content": f"已记录最早订单{order}，商品{product}，型号{sku}，本次咨询工单编号为{ticket}。"},
                {"role": "user", "content": f"我还有另一笔订单{other}，商品是{other_product}，型号{other_sku}。这两笔要分开记录。"},
                {"role": "assistant", "content": f"已将另一笔订单{other}与最早的订单分开记录，不混用两笔订单的商品与处理信息。"},
            ]
            current_order = order
            if kind == "corrected_order":
                history.extend([
                    {"role": "user", "content": f"更正一下：最早那笔订单号{order}是我报错了，正确订单号是{corrected}，原来的商品型号和工单编号不变。以后以更正后的订单号为准。"},
                    {"role": "assistant", "content": f"已更正为订单{corrected}，旧订单号{order}不再用于查询，该商品仍为{sku}，工单仍为{ticket}。"},
                ])
                current_order = corrected
            while len(history) // 2 < turns:
                turn = len(history) // 2
                # 多主题详细咨询形成超过 token 滑窗的长历史;不引入新的订单/型号。
                parts = [FILLERS[(turn + variant + group + offset) % len(FILLERS)] for offset in range(3)]
                user = "另外，".join(p[0] for p in parts)
                assistant = "此外，".join(p[1] for p in parts)
                history.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
            if kind == "recent_reference":
                history[-2:] = [
                    {"role": "user", "content": f"现在切换到另一笔订单{other}，先查这笔，不查最开始的那单。"},
                    {"role": "assistant", "content": f"好的，现在关注订单{other}，商品是{other_product}，型号{other_sku}。"},
                ]
                query = "那就查这笔订单的物流吧。"
                required, forbidden, intents = [[other]], [order], ["物流", "快递", "配送", "运输", "发货"]
            elif kind == "early_product":
                query = "最开始那款商品应该怎么清洗？"
                required, forbidden, intents = [[product, sku]], [other_sku, other_product], ["清洗", "清洁", "洗"]
            elif kind == "corrected_order":
                query = "那就查更正后那笔订单的物流吧。"
                required, forbidden, intents = [[corrected]], [order, other], ["物流", "快递", "配送", "运输", "发货"]
            else:
                query = "帮我查最开始那笔订单的物流。"
                required, forbidden, intents = [[order]], [other], ["物流", "快递", "配送", "运输", "发货"]
            cases.append({"id": f"ctx-{index:02d}", "kind": kind, "history_turns": turns,
                          "history": history, "query": query,
                          "historical_entities": [current_order, sku, ticket],
                          "required_entity_groups": required, "forbidden_entities": forbidden,
                          "intent_keywords": intents})
    return {"version": "context-engineering-v1", "seed": 20261016,
            "provenance": "24 independently constructed synthetic conversations; no customer data; deterministic labels, not model-generated labels",
            "cases": cases}


def present(entity, text):
    compact = lambda s: re.sub(r"[\s_\-]", "", unicodedata.normalize("NFKC", s)).lower()
    key, body = compact(entity), compact(text)
    pattern = re.escape(key)
    if key.isdigit():
        pattern = r"(?<!\d)" + pattern + r"(?!\d)"
    return re.search(pattern, body) is not None


def score_rewrite(case, text):
    return (all(any(present(e, text) for e in group) for group in case["required_entity_groups"])
            and not any(present(e, text) for e in case["forbidden_entities"])
            and any(k in text for k in case["intent_keywords"]))


def messages(case):
    out = []
    for i, row in enumerate(case["history"], 1):
        cls = HumanMessage if row["role"] == "user" else AIMessage
        out.append(cls(row["content"], id=f"db-{i}"))
    out.append(HumanMessage(case["query"], id=f"db-{len(out) + 1}"))
    return out


def contexts(case, summary, boundary):
    msgs = messages(case)
    full_text = "\n".join(f"{'用户' if r['role'] == 'user' else '客服'}:{r['content']}" for r in case["history"])
    states = {name: {"messages": msgs, "summary": summary if name == "summary_window" else "",
                     "summary_upto_msg_id": boundary if name == "summary_window" else 0}
              for name in STRATEGIES}
    histories = {"full_history": full_text,
                 "window_only": _history_text(states["window_only"]),
                 "summary_window": _history_text(states["summary_window"])}
    agent_contexts = {"full_history": [SystemMessage(AGENT_SYSTEM), *msgs],
                      "window_only": _agent_messages(states["window_only"]),
                      "summary_window": _agent_messages(states["summary_window"])}
    return histories, agent_contexts


class Usage(BaseCallbackHandler):
    def __init__(self):
        self.calls = []
        self.errors = []

    def on_llm_end(self, response, **kwargs):
        for batch in response.generations:
            for generation in batch:
                msg = getattr(generation, "message", None)
                data = getattr(msg, "usage_metadata", None) or {}
                self.calls.append({"input_tokens": data.get("input_tokens"),
                                   "output_tokens": data.get("output_tokens"),
                                   "cache_read": (data.get("input_token_details") or {}).get("cache_read")})

    def on_llm_error(self, error, **kwargs):
        self.errors.append(type(error).__name__)


async def tracked(fn, timeout):
    usage = Usage()
    async def invoke(_):
        return await fn()
    started = time.monotonic()
    try:
        result = await asyncio.wait_for(RunnableLambda(invoke).ainvoke(
            {}, config={"callbacks": [usage]}), timeout=timeout)
        return {"text": result, "calls": usage.calls, "errors": usage.errors,
                "elapsed_seconds": round(time.monotonic() - started, 3)}
    except Exception as exc:
        return {"text": None, "calls": usage.calls, "errors": usage.errors + [type(exc).__name__],
                "elapsed_seconds": round(time.monotonic() - started, 3)}


async def evaluate_case(case, path, timeout):
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "case_id": case["id"], "kind": case["kind"], "history_turns": case["history_turns"],
        "summary_steps": [], "strategies": {}}
    summary, old_upto = "", 0
    history = [SimpleNamespace(id=i, role=r["role"], content=r["content"])
               for i, r in enumerate(case["history"], 1)]
    for turn in range(1, case["history_turns"] + 1):
        prefix = history[:2 * turn]
        if len(prefix) - old_upto < settings.summary_trigger_messages:
            continue
        boundary = summarizer.compute_boundary(prefix, settings.context_window_turns)
        if boundary is None or boundary <= old_upto:
            continue
        cached = next((s for s in result["summary_steps"] if s["boundary"] == boundary and s.get("text")), None)
        if cached is None:
            dialog = "\n".join(f"{'用户' if m.role == 'user' else '客服'}:{m.content}"
                               for m in prefix if old_upto < m.id <= boundary)
            call = await tracked(lambda: summarizer.summarize_dialog(summary, dialog), timeout)
            step = call | {"turn": turn, "boundary": boundary, "previous_boundary": old_upto}
            result["summary_steps"] = [s for s in result["summary_steps"] if s["boundary"] != boundary] + [step]
            write_json(path, result)
            if not call["text"]:
                print(f"{case['id']} summary failed: {call['errors']}", flush=True)
                return result
            cached = step
        summary, old_upto = cached["text"], boundary
    result["final_summary"], result["summary_boundary"] = summary, old_upto
    histories, agent_contexts = contexts(case, summary, old_upto)
    order = list(STRATEGIES)
    random.Random(20261016 + int(case["id"].split("-")[1])).shuffle(order)
    for name in order:
        if result["strategies"].get(name, {}).get("text"):
            continue
        call = await tracked(lambda: coref.resolve(case["query"], histories[name]), timeout)
        # 只统计历史/摘要,不把固定人设中的示例实体计作会话记忆。
        all_context = "\n".join(str(m.content) for m in agent_contexts[name] if not isinstance(m, SystemMessage))
        hits = [e for e in case["historical_entities"] if present(e, all_context)]
        result["strategies"][name] = call | {
            "correct": bool(call["text"] and call["calls"] and score_rewrite(case, call["text"])),
            "historical_entities_retained": hits,
            "historical_entities_total": len(case["historical_entities"]),
            "agent_input_tokens_approx": count_tokens_approximately(agent_contexts[name]),
            "coref_history_chars": len(histories[name])}
        write_json(path, result)
    print(f"{case['id']} {case['kind']} turns={case['history_turns']} summaries={len(result['summary_steps'])} "
          + " ".join(f"{s}={'OK' if result['strategies'][s]['correct'] else 'MISS'}" for s in STRATEGIES), flush=True)
    return result


def total_usage(calls, key):
    values = [c.get(key) for c in calls]
    return sum(values) if values and all(v is not None for v in values) else None


def make_report(fixture, manifest, results):
    complete = [r for r in results if set(r.get("strategies", {})) == set(STRATEGIES)]
    report = {"manifest": manifest, "expected_cases": len(fixture["cases"]), "completed_cases": len(complete),
              "limitations": ["Synthetic text-only benchmark, not production QA success or industrial diagnosis accuracy.",
                  "Entity coverage is literal normalized order/SKU/ticket availability in assembled Agent input, not semantic fact accuracy.",
                  "Coreference score requires target entity, preserved intent keyword and absence of distractor entity; raw outputs are saved for manual review.",
                  "Agent token counts use the project's approximate counter and exclude tool schemas; coreference API tokens are provider-reported.",
                  "Summary generation overhead is recorded separately; summary jobs are awaited in this isolated simulation, not measured as live async response latency.",
                  "One model sample per case/strategy; provider/model nondeterminism can change fresh runs; cached results recompute exactly."],
              "strategies": {}}
    for name in STRATEGIES:
        rows = [r["strategies"][name] for r in complete]
        calls = [c for r in rows for c in r["calls"]]
        retained = sum(len(r["historical_entities_retained"]) for r in rows)
        entities = sum(r["historical_entities_total"] for r in rows)
        groups = {}
        for kind in sorted({r["kind"] for r in complete}):
            group = [r["strategies"][name] for r in complete if r["kind"] == kind]
            groups[kind] = {"correct": sum(r["correct"] for r in group), "n": len(group)}
        report["strategies"][name] = {
            "n": len(rows), "coref_correct": sum(r["correct"] for r in rows),
            "coref_accuracy": sum(r["correct"] for r in rows) / len(rows) if rows else None,
            "entities_retained": retained, "entities_total": entities,
            "entity_coverage": retained / entities if entities else None,
            "agent_tokens_approx_total": sum(r["agent_input_tokens_approx"] for r in rows),
            "coref_api_input_tokens": total_usage(calls, "input_tokens"),
            "coref_api_output_tokens": total_usage(calls, "output_tokens"),
            "api_calls": len(calls), "provider_error_count": sum(len(r["errors"]) for r in rows),
            "coref_latency_seconds_total": round(sum(r["elapsed_seconds"] for r in rows), 3),
            "by_kind": groups}
    summary_calls = [c for r in complete for s in r["summary_steps"] for c in s["calls"]]
    report["summary_overhead"] = {"calls": len(summary_calls),
        "input_tokens": total_usage(summary_calls, "input_tokens"),
        "output_tokens": total_usage(summary_calls, "output_tokens"),
        "seconds_total": round(sum(s["elapsed_seconds"] for r in complete for s in r["summary_steps"]), 3)}
    full = report["strategies"]["full_history"]
    for name in ("window_only", "summary_window"):
        row = report["strategies"][name]
        row["agent_token_reduction_vs_full_approx"] = (1 - row["agent_tokens_approx_total"] / full["agent_tokens_approx_total"]) if full["agent_tokens_approx_total"] else None
        row["coref_api_input_reduction_vs_full"] = (1 - row["coref_api_input_tokens"] / full["coref_api_input_tokens"]) if full["coref_api_input_tokens"] and row["coref_api_input_tokens"] is not None else None
    return report


async def main(args):
    if args.prepare:
        if CASES.exists() and json.loads(CASES.read_text(encoding="utf-8")) != make_cases():
            raise RuntimeError("Existing fixture differs; do not overwrite a frozen evaluation set")
        write_json(CASES, make_cases())
        print(f"Prepared {len(make_cases()['cases'])} cases: {CASES}")
        return
    output = Path(args.output)
    frozen_cases = output / "cases.json"
    fixture = json.loads((frozen_cases if args.recompute and frozen_cases.exists() else CASES).read_text(encoding="utf-8"))
    chosen = fixture["cases"][:args.limit] if args.limit else fixture["cases"]
    fixture = fixture | {"cases": chosen}
    manifest_path = output / "manifest.json"
    if args.recompute:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["dataset_sha256"] != digest(fixture):
            raise RuntimeError("Dataset hash mismatch")
        results = [json.loads((output / (c["id"] + ".json")).read_text(encoding="utf-8"))
                   for c in chosen if (output / (c["id"] + ".json")).exists()]
    else:
        config = {"chat_model": settings.chat_model, "summary_model": settings.summary_model or settings.chat_model,
                  "summary_trigger_messages": settings.summary_trigger_messages,
                  "context_window_turns": settings.context_window_turns,
                  "context_window_max_tokens": settings.context_window_max_tokens,
                  "coref_recent_message_limit": 6, "thinking": settings.chat_thinking,
                  "summary_and_coref_temperature": .3}
        hashes = {file: hashlib.sha256((ROOT / file).read_bytes()).hexdigest() for file in SOURCE_FILES}
        signature = {"dataset_sha256": digest(fixture), "config": config, "source_sha256": hashes}
        manifest = signature | {"created_at": datetime.now(timezone.utc).isoformat(),
                                "strategies": list(STRATEGIES), "concurrency": args.concurrency}
        if manifest_path.exists():
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            if any(saved[k] != signature[k] for k in signature):
                raise RuntimeError("Source/config/dataset changed; use a new output directory")
            manifest = saved
        write_json(manifest_path, manifest)
        write_json(frozen_cases, fixture)
        for file in SOURCE_FILES:
            snapshot = output / "source_snapshot" / file
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes((ROOT / file).read_bytes())
        sem = asyncio.Semaphore(args.concurrency)
        async def run(case):
            async with sem:
                return await evaluate_case(case, output / (case["id"] + ".json"), args.timeout)
        results = await asyncio.gather(*(run(case) for case in chosen))
    report = make_report(fixture, manifest, results)
    write_json(output / "report.json", report)
    print(json.dumps({"report": str(output / 'report.json'), "completed": report["completed_cases"],
                      "strategies": report["strategies"], "summary_overhead": report["summary_overhead"]}, ensure_ascii=False, indent=2))
    if report["completed_cases"] != len(chosen):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output", default="data/ch07/context-eval/20261016-v1")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=90)
    args = parser.parse_args()
    if args.concurrency < 1 or args.timeout <= 0 or (args.limit is not None and args.limit < 1):
        parser.error("concurrency, timeout and limit must be positive")
    asyncio.run(main(args))
