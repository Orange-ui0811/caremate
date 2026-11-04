"""Five memory strategies, production node inputs, rolling boundary keep3.

Only --run calls the authorized configured Gitee model. This process uses
isolated synthetic DB/knowledge/business fixtures and never deploys a policy.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

from langchain_core.messages import message_to_dict

from scripts import eval_context_unified as unified
from app.core import memory
from app.core.context_policy import estimate
from app.core.legacy_history_recall import old_dialogue_supplement

base = unified.base
ARMS = ("full", "window", "summary", "retrieval", "hybrid")
LABELS = {"full": "完整历史", "window": "仅滑动窗口", "summary": "滚动摘要＋滑窗",
          "retrieval": "相关原文检索＋近期窗口", "hybrid": "滚动摘要＋滑窗＋原文回查（本方案）"}
COMMON = {**unified.SETTINGS, "context_window_turns": 3, "summary_trigger_messages": 30,
          "context_window_max_tokens": 3000, "context_legacy_recall_enabled": False,
          "context_legacy_recall_mode": "archived"}
ORIGINAL_HISTORY, ORIGINAL_AGENT = base.nodes._history_text, base.nodes._agent_messages
ORIGINAL_ISOLATED, ORIGINAL_AUDIT = unified.isolated_workflow, unified.audit
EXTRA_SOURCES = ["scripts/eval_context_keep3.py", "scripts/context_keep3_cases.py",
    "scripts/audit_context_keep3.py", "scripts/analyze_context_keep3.py",
    "scripts/run_context_keep3_batch.py", "app/core/legacy_history_recall.py",
    "tests/test_context_keep3.py", "tests/test_legacy_history_recall.py"]


def dataset(pilot=False):
    from scripts import context_keep3_cases
    return context_keep3_cases.dataset(pilot)


def effective_config():
    return {"environment": base.config(), "overrides": COMMON, "labels": LABELS,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "unset"),
        "thinking": base.OFF, "failover": False, "keep_turns": 3, "trigger_messages": 30,
        "coref_intent_recent_messages": 6, "agent_original_window_tokens": 3000,
        "query_rewrite": "unchanged effective_query only, no added history",
        "recall": "same production supplement: summarized/retired prefix, user-only BM25 top3 plus next old unit, chronological, max6 units",
        "full": "all visible raw messages at history-consuming nodes, unchanged query rewrite protocol",
        "window_only": "same batch boundary progression as summary arms, but model-free and summary text empty",
        "full_request_cap": "no experiment-added token blocking; provider errors retained; original node window trimming kept except full",
        "replicates": "each output directory is one independent replicate; no semantic reruns"}


def replay_view(state, arm, function, max_turns=6, tool_schemas=None):
    """Pure projection under explicit local settings; returns JSON-ready views."""
    if arm not in ARMS or function not in {"history", "answer"}:
        raise ValueError("unknown projection policy")
    with ExitStack() as stack:
        stack.enter_context(patch.object(base.settings, "context_policy", "legacy"))
        stack.enter_context(patch.object(base.settings, "context_window_max_tokens", 3000))
        stack.enter_context(patch.object(base.settings, "context_legacy_recall_enabled", False))
        stack.enter_context(patch.object(base.settings, "context_legacy_recall_mode", "archived"))
        if arm == "full":
            stack.enter_context(patch.object(memory, "build_window", lambda messages, summary_upto_msg_id, max_tokens: list(messages)))

        def view():
            if function == "history":
                count = len(state.get("messages", [])) if arm == "full" else max_turns
                return {"history": ORIGINAL_HISTORY(state, count), "messages": []}
            messages = ORIGINAL_AGENT(state, tool_schemas=tool_schemas)
            return {"history": "\n".join(str(m.content) for m in messages),
                    "messages": [message_to_dict(m) for m in messages]}

        unchanged = view()
        supplement = {"text": "", "manifest": {"selected_message_ids": []}}
        if arm in {"retrieval", "hybrid"}:
            supplement = old_dialogue_supplement(state)
            with patch.object(base.settings, "context_legacy_recall_enabled", True):
                actual = view()
        else:
            actual = deepcopy(unchanged)
        return {"packet": actual, "base_packet": unchanged, "supplement": supplement}


@contextmanager
def isolated_workflow(fixture, arm, recorder):
    production_schedule = base.summarizer.maybe_schedule_summary
    with ORIGINAL_ISOLATED(fixture, "legacy", recorder) as (graph, _), ExitStack() as stack:
        for key, value in COMMON.items():
            stack.enter_context(patch.object(base.settings, key, value))
        projections, maintenance = [], []

        def measure_payload(payload):
            parts = {k: payload[k] for k in ("messages", "tools", "functions", "response_format") if k in payload}
            return estimate(json.dumps(parts, ensure_ascii=False, default=str))
        stack.enter_context(patch.object(base.context_runtime, "check_payload", measure_payload))

        def record_view(state, function, stage="answer", max_turns=6, tool_schemas=None):
            started = time.monotonic()
            result = replay_view(state, arm, function, max_turns, tool_schemas)
            serial = {k: ([message_to_dict(m) for m in v] if k == "messages" else deepcopy(v)) for k,v in state.items()}
            projections.append({"coords": deepcopy(base.COORDS.get()), "stage": stage, "function": function,
                "assembler_input": {"state": serial, "max_turns": max_turns, "tool_schemas": deepcopy(tool_schemas), "arm": arm},
                **deepcopy(result), "assembly_seconds": time.monotonic() - started})
            unified.write(recorder.out / "projections" / f"{arm}-{fixture.case['id']}.json", projections)
            return result["packet"]

        def history(state, max_turns=6, *, stage="coref"):
            return record_view(state, "history", stage, max_turns)["history"]

        def agent_messages(state, *, tool_schemas=None):
            from langchain_core.messages import messages_from_dict
            return messages_from_dict(record_view(state, "answer", tool_schemas=tool_schemas)["messages"])

        stack.enter_context(patch.object(base.nodes, "_history_text", history))
        stack.enter_context(patch.object(base.nodes, "_agent_messages", agent_messages))

        async def schedule(cid):
            started, before_calls = time.monotonic(), set(recorder.calls)
            old_summary, old_upto = fixture.summary, fixture.summary_upto
            n = await fixture.count(cid, old_upto)
            error, triggered = None, False
            mode = "summary" if arm in {"summary", "hybrid"} else "window" if arm in {"window", "retrieval"} else "none"
            if mode == "summary":
                await production_schedule(cid)
                task = base.summarizer._running.get(cid)
                triggered = task is not None
                if task is not None:
                    try:
                        await task
                    except Exception as exc:
                        error = type(exc).__name__
            elif mode == "window" and n >= 30:
                triggered = True
                boundary = base.summarizer.compute_boundary(await fixture.dialog(cid), 3)
                if boundary is None or boundary <= old_upto:
                    error = "InvalidWindowBoundary"
                else:
                    fixture.summary_upto = boundary
            maintenance.append({"coords": deepcopy(base.COORDS.get()), "policy": arm, "mode": mode,
                "triggered": triggered, "error_type": error, "seconds": time.monotonic() - started,
                "previous_summary": old_summary, "previous_summary_upto": old_upto,
                "legacy_summary": fixture.summary, "legacy_summary_upto": fixture.summary_upto,
                "count_after_boundary": n, "row_watermark": max((r["id"] for r in fixture.rows), default=0),
                "call_ids": sorted(set(recorder.calls) - before_calls), "new_segments": []})
            unified.write(recorder.out / "maintenance" / f"{arm}-{fixture.case['id']}.json", maintenance)
        stack.enter_context(patch.object(base.summarizer, "maybe_schedule_summary", schedule))
        yield graph, maintenance


@contextmanager
def binding():
    with ExitStack() as stack:
        for key, value in {"ARMS": ARMS, "SETTINGS": COMMON, "dataset": dataset,
                          "effective_config": effective_config, "isolated_workflow": isolated_workflow,
                          "audit": audit, "full_preflight": lambda case: {"history_estimate": estimate(json.dumps(case['initial_history'],ensure_ascii=False)),
                              "mode": "measurement only; full never silently clipped"}}.items():
            stack.enter_context(patch.object(unified, key, value))
        stack.enter_context(patch.object(base, "SOURCES", list(dict.fromkeys(base.SOURCES + EXTRA_SOURCES))))
        yield


def prepare(out, *, pilot=False, arms=ARMS, case_ids=None, replicate=1):
    with binding():
        manifest = unified.prepare(out, arms, case_ids, pilot, turn_timeout=180)
    manifest.update(version="keep3-five-memory-methods-v1", labels=LABELS, replicate=replicate,
        local_caps="No artificial full-request input cap. Original 3000-token window and six-message history slots apply to four compact arms; full removes only these truncations.",
        comparison="Four compact arms share batch window rules: summary on/off x old-recall on/off. Full is a resource reference. Independent model summaries and histories can diverge.",
        summary={"summary": "unchanged production rolling prompt;30messages keep3", "hybrid": "same",
                 "window": "no LLM; batch window retirement uses same 30messages keep3", "retrieval": "same local retirement, with old source recall", "full": "none"},
        token_scope="all seed summary initialization, three actual follow-ups, maintenance and retries. Excludes scripted historical dialogue generation and real retrieval/business service costs",
        latency_scope="test-only maintenance barrier; parallel worker/provider queues reported, not a production latency benchmark")
    unified.write(Path(out) / "manifest.json", manifest)
    return manifest


def audit(out):
    out = Path(out).resolve()
    report = ORIGINAL_AUDIT(out)
    report["labels"] = LABELS
    report["limits"] = ["Single internal synthetic run per directory; previous cases and new authored cases reported separately.",
        "Same trajectory's successive turns are correlated; rule pass is not full factuality or production accuracy.",
        "No-summary arms retire a batch window locally with the same 30/keep3 rule; they do not call a summary model.",
        "Summary failure may delay its boundary while no-summary windows advance; retain failures and diagnose resulting divergence.",
        "All model costs including maintenance/retries recorded. Missing usage is unknown cost, never assumed zero.",
        "Constraint tests are read-only; no claim about complete positive write authorization."]
    for arm,g in report["groups"].items():
        ms=[m for p in (out/"maintenance").glob(f"{arm}-*.json") for m in unified.read(p)]
        g["maintenance_failures"]=sum(bool(m.get("error_type")) for m in ms)
        g["cost_complete"]=g["unknown_usage_calls"]==0
    unified.write(out/"audited_report.json", report)
    # Avoid inherited five-arm prose describing obsolete budgets/strategies.
    lines=["# 保留3轮：五种上下文方法", "",f"记录完整：{report['complete']}；账本/评分审计：{report['audit_passed']}。", "",
           "|方法|任务规则通过|输入Token|输出Token|已知总Token|摘要调用|未知usage|", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm,g in report["groups"].items():
        prefix="≥" if not g["cost_complete"] else ""
        lines.append(f"|{LABELS[arm]}|{g['task_rule_pass']}/{g['planned_turns']}|{g['input_tokens']}|{g['output_tokens']}|{prefix}{g['total_tokens']}|{g['maintenance_model_calls']}|{g['unknown_usage_calls']}|")
    lines += ["", "真实模型和Workflow，隔离合成业务及知识夹具。含摘要初始化、维护和重试；不含预置历史生成及真实RAG服务成本。",
              "窗口组为30消息触发、留3轮的批次窗口；coref/intent仍取最近6条消息，回答保留原3000Token裁剪，查询改写输入方式不变。",
              "独立来源与窗口回放详见AUDIT_DETAIL.json；未生成该文件前不声称独立审计通过。"]
    (out/"AUDITED_REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--pilot",action="store_true")
    p.add_argument("--arms",nargs="+",choices=ARMS,default=list(ARMS))
    p.add_argument("--cases",nargs="+")
    p.add_argument("--replicate",type=int,default=1)
    m=p.add_mutually_exclusive_group();m.add_argument("--run",action="store_true");m.add_argument("--audit",action="store_true")
    a=p.parse_args()
    if a.run:
        with binding():result=asyncio.run(unified.run(a.output))
    elif a.audit:result=audit(a.output)
    else:result=prepare(a.output,pilot=a.pilot,arms=tuple(a.arms),case_ids=a.cases,replicate=a.replicate)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
