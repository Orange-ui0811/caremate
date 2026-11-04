"""Isolated comparison: legacy, rolling keep6, keep6+BM25, previous task view.

Reuse the frozen unified runner/recorder/scorer without changing those files.
Only --run calls the configured provider. New views have no token admission cap.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

from langchain_core.messages import message_to_dict

from scripts import eval_context_unified as unified
from scripts import context_hybrid_cases
from app.core.context_hybrid_simple import assemble_simple_context
from app.core.context_policy import estimate

base = unified.base
ARMS = ("legacy", "summary6", "hybrid6", "task")
ORIGINAL_ISOLATED = unified.isolated_workflow
ORIGINAL_AUDIT = unified.audit
EXTRA_SOURCES = ["scripts/eval_context_hybrid.py", "scripts/context_hybrid_cases.py",
                 "app/core/context_hybrid_simple.py", "tests/test_context_hybrid.py"]
LABELS = {"legacy": "原默认摘要+滑窗（保留8轮，原节点裁剪）",
          "summary6": "滚动摘要+全部未摘要原文（保留6轮）",
          "hybrid6": "保留6轮摘要滑窗+BM25旧问答前三+各自下一轮",
          "task": "此前近期原文+相关原文回查（原预算与扩展规则）"}


def effective_config():
    return {"environment": base.config(), "common_overrides": unified.SETTINGS,
        "thinking": base.OFF, "failover": False, "labels": LABELS,
        "simple": {"trigger_messages": 30, "keep_turns": 6, "summary_prompt": "unchanged production SUMMARY_PROMPT",
                   "old_hits": 3, "following_units_per_hit": 1, "token_admission": False,
                   "complete_payload_local_cap": None, "recent": "all rows after summary boundary"},
        "legacy": "unchanged keep8 maintenance and original per-node trimming",
        "task": "unchanged previous task selector, no summaries, original stage and 16000 total caps",
        "provider_limit": "provider errors preserved; new arms measure input but never trim/block by local token count"}


@contextmanager
def isolated_workflow(fixture, arm, recorder):
    if arm in {"legacy", "task"}:
        with ORIGINAL_ISOLATED(fixture, arm, recorder) as value:
            yield value
        return
    # Capture original scheduling before unified's task arm disables maintenance.
    production_schedule = base.summarizer.maybe_schedule_summary
    with ORIGINAL_ISOLATED(fixture, "task", recorder) as (graph, _), ExitStack() as stack:
        stack.enter_context(patch.object(base.settings, "context_window_turns", 6))
        projections, maintenance = [], []

        def measure_payload(payload):
            # All requests including summary and retries are still recorded.
            parts = {k: payload[k] for k in ("messages", "tools", "functions", "response_format") if k in payload}
            return estimate(json.dumps(parts, ensure_ascii=False, default=str))
        stack.enter_context(patch.object(base.context_runtime, "check_payload", measure_payload))

        def packet(state, query, stage="answer", budget=None):
            started = time.monotonic()
            inputs = {"snapshot": base.context_runtime.snapshot_from_state(state), "query": query,
                      "stage": stage, "summary": state.get("summary") or "",
                      "summary_upto": state.get("summary_upto_msg_id") or 0,
                      "retrieval": arm == "hybrid6"}
            if state.get("memory_error"):
                raise ValueError("source snapshot unavailable")
            result = assemble_simple_context(**inputs)
            saved = deepcopy(result)
            saved["messages"] = [message_to_dict(m) for m in saved["messages"]]
            projections.append({"coords": deepcopy(base.COORDS.get()), "stage": stage,
                "assembler_input": deepcopy(inputs), "packet": saved,
                "manifest": deepcopy(result["manifest"]), "caller_budget_ignored": budget,
                "assembly_seconds": time.monotonic() - started})
            unified.write(recorder.out / "projections" / f"{arm}-{fixture.case['id']}.json", projections)
            return result
        stack.enter_context(patch.object(base.context_runtime, "packet", packet))

        async def schedule(cid):
            started, before_calls = time.monotonic(), set(recorder.calls)
            before_upto, before_summary = fixture.summary_upto, fixture.summary
            n = await fixture.count(cid, before_upto)
            error, task = None, None
            # Test-only maintenance completion barrier. No simultaneous turns;
            # this patch cannot affect the running service or another process.
            with patch.object(base.settings, "context_policy", "legacy"):
                await production_schedule(cid)
                task = base.summarizer._running.get(cid)
                if task is not None:
                    try:
                        await task
                    except Exception as exc:
                        error = type(exc).__name__
            maintenance.append({"coords": deepcopy(base.COORDS.get()), "policy": arm,
                "triggered": task is not None, "error_type": error,
                "seconds": time.monotonic() - started, "count_after_boundary": n,
                "previous_summary_upto": before_upto, "previous_summary": before_summary,
                "legacy_summary": fixture.summary, "legacy_summary_upto": fixture.summary_upto,
                "new_segments": [], "row_watermark": max((r["id"] for r in fixture.rows), default=0),
                "call_ids": sorted(set(recorder.calls) - before_calls)})
            unified.write(recorder.out / "maintenance" / f"{arm}-{fixture.case['id']}.json", maintenance)
        stack.enter_context(patch.object(base.summarizer, "maybe_schedule_summary", schedule))
        yield graph, maintenance


@contextmanager
def binding():
    with ExitStack() as stack:
        for name, value in {"ARMS": ARMS, "dataset": context_hybrid_cases.dataset,
                "effective_config": effective_config, "isolated_workflow": isolated_workflow,
                "audit": audit,
                "full_preflight": lambda case: {"applicable": False, "reason": "no full-history arm; no local token cap in new arms"}}.items():
            stack.enter_context(patch.object(unified, name, value))
        stack.enter_context(patch.object(base, "SOURCES", list(dict.fromkeys(base.SOURCES + EXTRA_SOURCES))))
        yield


def prepare(out, pilot=False, arms=ARMS, case_ids=None, turn_timeout=180):
    with binding():
        manifest = unified.prepare(out, arms, case_ids, pilot, turn_timeout)
    manifest.update(version="rolling30-keep6-hybrid-v1", labels=LABELS,
        comparison="summary6 vs hybrid6 is retrieval addition; legacy and task are system references, not matched-budget ablations",
        summary={"legacy": "production rolling30 keep8", "summary6": "production rolling30 keep6",
                 "hybrid6": "same rolling30 keep6", "task": "none"},
        local_caps={"summary6": None, "hybrid6": None, "legacy": "original", "task": "original"},
        latency_scope="maintenance awaited only for reproducible accounting; end-to-end not production async latency")
    unified.write(Path(out) / "manifest.json", manifest)
    return manifest


def audit(out):
    out = Path(out).resolve()
    report = ORIGINAL_AUDIT(out)
    manifest = unified.read(out / "manifest.json")
    issues = list(report["issues"])
    replay_count = 0
    # Replay uses the checked current sources. Refuse to label it reproducible
    # if any source differs from its frozen copy; no new model calls involved.
    current_matches = all(hashlib.sha256((unified.ROOT / name).read_bytes()).hexdigest() == sha
                          for name, sha in manifest["sources"].items())
    if not current_matches:
        issues.append("current source differs from frozen sources; projection replay unavailable")
    from app.core.context_task import assemble_task_context
    for path in (out / "projections").glob("*.json"):
        items = unified.read(path)
        for item in items:
            arm = item["coords"]["arm"]
            if arm == "legacy" or not current_matches:
                continue
            expected = (assemble_task_context(**item["assembler_input"]) if arm == "task" else
                        assemble_simple_context(**item["assembler_input"]))
            expected["messages"] = [message_to_dict(m) for m in expected["messages"]]
            if expected != item["packet"]:
                issues.append("projection replay mismatch: " + path.name)
            replay_count += 1
    data = unified.read(out / "dataset.json")
    turns = [unified.read(p) for p in (out / "turns").glob("*.json")]
    for arm in manifest["arms"]:
        maintenance_records = [m for p in (out / "maintenance").glob(f"{arm}-*.json") for m in unified.read(p)]
        report["groups"][arm]["maintenance_failures"] = sum(bool(m.get("error_type")) for m in maintenance_records)
        for case in data["cases"]:
            prefix = f"{arm}-{case['id']}"
            mp = out / "maintenance" / f"{prefix}.json"
            if not mp.exists():
                continue
            rs = sorted([r for r in turns if r["coords"]["arm"] == arm and r["coords"]["case_id"] == case["id"]], key=lambda r:r["coords"]["turn"])
            seed_path = out / "seeds" / f"{prefix}.json"
            rows = rs[-1]["actual_dialog_rows"] if rs else (unified.read(seed_path)["rows"] if seed_path.exists() else [])
            checks = unified.read(mp)
            if arm not in {"summary6", "hybrid6"}:
                continue
            for m in checks:
                visible = [r for r in rows if r["id"] <= m["row_watermark"]]
                count = sum(r["id"] > m["previous_summary_upto"] for r in visible)
                if count != m["count_after_boundary"] or m["triggered"] != (count >= 30):
                    issues.append("maintenance trigger mismatch: " + prefix)
                if m["triggered"] and not m["error_type"]:
                    users = [i for i,r in enumerate(visible) if r["role"] == "user"]
                    boundary = visible[users[-6] - 1]["id"]
                    if m["legacy_summary_upto"] != boundary:
                        issues.append("keep6 boundary mismatch: " + prefix)
                    if not m["legacy_summary"] or not m["call_ids"]:
                        issues.append("summary missing output or call ledger: " + prefix)
                if not m["triggered"] or m["error_type"]:
                    if (m["legacy_summary"],m["legacy_summary_upto"]) != (m["previous_summary"],m["previous_summary_upto"]):
                        issues.append("failed/untriggered maintenance changed summary: " + prefix)
    report.update(audit_passed=not issues, complete=report["complete"] and not issues,
                  issues=issues, projection_replays=replay_count, labels=LABELS)
    report["limits"] = [x for x in report["limits"] if not x.startswith("Legacy uses")]
    report["limits"] += ["New summary6/hybrid6 arms have no local token-based truncation or admission cap. Legacy/task retain original controls.",
        "Rolling summaries generated independently per arm; stochastic summary differences can affect retrieval ablation.",
        "12 existing cases and 4 new internal long-history cases; not a blind external benchmark."]
    unified.write(out / "audited_report.json", report)
    lines = ["# 滚动摘要与旧问答回查实验", "", f"记录完整：{report['complete']}；审计通过：{report['audit_passed']}。",
             "", "|组别|任务通过|输入Token|输出Token|总Token|摘要调用|接口错误|运行失败轮|",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm,g in report["groups"].items():
        lines.append(f"|{LABELS[arm]}|{g['task_rule_pass']}/{g['planned_turns']}|{g['input_tokens']}|{g['output_tokens']}|{g['total_tokens']}|{g['maintenance_model_calls']}|{g['interface_error_calls']}|{g['failed_turns']}|")
    lines += ["", "Token含历史初始化摘要、连续追问全部节点、摘要维护及重试；不含预置原始对话生成、真实知识库与业务服务费用。",
              "新两组不按Token裁剪；旧两组保留原控制。摘要提示词不改，各组独立生成，语义错误不重跑。",
              "真实模型与Workflow，隔离合成知识/业务夹具；单次内部评测，不代表线上准确率。",
              f"离线上下文回放：{replay_count}；审计问题：{issues}"]
    (out / "AUDITED_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--cases", nargs="+")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.run:
        with binding():
            result = asyncio.run(unified.run(args.output))
    elif args.audit:
        result = audit(args.output)
    else:
        result = prepare(args.output, args.pilot, tuple(args.arms), args.cases)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
