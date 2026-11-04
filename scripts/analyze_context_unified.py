"""Read-only analysis of unified real-Workflow experiments, with no model calls.

Never rescoring answers: task scores are copied from frozen turn records.
Counterfactual task ablations only measure evidence selection and estimated
context size on an identical prefix. They do NOT estimate task pass rates.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(directory, *, lists=False):
    result = []
    for path in sorted(directory.glob("*.json")):
        item = read(path)
        for value in item if lists and isinstance(item, list) else [item]:
            result.append((path, value))
    return result


def coordinate(record):
    c = record.get("coords") or {}
    return str(c.get("arm", "unknown")), str(c.get("case_id", "unknown")), int(c.get("turn", 0))


def passing(record):
    score = record.get("score") or {}
    value = score.get("task_rule_pass") if isinstance(score, dict) else score
    return value if isinstance(value, bool) else None


def percentile(values, percent):
    values = sorted(v for v in values if isinstance(v, (int, float)))
    if not values:
        return None
    index = (len(values) - 1) * percent / 100
    low = int(index)
    return round(values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (index - low), 4)


def call_stats(calls):
    result = {"calls": len(calls), "known_input_tokens": 0, "known_output_tokens": 0,
              "known_total_tokens": 0, "missing_usage_calls": 0, "call_errors": 0,
              "invalid_structured_outputs": 0, "cache_read_reported": 0, "reasoning_output_reported": 0}
    for call in calls:
        usage = call.get("usage") or {}
        ins, outs = usage.get("input_tokens"), usage.get("output_tokens")
        if not isinstance(ins, (int, float)) or not isinstance(outs, (int, float)):
            result["missing_usage_calls"] += 1
        else:
            result["known_input_tokens"] += ins
            result["known_output_tokens"] += outs
            result["known_total_tokens"] += ins + outs
        result["call_errors"] += call.get("status") not in ("ok", "success", "completed")
        result["invalid_structured_outputs"] += (call.get("structured_response_check") or {}).get("status") == "invalid"
        result["cache_read_reported"] += (usage.get("input_token_details") or {}).get("cache_read", 0) or 0
        result["reasoning_output_reported"] += (usage.get("output_token_details") or {}).get("reasoning", 0) or 0
    return result


def group_stats(turns, calls):
    scores = [passing(t) for t in turns]
    scored = [s for s in scores if s is not None]
    data = {"turn_records": len(turns), "scored_turns": len(scored), "passed": sum(scored),
            "task_rule_rate": sum(scored) / len(scored) if scored else None,
            "run_incomplete_or_failed": sum(t.get("status") != "completed" for t in turns),
            "end_to_end_p50_seconds": percentile([t.get("end_to_end_seconds") for t in turns], 50),
            "end_to_end_p95_seconds": percentile([t.get("end_to_end_seconds") for t in turns], 95),
            "first_visible_delta_p50_seconds": percentile([t.get("first_visible_sse_delta_seconds") for t in turns], 50),
            **call_stats(calls)}
    data["known_tokens_per_scored_turn_including_maintenance"] = data["known_total_tokens"] / len(scored) if scored else None
    purposes = sorted({c.get("purpose", "unknown") for c in calls})
    data["by_purpose"] = {p: call_stats([c for c in calls if c.get("purpose", "unknown") == p]) for p in purposes}
    return data


def compare_pairs(turn_map, arms, cases):
    result = []
    for left, right in combinations(arms, 2):
        common = sorted({(case, turn) for arm, case, turn in turn_map if arm == left}
                        & {(case, turn) for arm, case, turn in turn_map if arm == right})
        details = []
        for case, turn in common:
            a, b = turn_map[(left, case, turn)], turn_map[(right, case, turn)]
            pa, pb = passing(a), passing(b)
            details.append({"case_id": case, "turn": turn, "family": cases.get(case, {}).get("family", case),
                            "stratum": cases.get(case, {}).get("stratum", "unspecified"),
                            "left_pass": pa, "right_pass": pb,
                            "both_completed": a.get("status") == b.get("status") == "completed"})
        scored = [d for d in details if isinstance(d["left_pass"], bool) and isinstance(d["right_pass"], bool)]
        result.append({"left": left, "right": right, "paired_records": len(common), "paired_scores": len(scored),
                       "right_wins": sum(not d["left_pass"] and d["right_pass"] for d in scored),
                       "right_losses": sum(d["left_pass"] and not d["right_pass"] for d in scored),
                       "both_pass": sum(d["left_pass"] and d["right_pass"] for d in scored),
                       "both_fail": sum(not d["left_pass"] and not d["right_pass"] for d in scored),
                       "details": details})
    return result


def offline_ablations(projections, manifest):
    """Replay frozen inputs, not independently diverging arm trajectories."""
    from app.core.context_task import assemble_task_context
    issues, result = [], []
    task_items = [item for _, item in projections if coordinate(item)[0] == "task"]
    if not task_items:
        return {"scope": "local assembly only; no model inference or task accuracy", "items": [], "totals": {}, "skipped": 0, "issues": []}
    hashes = manifest.get("sources") or {}
    for relative in ("app/core/context_task.py", "app/core/context_policy.py"):
        expected = hashes.get(relative)
        if expected is None or expected != digest(ROOT / relative):
            issues.append(f"Cannot faithfully replay: frozen hash absent or differs for {relative}")
    if issues:
        return {"scope": "local assembly only; no model inference or task accuracy", "items": [],
                "skipped": len(task_items), "issues": issues}
    for item in task_items:
        inputs = item.get("assembler_input")
        if not isinstance(inputs, dict) or not isinstance(inputs.get("snapshot"), dict):
            issues.append(f"Missing frozen assembler_input: {coordinate(item)} {item.get('stage')}")
            continue
        args = {k: v for k, v in inputs.items() if k in {
            "snapshot", "query", "stage", "policy", "budget", "summaries", "retrieval", "source_labels", "recent_turns", "current_message_id"}}
        args.setdefault("stage", item.get("stage", "answer"))
        args["policy"] = "task"
        baseline = assemble_task_context(**args)
        saved = item.get("manifest") or (item.get("packet") or {}).get("manifest") or {}
        same = all(baseline["manifest"].get(k) == saved.get(k) for k in ("selected_message_ids", "approx_tokens", "source_hash"))
        if not same:
            issues.append(f"Saved manifest does not replay: {coordinate(item)} {args['stage']}")
        variants = {}
        base_ids = set(baseline["manifest"]["selected_message_ids"])
        for name, changes in (("retrieval_off", {"retrieval": False}), ("source_labels_off", {"source_labels": False})):
            altered = assemble_task_context(**{**args, **changes})
            am = altered["manifest"]
            ids = set(am["selected_message_ids"])
            variants[name] = {"blocked": altered["blocked"], "reason": altered["reason"],
                              "approx_tokens": am["approx_tokens"], "selected_message_ids": sorted(ids),
                              "added_ids": sorted(ids - base_ids), "removed_ids": sorted(base_ids - ids),
                              "estimated_token_delta": am["approx_tokens"] - baseline["manifest"]["approx_tokens"]}
        result.append({"coords": item.get("coords"), "stage": args["stage"], "saved_manifest_reproduced": same,
                       "prefix_hash": baseline["manifest"]["source_hash"], "budget": baseline["manifest"]["budget"],
                       "baseline": {"blocked": baseline["blocked"], "approx_tokens": baseline["manifest"]["approx_tokens"],
                                    "selected_message_ids": sorted(base_ids)}, "variants": variants})
    totals = {}
    for name in ("retrieval_off", "source_labels_off"):
        totals[name] = {"views": len(result), "selection_changed_views": sum(bool(r["variants"][name]["added_ids"] or r["variants"][name]["removed_ids"]) for r in result),
                        "blocked_views": sum(r["variants"][name]["blocked"] for r in result),
                        "estimated_token_delta_sum": sum(r["variants"][name]["estimated_token_delta"] for r in result)}
    return {"scope": "local assembly only; no model inference or task accuracy", "items": result,
            "totals": totals, "skipped": len(task_items) - len(result), "issues": issues,
            "interpretation": "Token values are local context estimates, not API usage. Label removal may admit different messages; this is budget-aware assembly ablation, not a semantic causal-effect estimate."}


def analyze(output):
    manifest, dataset = read(output / "manifest.json"), read(output / "dataset.json")
    cases = {str(c["id"]): c for c in dataset["cases"]}
    arms = manifest["arms"]
    turn_files, call_files = records(output / "turns"), records(output / "calls")
    projections, maintenance = records(output / "projections", lists=True), records(output / "maintenance", lists=True)
    issues, turn_map, call_map = [], {}, {}
    for path, record in turn_files:
        key = coordinate(record)
        if key in turn_map:
            issues.append(f"Duplicate turn coordinates: {key}")
        turn_map[key] = record
    for path, call in call_files:
        cid = str(call.get("id", path.stem))
        if cid in call_map:
            issues.append(f"Duplicate call ID: {cid}")
        call_map[cid] = call
    expected = {(arm, cid, i) for arm in arms for cid, case in cases.items() for i in range(1, len(case["steps"]) + 1)}
    missing = sorted(expected - set(turn_map))
    extra = sorted(set(turn_map) - expected)
    if extra:
        issues.append(f"Unexpected turn coordinates: {extra}")
    ref_ids = {cid for _, t in turn_files + maintenance for cid in t.get("call_ids", [])}
    absent_calls = sorted(ref_ids - set(call_map))
    if absent_calls:
        issues.append(f"Referenced model call files missing: {absent_calls}")
    all_calls, all_turns = list(call_map.values()), list(turn_map.values())
    groups = {}
    for arm in arms:
        ts = [t for t in all_turns if coordinate(t)[0] == arm]
        cs = [c for c in all_calls if coordinate(c)[0] == arm]
        data = group_stats(ts, cs)
        for field in ("family", "stratum"):
            values = sorted({str(c.get(field, c["id"] if field == "family" else "unspecified")) for c in cases.values()})
            data["by_" + field] = {}
            for value in values:
                ids = {cid for cid, c in cases.items() if str(c.get(field, c["id"] if field == "family" else "unspecified")) == value}
                data["by_" + field][value] = group_stats([t for t in ts if coordinate(t)[1] in ids], [c for c in cs if coordinate(c)[1] in ids])
        ms = [m for _, m in maintenance if coordinate(m)[0] == arm]
        ps = [p for _, p in projections if coordinate(p)[0] == arm]
        data["summary"] = {"maintenance_records": len(ms), "triggered_records": sum(bool(m.get("triggered")) for m in ms),
                           "maintenance_errors": sum(bool(m.get("error_type")) for m in ms),
                           "new_segments": sum(len(m.get("new_segments") or []) for m in ms),
                           "legacy_summary_records_nonempty": sum(bool(m.get("legacy_summary")) for m in ms),
                           "maintenance_seconds": sum(m.get("seconds", 0) or 0 for m in ms),
                           "calls": call_stats([c for c in cs if c.get("purpose") == "conversation_summary"]),
                           "visible_views_by_stage": dict(Counter(p.get("stage", "unknown") for p in ps if (
                               (p.get("manifest") or (p.get("packet") or {}).get("manifest") or {}).get("summary_refs")
                               or (p.get("manifest") or (p.get("packet") or {}).get("manifest") or {}).get("legacy_summary_visible"))))}
        groups[arm] = data
    failures = []
    for path, record in turn_files:
        if passing(record) is False or record.get("status") != "completed":
            score = record.get("score") or {}
            failures.append({"coords": record.get("coords"), "file": str(path.resolve()), "status": record.get("status"),
                             "query": record.get("query"), "expected": record.get("expected"), "error": record.get("error"),
                             "frozen_score": score, "failed_checks": [k for k, v in score.get("checks", {}).items() if v is False],
                             "notice": "Original score preserved. No new semantic judgment was performed."})
    return {"created_at": datetime.now(timezone.utc).isoformat(), "output": str(output.resolve()),
            "analysis_version": "unified-offline-v1", "manifest_sha256": digest(output / "manifest.json"),
            "dataset_sha256": digest(output / "dataset.json"),
            "scope": "real Workflow records; synthetic cases; no rescoring or model API calls",
            "token_scope": manifest.get("token_scope", "all recorded follow-up and maintenance model calls; preloaded history generation excluded"),
            "completion": {"expected_turns": len(expected), "present_turns": len(turn_map), "missing_turns": missing,
                           "unscored_turns": [list(k) for k, v in turn_map.items() if passing(v) is None],
                           "all_turn_records_and_scores_present": not missing and not extra and all(passing(v) is not None for v in turn_map.values()),
                           "referenced_calls_missing": absent_calls,
                           "calls_not_in_turn_or_maintenance_call_ids": sorted(set(call_map) - ref_ids)},
            "groups": groups, "paired_comparisons": compare_pairs(turn_map, arms, cases),
            "frozen_failures": failures, "offline_task_ablations": offline_ablations(projections, manifest),
            "issues": issues,
            "limits": ["A report file does not prove the evaluation process has finished.",
                       "Known token totals exclude unknown usage; missing usage is not zero cost.",
                       "Rule pass rate is not human semantic accuracy; failure labels remain frozen.",
                       "Ablations use fixed prefixes but make no model calls, so they cannot establish task accuracy.",
                       "Maintenance barriers and synthetic tools limit latency and production-cost conclusions."]}


def markdown(data):
    lines = ["# 统一 Workflow 评测离线分析", "", "沿用原评分；未修改答案、评分或模型输出。Token 为所有记录的 API usage，包含维护与重试；缺失 usage 单列。", "",
             f"轮记录：{data['completion']['present_turns']}/{data['completion']['expected_turns']}。报告存在不代表评测进程已经结束。", "",
             "|组别|任务规则通过|已知总 Token|未知 usage 调用|接口/调用错误|结构化错误|P50 秒|", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, g in data["groups"].items():
        lines.append(f"|{arm}|{g['passed']}/{g['scored_turns']}|{g['known_total_tokens']:,}|{g['missing_usage_calls']}|{g['call_errors']}|{g['invalid_structured_outputs']}|{g['end_to_end_p50_seconds']}|")
    lines += ["", "## 配对比较", "", "右组胜/负仅比较相同会话同一追问的原始规则分数，包含执行失败；不能只展示共同成功子集。", "",
              "|左组|右组|共同评分|右胜|右负|均过|均未过|", "|---|---|---:|---:|---:|---:|---:|"]
    for p in data["paired_comparisons"]:
        lines.append(f"|{p['left']}|{p['right']}|{p['paired_scores']}|{p['right_wins']}|{p['right_losses']}|{p['both_pass']}|{p['both_fail']}|")
    lines += ["", "## 摘要维护与可见性", "", "|组别|维护检查/触发|新段|摘要调用|摘要已知 Token|可见阶段次数|", "|---|---:|---:|---:|---:|---|"]
    for arm, g in data["groups"].items():
        s = g["summary"]
        lines.append(f"|{arm}|{s['maintenance_records']}/{s['triggered_records']}|{s['new_segments']}|{s['calls']['calls']}|{s['calls']['known_total_tokens']:,}|{json.dumps(s['visible_views_by_stage'], ensure_ascii=False)}|")
    lines += ["", "## 固定前缀的装配消融", "", "下面仅离线重新选择原文，不调用模型，因此**没有消融后的任务通过率**。来源标签移除会释放预算并可能改变入选原文，不能当作纯提示措辞的因果结论。", ""]
    for name, a in data["offline_task_ablations"].get("totals", {}).items():
        lines.append(f"- {name}：{a['views']} 个视图，{a['selection_changed_views']} 个原文集合变化，估算 Token 差值合计 {a['estimated_token_delta_sum']:+,}；这不是 API 成本。")
    lines += ["", f"保留失败记录 {len(data['frozen_failures'])} 条；逐项原分、分场景族/长度层、分调用用途账单和逐视图原文 ID 差异见 JSON。", ""]
    for issue in data["issues"] + data["offline_task_ablations"].get("issues", []):
        lines.append(f"- 检查项：{issue}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="frozen experiment directory")
    parser.add_argument("--report-dir", type=Path, help="new analysis directory; defaults to output/analysis-unified")
    args = parser.parse_args()
    result = analyze(args.output)
    target = args.report_dir or args.output / "analysis-unified"
    target.mkdir(parents=True, exist_ok=True)
    (target / "unified_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "UNIFIED_ANALYSIS.md").write_text(markdown(result), encoding="utf-8")
    print(json.dumps({"report_dir": str(target), "completion": result["completion"], "issues": result["issues"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
