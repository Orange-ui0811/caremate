"""Read-only paired analysis of the frozen compact-context experiment.

No model calls or rescoring. Provider receipts, not heuristic context sizes,
determine Token totals. Artifacts are written only to a new analysis filename.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from scripts.context_state_analysis_stats import (
    family_cluster_difference, group_calls, reconcile_attempt_usage, usage,
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metrics(rows):
    rows = list(rows)
    return {"n": len(rows), "task_passed": sum(r["score"]["task_pass"] for r in rows),
            "target_passed": sum(r["score"]["target_pass"] for r in rows),
            "transport_failed": sum(r["score"]["transport_failure"] for r in rows),
            "protocol_failed": sum(r["score"]["protocol_failure"] for r in rows),
            "budget_blocked": sum(r["score"]["budget_blocked"] for r in rows)}


def analyze(out):
    audit = read(out / "audited_report.json")
    if not audit["complete"] or not audit["audit_passed"]:
        raise ValueError("completed passing audit is required before final analysis")
    manifest = read(out / "manifest.json")
    cases = {c["id"]: c for c in read(out / "dataset.json")["cases"]}
    results = {p.stem: read(p) for p in (out / "results").glob("*.json")}
    calls = {p.stem: read(p) for p in (out / "calls").glob("*.json")}
    attempts = {p.stem: read(p) for p in (out / "attempts").glob("*.json")}
    maintenance = {p.stem: read(p) for p in (out / "maintenance").glob("*.json")}
    arms = manifest["arms"]
    grouped, unassigned = group_calls(cases, arms, calls)
    if unassigned:
        raise ValueError("unassigned call receipts: " + str(unassigned))
    costs, strata, families = {}, {}, {}
    for arm in arms:
        records = [call for cid in cases for call in grouped[cid, arm]]
        stages = sorted({r["contract"]["stage"] for r in records})
        events = [e for cid in cases for e in maintenance[f"{cid}-{arm}"]]
        costs[arm] = {
            "all": usage(records),
            "stages": {s: usage(r for r in records if r["contract"]["stage"] == s) for s in stages},
            "call_status": dict(Counter(r["status"] for r in records)),
            "summary_events": len(events),
            "summary_accepted": sum(e.get("accepted", False) for e in events),
            "summary_rejected": sum(e.get("accepted") is False for e in events),
        }
        for field, target in (("length_stratum", strata), ("kind", families)):
            for value in sorted({c[field] for c in cases.values()}):
                ids = {cid for cid, c in cases.items() if c[field] == value}
                rr = [r for r in results.values() if r["arm"] == arm and r["case_id"] in ids]
                cc = [c for cid in ids for c in grouped[cid, arm]]
                target.setdefault(value, {})[arm] = {**metrics(rr), "usage": usage(cc)}
        expected = audit["strategies"][arm]["usage"]
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            if costs[arm]["all"][key] != expected[key]:
                raise ValueError("receipt total differs from audit: " + arm + " " + key)
    reconciliation = reconcile_attempt_usage(calls, attempts, costs)
    if not reconciliation["all_receipts_accounted"]:
        raise ValueError("attempt reconciliation failed")
    comparisons = {}
    for first, second in (("M", "CW"), ("M", "CS"), ("CW", "CS"), ("C", "CS")):
        if first not in arms or second not in arms:
            continue
        wins, losses, same = [], [], []
        fixed = []
        clean_first, clean_second, excluded = [], [], []
        for cid, case in cases.items():
            for step in case["steps"]:
                key = f"{cid}-{step['id']}"
                a, b = (results[key + "-" + arm] for arm in (first, second))
                av, bv = a["score"]["task_pass"], b["score"]["task_pass"]
                (wins if bv and not av else losses if av and not bv else same).append(key)
                if a["history_sha256"] == b["history_sha256"]:
                    fixed.append(key)
                pair_ok = all(r[s]["ok"] for r in (a, b) for s in ("coref", "plan", "reply"))
                maintenance_ok = all(
                    c["ok"] for arm in (first, second) for c in grouped[cid, arm]
                    if c["contract"]["stage"] == "summary"
                ) and not any(e.get("accepted") is False for arm in (first, second)
                              for e in maintenance[f"{cid}-{arm}"])
                if pair_ok and maintenance_ok:
                    clean_first.append(a)
                    clean_second.append(b)
                else:
                    excluded.append(key)
        a, b = costs[first]["all"], costs[second]["all"]
        comparisons[first + "->" + second] = {
            "first": first, "second": second,
            "total_token_difference": b["total_tokens"] - a["total_tokens"],
            "relative_total_token_change_pct": (b["total_tokens"] / a["total_tokens"] - 1) * 100,
            "family_cluster_task_difference": family_cluster_difference(cases, results, first, second),
            "improved_checkpoints": wins, "regressed_checkpoints": losses, "tied_checkpoints": same,
            "identical_history_checkpoints": fixed,
            "same_zero_call_failure_subset": {"first": metrics(clean_first), "second": metrics(clean_second),
                "excluded": excluded,
                "definition": "Both arms' three foreground calls are ok, and both conversations' maintenance calls are ok with no rejected segment.",
                "limits": "Post-treatment selection, potentially biased. Does not establish memory-only causality or replace the full denominator; earlier failed foreground turns can still affect later context."},
            "same_history_scope": "Prior raw history hash only; independently generated summaries can still differ. Later equal history is an observed subset, not a randomized causal sample.",
        }
    stage_visibility = {}
    for arm in ("CW", "CS"):
        if arm not in arms:
            continue
        stage_visibility[arm] = {}
        for stage in ("coref", "plan", "answer"):
            packets = [r["stage_contexts"][stage] for r in results.values() if r["arm"] == arm]
            stage_visibility[arm][stage] = {"packets": len(packets),
                "with_summary": sum(bool(p["manifest"].get("summary_refs")) for p in packets),
                "blocked": sum(p["blocked"] for p in packets),
                "average_estimated_context_tokens": sum(p["manifest"].get("approx_tokens", 0) for p in packets) / len(packets)}
    failures = []
    for key, row in sorted(results.items()):
        if row["score"]["task_pass"]:
            continue
        failures.append({"key": key, "arm": row["arm"], "case_id": row["case_id"],
                         "family": row["family"], "probe_id": row["probe_id"], "query": row["query"],
                         "coref": row["coref"]["value"], "plan": row["plan"]["value"],
                         "reply": row["reply"]["value"], "score": row["score"],
                         "call_status": {s: row[s]["status"] for s in ("coref", "plan", "reply")}})
    return {"experiment_version": manifest["version"], "audit_passed": True,
            "audit_sha256": hashlib.sha256((out / "audited_report.json").read_bytes()).hexdigest(),
            "case_count": len(cases), "families": len({c["kind"] for c in cases.values()}),
            "result_count": len(results), "arms": arms,
            "results": {a: metrics(r for r in results.values() if r["arm"] == a) for a in arms},
            "costs": costs, "strata": strata, "by_family": families,
            "comparisons": comparisons, "attempt_reconciliation": reconciliation,
            "stage_summary_visibility": stage_visibility,
            "all_rule_failures": failures,
            "limits": [
                "Previously inspected synthetic regression set, not unseen holdout or real production traffic.",
                "Frozen rule passes are not independently human-labelled semantic accuracy.",
                "All saved attempts and maintenance included; supplied history generation excluded. Cache reads are already included in input.",
                "Shared reader includes separate coref, planning and reply calls. Savings do not equal production Workflow savings.",
                "Single-run paired family intervals are descriptive; equal scores do not prove noninferiority or losslessness.",
                "Semantic failures retained; no rescoring or model requests in this analysis.",
            ]}


def markdown(report):
    lines = ["# 紧凑上下文回归：效果与 Token", "",
             f"{report['case_count']} 条已见合成会话，{report['families']} 个场景族，{report['result_count']} 条评分；离线审计及回执核算通过。", "",
             "|组|规则通过|输入含维护|输出含维护|总 Token|调用故障/预算阻断|",
             "|---|---:|---:|---:|---:|---:|"]
    for a in report["arms"]:
        r, u = report["results"][a], report["costs"][a]["all"]
        lines.append(f"|{a}|{r['task_passed']}/{r['n']}|{u['input_tokens']:,}|{u['output_tokens']:,}|{u['total_tokens']:,}|{r['transport_failed']+r['protocol_failed']}/{r['budget_blocked']}|")
    lines += ["", "|组|指代总 Token|规划总 Token|回答总 Token|摘要总 Token|", "|---|---:|---:|---:|---:|"]
    for a in report["arms"]:
        stage = report["costs"][a]["stages"]
        lines.append("|" + a + "|" + "|".join(f"{stage.get(s, {}).get('total_tokens', 0):,}" for s in ("coref", "plan", "reply", "summary")) + "|")
    lines += ["", "|比较：后者相对前者|总 Token 变化|规则通过率差/百分点|按场景族重采样的95%描述区间|改善/退步检查点|", "|---|---:|---:|---|---|"]
    for name, value in report["comparisons"].items():
        diff = value["family_cluster_task_difference"]
        interval = diff["descriptive_95_percentile_interval_pp"]
        ci = "—" if interval is None else f"[{interval[0]:.2f}, {interval[1]:.2f}]"
        lines.append(f"|{name}|{value['relative_total_token_change_pct']:+.2f}%|{diff['mean_difference_pp']:+.2f}|{ci}|{len(value['improved_checkpoints'])}/{len(value['regressed_checkpoints'])}|")
    lines += ["", "|历史长度|组|规则通过|含维护总 Token|", "|---|---|---:|---:|"]
    for kind, arms in report["strata"].items():
        for a, row in arms.items():
            lines.append(f"|{kind}|{a}|{row['task_passed']}/{row['n']}|{row['usage']['total_tokens']:,}|")
    lines += ["", "本轮仅检验按节点缩小输入及保留来源状态的效果，不跳过指代或合并回答。C 是原系统参照，不是与 CW/CS 等预算的组件消融。", "",
              "单次合成测试不能证明无损。逐条退步/改善、全部原规则失败及原始答案保存在同名 JSON；语义复核只能补充解释，不能替换冻结分数。",
              "成本包含历史摘要维护和三次新追问，不包含生成预置历史。缓存读取已属于输入；这里比较 Token 数量，不是账单费用。实际 Workflow 的结果应单独查看。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", default="COMPACT_ANALYSIS")
    args = parser.parse_args()
    out = Path(args.output).resolve()
    if Path(args.name).name != args.name:
        raise ValueError("analysis name must be a filename stem")
    report = analyze(out)
    dest = out / "analysis"
    dest.mkdir(exist_ok=True)
    for suffix, text in ((".json", json.dumps(report, ensure_ascii=False, indent=2)), (".md", markdown(report))):
        path = dest / (args.name + suffix)
        if path.exists():
            raise ValueError("existing analysis preserved: " + str(path))
        path.write_text(text, encoding="utf-8")
    print(json.dumps({"results": report["results"], "receipts_reconciled": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
