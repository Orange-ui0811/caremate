"""Read-only cost/failure analysis for completed compact Workflow runs."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def cost(calls):
    calls = list(calls)
    return {"calls": len(calls),
        "input_tokens": sum((c.get("usage") or {}).get("input_tokens", 0) for c in calls),
        "output_tokens": sum((c.get("usage") or {}).get("output_tokens", 0) for c in calls),
        "total_tokens": sum((c.get("usage") or {}).get("input_tokens", 0) + (c.get("usage") or {}).get("output_tokens", 0) for c in calls),
        "cache_read_tokens": sum(((c.get("usage") or {}).get("input_token_details") or {}).get("cache_read", 0) for c in calls),
        "reasoning_tokens": sum(((c.get("usage") or {}).get("output_token_details") or {}).get("reasoning", 0) for c in calls),
        "successful_calls_without_reasoning_usage": sum(c["status"] == "ok" and "reasoning" not in ((c.get("usage") or {}).get("output_token_details") or {}) for c in calls),
        "unknown_usage_calls": sum(not c.get("usage") for c in calls),
        "status": dict(Counter(c["status"] for c in calls)),
        "protocol_invalid_calls": sum(c.get("structured_response_check", {}).get("status") == "invalid" for c in calls)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args()
    out = Path(args.output).resolve()
    if (out / "writer.lock").exists():
        raise ValueError("writer still active")
    report = read(out / "audited_report.json")
    if not report["complete"] or not report["audit_passed"]:
        raise ValueError("completed passing audit required")
    manifest = read(out / "manifest.json")
    cases = {c["id"]: c for c in read(out / "dataset.json")["cases"]}
    calls = {f.stem: read(f) for f in (out / "calls").glob("*.json")}
    turns = {f.stem: read(f) for f in (out / "turns").glob("*.json")}
    cited = [cid for r in turns.values() for cid in r["call_ids"]]
    if set(cited) != set(calls) or len(cited) != len(set(cited)):
        raise ValueError("each SDK call must belong to exactly one turn")
    projections = [r for f in (out / "projections").glob("*.json") for r in read(f)]
    pool = {}
    unavailable_pool_turns = []
    for key, turn in turns.items():
        if turn.get("pool_attempts") is None:
            unavailable_pool_turns.append(key)
        for attempt in turn.get("pool_attempts") or []:
            if attempt["id"] in pool and pool[attempt["id"]] != attempt:
                raise ValueError("conflicting pool attempt")
            pool[attempt["id"]] = attempt
    groups = {}
    for arm in manifest["arms"]:
        acalls = [c for c in calls.values() if c["coords"]["arm"] == arm]
        aturns = [t for t in turns.values() if t["coords"]["arm"] == arm]
        info = cost(acalls)
        if any(info[k] != report["groups"][arm][k] for k in ("input_tokens", "output_tokens")):
            raise ValueError("known usage differs from audit")
        info["by_stage"] = {purpose: cost(c for c in acalls if c["purpose"] == purpose)
                            for purpose in sorted({c["purpose"] for c in acalls})}
        info["strata"] = {}
        for stratum in sorted({c["stratum"] for c in cases.values()}):
            tc = [t for t in aturns if cases[t["coords"]["case_id"]]["stratum"] == stratum]
            cc = [c for c in acalls if cases[c["coords"]["case_id"]]["stratum"] == stratum]
            info["strata"][stratum] = {**cost(cc), "turns": len(tc),
                "passed": sum(t["score"]["task_rule_pass"] for t in tc),
                "end_to_end_p50_seconds": statistics.median(t["end_to_end_seconds"] for t in tc)}
        pv = [r for r in projections if r["coords"]["arm"] == arm]
        info["actual_projection_budgets"] = {stage: dict(Counter(
            str(r["manifest"].get("budget")) for r in pv if r["stage"] == stage)) for stage in sorted({r["stage"] for r in pv})}
        info["failed_turns"] = [{"case_id": t["coords"]["case_id"], "turn": t["coords"]["turn"],
            "query": t["query"], "expected": t["expected"], "answer": t["answer"], "score": t["score"],
            "executed_tools": t["executed_tools"], "proposed_tools": t["proposed_tools"]}
            for t in aturns if not t["score"]["task_rule_pass"]]
        groups[arm] = info
    a, b = groups["source"], groups["compact_source"]
    result = {"complete_audit_verified": True, "groups": groups,
        "compact_relative_total_token_change_pct": (b["total_tokens"] / a["total_tokens"] - 1) * 100,
        "sdk_calls_uniquely_assigned": len(calls), "unique_pool_attempts": len(pool),
        "pool_outcomes": dict(Counter(r["outcome"] for r in pool.values())),
        "pool_http_status": dict(Counter(str(r["http_status"]) for r in pool.values())),
        "pool_attempts_unavailable_turns": unavailable_pool_turns,
        "pool_unsuccessful_or_unfinished": [r for r in pool.values() if r.get("http_status") != 200 or not r.get("finished")],
        "limits": [
            "All generated follow-up and any triggered maintenance callback usage is included; supplied history generation is excluded.",
            "Unknown usage remains unknown. Underlying pool failures can incur unknown provider work even if the SDK later succeeds.",
            "Fixed sequential groups and a small synthetic set do not identify causal production latency improvements.",
            "Tools/RAG/DB are fixtures; task pass is a rule score. The retraction pressure case only executes logistics, not write authorization.",
            "Read-only analysis does not call models, re-score, or replace any frozen output.",
        ]}
    target = out / "analysis" / "WORKFLOW_COST_ANALYSIS.json"
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        raise ValueError("existing analysis preserved")
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"groups": {a: {k: v for k, v in g.items() if k in ("calls", "total_tokens", "unknown_usage_calls", "strata")} for a, g in groups.items()},
                      "pool_outcomes": result["pool_outcomes"], "unsuccessful_pool_attempts": len(result["pool_unsuccessful_or_unfinished"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
