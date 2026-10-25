"""Uniform offline fix for a care-answer scoring bug: its own order is not contamination.

The original evaluator mistakenly forbids every order number in care replies,
including an order explicitly linked to the correct SKU in the visible history.
This audit changes only that condition, retains every raw result and initial
report, and applies the same correction to all arms without new model calls.
"""
import argparse
import hashlib
import json
from pathlib import Path

import scripts.eval_context_comparison as evaluation
from scripts.audit_context_comparison import audit as integrity_audit
from scripts.context_benchmark_strategies import STRATEGIES

original_score = evaluation.score_probe


def audited_score(case, probe, result):
    score = original_score(case, probe, result)
    if probe["expected"]["action"] != "care":
        return score
    sku = probe["expected"]["sku"]
    visible = case["history"][:2 * probe["at_turn"]]
    related_orders = {oid for row in visible if sku in row["content"]
                      for oid in case["all_orders"] if oid in row["content"]}
    reply = result.get("reply", {}).get("value") or ""
    answer_ok = (sku.upper() in reply.upper() and "断电" in reply and "软布" in reply
                 and not any(oid in reply for oid in set(case["all_orders"]) - related_orders))
    score["answer_rule_pass"] = answer_ok
    score["task_pass"] = bool(score["api_success"] and score["target_pass"] and answer_ok)
    return score


def audit(output):
    # The original auditor verifies frozen source, initial scores and causal inputs first.
    integrity = integrity_audit(output)
    if not integrity["passed"]:
        raise RuntimeError("Initial integrity audit failed: " + json.dumps(integrity["issues"]))
    originals = (("report.json", "initial_report.json"), ("REPORT.md", "INITIAL_REPORT.md"),
                 ("failures.json", "initial_failures.json"))
    for source, dest in originals:
        if not (output / dest).exists():
            (output / dest).write_bytes((output / source).read_bytes())
    changes = []
    fixture = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
    for case in fixture["cases"]:
        for name in STRATEGIES:
            path = output / "cases" / case["id"] / (name + ".json")
            row = json.loads(path.read_text(encoding="utf-8"))
            for result in row["probes"]:
                probe = next(p for p in case["probes"] if p["id"] == result["probe_id"])
                before, after = original_score(case, probe, result), audited_score(case, probe, result)
                if before != after:
                    changes.append({"case": case["id"], "strategy": name, "probe": probe["id"],
                                    "before": before, "after": after, "reply": result["reply"]["value"]})
    evaluation.score_probe = audited_score
    saved_bootstrap = evaluation.bootstrap_difference
    # The frozen evaluator iterates an unordered set before drawing. Canonicalize
    # the paired differences so identical files yield identical offline intervals.
    evaluation.bootstrap_difference = lambda pairs: saved_bootstrap(sorted(pairs))
    try:
        report = evaluation.aggregate(output)
    finally:
        evaluation.score_probe = original_score
        evaluation.bootstrap_difference = saved_bootstrap
    report["scoring_audit"] = {"rule": "Care answer may include orders explicitly paired with its correct SKU in visible history; unrelated orders still fail.",
                               "changed_rows": len(changes), "total_rows": len(fixture["cases"]) * 2 * len(STRATEGIES),
                               "new_model_calls": 0}
    evaluation.write_json(output / "audited_report.json", report)
    (output / "AUDITED_REPORT.md").write_bytes((output / "REPORT.md").read_bytes())
    evaluation.write_json(output / "score_changes.json", changes)
    # Keep original public report reproducible with the frozen evaluator; audited files are authoritative.
    for source, dest in originals:
        (output / source).write_bytes((output / dest).read_bytes())
    audit_source = Path(__file__)
    (output / "semantic_audit_source.py").write_bytes(audit_source.read_bytes())
    final = {"passed": True, "integrity": integrity, "changed_rows": len(changes),
             "new_model_calls": 0,
             "semantic_audit_sha256": hashlib.sha256(audit_source.read_bytes()).hexdigest(),
             "audited_report_sha256": hashlib.sha256((output / "audited_report.json").read_bytes()).hexdigest()}
    evaluation.write_json(output / "semantic_audit.json", final)
    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(Path(args.output)), ensure_ascii=False, indent=2))
