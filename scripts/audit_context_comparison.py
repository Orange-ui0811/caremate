"""Offline integrity/scoring audit for saved customer-service context comparisons."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.eval_context_comparison import aggregate, sha, score_probe, write_json
from scripts.context_benchmark_strategies import STRATEGIES


def audit(output):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    fixture = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
    issues = []
    if sha(fixture) != manifest["dataset_sha256"]:
        issues.append("dataset hash mismatch")
    for prefix, key in (("mewhelp", "source_sha256"), ("harness", "harness_sha256")):
        for name, expected in manifest[key].items():
            path = output / "source_snapshot" / prefix / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                issues.append("snapshot hash mismatch: " + prefix + "/" + name)
    checked, maintenance_checked = 0, 0
    for case in fixture["cases"]:
        for name in STRATEGIES:
            path = output / "cases" / case["id"] / (name + ".json")
            if not path.exists():
                issues.append(f"missing result {case['id']}/{name}")
                continue
            row = json.loads(path.read_text(encoding="utf-8"))
            if len(row["probes"]) != len(case["probes"]):
                issues.append(f"incomplete probes {case['id']}/{name}")
            for result in row["probes"]:
                probe = next(p for p in case["probes"] if p["id"] == result["probe_id"])
                if result["score"] != score_probe(case, probe, result):
                    issues.append(f"score mismatch {case['id']}/{name}/{probe['id']}")
                for stage in ("coref", "plan", "reply"):
                    for call in result[stage]["calls"]:
                        if call["input_tokens"] is None or call["output_tokens"] is None:
                            issues.append(f"missing usage {case['id']}/{name}/{stage}")
                checked += 1
            for probe in case["probes"]:
                snapshot = row["snapshots"].get(probe["id"])
                if snapshot is None or snapshot["ingested"] != 2 * probe["at_turn"]:
                    issues.append(f"wrong causal checkpoint {case['id']}/{name}/{probe['id']}")
                elif snapshot["boundary"] > 2 * probe["at_turn"]:
                    issues.append(f"future summary {case['id']}/{name}/{probe['id']}")
            for event in row.get("maintenance", []):
                # Exact test prompts must never be used to construct a query-independent memory.
                prompt = json.dumps(event.get("prompt"), ensure_ascii=False)
                if any(p["query"] in prompt for p in case["probes"]):
                    issues.append(f"probe leakage in maintenance {case['id']}/{name}")
                if event["ok"] and event["stage"] == "state":
                    if any(i > event["boundary"] or i < 1 for f in event["value"]["facts"] for i in f["source_ids"]):
                        issues.append(f"future source id {case['id']}/{name}")
                maintenance_checked += 1
        d = json.loads((output / "cases" / case["id"] / "D_state.json").read_text(encoding="utf-8"))
        e = json.loads((output / "cases" / case["id"] / "E_state_retrieval.json").read_text(encoding="utf-8"))
        if d["snapshots"] != e["snapshots"]:
            issues.append(f"D/E extraction differs {case['id']}")
    old = hashlib.sha256((output / "report.json").read_bytes()).hexdigest() if (output / "report.json").exists() else None
    report = aggregate(output)
    new = hashlib.sha256((output / "report.json").read_bytes()).hexdigest()
    result = {"passed": not issues and report["complete"], "issues": issues,
              "probes_checked": checked, "maintenance_records_checked": maintenance_checked,
              "report_sha256": new, "offline_recompute_identical": old == new,
              "physical_api_calls": report["physical_successful_api_calls"]}
    write_json(output / "audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(Path(args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
