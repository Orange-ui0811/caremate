"""Offline report for fully paired, provider-successful conversations only."""
import argparse
import json
import shutil
from pathlib import Path

from scripts import eval_context_comparison as evaluation
from scripts.audit_context_semantics import audit as semantic_audit
from scripts.context_benchmark_strategies import STRATEGIES


def paired_cases(output):
    fixture = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
    selected = []
    for case in fixture["cases"]:
        all_good = True
        for arm in STRATEGIES:
            path = output / "cases" / case["id"] / (arm + ".json")
            if not path.exists():
                all_good = False
                break
            row = json.loads(path.read_text(encoding="utf-8"))
            stages = [p[k] for p in row["probes"] for k in ("coref", "plan", "reply")]
            maintenance = row.get("maintenance", list(row["snapshots"].values())[-1]["maintenance"])
            good_stages = len(row["probes"]) == 2 and all(r["ok"] and r["calls"] and not r["errors"] for r in stages)
            good_memory = all(not r["errors"] and (r["ok"] or "validation_error" in r) for r in maintenance)
            if not (good_stages and good_memory):
                all_good = False
                break
        if all_good:
            selected.append(case)
    return fixture, selected


def main(output, dest):
    fixture, selected = paired_cases(output)
    if not selected:
        raise RuntimeError("No completely paired successful conversations")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(output / "source_snapshot", dest / "source_snapshot", dirs_exist_ok=True)
    for case in selected:
        shutil.copytree(output / "cases" / case["id"], dest / "cases" / case["id"], dirs_exist_ok=True)
    reduced = fixture | {"cases": selected}
    parent_report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    parent_complete = parent_report["complete"]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest |= {"dataset_sha256": evaluation.sha(reduced), "parent_run": str(output),
                 "incomplete_parent": not parent_complete, "planned_conversations": len(fixture["cases"]),
                 "selection": "All eight arms, both probes, no recorded call/maintenance errors; not selected by task score. Semantic memory validation failures are retained."}
    evaluation.write_json(dest / "dataset.json", reduced)
    evaluation.write_json(dest / "manifest.json", manifest)
    evaluation.aggregate(dest)
    result = semantic_audit(dest)
    evaluation.write_json(dest / "COVERAGE.json", {"parent_complete": parent_complete,
        "planned_conversations": len(fixture["cases"]), "selected_conversations": len(selected),
        "conversations": [{"id": c["id"], "kind": c["kind"], "turns": c["turns"]} for c in selected],
        "warning": ("Complete replay contains transport failures. This paired subset is selected only by error records; "
                    "coverage and selection bias limit generalization. Do not claim production performance."
                    if parent_complete else "Incomplete narrow pilot; no reliable ranking, generalization or resume-ready improvement claim.")})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/ch07/context-comparison/20261016-v2")
    parser.add_argument("--dest", default="data/ch07/context-comparison/20261016-v2-paired-pilot")
    args = parser.parse_args()
    main(Path(args.output), Path(args.dest))
