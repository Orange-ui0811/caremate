"""Prepare, run and merge at most two isolated keep3 evaluation workers.

Preparation and merging are offline. Only ``--run`` launches model-calling
subprocesses. Every case (all five arms and sequential follow-ups) belongs to
one worker. Existing attempts are never silently rerun or semantically fixed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RAW_DIRS = ("calls", "turns", "seeds", "maintenance", "projections")
VERSION = "keep3-case-sharded-batch-v1"


def _stamp():
    return datetime.now(timezone.utc).isoformat()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Only a derived status file is replaced, never an experimental record.
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), default=str).encode()).hexdigest()


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _within(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError("batch path is outside its output directory")
    return path


@contextmanager
def _lock(root, phase):
    held = []
    try:
        for name in ("batch.lock", "writer.lock"):
            path = root / name
            with path.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps({"pid": os.getpid(), "phase": phase, "at": _stamp()}))
            held.append(path)
        yield
    finally:
        for path in reversed(held):
            path.unlink(missing_ok=True)


def _check_frozen(root, *, current=True):
    manifest, data = _read(root / "manifest.json"), _read(root / "dataset.json")
    if _digest(data) != manifest["dataset_sha256"]:
        raise ValueError(f"frozen dataset changed: {root.name}")
    if not manifest.get("sources"):
        raise ValueError("source manifest is empty")
    for name, sha in manifest["sources"].items():
        frozen = root / "source_snapshot" / name
        if not frozen.is_file() or _hash(frozen) != sha:
            raise ValueError(f"frozen source changed: {root.name}/{name}")
        if current and (not (ROOT / name).is_file() or _hash(ROOT / name) != sha):
            raise ValueError(f"current source changed after freeze: {name}")
    return manifest, data


def _validate_batch(root, *, current=True):
    plan = _read(root / "batch.json")
    parent, data = _check_frozen(root, current=current)
    if plan["version"] != VERSION or not 1 <= len(plan["workers"]) <= 2:
        raise ValueError("unsupported batch plan")
    if _hash(root / "manifest.json") != plan["parent_manifest_sha256"]:
        raise ValueError("parent manifest changed after worker preparation")
    if plan["python_hash_seed"] != "0":
        raise ValueError("workers must use PYTHONHASHSEED=0")
    parent_cases = {case["id"]: case for case in data["cases"]}
    if len(parent_cases) != len(data["cases"]):
        raise ValueError("duplicate parent case ID")
    seen, paths = set(), set()
    for spec in plan["workers"]:
        directory = _within(root, spec["directory"])
        if directory in paths or directory.name in {p.name for p in paths}:
            raise ValueError("workers require different output names and directories")
        paths.add(directory)
        child, subset = _check_frozen(directory, current=current)
        if _hash(directory / "manifest.json") != spec["manifest_sha256"]:
            raise ValueError("worker manifest changed")
        for key in ("sources", "config", "arms", "replicate", "turn_timeout_seconds", "token_scope"):
            if child.get(key) != parent.get(key):
                raise ValueError(f"worker {directory.name} differs in {key}")
        ids = [case["id"] for case in subset["cases"]]
        if ids != spec["case_ids"] or len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError("worker datasets overlap or differ from their frozen partition")
        if {key: value for key, value in subset.items() if key != "cases"} != {
                key: value for key, value in data.items() if key != "cases"}:
            raise ValueError("worker dataset metadata differs")
        for case in subset["cases"]:
            if case != parent_cases.get(case["id"]):
                raise ValueError("worker case differs from the frozen parent case")
        seen.update(ids)
    if seen != set(parent_cases):
        raise ValueError("worker dataset union differs from the parent dataset")
    return plan, parent, data


def prepare(output, *, workers=2, pilot=False, case_ids=None):
    """Freeze once; derive disjoint worker manifests without calling a model."""
    root = Path(output).resolve()
    if workers not in (1, 2):
        raise ValueError("only one or two worker processes are supported")
    if root.exists() and any(root.iterdir()):
        raise ValueError("use an empty output directory for a new frozen batch")
    root.mkdir(parents=True, exist_ok=True)
    with _lock(root, "prepare"):
        from scripts import eval_context_keep3 as evaluation
        manifest = evaluation.prepare(root, pilot=pilot, arms=evaluation.ARMS,
                                      case_ids=case_ids, replicate=1)
        data = _read(root / "dataset.json")
        if not data["cases"]:
            raise ValueError("batch dataset is empty")
        workers = min(workers, len(data["cases"]))
        # Include this controller even if the underlying single-worker runner
        # does not yet list it among its additional sources.
        relative = "scripts/run_context_keep3_batch.py"
        snapshot = root / "source_snapshot" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, snapshot)
        manifest["sources"][relative] = _hash(snapshot)
        manifest["group_order"] = "case-major within each independent worker; local case index rotates arm order; at most two workers"
        manifest["batch_execution"] = {
            "version": VERSION, "workers": workers, "unit": "whole case across all five arms",
            "partition": "round-robin parent case order", "python_hash_seed": "0",
            "worker_processes": "separate Python interpreters; globals and model clients are not shared",
            "replicates": 1, "latency_caveat": "parallel provider queues; not a production latency benchmark"}
        _write(root / "manifest.json", manifest)
        specs = []
        for index in range(workers):
            name = f"{root.name}-worker-{index + 1:02d}"
            directory = _within(root, "workers/" + name)
            directory.mkdir(parents=True, exist_ok=False)
            subset = deepcopy(data)
            subset["cases"] = deepcopy(data["cases"][index::workers])
            child = deepcopy(manifest)
            child["dataset_sha256"] = _digest(subset)
            child["full_preflight"] = {case["id"]: manifest.get("full_preflight", {}).get(case["id"])
                                       for case in subset["cases"]}
            child["batch_worker"] = {"index": index + 1, "parent_dataset_sha256": manifest["dataset_sha256"],
                                     "case_ids": [case["id"] for case in subset["cases"]]}
            shutil.copytree(root / "source_snapshot", directory / "source_snapshot")
            _write(directory / "dataset.json", subset)
            _write(directory / "manifest.json", child)
            specs.append({"worker": index + 1, "directory": directory.relative_to(root).as_posix(),
                          "case_ids": child["batch_worker"]["case_ids"],
                          "manifest_sha256": _hash(directory / "manifest.json"),
                          "log": f"logs/{name}.log"})
        plan = {"version": VERSION, "prepared_at": _stamp(), "max_workers": workers,
                "python_hash_seed": "0", "parent_manifest_sha256": _hash(root / "manifest.json"),
                "workers": specs, "planned_cases": len(data["cases"]),
                "planned_turns": len(manifest["arms"]) * sum(len(case["steps"]) for case in data["cases"]),
                "replicates": 1}
        _write(root / "batch.json", plan)
        _validate_batch(root)
    return {"prepared": True, "model_calls": 0, "output": str(root), **plan}


def run(output):
    """Run the prepared workers once; keep each subprocess's entire log."""
    root = Path(output).resolve()
    with _lock(root, "run"):
        plan, _, _ = _validate_batch(root)
        status_path = root / "BATCH_STATUS.json"
        if status_path.exists():
            raise ValueError("this batch already has a run attempt; inspect saved status instead of rerunning")
        for spec in plan["workers"]:
            directory = _within(root, spec["directory"])
            if (directory / "writer.lock").exists() or any((directory / name).exists() for name in RAW_DIRS):
                raise ValueError("worker is active or already has live output; no automatic rerun")
            if _within(root, spec["log"]).exists():
                raise ValueError("worker log already exists; refusing to overwrite an earlier attempt")
        status = {"version": VERSION, "started_at": _stamp(), "phase": "running",
                  "execution_finished": False, "all_worker_processes_succeeded": False,
                  "records_complete": False, "workers": []}
        _write(status_path, status)
        children, handles = [], []
        controller_error = None
        try:
            for spec in plan["workers"]:
                directory = _within(root, spec["directory"])
                log = _within(root, spec["log"])
                log.parent.mkdir(parents=True, exist_ok=True)
                handle = log.open("xb")
                handles.append(handle)
                command = [sys.executable, "-m", "scripts.eval_context_keep3", "--output", str(directory), "--run"]
                env = os.environ.copy()
                env.update(PYTHONHASHSEED="0", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                                           stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                item = {**spec, "pid": process.pid, "command": command,
                        "started_at": _stamp(), "returncode": None, "phase": "running"}
                status["workers"].append(item)
                children.append((process, item))
                _write(status_path, status)
            while any(process.poll() is None for process, _ in children):
                for process, item in children:
                    code = process.poll()
                    if code is not None and item["returncode"] is None:
                        item.update(returncode=code, finished_at=_stamp(), phase="exited")
                        _write(status_path, status)
                time.sleep(1)
        except BaseException as exc:
            controller_error = {"type": type(exc).__name__, "message": str(exc)}
            for process, _ in children:
                if process.poll() is None:
                    process.terminate()
            for process, _ in children:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        finally:
            for process, item in children:
                item.update(returncode=process.poll(), finished_at=item.get("finished_at") or _stamp(), phase="exited")
                report_path = _within(root, item["directory"]) / "audited_report.json"
                try:
                    report = _read(report_path) if report_path.is_file() else {}
                except (ValueError, OSError) as exc:
                    report = {}
                    item["audit_report_error"] = type(exc).__name__
                item["generic_audit_passed"] = report.get("audit_passed", False)
                item["records_complete"] = report.get("complete", False)
                item["failed_turns"] = sum(group.get("failed_turns", 0) for group in report.get("groups", {}).values())
                item["interface_error_calls"] = sum(group.get("interface_error_calls", 0) for group in report.get("groups", {}).values())
            for handle in handles:
                handle.close()
            status["controller_error"] = controller_error
            status["execution_finished"] = len(children) == len(plan["workers"]) and all(
                process.poll() is not None for process, _ in children)
            status["all_worker_processes_succeeded"] = (not controller_error and status["execution_finished"]
                                                        and all(item["returncode"] == 0 for _, item in children))
            status["records_complete"] = (status["all_worker_processes_succeeded"]
                                          and all(item["records_complete"] and item["generic_audit_passed"]
                                                  for _, item in children))
            status["phase"] = "exited" if status["all_worker_processes_succeeded"] else "failed"
            status["finished_at"] = _stamp()
            status["result_notice"] = "Worker exit does not establish independent audit or model-task success; merge and audit the complete evidence next."
            _write(status_path, status)
    return status


def merge(output):
    """Copy all saved raw evidence; never overwrite differing or duplicate data."""
    root = Path(output).resolve()
    with _lock(root, "merge"):
        plan, parent, data = _validate_batch(root, current=False)
        status = _read(root / "BATCH_STATUS.json")
        if status.get("phase") == "running" or any(item.get("returncode") is None for item in status.get("workers", [])):
            raise ValueError("worker execution is unfinished; cannot merge live evidence")
        transfers, destinations, chains, chain_sources = [], set(), [], []
        stale_locks = []
        exited = {item["directory"]: item for item in status.get("workers", [])
                  if item.get("returncode") is not None and item.get("phase") == "exited"}
        raw_counts = {name: 0 for name in RAW_DIRS}
        for spec in plan["workers"]:
            directory = _within(root, spec["directory"])
            if (directory / "writer.lock").exists():
                if spec["directory"] not in exited:
                    raise ValueError("worker lock remains without a recorded subprocess exit")
                # A killed interpreter may leave its lock. Its parent observed
                # the process exit, so retain the stale lock as failure evidence
                # while copying the records; never remove it or rerun the arm.
                stale_locks.append((directory / "writer.lock").relative_to(root).as_posix())
            allowed_cases = set(spec["case_ids"])
            for category in RAW_DIRS:
                for source in sorted((directory / category).glob("**/*")):
                    if source.is_dir():
                        continue
                    if source.suffix != ".json" or source.parent != directory / category:
                        raise ValueError("unrecognized raw record layout; refusing to silently skip evidence")
                    target = root / category / source.name
                    key = target.relative_to(root).as_posix()
                    if key in destinations:
                        raise ValueError("raw record name collides between workers: " + key)
                    destinations.add(key)
                    value = _read(source)
                    records = value if category in {"maintenance", "projections"} else [value]
                    if not isinstance(records, list):
                        raise ValueError("invalid raw record list")
                    for item in records:
                        coordinates = item.get("coords") or {}
                        if coordinates.get("arm") not in parent["arms"] or coordinates.get("case_id") not in allowed_cases:
                            raise ValueError("raw evidence belongs to another worker's case or unknown arm")
                    if category == "calls" and value.get("id") != source.stem:
                        raise ValueError("call filename does not match its identity")
                    sha = _hash(source)
                    if target.exists() and _hash(target) != sha:
                        raise ValueError("existing parent raw record differs: " + key)
                    transfers.append({"source": source.relative_to(root).as_posix(), "destination": key,
                                      "sha256": sha, "bytes": source.stat().st_size})
                    raw_counts[category] += 1
            errors = directory / "chain_errors.json"
            if errors.exists():
                values = _read(errors)
                if not isinstance(values, list):
                    raise ValueError("chain errors must be a JSON list")
                for value in values:
                    coord = value.get("coords") or {}
                    if coord.get("arm") not in parent["arms"] or coord.get("case_id") not in allowed_cases:
                        raise ValueError("chain error has invalid worker coordinates")
                chains.extend(values)
                chain_sources.append({"source": errors.relative_to(root).as_posix(),
                                      "sha256": _hash(errors), "records": len(values)})
        # Detect unrelated/pre-existing records, not only missing worker files.
        actual_parent = {path.relative_to(root).as_posix()
                         for name in RAW_DIRS for path in (root / name).glob("**/*") if path.is_file()}
        if actual_parent - destinations:
            raise ValueError("parent contains raw records not present in the frozen worker union")
        if (root / "chain_errors.json").exists() and _read(root / "chain_errors.json") != chains:
            raise ValueError("parent chain errors differ from worker evidence")
        for transfer in transfers:
            source, target = _within(root, transfer["source"]), _within(root, transfer["destination"])
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with source.open("rb") as incoming, target.open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
            if _hash(target) != transfer["sha256"] or _hash(source) != transfer["sha256"]:
                raise ValueError("raw evidence changed while merging")
        _write(root / "chain_errors.json", chains)
        expected_turns = {f"turns/{arm}-{case['id']}-{i}.json" for arm in parent["arms"]
                          for case in data["cases"] for i in range(1, len(case["steps"]) + 1)}
        expected_aux = {f"{kind}/{arm}-{case['id']}.json" for kind in ("seeds", "maintenance", "projections")
                        for arm in parent["arms"] for case in data["cases"]}
        missing = sorted((expected_turns | expected_aux) - destinations)
        merged = {"version": VERSION, "merged_at": _stamp(), "merged": True,
                  "records_complete": bool(status.get("records_complete") and not missing),
                  "worker_processes_succeeded": status.get("all_worker_processes_succeeded", False),
                  "missing_records": missing, "raw_file_counts": raw_counts,
                  "files": transfers, "chain_error_sources": chain_sources,
                  "chain_error_records": len(chains), "parent_dataset_sha256": parent["dataset_sha256"],
                  "stale_worker_locks_preserved": stale_locks,
                  "workers_preserved": True, "worker_logs_preserved": True,
                  "next_steps": ["scripts.eval_context_keep3 --audit", "scripts.audit_context_keep3", "scripts.analyze_context_keep3"],
                  "notice": "Byte-preserving merge only. Failed calls, failed turns and missing usage remain unchanged; independent audit is still required."}
        _write(root / "BATCH_MERGE.json", merged)
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--cases", nargs="+")
    phases = parser.add_mutually_exclusive_group()
    phases.add_argument("--prepare", action="store_true")
    phases.add_argument("--run", action="store_true")
    phases.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.run:
        result = run(args.output)
        ok = result["all_worker_processes_succeeded"] and result["records_complete"]
    elif args.merge:
        result = merge(args.output)
        ok = result["records_complete"]
    else:
        result = prepare(args.output, workers=args.workers, pilot=args.pilot, case_ids=args.cases)
        ok = True
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
