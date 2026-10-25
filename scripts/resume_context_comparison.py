"""Resume only missing/provider-failed work; stop on provider errors instead of flooding requests.

Model, original prompts, frozen fixtures and source signature remain unchanged.
Old failed records are moved to a timestamped recovery archive, never discarded.
"""
import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
import hashlib

import scripts.eval_context_comparison as evaluation
from scripts.context_benchmark_strategies import STRATEGIES

PROVIDER_ERRORS = {"APIStatusError", "RateLimitError", "AuthenticationError", "PermissionDeniedError",
                   "APIConnectionError", "APITimeoutError", "InternalServerError", "NotFoundError"}


def provider_failed(record):
    errors = set(record.get("errors", [])) | {record.get("error_type")}
    return bool(errors & PROVIDER_ERRORS) or (not record.get("ok", False) and not record.get("validation_error"))


def repair_cache(output):
    """Retain good checkpoints; archive failed suffixes before any new model call."""
    fixture = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = output / "recovery_archive" / stamp
    changes = []
    for case in fixture["cases"]:
        folder = output / "cases" / case["id"]
        shared_path = folder / "shared_state.json"
        shared_bad_from = None
        if shared_path.exists():
            shared = json.loads(shared_path.read_text(encoding="utf-8"))
            kept = {}
            for probe in case["probes"]:
                snapshot = shared.get(probe["id"])
                if snapshot is not None and any(provider_failed(r) for r in snapshot["maintenance"]):
                    shared_bad_from = probe["at_turn"]
                    break
                if snapshot is not None:
                    kept[probe["id"]] = snapshot
            if shared_bad_from is not None:
                dest = archive / case["id"] / shared_path.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shared_path.rename(dest)
                if kept:
                    evaluation.write_json(shared_path, kept)
                changes.append(str(dest.relative_to(output)))
        for arm in STRATEGIES:
            path = folder / (arm + ".json")
            if not path.exists():
                continue
            row = json.loads(path.read_text(encoding="utf-8"))
            snapshots, probes = {}, []
            changed = False
            for probe in case["probes"]:
                snapshot = row["snapshots"].get(probe["id"])
                bad = snapshot is not None and any(provider_failed(r) for r in snapshot["maintenance"])
                if arm in ("D_state", "E_state_retrieval") and shared_bad_from is not None:
                    bad = bad or probe["at_turn"] >= shared_bad_from
                if snapshot is not None and not bad:
                    snapshots[probe["id"]] = snapshot
                elif snapshot is not None:
                    changed = True
                result = next((p for p in row["probes"] if p["probe_id"] == probe["id"]), None)
                if result is not None:
                    if bad or any(provider_failed(result[k]) or not result[k].get("calls") for k in ("coref", "plan", "reply")):
                        changed = True
                    else:
                        probes.append(result)
            if changed:
                dest = archive / case["id"] / path.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                path.rename(dest)
                if snapshots:
                    evaluation.write_json(path, {"strategy": arm, "probes": probes, "snapshots": snapshots,
                                                 "maintenance": list(snapshots.values())[-1]["maintenance"]})
                changes.append(str(dest.relative_to(output)))
    if changes:
        evaluation.write_json(archive / "repair.json", {"changed_files": changes,
            "scope": "Provider-failed checkpoints/probes only; semantic failures are retained.",
            "billing_note": "Archived successful calls are additional physical spend; do not hide them in final standalone-policy totals."})
    return changes


class ThrottledCalls(evaluation.Calls):
    """Shared start spacing and a circuit breaker for unavailable providers."""
    interval = .75
    error_path = None

    def __init__(self, timeout, concurrency):
        super().__init__(timeout, concurrency)
        self.gate = asyncio.Lock()
        self.next_start = 0

    async def __call__(self, stage, fn, prompt=None):
        async with self.gate:
            delay = self.next_start - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self.next_start = time.monotonic() + self.interval
        result = await super().__call__(stage, fn, prompt)
        if set(result.get("errors", [])) & PROVIDER_ERRORS:
            if self.error_path is not None:
                evaluation.write_json(self.error_path, result)
            raise RuntimeError("Provider failure: stopped the run; details preserved. Restore provider availability before resuming.")
        return result


async def main(args):
    output = Path(args.output)
    # Verify signature before modifying any saved records or calling the provider.
    if args.new_run:
        if (output / "manifest.json").exists():
            raise RuntimeError("--new-run requires a new output directory")
        fixture = evaluation.make_suite()
        if args.limit:
            fixture["cases"] = fixture["cases"][:args.limit]
    else:
        fixture = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
    original_args = argparse.Namespace(output=args.output, harness=args.harness, capacity=3750,
        concurrency=args.concurrency, case_workers=args.concurrency, timeout=args.timeout, limit=args.limit,
        prepare=False, recompute=False)
    evaluation.prepare(output, original_args, fixture)
    host = urlsplit(evaluation.settings.chat_base_url).hostname
    # Keep transport phases separate: a longer retry timeout is not the original run's latency policy.
    phase = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    history = output / "transport_history" / phase
    history.mkdir(parents=True, exist_ok=True)
    for name in ("transport.json", "run_launcher.py"):
        if (output / name).exists():
            (history / ("previous_" + name)).write_bytes((output / name).read_bytes())
    transport = {"api_host": host, "model": evaluation.settings.chat_model,
        "interval_seconds": args.interval, "concurrency": args.concurrency,
        "timeout_seconds": args.timeout, "started_at_utc": phase,
        "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    evaluation.write_json(output / "transport.json", transport)
    evaluation.write_json(history / "transport.json", transport)
    (output / "run_launcher.py").write_bytes(Path(__file__).read_bytes())
    (history / "run_launcher.py").write_bytes(Path(__file__).read_bytes())
    if args.repair_only:
        print(json.dumps({"changed": repair_cache(output), "new_model_calls": 0}, ensure_ascii=False, indent=2))
        return
    # One tiny health request fails immediately on quota problems before the batch starts.
    msg = await evaluation.llm.get_chat_model(temperature=0).bind(max_tokens=8).ainvoke("Reply OK only.")
    evaluation.write_json(output / "resume_health.json", {"text": msg.content, "usage": msg.usage_metadata})
    repair_cache(output)
    ThrottledCalls.interval = args.interval
    ThrottledCalls.error_path = output / "provider_stop.json"
    evaluation.Calls = ThrottledCalls
    await evaluation.main(original_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/ch07/context-comparison/20261016-v2")
    parser.add_argument("--harness", default="D:/harness/deepseek-harness")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--interval", type=float, default=.75)
    parser.add_argument("--timeout", type=float, default=100)
    parser.add_argument("--repair-only", action="store_true")
    parser.add_argument("--new-run", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.concurrency < 1 or min(args.interval, args.timeout) <= 0:
        parser.error("concurrency, interval and timeout must be positive")
    asyncio.run(main(args))
