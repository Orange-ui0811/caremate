"""Independent offline checks against frozen sources; never calls a model.

This verifies provenance, assembly, transport settings and deterministic scores.
It cannot prove the semantic truth of generated summaries or answer text.
Only AUDIT_DETAIL.json is written; raw experiment evidence is never modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch
from uuid import uuid4

from langchain_core.messages import message_to_dict, messages_from_dict

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), default=str).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_view(rows):
    return [{"id": r["id"], "role": r["role"], "content": r["content"]} for r in rows]


def coords(item):
    c = item.get("coords") or {}
    return c.get("arm"), c.get("case_id"), c.get("turn")


def load_module(path, label):
    name = "_frozen_context_audit_" + label + "_" + uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses needs the defining module registered.
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@contextmanager
def frozen_functions(output):
    base = output / "source_snapshot"
    policy = load_module(base / "app/core/context_policy.py", "policy")
    with patch.dict(sys.modules, {"app.core.context_policy": policy}):
        task = load_module(base / "app/core/context_task.py", "task")
        state = load_module(base / "app/core/context_state.py", "state")
    score = load_module(base / "scripts/context_unified_cases.py", "score")
    try:
        yield policy, task, state, score
    finally:
        for module in (policy, task, state, score):
            sys.modules.pop(module.__name__, None)


def _read_records(directory, lists=False):
    result = []
    for path in sorted(directory.glob("*.json")):
        value = read(path)
        result.extend((path, item) for item in value) if lists else result.append((path, value))
    return result


def _string_values(value, key="summary"):
    found = []
    if isinstance(value, dict):
        if isinstance(value.get(key), str):
            found.append(value[key])
        for child in value.values():
            found.extend(_string_values(child, key))
    elif isinstance(value, list):
        for child in value:
            found.extend(_string_values(child, key))
    return found


def _message_contents(value):
    if isinstance(value, dict):
        if isinstance(value.get("content"), str):
            yield value["content"]
        for child in value.values():
            yield from _message_contents(child)
    elif isinstance(value, list):
        for child in value:
            yield from _message_contents(child)


def _forbidden_keys(value, path=""):
    result = []
    if isinstance(value, dict):
        for key, child in value.items():
            here = f"{path}.{key}"
            if key in {"expected", "gold", "gold_answer", "gold_target", "task_rule_pass", "failed_checks", "score_contract"}:
                result.append(here)
            result.extend(_forbidden_keys(child, here))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(_forbidden_keys(child, f"{path}[{index}]"))
    return result


def _legacy_replay(item, output, manifest):
    """Frozen node functions, with current dependencies allowed only by hashes."""
    for name, sha in manifest["sources"].items():
        if name.startswith("app/") and (not (ROOT / name).exists() or file_hash(ROOT / name) != sha):
            raise ValueError("legacy dependency differs from frozen source: " + name)
    module = load_module(output / "source_snapshot/app/graph/nodes.py", "legacy_nodes")
    inputs = item["assembler_input"]
    state = deepcopy(inputs["state"])
    state["messages"] = messages_from_dict(state.get("messages") or [])
    try:
        with patch.object(module.settings, "context_policy", "legacy"), patch.object(
                module.settings, "context_window_max_tokens", inputs["context_window_max_tokens"]):
            if item["function"] == "history":
                history = module._history_text(state, inputs.get("max_turns", 6), stage=item["stage"])
                messages = []
            elif item["function"] == "answer":
                values = module._agent_messages(state, tool_schemas=inputs.get("tool_schemas"))
                history = "\n".join(str(m.content) for m in values)
                messages = [message_to_dict(m) for m in values]
            else:
                raise ValueError("unknown legacy projection function")
        return history, messages
    finally:
        sys.modules.pop(module.__name__, None)


def audit(output):
    output = Path(output).resolve()
    if (output / "writer.lock").exists():
        raise ValueError("evaluation writer is active; wait until it exits")
    manifest, dataset = read(output / "manifest.json"), read(output / "dataset.json")
    issues, warnings, checked = [], [], Counter()

    def check(ok, code, location, detail=None):
        checked[code] += 1
        if not ok:
            issues.append({"code": code, "location": str(location), "detail": detail})
        return bool(ok)

    check(digest(dataset) == manifest.get("dataset_sha256"), "dataset_hash", "dataset.json")
    source_ok = True
    for name, expected in manifest["sources"].items():
        path = output / "source_snapshot" / name
        source_ok &= check(path.exists() and file_hash(path) == expected, "frozen_source_hash", name)
    required_sources = {"app/core/context_policy.py", "app/core/context_task.py", "app/core/context_state.py", "scripts/context_unified_cases.py"}
    source_ok &= check(required_sources.issubset(manifest["sources"]), "frozen_required_sources", "manifest.json")
    if not source_ok:
        return {"version": "context-unified-independent-audit-v1", "audit_ok": False, "complete": False,
                "issues": issues, "warnings": warnings, "checked": dict(checked),
                "note": "Frozen sources failed integrity checks; no snapshot code executed."}

    turn_items, call_items = _read_records(output / "turns"), _read_records(output / "calls")
    seed_items = _read_records(output / "seeds")
    projections, maintenance = _read_records(output / "projections", True), _read_records(output / "maintenance", True)
    turns, seeds, calls = {coords(v): v for _, v in turn_items}, {coords(v): v for _, v in seed_items}, {str(v.get("id", p.stem)): v for p, v in call_items}
    check(len(turns) == len(turn_items), "unique_turn_coordinates", "turns")
    check(len(seeds) == len(seed_items), "unique_seed_coordinates", "seeds")
    check(len(calls) == len(call_items), "unique_call_ids", "calls")
    cases = {c["id"]: c for c in dataset["cases"]}
    expected_turns = {(arm, cid, i) for arm in manifest["arms"] for cid, c in cases.items() for i in range(1, len(c["steps"]) + 1)}
    expected_seeds = {(arm, cid, 0) for arm in manifest["arms"] for cid in cases}
    missing_turns, missing_seeds = sorted(expected_turns - turns.keys()), sorted(expected_seeds - seeds.keys())
    check(not (turns.keys() - expected_turns), "unexpected_turns", "turns")
    check(not (seeds.keys() - expected_seeds), "unexpected_seeds", "seeds")
    for path, value in turn_items + call_items:
        check(value.get("record_sha256") == digest({k: v for k, v in value.items() if k != "record_sha256"}), "record_hash", path.name)
    for path, seed in seed_items:
        key = coords(seed)
        if key[1] in cases:
            expected = cases[key[1]].get("initial_history", [])
            check([{"role": r["role"], "content": r["content"]} for r in seed["rows"]] == expected, "seed_matches_dataset", path.name)
    assigned = [str(cid) for _, record in turn_items + seed_items for cid in record.get("call_ids", [])]
    check(len(assigned) == len(set(assigned)) and set(assigned) == set(calls), "call_assignment", "seeds+turns")
    for path, record in turn_items + seed_items:
        for cid in record.get("call_ids", []):
            check(str(cid) in calls and coords(calls[str(cid)]) == coords(record), "call_coordinate_binding", path.name, str(cid))

    with frozen_functions(output) as (policy, task, state_module, scoring):
        for path, turn in turn_items:
            key = coords(turn)
            if key not in expected_turns:
                continue
            step = cases[key[1]]["steps"][key[2] - 1]
            check(turn.get("query") == step["query"] and turn.get("expected") == step["expected"], "frozen_step", path.name)
            try:
                check(scoring.score_turn(turn) == turn.get("score"), "frozen_score_replay", path.name)
            except Exception as exc:
                check(False, "frozen_score_replay", path.name, type(exc).__name__)
            previous = turns.get((key[0], key[1], key[2] - 1)) if key[2] > 1 else seeds.get((key[0], key[1], 0))
            if previous:
                old = previous.get("actual_dialog_rows", previous.get("rows", []))
                check(row_view(turn["actual_dialog_rows"][:len(old)]) == row_view(old), "dialog_continuity", path.name)
            if turn.get("status") == "completed":
                mid = (turn.get("state") or {}).get("current_message_id")
                current = next((r for r in turn.get("actual_dialog_rows", []) if r["id"] == mid), None)
                check(current is not None and current["role"] == "user" and current["content"] == turn["query"], "current_original_matches_frozen_query", path.name)

        visibility, projection_counts = Counter(), Counter()
        for path, item in projections:
            key, stage = coords(item), item.get("stage")
            label = f"{path.name}:{key}:{stage}"
            turn = turns.get(key)
            if not check(turn is not None, "projection_has_turn", label):
                continue
            inputs, saved = item.get("assembler_input"), item.get("packet") or {}
            saved_manifest = item.get("manifest") or {}
            projection_counts[key[0]] += 1
            check(saved_manifest == saved.get("manifest"), "duplicate_projection_manifest", label)
            watermark = (turn.get("state") or {}).get("current_message_id", 0)
            actual = [r for r in turn.get("actual_dialog_rows", []) if r["id"] <= watermark]
            if key[0] == "legacy":
                if not check(isinstance(inputs, dict), "legacy_replay_input_present", label):
                    continue
                try:
                    history, messages = _legacy_replay(item, output, manifest)
                    check(history == saved.get("history") and messages == saved.get("messages"), "legacy_projection_replay", label)
                    s = inputs["state"]
                    original = [m["data"] for m in s.get("messages", []) if str(m.get("data", {}).get("id", "")).startswith("db-")]
                    mapped = [{"id": int(m["id"][3:]), "role": "user" if m["type"] == "human" else "assistant", "content": m["content"]} for m in original]
                    check(mapped == row_view(actual), "legacy_original_prefix", label)
                    check(s.get("current_message_id") == watermark, "legacy_watermark", label)
                    visible = bool(s.get("summary") and s["summary"] in history)
                    check(saved_manifest.get("legacy_summary_visible") == visible, "legacy_summary_visibility", label)
                    if visible:
                        visibility[(key[0], stage)] += 1
                except Exception as exc:
                    check(False, "legacy_projection_replay", label, f"{type(exc).__name__}: {exc}")
                continue
            if inputs is None:
                check(bool(saved.get("blocked") and (turn.get("state") or {}).get("memory_error")), "blocked_projection_without_input", label)
                continue
            check(not _forbidden_keys(inputs), "no_gold_fields_in_assembler_input", label, _forbidden_keys(inputs))
            snapshot = inputs["snapshot"]
            check(inputs.get("policy") == key[0] and inputs.get("stage") == stage, "projection_policy_stage", label)
            check(row_view(snapshot["rows"]) == row_view(actual), "projection_matches_actual_prefix", label)
            check(snapshot.get("watermark") == watermark and all(r["id"] <= watermark for r in snapshot["rows"]), "projection_watermark", label)
            check(str(snapshot.get("conversation_id")) == str((turn.get("state") or {}).get("conversation_id")), "projection_conversation_scope", label)
            check(inputs.get("query") in {turn.get("query"), (turn.get("state") or {}).get("resolved_query")}, "projection_query_is_current", label)
            try:
                fresh_snapshot = state_module.build_snapshot(actual, snapshot["conversation_id"])
                for field in ("rows", "source_hash", "sources", "candidate", "active_constraints", "events"):
                    check(snapshot.get(field) == fresh_snapshot.get(field), "snapshot_replay_" + field, label)
                recomputed = task.assemble_task_context(**inputs)
                serial = {**recomputed, "messages": [message_to_dict(m) for m in recomputed["messages"]]}
                check(serial == saved, "projection_exact_replay", label)
            except Exception as exc:
                check(False, "projection_exact_replay", label, f"{type(exc).__name__}: {exc}")
                continue
            pm = saved_manifest
            cap = 16000 if key[0] == "full" else task.STAGE_BUDGETS[stage]
            caller = inputs.get("budget")
            check(pm.get("budget") == min(cap, caller) if caller is not None else pm.get("budget") == cap, "effective_projection_budget", label)
            if not saved.get("blocked"):
                check(pm.get("approx_tokens", cap + 1) <= cap and (caller is None or pm["approx_tokens"] <= caller), "admitted_projection_budget", label)
            source_rows = {r["id"]: r for r in actual}
            for ref in pm.get("source_refs", []):
                if not check(isinstance(ref, dict), "source_ref_shape", label):
                    continue
                row = source_rows.get(ref.get("message_id"))
                good = row is not None and ref.get("role") == row["role"] and ref.get("content_hash") == hashlib.sha256(row["content"].encode()).hexdigest()
                good = good and 0 <= ref.get("start", -1) <= ref.get("end", -1) <= len(row["content"])
                check(good, "source_ref_original_hash_offsets", label, ref.get("ref"))
            if key[0] == "full" and not saved.get("blocked"):
                check(pm.get("selected_message_ids") == [r["id"] for r in actual if r["id"] != watermark], "full_contains_all_prior_rows", label)
            segments = inputs.get("summaries") or []
            for sid in pm.get("summary_refs", []):
                segment = next((s for s in segments if str(s.get("id")) == str(sid)), None)
                check(segment is not None and segment["summary"] in saved.get("summary_text", "") and segment["summary"] in saved.get("history", ""), "admitted_summary_is_visible", label, sid)
            if pm.get("summary_refs"):
                visibility[(key[0], stage)] += 1
            state_segments = (turn.get("state") or {}).get("memory_segments") or []
            check(segments == state_segments, "summary_visibility_matches_turn_snapshot", label)

        present_stages = {(coords(item), item.get("stage")) for _, item in projections}
        for path, turn in turn_items:
            key = coords(turn)
            if turn.get("status") == "completed":
                check((key, "coref") in present_stages, "completed_turn_has_coref_projection", path.name)
        purpose_stage = {"coreference": "coref", "intent_classification": "intent",
                         "query_rewrite": "retrieval", "answer_generation": "answer"}
        for path, call in call_items:
            stage = purpose_stage.get(call.get("purpose"))
            if stage and not (coords(call)[0] == "legacy" and stage == "retrieval"):
                check((coords(call), stage) in present_stages, "model_call_has_stage_projection", path.name)

        # Check each immutable summary against its original interval and the
        # output of the maintenance call that created it; never grade its prose.
        for path, item in maintenance:
            key = coords(item)
            record = seeds.get(key) if key[2] == 0 else turns.get(key)
            if not check(record is not None, "maintenance_has_trajectory_record", path.name, key):
                continue
            all_rows = record.get("actual_dialog_rows", record.get("rows", []))
            boundary = item.get("row_watermark", 0)
            prefix = [r for r in all_rows if r["id"] <= boundary]
            check(bool(prefix) and prefix[-1]["id"] == boundary, "maintenance_visible_watermark", path.name, boundary)
            matching_calls = [calls[cid] for cid in item.get("call_ids", []) if cid in calls]
            check(len(matching_calls) == len(item.get("call_ids", [])) and all(coords(c) == key for c in matching_calls), "maintenance_call_binding", path.name)
            output_summaries = [s for c in matching_calls for s in _string_values(c.get("outputs", []))]
            prompt_texts = [content for c in matching_calls for content in _message_contents(c.get("input", {}))]
            for segment in item.get("new_segments") or []:
                start, end = segment["from_msg_id"], segment["upto_msg_id"]
                span = [r for r in prefix if start <= r["id"] <= end]
                valid_span = bool(span) and span[0]["id"] == start and span[-1]["id"] == end and end <= boundary
                check(valid_span, "summary_range_within_trigger_watermark", path.name, segment.get("id"))
                check(digest(row_view(span)) == segment.get("source_hash"), "summary_original_hash", path.name, segment.get("id"))
                refs = segment.get("source_refs") or []
                check([r["message_id"] for r in refs] == [r["id"] for r in span], "summary_source_coverage", path.name)
                for ref, row in zip(refs, span):
                    check(ref.get("start") == 0 and ref.get("end") == len(row["content"]) and ref.get("content_hash") == hashlib.sha256(row["content"].encode()).hexdigest(), "summary_source_ref", path.name)
                check(segment.get("summary") in output_summaries, "summary_matches_model_output", path.name, segment.get("id"))
                dialog = "\n".join(f"{r['role']}: {r['content']}" for r in span)
                check(any(dialog in text for text in prompt_texts), "summary_prompt_contains_exact_source_interval", path.name)
            check(item.get("legacy_summary_upto", 0) <= boundary, "legacy_summary_boundary", path.name)
            if key[0] == "legacy" and item.get("triggered") and not item.get("error_type"):
                check(item.get("legacy_summary") in output_summaries, "legacy_summary_matches_model_output", path.name)

        usage_unknown, call_errors, wire_missing = [], [], []
        config = manifest.get("config") or {}
        environment = config.get("environment", config)
        expected_model = environment.get("chat_model")
        payload_limit = (config.get("overrides") or {}).get("context_input_budget", 16000)
        output_limit = (config.get("overrides") or {}).get("context_output_budget", 8192)
        wire_estimates = []
        for path, call in call_items:
            name = path.name
            check(digest(call.get("input")) == call.get("input_sha256"), "call_input_hash", name)
            usage = call.get("usage") or {}
            if not all(isinstance(usage.get(k), (int, float)) for k in ("input_tokens", "output_tokens")):
                usage_unknown.append(name)
            if call.get("status") != "ok":
                call_errors.append({"file": name, "status": call.get("status")})
            wire = call.get("wire_payload")
            if not isinstance(wire, dict):
                wire_missing.append(name)
                check(call.get("status") != "ok", "successful_call_has_wire_payload", name)
                continue
            check(digest(wire) == call.get("wire_sha256"), "wire_hash", name)
            parts = {k: wire[k] for k in ("messages", "tools", "functions", "response_format") if k in wire}
            size = policy.estimate(json.dumps(parts, ensure_ascii=False, default=str))
            wire_estimates.append(size)
            check(size <= payload_limit, "complete_wire_input_budget", name, size)
            parameters = (call.get("input") or {}).get("parameters") or {}
            check(expected_model is not None and wire.get("model", parameters.get("model")) == expected_model, "configured_model", name)
            extras = wire.get("extra_body") or parameters.get("extra_body") or {}
            thinking = extras.get("chat_template_kwargs") or {}
            check(thinking.get("thinking") is False and thinking.get("enable_thinking") is False, "thinking_disabled", name)
            check(wire.get("max_completion_tokens", wire.get("max_tokens")) == output_limit, "configured_output_limit", name)
            check(not _forbidden_keys(parts), "no_gold_fields_in_wire_structure", name, _forbidden_keys(parts))

    missing_maintenance = [(arm, cid) for arm in manifest["arms"] for cid in cases if not (output / "maintenance" / f"{arm}-{cid}.json").exists()]
    unfinished = [p.name for p, c in call_items if c.get("status") == "running"]
    complete = not (missing_turns or missing_seeds or missing_maintenance or unfinished or issues)
    return {"version": "context-unified-independent-audit-v1", "created_at": datetime.now(timezone.utc).isoformat(),
            "output": str(output), "audit_ok": not issues, "complete": complete,
            "checked": dict(checked), "issues": issues, "warnings": warnings,
            "manifest_file_sha256": file_hash(output / "manifest.json"), "auditor_file_sha256": file_hash(Path(__file__)),
            "missing_turns": missing_turns, "missing_seeds": missing_seeds, "missing_maintenance": missing_maintenance,
            "unfinished_calls": unfinished, "unknown_usage_calls": usage_unknown, "call_errors": call_errors,
            "wire_missing_calls": wire_missing, "max_estimated_wire_input": max(wire_estimates, default=None),
            "projection_counts": dict(projection_counts),
            "visible_summary_views": [{"arm": arm, "stage": stage, "count": count} for (arm, stage), count in sorted(visibility.items())],
            "limits": ["Exact replay verifies deterministic transformations, not semantic correctness of summaries or answers.",
                       "Gold exclusion checks structural fields and original-prefix binding; it does not claim a blinded external benchmark.",
                       "Unknown usage remains unknown, never zero cost; input estimates are not provider tokenizer counts.",
                       "Legacy replay requires current imported app dependencies to match frozen source hashes; drift is reported rather than ignored."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.output)
    path = args.output / "AUDIT_DETAIL.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(path), "audit_ok": result["audit_ok"], "complete": result["complete"],
                      "issue_count": len(result["issues"])}, ensure_ascii=True))


if __name__ == "__main__":
    main()
