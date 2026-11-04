"""Offline all-visible +/-1 recall and per-node raw-message dedup audit.

Only AUDIT_DETAIL.json is written. Model calls, scores and raw evidence are
never changed; general score/usage/hash checks belong to the runner's audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import message_to_dict, messages_from_dict

ROOT = Path(__file__).resolve().parents[1]
ARMS = {"retrieval", "hybrid"}


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(values):
    return [{key: value[key] for key in ("id", "role", "content")} for value in values]


def _anchored(messages):
    result = []
    for message in messages:
        data = message.get("data", {})
        mid = str(data.get("id") or "")
        if not mid.startswith("db-") or not mid[3:].isdigit():
            continue
        if message.get("type") not in {"human", "ai"}:
            raise ValueError("persistent source is not user/assistant")
        result.append({"id": int(mid[3:]),
                       "role": "user" if message["type"] == "human" else "assistant",
                       "content": data.get("content")})
    return result


def _gold_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"expected", "gold", "gold_answer", "gold_target", "task_rule_pass", "failed_checks"}:
                yield key
            yield from _gold_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _gold_keys(child)


def _summary_values(value):
    if isinstance(value, dict):
        if isinstance(value.get("summary"), str):
            yield value["summary"].strip()
        for child in value.values():
            yield from _summary_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _summary_values(child)
    elif isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return
        if isinstance(parsed, (dict, list)):
            yield from _summary_values(parsed)


def _serial_packet(packet):
    result = deepcopy(packet)
    if "messages" in result:
        result["messages"] = [message_to_dict(m) if not isinstance(m, dict) else m
                              for m in result["messages"]]
    return result


def audit(root):
    root = Path(root).resolve()
    if (root / "writer.lock").exists():
        raise ValueError("evaluation writer is active; audit only after it exits")
    manifest, data = _read(root / "manifest.json"), _read(root / "dataset.json")
    cases = {case["id"]: case for case in data["cases"]}
    issues, checked = [], Counter()
    counters = Counter()

    def check(ok, code, location, detail=None):
        checked[code] += 1
        if not ok:
            issue = {"code": code, "location": str(location)}
            if detail is not None:
                issue["detail"] = detail
            issues.append(issue)
        return bool(ok)

    def record(path):
        if not check(path.exists(), "evidence_file_exists", path):
            return None
        try:
            return _read(path)
        except (ValueError, OSError) as exc:
            check(False, "evidence_file_readable", path, type(exc).__name__)
            return None

    # Replaying changed current code would test a different algorithm. Also
    # check frozen copies, so a valid current tree cannot mask a bad snapshot.
    sources_match = bool(manifest.get("sources"))
    check(sources_match, "source_manifest_present", root)
    for name, expected in manifest.get("sources", {}).items():
        for path, label in ((ROOT / name, "current_source_hash"),
                            (root / "source_snapshot" / name, "frozen_source_hash")):
            actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            sources_match = check(actual == expected, label, name) and sources_match
    replay_view = nodes = settings = summary_prompt = None
    if sources_match:
        from scripts.eval_context_recall_v2 import replay_view
        from app.graph import nodes
        from app.config import settings
        from app.core.prompts import SUMMARY_PROMPT as summary_prompt

    for path in sorted((root / "projections").glob("*.json")):
        entries = record(path)
        if entries is None:
            continue
        for index, entry in enumerate(entries):
            location = f"{path.name}:{index}"
            try:
                coord, inp = entry["coords"], entry["assembler_input"]
                arm, case_id, number = coord["arm"], coord["case_id"], coord["turn"]
                state = inp["state"]
                if not check(arm in ARMS and inp["arm"] == arm, "projection_arm", location):
                    continue
                check(not list(_gold_keys(inp)), "no_gold_in_assembler", location)
                check(entry["function"] in {"history", "answer"}, "projection_function", location)
                seed = record(root / "seeds" / f"{arm}-{case_id}.json")
                previous = (record(root / "turns" / f"{arm}-{case_id}-{number - 1}.json")
                            if number > 1 else None)
                if seed is None or (number > 1 and previous is None):
                    continue
                before = previous if previous is not None else seed
                prior = _rows(before["actual_dialog_rows"] if previous is not None else seed["rows"])
                current_id = state["current_message_id"]
                query = cases[case_id]["steps"][number - 1]["query"]
                raw = _anchored(state.get("messages", []))
                expected_rows = [*prior, {"id": current_id, "role": "user", "content": query}]
                check(raw == expected_rows, "source_prefix_and_current_query", location)
                check(all(r["id"] <= current_id for r in raw), "no_future_source", location)
                check(not prior or current_id > prior[-1]["id"], "current_watermark_after_history", location)
                expected_summary = before.get("summary") or ""
                expected_upto = before.get("summary_upto") or 0
                check((state.get("summary") or "", state.get("summary_upto_msg_id") or 0)
                      == (expected_summary, expected_upto), "previous_summary_lineage", location)
                if arm in {"full", "window", "retrieval"}:
                    check(not expected_summary, "no_summary_in_no_summary_arm", location)
                if arm == "full":
                    check(expected_upto == 0, "full_has_no_window_boundary", location)
                packet, base_packet = entry["packet"], entry["base_packet"]
                supplement = entry["supplement"]
                text = supplement.get("text") or ""
                if arm not in {"retrieval", "hybrid"}:
                    check(not text and not supplement.get("manifest", {}).get("selected_message_ids"),
                          "control_has_no_recall", location)
                if entry["function"] == "history":
                    expected_history = base_packet["history"] + ("\n\n" + text if text else "")
                    check(packet["history"] == expected_history, "history_is_base_plus_supplement", location)
                    check(not packet.get("messages"), "history_packet_has_no_new_messages", location)
                else:
                    stripped = deepcopy(packet["messages"])
                    count = 0
                    for message in stripped:
                        content = message.get("data", {}).get("content")
                        if text and isinstance(content, str) and "\n\n" + text in content:
                            check(message["data"].get("id") == "turn-ctx", "supplement_only_in_turn_context", location)
                            count += content.count("\n\n" + text)
                            message["data"]["content"] = content.replace("\n\n" + text, "", 1)
                    check(count == (1 if text else 0), "supplement_once_in_answer", location)
                    check(stripped == base_packet["messages"], "answer_preserves_base_messages", location)
                    if arm == "full":
                        check(_anchored(packet["messages"]) == expected_rows, "full_answer_has_all_original_messages", location)
                if arm == "full" and entry["function"] == "history":
                    check(all(not row["content"] or row["content"] in packet["history"] for row in prior),
                          "full_history_has_all_prior_text", location)
                selection = supplement.get("manifest", {})
                selected = selection.get("selected_message_ids", [])
                check(selected == sorted(set(selected)), "recall_source_unique_and_ordered", location)
                by_id = {row["id"]: row for row in prior}
                check(all(mid in by_id and mid < current_id for mid in selected),
                      "recall_stays_in_visible_prior_history", location)
                if entry['function']=='history':
                    from app.core.memory import build_window
                    window=build_window(messages_from_dict(state['messages']),expected_upto,3000)
                    candidate_messages=[message_to_dict(m) for m in window[:-1][-inp.get('max_turns',6):]]
                else:
                    candidate_messages=base_packet['messages']
                expected_visible=sorted(set(r['id'] for r in _anchored(candidate_messages) if r['content']))
                check(entry.get('visible_message_ids')==expected_visible,
                      'node_visible_ids_match_actual_base',location)
                check(selection.get('visible_message_ids')==expected_visible,
                      'recall_received_actual_visible_ids',location)
                check(not set(selected).intersection(expected_visible),
                      'no_duplicate_raw_message_in_supplement',location)
                expanded=selection.get('expanded_message_ids',[])
                removed=selection.get('deduplicated_message_ids',[])
                check(set(removed)==set(expanded).intersection(expected_visible)
                      and set(selected)==set(expanded)-set(expected_visible),
                      'dedup_exact_id_difference',location)
                check(selection.get('candidate_scope')=='all_visible_prior_dialogue',
                      'all_visible_search_scope',location)
                check(selection.get("selected_units", 0) <= 9, "recall_max_nine_units", location)
                refs = selection.get("source_refs", [])
                check([ref["message_id"] for ref in refs] == selected, "recall_reference_coverage", location)
                for ref in refs:
                    row = by_id.get(ref["message_id"])
                    if row is None:
                        continue
                    digest = hashlib.sha256(row["content"].encode("utf-8")).hexdigest()
                    check(str(ref["conversation_id"]) == str(state["conversation_id"])
                          and ref["start"] == 0 and ref["end"] == len(row["content"])
                          and ref["content_hash"] == digest and ref["role"] == row["role"],
                          "recall_reference_original_hash", location)
                counters["source_views"] += 1
                if sources_match:
                    real_state = deepcopy(state)
                    real_state["messages"] = messages_from_dict(real_state.get("messages", []))
                    replayed = replay_view(real_state, arm, entry["function"],
                                           max_turns=inp.get("max_turns", 6),
                                           tool_schemas=inp.get("tool_schemas"))
                    for key in ("packet", "base_packet"):
                        check(_serial_packet(replayed[key]) == entry[key], key + "_replay", location)
                    check(replayed["supplement"] == supplement, "supplement_replay", location)
                    counters["view_replays"] += 1
                    # Independently call the production legacy view, without
                    # going through the evaluation view factory.
                    if arm != "full":
                        with patch.object(settings, "context_policy", "legacy"), patch.object(
                                settings, "context_legacy_recall_enabled", False), patch.object(
                                settings, "context_window_max_tokens", 3000):
                            if entry["function"] == "history":
                                check(inp.get("max_turns", 6) == 6, "default_six_message_tail", location)
                                original = nodes._history_text(real_state, 6, stage=entry["stage"])
                                check(original == base_packet["history"], "production_base_history", location)
                            else:
                                original = nodes._agent_messages(real_state, tool_schemas=inp.get("tool_schemas"))
                                check([message_to_dict(m) for m in original] == base_packet["messages"],
                                      "production_base_answer", location)
                        counters["production_base_checks"] += 1
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                check(False, "projection_read_or_replay", location, f"{type(exc).__name__}: {exc}")

    for arm in manifest["arms"]:
        for case_id in cases:
            prefix = f"{arm}-{case_id}"
            path = root / "maintenance" / f"{prefix}.json"
            records = record(path)
            if records is None:
                continue
            last_summary, last_upto, last_watermark = "", 0, 0
            for index, item in enumerate(records):
                location = f"{path.name}:{index}"
                try:
                    coord = item["coords"]
                    check((coord["arm"], coord["case_id"]) == (arm, case_id), "maintenance_coordinates", location)
                    number = coord["turn"]
                    origin = record(root / ("turns" if number else "seeds") /
                                    (f"{prefix}-{number}.json" if number else f"{prefix}.json"))
                    if origin is None:
                        continue
                    rows = _rows(origin["actual_dialog_rows"] if number else origin["rows"])
                    visible = [row for row in rows if row["id"] <= item["row_watermark"]]
                    check(bool(visible) and visible[-1]["id"] == item["row_watermark"]
                          and item["row_watermark"] >= last_watermark, "maintenance_visible_watermark", location)
                    check((item["previous_summary"], item["previous_summary_upto"])
                          == (last_summary, last_upto), "maintenance_state_chain", location)
                    count = sum(row["id"] > item["previous_summary_upto"] for row in visible)
                    check(count == item["count_after_boundary"], "maintenance_message_count", location)
                    mode = "summary" if arm in {"summary", "hybrid"} else "none" if arm == "full" else "window"
                    check(item["mode"] == mode, "maintenance_mode", location)
                    should_trigger = mode != "none" and count >= 30
                    check(item["triggered"] == should_trigger, "maintenance_trigger_30", location)
                    users = [i for i, row in enumerate(visible) if row["role"] == "user"]
                    boundary = visible[users[-3] - 1]["id"] if len(users) > 3 else None
                    changed = (item["legacy_summary"], item["legacy_summary_upto"])
                    if not item["triggered"] or item.get("error_type"):
                        check(changed == (item["previous_summary"], item["previous_summary_upto"]),
                              "failed_or_idle_maintenance_preserves_state", location)
                    else:
                        check(boundary is not None and item["legacy_summary_upto"] == boundary,
                              "maintenance_keep_three_boundary", location)
                        if mode == "summary":
                            check(bool(item["legacy_summary"] and item["call_ids"]),
                                  "summary_success_has_output_and_calls", location)
                    if mode != "summary":
                        check(not item["legacy_summary"] and not item["call_ids"],
                              "no_summary_mode_has_no_model_maintenance", location)
                    elif item["triggered"]:
                        calls = [record(root / "calls" / f"{call_id}.json") for call_id in item["call_ids"]]
                        calls = [call for call in calls if call is not None]
                        for call in calls:
                            check(call.get("purpose") == "conversation_summary" and call.get("coords") == coord,
                                  "summary_call_purpose_and_coordinates", location)
                        wired = sorted((call for call in calls if call.get("wire_payload")),
                                       key=lambda call: call.get("started_at", ""))
                        if sources_match and wired and boundary is not None:
                            segment = [row for row in visible if item["previous_summary_upto"] < row["id"] <= boundary]
                            dialog = "\n".join(("用户" if row["role"] == "user" else "客服") + ":" + row["content"]
                                               for row in segment if row["content"])
                            expected_messages = summary_prompt.format_messages(
                                old_summary=item["previous_summary"] or "(无)", dialog=dialog)
                            actual_messages = wired[0]["wire_payload"].get("messages", [])
                            # Fallback structured-output machinery may add schema
                            # messages; the original human source must be exact.
                            expected_human = next(m.content for m in expected_messages if m.type == "human")
                            check(any(m.get("role") == "user" and m.get("content") == expected_human
                                      for m in actual_messages), "summary_wire_exact_original_segment", location)
                            counters["summary_wire_checks"] += 1
                        if not item.get("error_type"):
                            check(bool(wired), "successful_summary_has_wire", location)
                            possible = {text for call in calls for text in _summary_values(call.get("outputs", []))}
                            check(item["legacy_summary"].strip() in possible, "saved_summary_from_model_output", location)
                    last_summary, last_upto = changed
                    last_watermark = item["row_watermark"]
                    counters["maintenance_checks"] += 1
                except (KeyError, TypeError, ValueError, IndexError) as exc:
                    check(False, "maintenance_read_or_replay", location, f"{type(exc).__name__}: {exc}")
            final_seed = record(root / "seeds" / f"{prefix}.json")
            final_turns = [record(root / "turns" / f"{prefix}-{i}.json")
                           for i in range(1, len(cases[case_id]["steps"]) + 1)]
            finals = [item for item in final_turns if item is not None]
            final = finals[-1] if finals else final_seed
            if final is not None:
                check((last_summary, last_upto) == (final.get("summary") or "", final.get("summary_upto") or 0),
                      "maintenance_matches_final_persistence", prefix)

    for path in sorted((root/'calls').glob('*.json')):
        call=record(path)
        if call is None:continue
        wire=call.get('wire_payload')
        if not isinstance(wire,dict):
            check(call.get('status')!='ok','successful_call_has_wire',path.name)
            continue
        parameters=(call.get('input') or {}).get('parameters') or {}
        expected=manifest['config']['environment']['chat_model']
        check(wire.get('model',parameters.get('model'))==expected,'same_model',path.name)
        extra=wire.get('extra_body') or parameters.get('extra_body') or {}
        thinking=extra.get('chat_template_kwargs') or {}
        check(thinking.get('thinking') is False and thinking.get('enable_thinking') is False,
              'thinking_disabled',path.name)
        check(wire.get('max_completion_tokens',wire.get('max_tokens'))==8192,
              'same_output_limit',path.name)
        check(not list(_gold_keys(wire)),'no_gold_fields_in_wire',path.name)
    result = {"passed": not issues, "issues": issues, "current_and_frozen_sources_match": sources_match,
              "view_replays": counters["view_replays"], "source_views": counters["source_views"],
              "production_base_checks": counters["production_base_checks"],
              "maintenance_checks": counters["maintenance_checks"],
              "summary_wire_checks": counters["summary_wire_checks"], "checks": dict(checked),
              "scope": "Visible source lineage, all-visible BM25 +/-1 replay, per-node actual raw-ID dedup, unchanged production base views, source hashes, keep3/30-message maintenance, exact summary inputs/outputs and wire model/thinking. General score/usage audit separate.",
              "limits": ["Does not judge semantic correctness of summaries or generated replies.",
                         "No-summary arms move a local archive boundary without a model call; retained originals grow from 3 to 14 ordinary completed turns before the next successful maintenance check.",
                         "Summary failures preserve the old boundary; actual arm trajectories can consequently diverge."]}
    (root / "AUDIT_DETAIL.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    report = audit(parser.parse_args().output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
