from copy import deepcopy
import json
from pathlib import Path
import shutil

from langchain_core.messages import message_to_dict

from app.core.context_state import build_snapshot
from app.core.context_task import assemble_task_context
from scripts.audit_context_unified import ROOT, audit, digest, file_hash
from scripts.context_unified_cases import score_turn


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def fixture(tmp_path):
    output = tmp_path / "audit-fixture"
    names = ["app/core/context_task.py", "app/core/context_policy.py", "app/core/context_state.py",
             "scripts/context_unified_cases.py"]
    sources = {}
    for name in names:
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        sources[name] = file_hash(target)
    history = [{"role": "user", "content": "订单710001和710002都属于我，还没有选择。"},
               {"role": "assistant", "content": "请确定想查询哪一单。"}]
    query, expected = "查哪一单还没决定。", {"kind": "clarify"}
    data = {"cases": [{"id": "unit", "family": "ambiguous", "stratum": "short", "initial_history": history,
                       "steps": [{"query": query, "expected": expected}]}]}
    write(output / "dataset.json", data)
    write(output / "manifest.json", {"arms": ["task"], "dataset_sha256": digest(data), "sources": sources,
                                     "config": {"environment": {"chat_model": "DeepSeek-V4-Flash"}}})
    rows = [{"id": i + 1, "conversation_id": 7, **row} for i, row in enumerate(history)]
    seed = {"coords": {"arm": "task", "case_id": "unit", "turn": 0}, "rows": deepcopy(rows), "call_ids": []}
    write(output / "seeds/task-unit.json", seed)
    rows.append({"id": 3, "conversation_id": 7, "role": "user", "content": query})
    snapshot = build_snapshot(rows, 7)
    inputs = {"snapshot": snapshot, "query": query, "policy": "task", "stage": "coref", "budget": None, "summaries": []}
    packet = assemble_task_context(**inputs)
    packet["messages"] = [message_to_dict(m) for m in packet["messages"]]
    coordinate = {"arm": "task", "case_id": "unit", "turn": 1}
    projection = {"coords": coordinate, "stage": "coref", "packet": packet,
                  "manifest": deepcopy(packet["manifest"]), "assembler_input": inputs}
    write(output / "projections/task-unit.json", [projection])
    rows.append({"id": 4, "conversation_id": 7, "role": "assistant", "content": "请确认想查询哪一笔订单。"})
    record = {"coords": coordinate, "query": query, "expected": expected, "status": "completed", "error": None,
              "interrupt": None, "answer": rows[-1]["content"], "call_ids": [], "actual_dialog_rows": rows,
              "state": {"current_message_id": 3, "conversation_id": 7, "resolved_query": query,
                        "memory_segments": [], "conversation_memory": snapshot},
              "executed_tools": [], "proposed_tools": [], "retrieval": []}
    record["score"] = score_turn(record)
    record["record_sha256"] = digest(record)
    write(output / "turns/task-unit-1.json", record)
    write(output / "maintenance/task-unit.json", [{"coords": coordinate, "row_watermark": 4,
          "triggered": False, "new_segments": [], "legacy_summary": "", "legacy_summary_upto": 0, "call_ids": []}])
    return output


def test_independent_audit_replays_frozen_inputs_without_api(tmp_path):
    output = fixture(tmp_path)
    result = audit(output)
    assert result["audit_ok"], result["issues"]
    assert result["complete"]
    assert result["checked"]["projection_exact_replay"] == 1
    assert result["checked"]["frozen_score_replay"] == 1


def test_projection_tampering_is_detected_even_without_record_hash(tmp_path):
    output = fixture(tmp_path)
    path = output / "projections/task-unit.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value[0]["packet"]["history"] += "被篡改的历史"
    write(path, value)
    codes = {i["code"] for i in audit(output)["issues"]}
    assert "projection_exact_replay" in codes


def test_source_ref_tampering_is_independently_detected(tmp_path):
    output = fixture(tmp_path)
    path = output / "projections/task-unit.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value[0]["manifest"]["source_refs"][0]["content_hash"] = "forged"
    value[0]["packet"]["manifest"] = deepcopy(value[0]["manifest"])
    write(path, value)
    codes = {i["code"] for i in audit(output)["issues"]}
    assert {"source_ref_original_hash_offsets", "projection_exact_replay"}.issubset(codes)


def test_future_prefix_and_gold_fields_are_detected(tmp_path):
    output = fixture(tmp_path)
    path = output / "projections/task-unit.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value[0]["assembler_input"]["snapshot"]["rows"].append({"id": 5, "role": "user", "content": "未来消息"})
    value[0]["assembler_input"]["snapshot"]["expected"] = {"target": "710001"}
    write(path, value)
    codes = {i["code"] for i in audit(output)["issues"]}
    assert {"projection_matches_actual_prefix", "projection_watermark", "no_gold_fields_in_assembler_input"}.issubset(codes)


def test_frozen_score_is_recomputed_and_not_repaired(tmp_path):
    output = fixture(tmp_path)
    path = output / "turns/task-unit-1.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["score"]["task_rule_pass"] = False
    value["record_sha256"] = digest({k: v for k, v in value.items() if k != "record_sha256"})
    write(path, value)
    before = path.read_bytes()
    result = audit(output)
    assert "frozen_score_replay" in {i["code"] for i in result["issues"]}
    assert path.read_bytes() == before
