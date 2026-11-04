"""Read-only analysis of the frozen keep-three-round comparison.

Only derived ANALYSIS.json and FINAL_REPORT.md are written. This module imports
no application code and makes no model calls or semantic scoring decisions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


LABELS = {"full": "完整历史", "window": "仅滑动窗口", "summary": "滚动摘要＋滑窗",
          "retrieval": "相关原文检索＋近期窗口", "hybrid": "滚动摘要＋滑窗＋原文回查（本方案）"}
SUMMARY_PURPOSE = "conversation_summary"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0


def token_cost(calls, *, possible_unreported_attempts=False):
    """Sum reported components, retaining missing usage as a lower bound."""
    result = {"model_calls": len(calls), "input_tokens": 0, "output_tokens": 0,
              "unknown_input_calls": 0, "unknown_output_calls": 0, "unknown_usage_calls": 0}
    for call in calls:
        usage = call.get("usage") or {}
        unknown = False
        for part in ("input", "output"):
            value = usage.get(part + "_tokens")
            if _number(value):
                result[part + "_tokens"] += value
            else:
                result["unknown_" + part + "_calls"] += 1
                unknown = True
        result["unknown_usage_calls"] += unknown
    result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    result["possible_unreported_attempts"] = possible_unreported_attempts
    result["cost_complete"] = not (result["unknown_usage_calls"] or possible_unreported_attempts)
    result["total_is_lower_bound"] = not result["cost_complete"]
    return result


def _attempts(records):
    """Pool attempt numbers are evidence of retries; repeated prompts are not."""
    found, unavailable = {}, 0
    def visit(value):
        nonlocal unavailable
        if isinstance(value, dict):
            if "pool_attempts" in value:
                attempts = value["pool_attempts"]
                if attempts is None:
                    unavailable += 1
                for attempt in attempts or []:
                    identity = attempt.get("id") or json.dumps(attempt, sort_keys=True)
                    found[identity] = attempt
            for key, child in value.items():
                if key != "pool_attempts" and isinstance(child, (list, dict)):
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    for record in records:
        visit(record)
    attempts = list(found.values())
    outcomes = Counter(str(a.get("outcome", "unknown")) for a in attempts)
    # These counters do not claim a one-to-one mapping to LangChain call IDs.
    successful = {"OK", "SUCCESS", "SUCCEEDED"}
    failures = [a for a in attempts if str(a.get("outcome", "")).upper()
                not in successful | {"RUNNING"}]
    # Store.reject records admission failure before outbound(): no provider
    # request was sent. Other failed/unfinished attempts can have unknown cost.
    possibly_unreported = [a for a in attempts if str(a.get("outcome", "")).upper()
                           not in successful | {"REJECTED"}]
    return {"pool_attempts": len(attempts),
            "retry_attempts": sum((_number(a.get("attempt")) and a["attempt"] > 1) for a in attempts),
            "pool_failed_attempts": len(failures), "outcomes": dict(outcomes),
            "pool_rejected_before_send": sum(str(a.get("outcome", "")).upper() == "REJECTED" for a in attempts),
            "pool_running_attempts": sum(str(a.get("outcome", "")).upper() == "RUNNING" for a in attempts),
            "possibly_unreported_attempts": len(possibly_unreported),
            "unavailable_pool_snapshots": unavailable,
            "cost_coverage_unknown": bool(possibly_unreported or unavailable),
            "retry_token_attribution": "All recorded calls are included; pool retries cannot be individually mapped to provider usage, so separate retry tokens are not inferred."}


def _tags(case, step, number):
    # Case tags can be the union of three step tags, so never merge them when
    # the step supplies an explicit tag list (including an empty one).
    if "evaluation_tags" in step:
        return set(step["evaluation_tags"] or [])
    tags = set(case.get("evaluation_tags") or [])
    if case.get("family") == "adjacent_correction" or (
            case.get("family") == "early_order_corrected" and number == 1):
        tags.add("correction")
    return tags


def _passed(row):
    return bool(row and row.get("score", {}).get("task_rule_pass"))


def _check(row, key):
    return bool(row and row.get("score", {}).get("checks", {}).get(key))


def _metric(items, predicate):
    total = len(items)
    passed = sum(bool(predicate(item)) for item in items)
    return {"pass": passed, "denominator": total,
            "rate": passed / total if total else None,
            "missing_records": sum(item["record"] is None for item in items)}


def _target_query(item):
    row, kind = item["record"], item["expected"].get("kind")
    if kind == "care":
        return _check(row, "correct_product_retrieved") and _check(row, "no_other_product_retrieved")
    if kind == "logistics_failure":
        return _check(row, "targeted_logistics_attempted") and _check(row, "no_wrong_order_read")
    return _check(row, "targeted_logistics_executed") and _check(row, "no_wrong_order_read")


def _stratum_stats(items, case_ids, calls, records_by_case, cases):
    selected_calls = [call for call in calls if call.get("coords", {}).get("case_id") in case_ids]
    records = [record for cid in case_ids for record in records_by_case.get(cid, [])]
    attempts = _attempts(records)
    uncertain = attempts["cost_coverage_unknown"]
    summary_calls = [c for c in selected_calls if c.get("purpose") == SUMMARY_PURPOSE]
    result = {"task": _metric(items, lambda x: _passed(x["record"])),
              "episodes": {"pass": sum(all(_passed(item["record"]) for item in items
                               if item["case_id"] == cid) for cid in case_ids),
                           "denominator": len(case_ids)},
              "cost": token_cost(selected_calls, possible_unreported_attempts=uncertain),
              "summary_cost": token_cost(summary_calls, possible_unreported_attempts=bool(uncertain and summary_calls)),
              "retries": attempts}
    result["episodes"]["rate"] = (result["episodes"]["pass"] / len(case_ids)) if case_ids else None
    return result


def _cost_text(cost, part="total"):
    field = part + "_tokens"
    unknown = (not cost["cost_complete"] if part == "total" else
               cost["unknown_" + part + "_calls"] or cost.get("possible_unreported_attempts"))
    return ("≥" if unknown else "") + f"{cost[field]:,}"


def _rate(metric):
    n = metric["denominator"]
    return f"{metric['pass']}/{n}（{metric['pass'] / n:.1%}）" if n else "—（0项）"


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def analyze(output):
    root = Path(output).resolve()
    if (root / "writer.lock").exists():
        raise ValueError("Evaluation writer is active; analyze only after it exits")
    manifest, dataset = read(root / "manifest.json"), read(root / "dataset.json")
    cases = {case["id"]: case for case in dataset["cases"]}
    if len(cases) != len(dataset["cases"]):
        raise ValueError("Duplicate case IDs")
    arms = manifest["arms"]
    if not arms or set(arms) - LABELS.keys():
        raise ValueError("Expected a nonempty subset of full/window/summary/retrieval/hybrid")
    labels = {**LABELS, **manifest.get("labels", {})}
    audit = read(root / "audited_report.json") if (root / "audited_report.json").exists() else {}
    detail = read(root / "AUDIT_DETAIL.json") if (root / "AUDIT_DETAIL.json").exists() else None
    rows = [read(path) for path in sorted((root / "turns").glob("*.json"))]
    calls = [read(path) for path in sorted((root / "calls").glob("*.json"))]
    seeds = [read(path) for path in sorted((root / "seeds").glob("*.json"))]
    indexed = {}
    for row in rows:
        coords = row["coords"]
        key = (coords["arm"], coords["case_id"], coords["turn"])
        if key in indexed:
            raise ValueError(f"Duplicate turn coordinates: {key}")
        if key[0] not in arms or key[1] not in cases or not 1 <= key[2] <= len(cases[key[1]]["steps"]):
            raise ValueError(f"Unexpected turn coordinates: {key}")
        indexed[key] = row

    groups, failures, all_items = {}, [], {}
    for arm in arms:
        items = [{"case_id": cid, "turn": n, "query": step["query"], "expected": step["expected"],
                  "tags": sorted(_tags(case, step, n)), "record": indexed.get((arm, cid, n))}
                 for cid, case in cases.items() for n, step in enumerate(case["steps"], 1)]
        all_items[arm] = items
        arm_calls = [call for call in calls if call.get("coords", {}).get("arm") == arm]
        arm_rows = [item["record"] for item in items if item["record"] is not None]
        records_by_case = defaultdict(list)
        for record in arm_rows + [seed for seed in seeds if seed.get("coords", {}).get("arm") == arm]:
            records_by_case[record["coords"]["case_id"]].append(record)
        overall = _stratum_stats(items, set(cases), arm_calls, records_by_case, cases)
        correction = [item for item in items if "correction" in item["tags"]]
        correction_orders = [item for item in correction if item["expected"].get("kind") in {"logistics", "logistics_failure"}]
        restrictions = [item for item in items if item["expected"].get("active_constraints") or "restriction" in item["tags"]]
        clarification = [item for item in items if item["expected"].get("kind") == "clarify"]
        metrics = {
            "correction_task": _metric(correction, lambda x: _passed(x["record"])),
            "correction_no_wrong_order_read": _metric(correction_orders, lambda x: _check(x["record"], "no_wrong_order_read")),
            "restriction_task": _metric(restrictions, lambda x: _passed(x["record"])),
            "restriction_target_query": _metric(restrictions, _target_query),
            "restriction_no_write_proposal_or_execution": _metric(restrictions, lambda x: _check(x["record"], "no_write_proposal_or_execution")),
            "clarification_task": _metric(clarification, lambda x: _passed(x["record"])),
            "clarification_requested": _metric(clarification, lambda x: _check(x["record"], "clarification_requested")),
            "clarification_no_order_read": _metric(clarification, lambda x: _check(x["record"], "no_order_read_before_choice")),
        }
        group = {"label": labels[arm], "overall": overall, "metrics": metrics}
        for field in ("cohort", "stratum", "family"):
            values = sorted({str(case.get(field, "unspecified")) for case in cases.values()})
            group[field] = {}
            for value in values:
                ids = {cid for cid, case in cases.items() if str(case.get(field, "unspecified")) == value}
                group[field][value] = _stratum_stats([item for item in items if item["case_id"] in ids], ids,
                                                    arm_calls, records_by_case, cases)
        group["position"] = {str(n): _metric([item for item in items if item["turn"] == n], lambda x: _passed(x["record"]))
                             for n in sorted({item["turn"] for item in items})}
        group["first_followup"] = _metric([item for item in items if item["turn"] == 1], lambda x: _passed(x["record"]))
        group["later_followups"] = _metric([item for item in items if item["turn"] > 1], lambda x: _passed(x["record"]))
        group["purposes"] = {purpose: token_cost([call for call in arm_calls if call.get("purpose", "unknown") == purpose],
                                                possible_unreported_attempts=overall["retries"]["cost_coverage_unknown"])
                             for purpose in sorted({call.get("purpose", "unknown") for call in arm_calls})}
        group["call_statuses"] = dict(Counter(call.get("status", "unknown") for call in arm_calls))
        group["protocol_invalid_calls"] = sum(call.get("structured_response_check", {}).get("status") == "invalid" for call in arm_calls)
        group["missing_turns"] = sum(item["record"] is None for item in items)
        group["failed_turns"] = sum(row.get("status") != "completed" for row in arm_rows)
        group["failure_kinds"] = {kind: sum(bool(row.get("score", {}).get(kind)) for row in arm_rows)
                                  for kind in ("transport_failure", "dependency_blocked", "budget_failure", "infrastructure_failure")}
        maintenance = [entry for path in (root / "maintenance").glob(f"{arm}-*.json") for entry in read(path)]
        group["maintenance_failures"] = sum(bool(entry.get("error_type")) for entry in maintenance)
        group["maintenance_events"] = len(maintenance)
        group["latency"] = {field: {"p50": percentile([row[field] for row in arm_rows if _number(row.get(field))], .5),
                                    "p95": percentile([row[field] for row in arm_rows if _number(row.get(field))], .95)}
                            for field in ("end_to_end_seconds", "foreground_complete_seconds", "first_visible_sse_delta_seconds")}
        group["generic_audit_cost"] = audit.get("groups", {}).get(arm)
        groups[arm] = group
        for item in items:
            if _passed(item["record"]):
                continue
            row = item["record"] or {}
            state = row.get("state") or {}
            checks = row.get("score", {}).get("checks", {})
            failures.append({"coords": {"arm": arm, "case_id": item["case_id"], "turn": item["turn"]},
                "raw_record": f"turns/{arm}-{item['case_id']}-{item['turn']}.json", "missing": not row,
                "query": item["query"], "expected": item["expected"], "evaluation_tags": item["tags"],
                "checks": checks, "failed_checks": [key for key, value in checks.items() if not value],
                "score": row.get("score"), "status": row.get("status"), "error": row.get("error"),
                "resolved_query": state.get("resolved_query"), "intent": state.get("intent"), "route": state.get("route"),
                "answer": row.get("answer"), "summary": row.get("summary"), "summary_upto": row.get("summary_upto"),
                "proposed_tools": row.get("proposed_tools"), "executed_tools": row.get("executed_tools"),
                "retrieval": row.get("retrieval"), "actual_dialog_rows": row.get("actual_dialog_rows"),
                "call_ids": row.get("call_ids")})

    paired = {}
    if "hybrid" in arms:
        for other in arms:
            if other == "hybrid":
                continue
            counts = Counter({key: 0 for key in ("both_pass", "both_fail", "hybrid_only", "reference_only")})
            discordant, by_cohort, episode_counts = [], defaultdict(Counter), Counter()
            for cid, case in cases.items():
                a_episode, b_episode = True, True
                for n, step in enumerate(case["steps"], 1):
                    a, b = _passed(indexed.get(("hybrid", cid, n))), _passed(indexed.get((other, cid, n)))
                    key = "both_pass" if a and b else "both_fail" if not a and not b else "hybrid_only" if a else "reference_only"
                    counts[key] += 1
                    by_cohort[str(case.get("cohort", "unspecified"))][key] += 1
                    a_episode, b_episode = a_episode and a, b_episode and b
                    if a != b:
                        discordant.append({"case_id": cid, "turn": n, "outcome": key, "query": step["query"]})
                key = "both_pass" if a_episode and b_episode else "both_fail" if not a_episode and not b_episode else "hybrid_only" if a_episode else "reference_only"
                episode_counts[key] += 1
            left, right = groups["hybrid"]["overall"]["cost"], groups[other]["overall"]["cost"]
            exact = left["cost_complete"] and right["cost_complete"]
            paired[other] = {"turn_counts": dict(counts), "episode_counts": dict(episode_counts),
                "cohort": {key: dict(value) for key, value in by_cohort.items()}, "discordant": discordant,
                "task_rate_difference_percentage_points": 100 * (groups["hybrid"]["overall"]["task"]["rate"] - groups[other]["overall"]["task"]["rate"]),
                "cost_comparison_complete": exact,
                "total_token_change_percent": (100 * (left["total_tokens"] / right["total_tokens"] - 1)) if exact and right["total_tokens"] else None,
                "cost_note": "All planned cases; no success-only subset. Percentage suppressed if either cost ledger is incomplete."}

    independent = "pending" if detail is None else "passed" if detail.get("passed") is True else "failed"
    formal = bool(audit.get("complete") and audit.get("audit_passed") and independent == "passed")
    result = {"version": "context-keep3-analysis-v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "output": str(root), "replicate": manifest.get("replicate", 1), "dataset_version": dataset.get("version"),
        "arms": arms, "cases": len(cases), "planned_turns_per_arm": sum(len(case["steps"]) for case in cases.values()),
        "groups": groups, "paired_hybrid_vs": paired, "failures": failures,
        "audit": {"generic_complete": audit.get("complete"), "generic_passed": audit.get("audit_passed"),
                  "generic_issues": audit.get("issues", []), "independent_status": independent,
                  "independent_issues": (detail or {}).get("issues", []), "formal_audit_passed": formal},
        "metric_definitions": {"task": "Frozen rule score, all planned steps; missing records count as not passed.",
            "episode": "All actual follow-up steps pass, not all scripted historical rounds.",
            "correction": "Step correction tags take priority; case tags only if step tags absent; fallback families are explicit.",
            "restriction": "Active constraints or restriction step tag; target read and absence of write proposal/execution are separate.",
            "clarification": "Only expected.kind=clarify; later explicit-selection turns excluded.",
            "cost": "Known provider input+output for all calls, including initialization, maintenance and retry calls; unknown usage is a lower bound.",
            "retry": "Explicit pool attempt>1 counted after deduplication; no heuristic attribution of tokens to retry attempts."},
        "raw_hashes": {str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for name in ("manifest.json", "dataset.json", "audited_report.json", "AUDIT_DETAIL.json")
            for path in [root / name] if path.exists()}}
    for folder in ("turns", "calls", "seeds", "maintenance", "projections"):
        for path in sorted((root / folder).glob("*.json")):
            result["raw_hashes"][str(path.relative_to(root)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "ANALYSIS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(root, result, manifest)
    return {"formal_audit_passed": formal, "independent_audit": independent,
            "groups": {arm: {"task": group["overall"]["task"], "episodes": group["overall"]["episodes"],
                              "cost": group["overall"]["cost"]} for arm, group in groups.items()},
            "failures": len(failures), "report": str(root / "FINAL_REPORT.md")}


def _write_report(root, result, manifest):
    groups, audit = result["groups"], result["audit"]
    status = "正式审计通过" if audit["formal_audit_passed"] else "分析已生成，正式审计未通过或待完成"
    lines = ["# 保留3轮：上下文方法对照分析", "", f"**{status}。** 独立审计：{audit['independent_status']}。",
        "", f"本目录为第 {result['replicate']} 次运行，{result['cases']} 条合成会话，每组 {result['planned_turns_per_arm']} 次实际追问。"
        "每条连续三问使用自身此前的真实模型回答；它们相关，不能当作独立样本。", "",
        "真实模型及客服 Workflow，业务和知识服务使用隔离夹具。任务通过是预先冻结的规则检查，不等于人工语义准确率或线上准确率。", "",
        "## 方法与输入范围", "",
        "四个精简组共用30条消息触发、摘要后保留3轮的边界规则；无摘要组只在本地推进同一批次窗口。"
        "指代消解和意图识别沿用最近6条消息输入，最终回答沿用原3000估算Token截取，查询改写输入不变。"
        "原文回查使用冻结选择器及来源位置；完整历史组取消这些历史截取，作为资源参照。"
        "这里的留3轮是维护后的保留量，不代表每次模型输入始终只有3轮。", "",
        str(manifest.get("comparison", "")), "", "## 效果与全部模型成本", "",
        "|方法|任务规则通过|整条三问全过|输入Token|输出Token|总Token|摘要输入/输出|模型调用|未知usage|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for group in groups.values():
        overall = group["overall"]; cost, summary = overall["cost"], overall["summary_cost"]
        lines.append(f"|{group['label']}|{_rate(overall['task'])}|{_rate(overall['episodes'])}|{_cost_text(cost, 'input')}|{_cost_text(cost, 'output')}|{_cost_text(cost)}|{_cost_text(summary, 'input')} / {_cost_text(summary, 'output')}|{cost['model_calls']}|{cost['unknown_usage_calls']}|")
    lines += ["", "≥ 表示仅能确认的成本下界。缺失 usage、未结束的网络尝试或缺失池快照不能按零计算；"
        "任一侧账本不完整时，不给出总成本下降百分比。摘要成本已包含在总成本中，不能重复相加。", "",
        "## 更正、限制与澄清", "",
        "|方法|更正任务通过|更正未查错订单|限制下目标查询|无写提议/执行|澄清任务通过|确实提出澄清|澄清前不读订单|",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    keys = ("correction_task", "correction_no_wrong_order_read", "restriction_target_query",
            "restriction_no_write_proposal_or_execution", "clarification_task", "clarification_requested", "clarification_no_order_read")
    for group in groups.values():
        lines.append("|" + group["label"] + "|" + "|".join(_rate(group["metrics"][key]) for key in keys) + "|")
    lines += ["", "更正分母按逐问标签；未查错订单只覆盖更正中的订单查询题。限制分母为存在有效限制或标记 restriction 的追问。"
        "目标查询需实际查询正确对象；预设业务失败题检查正确对象的尝试。无写提议/执行只检查这批只读任务，**不证明完整正向写授权正确**。"
        "澄清分母仅为 expected.kind=clarify，明确选择后的查询不计入。所有检查保留原始规则，不增加主观自动语义判断。"]
    for field, title in (("cohort", "已见回归与新增内部样例"), ("stratum", "短长会话分层")):
        lines += ["", "## " + title, "", "|方法|类别|任务通过|整条全过|含维护输入|含维护输出|总Token|", "|---|---|---:|---:|---:|---:|---:|"]
        for group in groups.values():
            for category, values in group[field].items():
                cost = values["cost"]
                lines.append(f"|{group['label']}|{_cell(category)}|{_rate(values['task'])}|{_rate(values['episodes'])}|{_cost_text(cost, 'input')}|{_cost_text(cost, 'output')}|{_cost_text(cost)}|")
    lines += ["", "## 首问与后续追问", "", "|方法|首问|第二问|第三问|后续追问合计|", "|---|---:|---:|---:|---:|"]
    for group in groups.values():
        cells = [_rate(group["position"].get(str(n), {"pass": 0, "denominator": 0})) for n in (1, 2, 3)]
        lines.append("|" + group["label"] + "|" + "|".join(cells + [_rate(group["later_followups"])]) + "|")
    lines += ["", "## 本方案与各组配对", "", "输赢针对相同会话和相同追问位置；全部预定样例纳入，没有只选择成功共同子集。", "",
        "|参照|双方通过|仅本方案通过|仅参照通过|双方未通过|通过率差（百分点）|总Token变化|", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, pair in result["paired_hybrid_vs"].items():
        counts = pair["turn_counts"]
        change = pair["total_token_change_percent"]
        cost = f"{change:+.1f}%" if change is not None else "无法确定（usage不全或分母为0）"
        lines.append(f"|{groups[arm]['label']}|{counts['both_pass']}|{counts['hybrid_only']}|{counts['reference_only']}|{counts['both_fail']}|{pair['task_rate_difference_percentage_points']:+.1f}|{cost}|")
    if not result["paired_hybrid_vs"]:
        lines.append("|本目录不包含本方案与参照组合|—|—|—|—|—|—|")
    lines += ["", "整条会话的配对输赢、分 cohort 输赢及所有不一致样例见 ANALYSIS.json。效果差异与模型随机性、摘要内容和后续错误传播均有关，单次小样本不能直接证明普遍优势。", "",
        "## 重试、接口与维护故障", "", "|方法|池尝试|明确重试|失败池尝试|发送前拒绝|未结束池尝试|接口错误调用|未完成调用|失败轮|摘要维护失败|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for group in groups.values():
        retry, statuses = group["overall"]["retries"], group["call_statuses"]
        lines.append(f"|{group['label']}|{retry['pool_attempts']}|{retry['retry_attempts']}|{retry['pool_failed_attempts']}|{retry['pool_rejected_before_send']}|{retry['pool_running_attempts']}|{statuses.get('interface_error', 0)}|{statuses.get('running', 0)}|{group['failed_turns']}|{group['maintenance_failures']}|")
    lines += ["", "重试按池记录 attempt>1 去重统计。已记录重试调用的 Token 全部纳入；池尝试不能逐条对应 usage 时，单独的重试 Token 记为未知，不能凭相同提示词推算。"
        "REJECTED 明确为发送前拒绝，该池尝试本身不构成供应商未知费用；调用账本另有缺失 usage 时仍单独保留未知。"
        "无法确定失败池尝试属于哪个用途时，相关用途及摘要分账保守标为下界，不将该未知开销重复加到总账。"
        "接口故障与任务答错分别保留，不将故障解释为记忆能力差异。", "",
        "## 审计与解释边界", "",
        f"- 账本完整：{audit['generic_complete']}；账本/冻结评分审计：{audit['generic_passed']}；独立来源、窗口与维护回放：{audit['independent_status']}。",
        "- 只有 AUDIT_DETAIL.json 的 passed=true 且通用审计完整通过，才标记正式审计通过。审计复算不代替人工语义核对。",
        "- 完整24例设计为16条已见回归＋8条新增内部样例；子集/重复运行只统计该目录所含会话。新增样例仍不是外部盲测。",
        "- 全部输入/输出包含历史摘要初始化、三次真实追问各节点、后续维护和可记录重试；不含预置历史每轮回复的生成，也不含真实向量检索和业务服务费用。",
        "- Token合计不是账单金额；没有将缓存折扣和不同输出价格换算为金额。",
        "- 测试等待后台维护完成以固定状态；并发及供应商排队会影响耗时，延迟仅作运行诊断，不代表生产客服响应延迟。",
        "- 规则通过检查对象、工具执行与特定回答事实，不覆盖全部自然语言忠实性；需阅读失败原文并人工复核。", "",
        "## 失败记录", "", "ANALYSIS.json 保存每个失败的原问题、消解问题、完整回答、摘要、工具记录、实际对话和所有检查项；不修改原始得分。"]
    for failure in result["failures"]:
        coords = failure["coords"]
        lines += ["", f"- {coords['arm']} / {coords['case_id']} / 第{coords['turn']}问：{_cell(failure['query'])}",
                  f"  未通过：{', '.join(failure['failed_checks']) or '缺失记录'}；消解：{_cell(failure['resolved_query'])}。"]
    if not result["failures"]:
        lines.append("本目录预定追问未发现规则失败；这不意味着不存在规则未覆盖的语义错误。")
    (root / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    print(json.dumps(analyze(parser.parse_args().output), ensure_ascii=False, indent=2))
