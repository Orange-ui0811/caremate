"""Offline score audit: reject rewrites with fabricated order IDs or wrong products.

The initial entity-hit rule can pass an answer containing a correct SKU alongside
an invented order. Apply the same stricter checks to ALL saved strategies; never
change raw outputs, fixed cases or production prompts.
"""
import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

from scripts.eval_context_engineering import PRODUCTS, make_report, present, score_rewrite, write_json

FAMILIES = {"MH-LP100": "猫砂盆", "MH-W40": "饮水机", "MH-FD10": "喂食器",
            "MH-CT30": "猫爬架", "MH-NEST20": "猫窝", "MH-CAM1": "摄像头"}


def audit_rewrite(case, text):
    reasons = []
    if not score_rewrite(case, text):
        reasons.append("target_or_intent_missing_or_distractor_present")
    numbers = lambda value: set(re.findall(r"(?<!\d)\d{4,}(?!\d)", value))
    allowed = numbers("\n".join(r["content"] for r in case["history"]) + case["query"])
    for value in sorted(numbers(text) - allowed):
        reasons.append("fabricated_numeric_entity:" + value)
    target_sku = case["historical_entities"][1]
    if case["kind"] == "recent_reference":
        target_sku = next(sku for _, sku in PRODUCTS if present(sku, case["history"][-1]["content"]))
    for _, sku in PRODUCTS:
        if sku != target_sku and present(sku, text):
            reasons.append("wrong_product_sku:" + sku)
    for sku, family in FAMILIES.items():
        if sku != target_sku and family in text:
            reasons.append("wrong_product_family:" + family)
    return {"passed": not reasons, "reasons": reasons}


def audit(output):
    fixture = json.loads((output / "cases.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    results, decisions = [], []
    for case in fixture["cases"]:
        raw = json.loads((output / (case["id"] + ".json")).read_text(encoding="utf-8"))
        result = copy.deepcopy(raw)
        for strategy, row in result["strategies"].items():
            decision = audit_rewrite(case, row.get("text") or "")
            valid = bool(row.get("calls") and row.get("text"))
            final = valid and decision["passed"]
            decisions.append({"case_id": case["id"], "kind": case["kind"], "strategy": strategy,
                              "text": row.get("text"), "initial_pass": row["correct"],
                              "audited_pass": final, "reasons": decision["reasons"]})
            row["correct"] = final
        results.append(result)
    report = make_report(fixture, manifest, results)
    report["scoring_audit"] = {
        "version": "strict-entity-consistency-v1", "outputs_reviewed": len(decisions),
        "initial_false_positives": sum(d["initial_pass"] and not d["audited_pass"] for d in decisions),
        "reason": "Initial rule accepted a correct SKU alongside invented order 1001; all outputs uniformly rechecked for fabricated numbers and conflicting product identity.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    report["limitations"].append("Strict offline audit additionally rejects unsupported 4+-digit entities and wrong product SKU/family; this remains rule-based scoring, not independent human semantic judgment.")
    summary_input = report["summary_overhead"]["input_tokens"]
    compact_input = report["strategies"]["summary_window"]["coref_api_input_tokens"]
    full_input = report["strategies"]["full_history"]["coref_api_input_tokens"]
    combined = summary_input + compact_input if summary_input is not None and compact_input is not None else None
    report["cost_scope"] = {
        "all_summary_preprocessing_plus_one_final_coref_input_tokens": combined,
        "full_history_final_coref_input_tokens": full_input,
        "input_change_for_this_accounting": combined / full_input - 1 if combined is not None and full_input else None,
        "interpretation": "Includes all historical summary construction but only the final coreference query; earlier-turn coreference/Agent calls are not measured. Do not claim full-conversation cost savings."}
    write_json(output / "score_audit.json", decisions)
    write_json(output / "audited_report.json", report)
    (output / "source_snapshot/scripts/audit_context_eval.py").write_bytes(Path(__file__).read_bytes())
    lines = ["上下文工程评测（固定合成长对话，规则判分）", "",
             f"样例：{len(fixture['cases'])} 条，24/32/40/48/56/64 轮，四类各六条；不读写真实客服数据库。",
             "四类：早期订单追问、早期商品追问、订单号更正、近期话题切换。",
             "模型：" + manifest["config"]["chat_model"],
             "配置：" + json.dumps(manifest["config"], ensure_ascii=False),
             "数据 SHA256：" + manifest["dataset_sha256"], "",
             "策略 | 追问规则通过 | Agent历史实体覆盖 | 指代API输入Token均值 | Agent近似Token均值"]
    names = {"full_history": "完整历史", "window_only": "仅滑窗", "summary_window": "摘要+滑窗"}
    for name, row in report["strategies"].items():
        tokens = row["coref_api_input_tokens"]
        lines.append(f"{names[name]} | {row['coref_correct']}/{row['n']} ({row['coref_accuracy']:.1%}) | "
                     f"{row['entities_retained']}/{row['entities_total']} ({row['entity_coverage']:.1%}) | "
                     f"{tokens / row['n']:.1f}" if tokens is not None else f"{names[name]} | API用量缺失")
        lines[-1] += f" | {row['agent_tokens_approx_total'] / row['n']:.1f}"
    row = report["strategies"]["summary_window"]
    lines += ["", f"摘要策略相比完整历史：指代API输入Token减少 {row['coref_api_input_reduction_vs_full']:.2%}；"
              f"主Agent近似输入Token减少 {row['agent_token_reduction_vs_full_approx']:.2%}。",
              "摘要预处理开销（另计）：" + json.dumps(report["summary_overhead"], ensure_ascii=False),
              "把全部摘要预处理加上最后一次指代改写，输入Token为 " + str(combined)
              + "，完整历史最后一次改写为 " + str(full_input) + "。这不是全会话成本比较，因为没有评测此前每轮改写/Agent调用。",
              "", "评分解释：实体覆盖只表示订单号/型号/工单号出现在Agent历史或摘要中，不证明实体关系正确。",
              "追问规则通过同时要求目标实体正确、意图关键词保留、无干扰实体、无编造编号和商品串线。",
              "初筛假阳性数：" + str(report["scoring_audit"]["initial_false_positives"]),
              "", "限制：单次采样、合成模板压力测试；不代表端到端客服正确率、生产效果或整体成本下降。",
              "API Token为上游实报，Agent输入Token为项目近似计数（不含工具schema）。",
              "摘要任务在隔离测试中等待完成，不等同于线上异步时延。",
              "", "复算：python -m scripts.eval_context_engineering --output " + output.as_posix() + " --recompute",
              "严格审计：python -m scripts.audit_context_eval --output " + output.as_posix(),
              "重测：使用 eval_context_engineering 的新 --output 目录重新调用模型。"]
    (output / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output / "audited_report.json"), "audit": report["scoring_audit"],
                      "strategies": report["strategies"], "summary_overhead": report["summary_overhead"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    audit(Path(parser.parse_args().output))
