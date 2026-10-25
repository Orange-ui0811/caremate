"""Seal an audited replay, separating transport failures from paired policy results.

Offline only. Never calls a provider or changes cached model responses or scores.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main(output, paired):
    full = read(output / "audited_report.json")
    clean = read(paired / "audited_report.json")
    for folder, report in ((output, full), (paired, clean)):
        if not report["complete"] or not read(folder / "semantic_audit.json")["passed"]:
            raise RuntimeError("Incomplete or unaudited report: " + str(folder))
    fixture = read(output / "dataset.json")
    failed_stages, failed_memory, warning_records = Counter(), Counter(), Counter()
    per_arm = {}
    for arm in full["strategies"]:
        probe_errors = memory_errors = 0
        for case in fixture["cases"]:
            row = read(output / "cases" / case["id"] / (arm + ".json"))
            if len(row["probes"]) != 2:
                raise RuntimeError("Missing probe in " + case["id"] + "/" + arm)
            for probe in row["probes"]:
                bad = False
                for stage in ("coref", "plan", "reply"):
                    record = probe[stage]
                    if not record["ok"]:
                        failed_stages[stage + ":" + str(record.get("error_type"))] += 1
                        bad = True
                    if record["errors"]:
                        warning_records[stage] += 1
                probe_errors += bad
            for record in row["maintenance"]:
                if not record["ok"] and not record.get("validation_error"):
                    memory_errors += 1
                    # D/E share the same physical state extraction.
                    if arm != "E_state_retrieval":
                        failed_memory[record["stage"] + ":" + str(record.get("error_type"))] += 1
        per_arm[arm] = {"failed_probe_pipelines": probe_errors,
                        "failed_memory_events_attributed_to_arm": memory_errors}
    coverage = read(paired / "COVERAGE.json")
    selected_count = coverage["selected_conversations"]
    all_count = len(fixture["cases"])
    arm_count = len(full["strategies"])
    probe_count = clean["strategies"]["C_current"]["n"]
    turns = [c["turns"] for c in fixture["cases"]]
    family_count = len({c["kind"] for c in fixture["cases"]})
    archived = [str(p.relative_to(output)) for p in (output / "recovery_archive").glob("*/*/*.json")]
    status = {
        "status": "completed_with_transport_errors",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "planned_conversations": len(fixture["cases"]),
        "planned_probes_per_arm": len(fixture["cases"]) * 2,
        "total_scored_probes": len(fixture["cases"]) * 2 * len(full["strategies"]),
        "all_records_present_and_audited": True,
        "paired_error_free_conversations": coverage["selected_conversations"],
        "paired_probes_per_arm": probe_count,
        "failed_model_stages": dict(failed_stages),
        "failed_physical_memory_events": dict(failed_memory),
        "records_with_internal_error_callbacks": dict(warning_records),
        "per_arm": per_arm,
        "paired_family_counts": dict(Counter(c["kind"] for c in coverage["conversations"])),
        "archived_result_files": len(archived),
        "billing_scope": "Retained comparative workload only. Archived attempts are additional spend; timeout responses may lack usage, so exact total billed tokens are unavailable.",
        "interpretation": f"Use the same {selected_count} conversations for policy comparison. Selection by error records can bias the subset; no significant superiority is established by these intervals.",
        "full_audited_report_sha256": hashlib.sha256((output / "audited_report.json").read_bytes()).hexdigest(),
        "paired_audited_report_sha256": hashlib.sha256((paired / "audited_report.json").read_bytes()).hexdigest(),
        "finalizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "new_model_calls": 0,
    }
    (output / "FINAL_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 上下文八组对照评测封存结果", "",
             f"{all_count}条固定合成会话、{family_count}个场景族、每条{min(turns)}～{max(turns)}轮；每条2次追问、{arm_count}组，共{all_count*2*arm_count}条评分，离线审计通过。",
             "完成代表所有案例已执行和评分，不代表所有接口调用成功。首轮和一次超时补跑均已结束，不再无限重试。", "",
             f"## 完整{all_count}条的系统表现（包含接口失败）", "",
             "下表不能直接作为上下文能力排名；超时和维护失败另列。D/E的维护由同一组调用产生，两个方案分别计入部署成本。", "",
             "|方案|任务规则通过|追问调用失败|接口导致的维护失败|", "|---|---:|---:|---:|"]
    for arm, m in full["strategies"].items():
        e = per_arm[arm]
        lines.append(f"|{m['label']}|{m['task_pass']}/{m['n']} ({100*m['task_rate']:.1f}%)|{e['failed_probe_pipelines']}|{e['failed_memory_events_attributed_to_arm']}|")
    lines += ["", "## 无调用错误的共同子集", "",
              f"{selected_count}/{all_count}条会话，每组{probe_count}次追问，共{probe_count*arm_count}条评分。按八组调用和维护错误记录筛选，未按回答正确与否筛选；语义校验失败不因此被剔除。场景覆盖见FINAL_STATUS.json；存在选择偏差。", "",
              "|方案|任务规则通过|相对当前方案/百分点|输入Token含维护|输出Token|", "|---|---:|---:|---:|---:|"]
    for arm, m in clean["strategies"].items():
        diff = clean["comparisons"][arm]["task_rate_difference_vs_current"] * 100
        lines.append(f"|{m['label']}|{m['task_pass']}/{m['n']} ({100*m['task_rate']:.1f}%)|{diff:+.1f}|{m['usage']['input_tokens']:,}|{m['usage']['output_tokens']:,}|")
    reduction = 1-clean["strategies"]["C_current"]["usage"]["input_tokens"]/clean["strategies"]["A_full"]["usage"]["input_tokens"]
    full_gap = clean["comparisons"]["A_full"]["task_rate_difference_vs_current"] * 100
    harness_original_gap = clean["comparisons"]["H_original"]["task_rate_difference_vs_current"] * 100
    harness_customer_gap = clean["comparisons"]["H_customer"]["task_rate_difference_vs_current"] * 100
    lines += ["", "## 如何解释", "",
              f"- 当前摘要方案比完整历史少用{100*reduction:.1f}%输入Token，完整历史任务规则通过率相对当前方案高{full_gap:.1f}个百分点，不能称为无损压缩。",
              f"- Harness原提示词迁移/客服适配相对当前方案差值为{harness_original_gap:+.1f}/{harness_customer_gap:+.1f}个百分点。统计区间见审计报告；本轮区间包含0，不能声称显著优势。",
              "- 结构化状态维护成本较高，未观察到其最终任务效果超过当前方案；增加原文检索后有所恢复，仍需更大且更真实的数据验证。",
              "- 全量残留接口失败见FINAL_STATUS.json。普通成功阶段通常为秒级，长尾超时不能作为上下文推理能力差异。", "",
              "## 指标边界与复算", "",
              "成本覆盖历史维护与两个追问的指代、规划和回答；不是24～64轮历史全部在线生成的成本。已归档的失败/被替换尝试属于额外实际开销；超时可能没有Token回执，不能给出精确的账单总量。",
              "Harness两组为Python文本策略迁移，未运行原TypeScript运行时；默认80万Token触发点在本集不会触发，迁移组使用约3000Token触发点。",
              "数据仅来自8个场景族的合成变体，评分为动作、实体和回答锚点规则，不能代表生产客服语义正确率。延迟也不是生产异步摘要和全链路延迟。", "",
              "原始完整结果：audited_report.json、AUDITED_REPORT.md；共同子集：../20261016-gitee-v1-paired-clean/。", "",
              "```powershell",
              f".\\.venv\\Scripts\\python.exe -m scripts.audit_context_semantics --output {output.as_posix()}",
              f".\\.venv\\Scripts\\python.exe -m scripts.audit_context_semantics --output {paired.as_posix()}",
              "```", "",
              f"简历建议：构建{all_count}条、{min(turns)}～{max(turns)}轮合成长会话评测，覆盖{family_count}类场景、{arm_count}种上下文策略与{all_count*2*arm_count}次追问；建立含记忆维护的Token核算、调用失败分离和离线复算流程，定位实体更正、否定和时间更新中的记忆失效。",
              "不要将本轮写成摘要方案显著提高最终任务准确率，也不要将旧24条集上的指代规则通过率当成本轮最终任务指标。", ""]
    (output / "FINAL_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paired", type=Path, required=True)
    args = parser.parse_args()
    main(args.output, args.paired)
