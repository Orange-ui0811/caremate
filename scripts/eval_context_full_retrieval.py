"""Supplement the frozen 24-case study with the previous complete task selector.

Only --run invokes the configured model. Previous experiments are read-only.
No production module, prompt, selector or scoring function is modified.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

from scripts import eval_context_unified as unified

ROOT = unified.ROOT
PARENT = ROOT / "data/ch07/context-keep3/20261027-formal-v1"
PREVIOUS = ROOT / "data/ch07/context-hybrid/20261027-formal-v1"
ARMS = ("task",)
LABEL = "完整原文检索＋近期窗口（此前方案）"
ORIGINAL_AUDIT = unified.audit
ORIGINAL_ISOLATED = unified.isolated_workflow
SETTINGS = deepcopy(unified.SETTINGS)
EXTRA_SOURCES = ["scripts/eval_context_full_retrieval.py", "scripts/analyze_context_keep3.py",
                 "scripts/audit_context_unified.py", "tests/test_context_full_retrieval.py"]
IDENTITY_SOURCES = ["app/core/context_task.py", "app/core/context_policy.py",
    "app/core/context_state.py", "app/core/context_runtime.py", "app/core/prompts.py",
    "app/core/coref.py", "app/core/intent.py", "app/core/query_understanding.py",
    "scripts/eval_context_unified.py", "scripts/context_unified_cases.py"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset(pilot=False):
    if pilot:
        # Separate existing smoke case; never part of the formal denominator.
        return unified.read(ROOT / "data/ch07/context-keep3/20261027-pilot-v1/dataset.json")
    return unified.read(PARENT / "dataset.json")


def lineage():
    parent = unified.read(PARENT / "manifest.json")
    previous = unified.read(PREVIOUS / "manifest.json")
    checks = {name: {"previous": previous["sources"][name], "current": sha(ROOT/name)}
              for name in IDENTITY_SOURCES}
    if any(x["previous"] != x["current"] for x in checks.values()):
        raise ValueError("previous selector or shared task pipeline source changed")
    return {"parent": str(PARENT), "previous": str(PREVIOUS),
        "parent_dataset_digest": parent["dataset_sha256"],
        "parent_dataset_file_sha256": sha(PARENT/"dataset.json"),
        "parent_analysis_sha256": sha(PARENT/"ANALYSIS.json"),
        "previous_manifest_sha256": sha(PREVIOUS/"manifest.json"),
        "identity_sources": checks,
        "known_changes_since_previous": "app/config.py and app/graph/nodes.py gained optional legacy-only recall; task uses source path, not legacy recall"}


def effective_config():
    return {"environment": unified.base.config(), "overrides": SETTINGS,
        "thinking": unified.base.OFF, "failover": False,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "unset"),
        "selector": "unchanged context_task.assemble_task_context(policy=task)",
        "search": "all visible original Q&A, user-text BM25, top3, protected latest+strongest, recent2, previous+next units, one-hop shared entities",
        "input": "previous per-node packet path and original stage caps; original full serialized input cap 16000",
        "stage_caps": unified.CAPS, "summary": "none; no model maintenance",
        "comparison": "Supplemental system reference, not BM25-only ablation; five frozen controls are reused read-only",
        "lineage": lineage()}


@contextmanager
def binding():
    with ExitStack() as stack:
        for name, value in {"ARMS": ARMS, "SETTINGS": SETTINGS, "dataset": dataset,
                            "effective_config": effective_config, "audit": audit,
                            "isolated_workflow": ORIGINAL_ISOLATED}.items():
            stack.enter_context(patch.object(unified, name, value))
        stack.enter_context(patch.object(unified.base, "SOURCES",
            list(dict.fromkeys(unified.base.SOURCES + EXTRA_SOURCES))))
        yield


def prepare(output, *, pilot=False, case_ids=None):
    parent_environment=unified.read(PARENT/"manifest.json")["config"]["environment"]
    current=unified.base.config()
    if any(current[key]!=parent_environment[key] for key in ("chat_model","chat_base_url")):
        raise ValueError("model or endpoint differs from the frozen five-arm controls")
    with binding():
        manifest = unified.prepare(output, ARMS, case_ids, pilot, turn_timeout=180)
    manifest.update(version="previous-full-retrieval-on-keep3-cases-v1", labels={"task": LABEL},
        comparison="Previous complete task selector on the exact current cases and gold. Separate supplemental run; no control reruns or semantic reruns.",
        summary={"task": "none"},
        token_scope="all actual three-follow-up node calls and recorded retries; no summary; excludes scripted history generation and real retrieval/business services",
        latency_scope="single supplemental process; not comparable as a production latency benchmark",
        source_lineage=lineage())
    unified.write(Path(output)/"manifest.json", manifest)
    return manifest


def audit(output):
    output = Path(output).resolve()
    result = ORIGINAL_AUDIT(output)
    result["labels"] = {"task": LABEL}
    result["limits"] = ["Same frozen synthetic cases and scoring as the five-arm study, but a separate single supplemental run.",
        "Preserves the earlier whole selector, node packet inputs and caps; not a matched-input BM25-only ablation.",
        "All calls and failures retained; missing usage is an unknown cost lower bound.",
        "Three follow-ups are correlated; raw historical dialogue generation and real service costs excluded."]
    unified.write(output/"audited_report.json", result)
    return result


def finalize(output):
    """Offline audits and six-way comparison; never touches prior output files."""
    from scripts.audit_context_unified import audit as independent
    from scripts import analyze_context_keep3 as analyzer
    output = Path(output).resolve()
    generic = audit(output)
    detail = independent(output)
    manifest = unified.read(output/"manifest.json")
    current_match = all((ROOT/name).is_file() and sha(ROOT/name)==value
                        for name,value in manifest["sources"].items())
    detail["current_sources_match"] = current_match
    detail["passed"] = detail["complete"] and detail["audit_ok"] and current_match
    unified.write(output/"AUDIT_DETAIL.json", detail)
    with patch.dict(analyzer.LABELS, {"task": LABEL}), patch.object(analyzer,"_write_report",lambda *a: None):
        analyzer.analyze(output)
    report = comparison(output)
    return {"complete": generic["complete"], "independent_passed": detail["passed"],
            "issues": detail["issues"], "groups": report["groups"],
            "report": str(output/"COMPARISON.md")}


def comparison(output):
    """Read-only cross-run pairing, with exact case/gold and parent hashes."""
    output = Path(output)
    data, manifest = unified.read(output/"dataset.json"), unified.read(output/"manifest.json")
    parent_data = unified.read(PARENT/"dataset.json")
    ancestry = manifest.get("source_lineage") or manifest["config"]["lineage"]
    if sha(PARENT/"dataset.json")!=ancestry["parent_dataset_file_sha256"] or sha(PARENT/"ANALYSIS.json")!=ancestry["parent_analysis_sha256"]:
        raise ValueError("frozen control reference changed")
    parent_cases = {c["id"]:c for c in parent_data["cases"]}
    same = (set(parent_cases)=={c["id"] for c in data["cases"]}
            and all(parent_cases.get(c["id"])==c for c in data["cases"]))
    # Pilot uses an independent case and is never compared to formal controls.
    formal = not manifest["pilot"]
    if formal and not same:
        raise ValueError("supplemental cases or gold differ from controls")
    child = unified.read(output/"ANALYSIS.json")
    parent = unified.read(PARENT/"ANALYSIS.json") if formal else {"groups":{}}
    groups = {**deepcopy(parent["groups"]), **deepcopy(child["groups"])}
    pairs = {}
    if formal:
        left = {(r["coords"]["case_id"],r["coords"]["turn"]):r for p in (output/"turns").glob("task-*.json") for r in [unified.read(p)]}
        for arm in parent["groups"]:
            right = {(r["coords"]["case_id"],r["coords"]["turn"]):r for p in (PARENT/"turns").glob(f"{arm}-*.json") for r in [unified.read(p)]}
            counts={k:0 for k in ("both_pass","both_fail","task_only","reference_only")}
            differing=[]
            for key in sorted(set(left)|set(right)):
                a=bool(left.get(key,{}).get("score",{}).get("task_rule_pass"))
                b=bool(right.get(key,{}).get("score",{}).get("task_rule_pass"))
                tag="both_pass" if a and b else "both_fail" if not a and not b else "task_only" if a else "reference_only"
                counts[tag]+=1
                if a!=b:differing.append({"case_id":key[0],"turn":key[1],"outcome":tag})
            lc,rc=groups["task"]["overall"]["cost"],groups[arm]["overall"]["cost"]
            pairs[arm]={"turn_counts":counts,"differing":differing,
                "total_token_change_percent":100*(lc["total_tokens"]/rc["total_tokens"]-1) if lc["cost_complete"] and rc["cost_complete"] and rc["total_tokens"] else None}
    result={"version":"full-retrieval-supplement-comparison-v1","formal":formal,"same_cases_and_gold":same if formal else None,
        "supplement_independent_audit_passed":child["audit"]["formal_audit_passed"],
        "controls_audit_passed":parent.get("audit",{}).get("formal_audit_passed") if formal else None,
        "controls_reused_not_rerun":formal,"groups":groups,"paired_task_vs":pairs,
        "limits":["Separate supplemental single run, model responses and trajectories can vary.",
                  "The previous selector includes different node context assembly and caps; do not attribute system differences to BM25 alone."]}
    unified.write(output/"COMPARISON.json",result)
    fmt=lambda m:f"{m['pass']}/{m['denominator']}（{m['rate']:.1%}）" if m.get('rate') is not None else f"{m['pass']}/{m['denominator']}"
    cost=lambda c,k:("≥" if not c['cost_complete'] else "")+f"{c[k]:,}"
    lines=[f"# 此前完整检索方案：{len(data['cases'])}段会话补充实验","",
        f"补充组独立审计通过：{result['supplement_independent_audit_passed']}。预检不计入正式结果。","",
        "正式补充组复用上一轮24段会话及评分金标，单独调用模型；五个参照组读取冻结结果，没有重跑或改分。"
        if formal else "这是独立预检案例，无正式对照比较。","",
        "完整检索组不生成摘要，保留旧实现的全历史BM25、最近2轮、最强命中保护、前后邻轮和实体单跳扩展，以及原分节点输入和预算。它是完整系统参照，不能解释成仅BM25算法的消融。","",
        "|方法|任务通过|整段三问全过|全部输入|全部输出|全部总Token|","|---|---:|---:|---:|---:|---:|"]
    for g in groups.values():
        o=g['overall'];c=o['cost'];lines.append(f"|{g['label']}|{fmt(o['task'])}|{fmt(o['episodes'])}|{cost(c,'input_tokens')}|{cost(c,'output_tokens')}|{cost(c,'total_tokens')}|")
    lines += ["","|方法|更正任务|限制目标查询|歧义澄清|","|---|---:|---:|---:|"]
    for g in groups.values():
        m=g['metrics'];lines.append(f"|{g['label']}|{fmt(m['correction_task'])}|{fmt(m['restriction_target_query'])}|{fmt(m['clarification_task'])}|")
    lines += ["","成本包含本工作负载所有记录调用和重试，包括控制组摘要维护；不含预置历史生成和真实检索/业务服务费用。≥代表未知usage导致的下界，未知费用不能当0。","",
        "## 短长会话与已见/新增案例","","|方法|分层|任务通过|整段通过|总Token|","|---|---|---:|---:|---:|"]
    for arm,g in groups.items():
        for field in ("stratum","cohort"):
            for name,v in g[field].items():lines.append(f"|{g['label']}|{name}|{fmt(v['task'])}|{fmt(v['episodes'])}|{cost(v['cost'],'total_tokens')}|")
    lines += ["","## 调用与故障","","|方法|模型调用|接口错误|协议无效|未知usage|明确重试|","|---|---:|---:|---:|---:|---:|"]
    for g in groups.values():
        lines.append(f"|{g['label']}|{g['overall']['cost']['model_calls']}|{g['call_statuses'].get('interface_error',0)}|{g['protocol_invalid_calls']}|{g['overall']['cost']['unknown_usage_calls']}|{g['overall']['retries']['retry_attempts']}|")
    lines += ["","规则通过不等于完整语义正确或线上准确率；整段指标只检查三次实际追问。16条已见回归与8条新内部案例分别报告。后续回答和摘要独立生成，存在错误传播，单次比较不证明统计显著性。","",
        "ANALYSIS.json保留补充组全部失败、COMPARISON.json保留与各参照的逐题配对；AUDIT_DETAIL.json验证冻结评分、来源前缀、选择器回放、实际请求模型与思考设置。"]
    (output/"COMPARISON.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--pilot",action="store_true")
    p.add_argument("--cases",nargs="+")
    mode=p.add_mutually_exclusive_group()
    mode.add_argument("--run",action="store_true")
    mode.add_argument("--audit",action="store_true")
    a=p.parse_args()
    if a.run:
        with binding():r=asyncio.run(unified.run(a.output))
        print(json.dumps({"complete":r["complete"],"audit_passed":r["audit_passed"],"groups":r['groups']},ensure_ascii=False))
    elif a.audit:
        r=finalize(a.output)
        print(json.dumps({k:v for k,v in r.items() if k!='groups'},ensure_ascii=False))
    else:
        r=prepare(a.output,pilot=a.pilot,case_ids=a.cases)
        print(json.dumps({"output":str(a.output),"dataset_sha256":r['dataset_sha256'],"sources":len(r['sources'])},ensure_ascii=False))


if __name__=="__main__":main()
