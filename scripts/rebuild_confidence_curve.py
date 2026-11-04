"""从置信度实验保存的真实API候选重建当前评分曲线，不调用模型、不改变运行阈值。

运行：python -m scripts.rebuild_confidence_curve --study data/confidence-study/20261027
"""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from app.core.confidence import SCORING_METHOD, compute_evidence_confidence
from scripts.calibrate_confidence import ANSWERABLE, _write_atomic

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rates(positive, negative, threshold):
    passed = sum(s >= threshold for s in positive) / len(positive)
    leaked = sum(s >= threshold for s in negative) / len(negative)
    return {"pass_rate": passed, "leak_rate": leaked, "youden_j": passed - leaked}


def distribution(scores):
    xs = sorted(scores)
    return {"n": len(xs), "min": xs[0], "max": xs[-1],
            **{name: xs[min(len(xs) - 1, int(q * len(xs)))]
               for name, q in (("p25", .25), ("p50", .5), ("p75", .75))}}


def build_report(study):
    samples = read(study / "samples.json")
    protocol = read(study / "protocol.json")
    if len(samples) != protocol["sample_count"] or len({s["id"] for s in samples}) != len(samples):
        raise ValueError("题集数量或ID不完整，保留现有报告。")
    positive, negative, score_ids = [], [], {"answerable": [], "absent": []}
    for sample in samples:
        if sample["bucket"] not in ANSWERABLE | {"D_absent"}:
            continue
        case = read(study / "cases" / (sample["id"] + ".json"))
        if (case["status"] != "ok" or case["id"] != sample["id"]
                or case["bucket"] != sample["bucket"] or case["original_query"] != sample["query"]):
            raise ValueError(f"逐题记录与题集不一致：{sample['id']}")
        score = compute_evidence_confidence(case["hits"]).score
        if score != case["features"]["variants"]["top1_max"]:
            raise ValueError(f"当前评分与实验最高分不一致：{sample['id']}")
        key = "answerable" if sample["bucket"] in ANSWERABLE else "absent"
        (positive if key == "answerable" else negative).append(score)
        score_ids[key].append(sample["id"])
    if not positive or not negative:
        raise ValueError("缺少可答或应拒分数，保留现有报告。")
    fit = read(study / "analysis.json")["results"]["0.05"]["top1_max"]["fit_all_abcd"]
    threshold = fit["threshold"]
    measured = rates(positive, negative, threshold)
    if (measured["leak_rate"] > .05 + 1e-12
            or sum(s < threshold for s in positive) != fit["calibration"]["false_reject_n"]
            or sum(s >= threshold for s in negative) != fit["calibration"]["false_accept_n"]):
        raise ValueError("实验阈值与重算数据不一致，保留现有报告。")
    scan = [{"t": t, **rates(positive, negative, t)}
            for t in sorted({i / 100 for i in range(5, 96)} | {threshold})]
    return {
        "meta": {"generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M"),
                 "strategy": "hybrid_rerank", "scoring_method": SCORING_METHOD,
                 "source": "recorded_api_replay", "source_study": str(study.resolve()),
                 "source_label": "2026-10-27真实接口检索记录回放；ABC 180题、D 60题；E桶不参与曲线和阈值选择。",
                 "dataset": "tests/data/eval_ch04.jsonl",
                 "dataset_sha256": protocol["dataset_sha256"],
                 "samples_sha256": hashlib.sha256((study / "samples.json").read_bytes()).hexdigest(),
                 "knowledge_rows": read(study / "knowledge-before.json")["rows"],
                 "n_answerable": len(positive), "n_absent": len(negative),
                 "limitation": "内部校准集原始标签含争议；曲线不是留出测试结果或最终答案质量。"},
        "distribution": {"answerable": distribution(positive), "absent": distribution(negative)},
        "scores": {"answerable": positive, "absent": negative}, "score_ids": score_ids,
        "scan": scan,
        "recommended": {"threshold": threshold, **measured,
                        "selection_rule": "false_accept_cap", "max_false_accept_rate": .05,
                        "selection_rule_label": "校准集应拒放行≤5%，优先减少可答误拒"},
        "read_notes": {"confidence_calibration":
            f"按最高重排分重算，阈值为{threshold:.4f}，校准集可答通过率为{measured['pass_rate']:.2%}，"
            f"应拒放行率为{measured['leak_rate']:.2%}。这是内部校准集回放，使用原始标签，"
            "不代表独立测试或最终答案质量；运行阈值保持原值。"},
    }


def publish(study, output):
    report = build_report(study)
    output.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, ensure_ascii=False, indent=1) + "\n"
    _write_atomic(output / "confidence_calibration.txt", "最高重排分阈值曲线（真实API记录回放）\n" + text)
    _write_atomic(output / "confidence_calibration.json", text)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data/ch09/reports")
    args = parser.parse_args()
    report = publish(args.study, args.output)
    print(json.dumps({"scoring_method": SCORING_METHOD, "scan_points": len(report["scan"]),
                      "recommended": report["recommended"], "meta": report["meta"]}, ensure_ascii=False))
