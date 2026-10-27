import json

import pytest

from app.core import runtime_settings
from scripts import rebuild_confidence_curve as curve


def save(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def study(tmp_path):
    (tmp_path / "cases").mkdir()
    samples = [{"id": sid, "bucket": bucket, "query": sid} for sid, bucket in
               [("A1", "A_policy"), ("B1", "B_model"), ("D1", "D_absent"), ("E1", "E_multi")]]
    save(tmp_path / "samples.json", samples)
    save(tmp_path / "protocol.json", {"sample_count": 4, "dataset_sha256": "fixed"})
    save(tmp_path / "knowledge-before.json", {"rows": 54})
    for sample, score in zip(samples, [.8, .2, .1, .99]):
        save(tmp_path / "cases" / (sample["id"] + ".json"),
             {**sample, "status": "ok", "original_query": sample["query"],
              "hits": [{"rerank_score": score}], "features": {"variants": {"top1_max": score}}})
    save(tmp_path / "analysis.json", {"results": {"0.05": {"top1_max": {"fit_all_abcd": {
        "threshold": .2, "calibration": {"false_reject_n": 0, "false_accept_n": 0}}}}}})
    return tmp_path


def test_curve_uses_current_scores_and_excludes_E(study):
    report = curve.build_report(study)
    assert report["scores"] == {"answerable": [.8, .2], "absent": [.1]}
    assert report["score_ids"] == {"answerable": ["A1", "B1"], "absent": ["D1"]}
    assert report["meta"]["scoring_method"] == "max_rerank_v1"
    at_boundary = next(row for row in report["scan"] if row["t"] == .2)
    assert at_boundary["pass_rate"] == 1 and at_boundary["leak_rate"] == 0
    assert report["recommended"]["selection_rule"] == "false_accept_cap"


def test_every_curve_point_is_counted_from_saved_scores(study):
    report = curve.build_report(study)
    for row in report["scan"]:
        assert row["pass_rate"] == sum(s >= row["t"] for s in [.8, .2]) / 2
        assert row["leak_rate"] == float(.1 >= row["t"])


def test_publish_does_not_change_online_threshold(study, tmp_path):
    runtime_settings.save_confidence_threshold(.2260993940477108)
    before = runtime_settings.SETTINGS_PATH.read_bytes()
    curve.publish(study, tmp_path / "output")
    assert runtime_settings.SETTINGS_PATH.read_bytes() == before


@pytest.mark.parametrize("failure", ["missing_case", "wrong_score", "wrong_query"])
def test_incomplete_or_mismatched_records_cannot_overwrite_report(study, tmp_path, failure):
    case_path = study / "cases/A1.json"
    if failure == "missing_case":
        case_path.unlink()
    else:
        case = curve.read(case_path)
        if failure == "wrong_score":
            case["hits"][0]["rerank_score"] = .01
        else:
            case["original_query"] = "different question"
        save(case_path, case)
    output = tmp_path / "output"
    output.mkdir()
    previous = output / "confidence_calibration.json"
    previous.write_text("previous report", encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        curve.publish(study, output)
    assert previous.read_text(encoding="utf-8") == "previous report"
