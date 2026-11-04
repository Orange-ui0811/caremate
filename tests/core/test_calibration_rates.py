import json

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from app.api import observability as obs


@pytest.mark.parametrize("threshold,expected", [
    (0, (1, 1)), (1, (0, 0)), (.4, (1, .5)), (.400005, (1, 0)), (.40002, (.5, 0)),
])
def test_current_rates_use_unrounded_scores_and_inclusive_gate(threshold, expected):
    result = obs._rates_at_threshold({"scores": {
        "answerable": [.40001, .9], "absent": [.1, .4]}}, threshold)
    assert (result["pass_rate"], result["leak_rate"]) == expected
    assert result["threshold"] == threshold


def test_legacy_scan_requires_exact_threshold():
    report = {"scan": [{"t": .26, "pass_rate": .8, "leak_rate": .1}]}
    assert obs._rates_at_threshold(report, .26)["pass_rate"] == .8
    assert obs._rates_at_threshold(report, .2601) is None
    assert obs._rates_at_threshold(report, 0) is None
    assert obs._rates_at_threshold({}, .26) is None


def test_empty_or_invalid_scores_do_not_invent_rates():
    for scores in [{"answerable": [], "absent": []},
                   {"answerable": [float("nan")], "absent": [.2]}]:
        assert obs._rates_at_threshold({"scores": scores}, .26) is None


async def test_overview_uses_newly_saved_threshold_not_recommendation(tmp_path, monkeypatch):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"scores": {"answerable": [.4, .9], "absent": [.1, .4]},
                               "recommended": {"threshold": .5}}), encoding="utf-8")
    monkeypatch.setattr(obs, "CALIB", path)
    async def trend():
        return {}
    monkeypatch.setattr(obs, "_trend_block", trend)
    monkeypatch.setattr(obs, "_cost_block", lambda: {})
    app = FastAPI()
    app.include_router(obs.router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for t, rates in [(.4, (1, .5)), (.4001, (.5, 0))]:
            assert (await client.put("/api/observability/confidence-threshold", json={"threshold": t})).status_code == 200
            data = (await client.get("/api/observability/overview")).json()["calibration"]
            assert data["in_use"] == data["in_use_rates"]["threshold"] == t
            assert (data["in_use_rates"]["pass_rate"], data["in_use_rates"]["leak_rate"]) == rates
