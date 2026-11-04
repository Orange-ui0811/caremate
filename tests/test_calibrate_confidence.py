"""Calibration uses background admission without self-induced queue bursts."""
import asyncio
import json
from types import SimpleNamespace

import httpx
from openai import APIConnectionError
import pytest

from app.core import model_pool
from scripts import calibrate_confidence as cal


def busy(wrapped):
    if not wrapped:
        raise model_pool.PoolBusy("busy")
    try:
        raise model_pool.PoolBusy("busy")
    except model_pool.PoolBusy as exc:
        raise APIConnectionError(request=httpx.Request("POST", "http://test/embeddings")) from exc


@pytest.mark.parametrize("wrapped", [False, True])
async def test_busy_retry_recognizes_sdk_wrapping(monkeypatch, wrapped):
    attempts, delays = [], []
    async def search(*args, **kwargs):
        attempts.append(model_pool.priority())
        if len(attempts) < 3:
            busy(wrapped)
        return []
    async def sleep(seconds):
        delays.append(seconds)
    monkeypatch.setattr(cal.retrieval, "search_knowledge", search)
    monkeypatch.setattr(cal.asyncio, "sleep", sleep)
    assert (await cal._conf({"query": "test", "bucket": "D_absent"}))[0] == "D_absent"
    assert attempts == ["background"] * 3
    assert delays == [1, 2]


async def test_busy_retries_are_bounded(monkeypatch):
    attempts = []
    async def search(*args, **kwargs):
        attempts.append(1)
        busy(True)
    async def sleep(seconds):
        pass
    monkeypatch.setattr(cal.retrieval, "search_knowledge", search)
    monkeypatch.setattr(cal.asyncio, "sleep", sleep)
    with pytest.raises(RuntimeError, match="旧报告已保留"):
        await cal._conf({"query": "test", "bucket": "A_policy"})
    assert len(attempts) == 5


@pytest.fixture
def report_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(cal, "_OUT_DIR", tmp_path)
    monkeypatch.setattr(cal, "_OUT", tmp_path / "report.txt")
    monkeypatch.setattr(cal, "_OUT_JSON", tmp_path / "report.json")
    monkeypatch.setattr(cal, "_DATASET", tmp_path / "samples.jsonl")
    monkeypatch.setattr(cal.settings, "model_pool_enabled", False)
    samples = [{"query": str(score), "bucket": bucket} for score, bucket in [
        (.40001, "A_policy"), (.9, "B_model"), (.1, "D_absent"), (.4, "D_absent"),
        (.7, "E_cross_doc")]]
    cal._DATASET.write_text("\n".join(json.dumps(s) for s in samples), encoding="utf-8")
    return tmp_path


async def test_main_serial_and_publishes_full_precision_scores(report_paths, monkeypatch):
    active, peak, seen = 0, 0, []
    async def search(query, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(.001)
        seen.append((query, model_pool.priority()))
        active -= 1
        return [float(query)]
    async def note(*args, **kwargs):
        return None
    monkeypatch.setattr(cal.retrieval, "search_knowledge", search)
    monkeypatch.setattr(cal, "compute_evidence_confidence", lambda h: SimpleNamespace(score=h[0]))
    monkeypatch.setattr(cal.read_notes, "generate", note)
    await cal.main()
    data = json.loads(cal._OUT_JSON.read_text(encoding="utf-8"))
    assert peak == 1 and len(seen) == 4
    assert all(priority == "background" for _, priority in seen)
    assert data["scores"] == {"answerable": [.40001, .9], "absent": [.1, .4]}
    assert data["meta"]["n_answerable"] == data["meta"]["n_absent"] == 2
    assert data["meta"]["scoring_method"] == "max_rerank_v1"
    assert data["meta"]["source"] == "live_retrieval"
    assert data["recommended"]["selection_rule"] == "youden_j"
    assert data["scan"][35] == {"t": .4, "pass_rate": 1, "leak_rate": .5, "youden_j": .5}
    assert not list(report_paths.glob("*.tmp"))


async def test_non_pool_failure_is_not_retried_or_published(report_paths, monkeypatch):
    cal._OUT_JSON.write_text('{"previous":true}', encoding="utf-8")
    cal._OUT.write_text("previous", encoding="utf-8")
    attempts = []
    async def search(query, **kwargs):
        attempts.append(query)
        if len(attempts) == 1:
            return []
        raise APIConnectionError(request=httpx.Request("POST", "http://test/embeddings"))
    monkeypatch.setattr(cal.retrieval, "search_knowledge", search)
    with pytest.raises(APIConnectionError):
        await cal.main()
    assert len(attempts) == 2
    assert json.loads(cal._OUT_JSON.read_text()) == {"previous": True}
    assert cal._OUT.read_text() == "previous"


async def test_paused_background_pool_fails_before_requests(report_paths, monkeypatch):
    monkeypatch.setattr(cal.settings, "model_pool_enabled", True)
    monkeypatch.setattr(model_pool, "get_store", lambda: SimpleNamespace(
        policy=lambda: SimpleNamespace(background_limit=0)))
    with pytest.raises(RuntimeError, match="后台模型调用已暂停"):
        await cal.main()
