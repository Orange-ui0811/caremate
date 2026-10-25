import importlib.util
import json

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api import observability
from app.core import runtime_settings


@pytest.fixture
async def client():
    app = FastAPI()
    app.include_router(observability.router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


async def test_saved_threshold_is_used_without_restart_and_survives_fresh_import(client, monkeypatch):
    monkeypatch.setattr(runtime_settings.settings, "evidence_confidence_threshold", .26)
    assert runtime_settings.get_evidence_confidence_threshold() == .26
    response = await client.put("/api/observability/confidence-threshold", json={"threshold": .17})
    assert response.status_code == 200
    assert response.json()["threshold"] == .17 and response.json()["updated_at"]
    assert runtime_settings.get_evidence_confidence_threshold() == .17
    assert observability._calibration_block()["in_use"] == .17
    spec = importlib.util.spec_from_file_location("fresh_runtime_settings", runtime_settings.__file__)
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    fresh.SETTINGS_PATH = runtime_settings.SETTINGS_PATH
    assert fresh.get_evidence_confidence_threshold() == .17


@pytest.mark.parametrize("value", [-.01, 1.01, None, "NaN", "Infinity", True, "0.17"])
async def test_invalid_threshold_does_not_overwrite_saved_value(client, value):
    runtime_settings.save_confidence_threshold(.26)
    response = await client.put("/api/observability/confidence-threshold", json={"threshold": value})
    assert response.status_code == 422
    assert runtime_settings.get_evidence_confidence_threshold() == .26


@pytest.mark.parametrize("value", [0, 1, .1705])
async def test_threshold_accepts_boundaries_and_custom_precision(client, value):
    response = await client.put("/api/observability/confidence-threshold", json={"threshold": value})
    assert response.status_code == 200 and response.json()["threshold"] == value


async def test_failed_atomic_save_keeps_previous_setting(client, monkeypatch):
    runtime_settings.save_confidence_threshold(.26)
    def fail(*args):
        raise PermissionError("disk not writable")
    monkeypatch.setattr(runtime_settings.os, "replace", fail)
    response = await client.put("/api/observability/confidence-threshold", json={"threshold": .17})
    assert response.status_code == 503
    assert runtime_settings.get_evidence_confidence_threshold() == .26
    assert not list(runtime_settings.SETTINGS_PATH.parent.glob("runtime-settings-*.tmp"))


def test_corrupt_saved_setting_falls_back_to_default(monkeypatch):
    monkeypatch.setattr(runtime_settings.settings, "evidence_confidence_threshold", .26)
    runtime_settings.SETTINGS_PATH.write_text(json.dumps({"threshold": 2}), encoding="utf-8")
    assert runtime_settings.get_evidence_confidence_threshold() == .26
