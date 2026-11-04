"""Offline bundle integrity and narrow diagnostic comparison regressions."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import replay_context_snapshot as replay


@pytest.mark.parametrize("name", ["../secret.py", "E:/secret.py", "app/../../secret.py",
                                 "app\\config.py", ".env", "/app/config.py", "app/config.json"])
def test_source_names_reject_escape_and_non_source_files(name):
    with pytest.raises(ValueError):
        replay.source_name(name)


def test_diagnostic_normalization_cannot_hide_prompt_or_score_changes():
    expected = {"body": ["earlier", "later"], "score": {"task_pass": False,
        "nonuse_mentions": [{"value": "810201"}, {"value": "810202"}, {"value": "810201"}]}}
    equivalent = json.loads(json.dumps(expected))
    equivalent["score"]["nonuse_mentions"].reverse()
    assert replay.diagnostic_order_equal(expected, equivalent)
    equivalent["body"].reverse()
    assert not replay.diagnostic_order_equal(expected, equivalent)
    equivalent = json.loads(json.dumps(expected))
    equivalent["score"]["task_pass"] = True
    assert not replay.diagnostic_order_equal(expected, equivalent)
    equivalent = json.loads(json.dumps(expected))
    equivalent["score"]["nonuse_mentions"].pop()
    assert not replay.diagnostic_order_equal(expected, equivalent)


@pytest.fixture
def synthetic_run(tmp_path):
    out, project = tmp_path / "original", tmp_path / "project"
    frozen = out / "source_snapshot/app/main.py"
    frozen.parent.mkdir(parents=True)
    frozen.write_text("from app.core import helper\n", encoding="utf-8")
    helper = project / "app/core/helper.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("from . import other\n", encoding="utf-8")
    (helper.parent / "other.py").write_text("VALUE = 1\n", encoding="utf-8")
    (project / ".env").write_text("DO_NOT_COPY=synthetic-sensitive-marker", encoding="utf-8")
    manifest = {"version": "context-state-eval-v1", "sources": {"app/main.py": replay.sha(frozen)},
                "config": {}, "packages": {}}
    replay.write(out / "manifest.json", manifest)
    for name in ("dataset.json", "preflight_evidence.json"):
        replay.write(out / name, {})
    replay.write(out / "audited_report.json", {"complete": True, "audit_passed": True})
    return out, project


def test_support_closure_is_dated_and_does_not_copy_environment(synthetic_run, tmp_path):
    out, project = synthetic_run
    support = tmp_path / "support"
    manifest = replay.capture_support(out, support, project_root=project)
    assert set(manifest["sources"]) == {"app/core/helper.py", "app/core/other.py"}
    assert manifest["captured_at"] and "not pre-run" in manifest["limitation"]
    assert not (support / ".env").exists()
    assert manifest["source_manifest_sha256"] == replay.sha(out / "manifest.json")


def test_bundle_is_independent_and_detects_source_or_evidence_changes(synthetic_run, tmp_path):
    out, project = synthetic_run
    support, bundle = tmp_path / "support", tmp_path / "bundle"
    replay.capture_support(out, support, project_root=project)
    before = replay.inventory(out)
    replay.create_bundle(out, support, bundle)
    assert replay.inventory(out) == before
    (project / "app/core/helper.py").write_text("VALUE = 'later edit'\n", encoding="utf-8")
    replay.verify_bundle(bundle)  # Current project edits are irrelevant.
    source = bundle / "code/app/core/helper.py"
    old = source.read_bytes()
    source.write_text("VALUE = 'tampered'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bundled source differs"):
        replay.verify_bundle(bundle)
    source.write_bytes(old)
    replay.write(bundle / "evidence/dataset.json", {"tampered": True})
    with pytest.raises(ValueError, match="copied evidence differs"):
        replay.verify_bundle(bundle)


def test_active_or_incomplete_run_is_not_copied(synthetic_run, tmp_path):
    out, project = synthetic_run
    support = tmp_path / "support"
    replay.capture_support(out, support, project_root=project)
    replay.write(out / "audited_report.json", {"complete": False})
    with pytest.raises(ValueError, match="completed evidence"):
        replay.create_bundle(out, support, tmp_path / "bundle")
    with pytest.raises(ValueError, match="outside original"):
        replay.create_bundle(out, support, out / "bundle")


def test_replay_child_receives_no_inherited_credentials(synthetic_run, tmp_path, monkeypatch):
    out, project = synthetic_run
    support, bundle = tmp_path / "support", tmp_path / "bundle"
    replay.capture_support(out, support, project_root=project)
    replay.create_bundle(out, support, bundle)
    monkeypatch.setenv("CHAT_API_KEY", "synthetic-secret-sentinel")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "synthetic-secret-sentinel")
    captured = {}

    def launch(command, **kwargs):
        captured.update(kwargs)
        assert "-I" in command
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(replay.subprocess, "run", launch)
    assert replay.replay(bundle) == 0
    assert "AWS_SECRET_ACCESS_KEY" not in captured["env"]
    assert captured["env"]["CHAT_API_KEY"] == "offline-no-credential"
    assert "synthetic-secret-sentinel" not in json.dumps(captured["env"])


def test_network_and_child_processes_are_denied_in_fresh_interpreter():
    path = Path(replay.__file__).resolve()
    script = f"""
import importlib.util, socket, subprocess, sys
spec = importlib.util.spec_from_file_location('offline_helper', {str(path)!r})
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
deny = module.deny_network()
for operation in (socket.socket, lambda: socket.getaddrinfo('invalid.invalid', 80),
                  lambda: subprocess.run([sys.executable, '-c', 'pass']), deny):
    try:
        operation()
    except RuntimeError:
        continue
    raise AssertionError('offline boundary was bypassed')
print('all-denied')
"""
    result = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "all-denied"
