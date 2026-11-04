"""Portable compact replay must preserve the original helper's isolation."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import replay_context_compact_snapshot as replay
from scripts import replay_context_snapshot as original


@pytest.fixture
def synthetic_run(tmp_path):
    out, project = tmp_path / "original", tmp_path / "project"
    files = {
        "app/__init__.py": "",
        "app/core/__init__.py": "",
        "scripts/__init__.py": "",
        "app/config.py": "from types import SimpleNamespace\nsettings=SimpleNamespace()\n",
        "app/core/llm.py": "def get_chat_model(*a,**kw): raise AssertionError('model called')\ndef structured(*a,**kw): raise AssertionError('model called')\n",
        "scripts/eval_context_compact.py": "from app.config import settings\n\ndef audit(out):\n    assert settings.chat_model == 'synthetic-frozen-model'\n    return {'complete':True,'audit_passed':True,'counts':{'results':0},'issues':[],'missing':[]}\n",
        replay.DESIGN_SOURCE: "# Frozen design evidence\n",
    }
    hashes = {}
    for name, text in files.items():
        target = out / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        hashes[name] = original.sha(target)
    original.write(out / "manifest.json", {"version": replay.EXPERIMENT_VERSION,
        "sources": hashes, "config": {"chat_model": "synthetic-frozen-model"}, "packages": {}})
    original.write(out / "dataset.json", {"cases": []})
    original.write(out / "preflight_evidence.json", {})
    original.write(out / "audited_report.json", {"complete": True, "audit_passed": True,
        "counts": {"results": 0}, "issues": [], "missing": []})
    project.mkdir()
    (project / ".env").write_text("CHAT_API_KEY=synthetic-secret-sentinel", encoding="utf-8")
    return out, project


def bundle(synthetic_run, tmp_path):
    out, project = synthetic_run
    support, dest = tmp_path / "support", tmp_path / "bundle"
    replay.capture_support(out, support, project_root=project)
    replay.create_bundle(out, support, dest)
    return dest


def test_adapter_allowlist_does_not_modify_old_helper(synthetic_run):
    out, _ = synthetic_run
    assert replay.verified_sources(out)["version"] == replay.EXPERIMENT_VERSION
    with pytest.raises(ValueError, match="unsupported"):
        original.verified_sources(out)
    assert replay.source_name(replay.DESIGN_SOURCE) == replay.DESIGN_SOURCE
    with pytest.raises(ValueError):
        original.source_name(replay.DESIGN_SOURCE)
    for name in ("docs/secret.md", "docs/experiments/../../.env", "app/config.json", ".env", "../secret.py"):
        with pytest.raises(ValueError):
            replay.source_name(name)
    assert replay._base is not original


def test_bundle_keeps_evidence_and_validates_both_helpers(synthetic_run, tmp_path):
    out, _ = synthetic_run
    before = original.inventory(out)
    dest = bundle(synthetic_run, tmp_path)
    info = replay.verify_bundle(dest)
    assert original.inventory(out) == before
    assert info["support"]["captured_at"]
    assert "not pre-run" in info["support"]["limitation"]
    assert not (dest / "code/.env").exists()
    support = dest / "replay_support.py"
    support.write_bytes(support.read_bytes() + b"\n# changed\n")
    with pytest.raises(ValueError, match="support helper differs"):
        replay.verify_bundle(dest)


def test_portable_worker_dispatches_compact_audit_without_project_imports(synthetic_run, tmp_path):
    dest = bundle(synthetic_run, tmp_path)
    assert replay.replay(dest) == 0
    report = original.read(dest / "replay_report.json")
    assert report["complete"] and report["strict_audit_passed"]
    assert report["all_metrics_match_original"] and report["copied_inputs_unchanged"]
    assert report["model_calls"] == 0 and report["network_denied"]
    assert "scripts.eval_context_compact" in report["local_imports"]


def test_child_environment_drops_credentials_and_uses_isolated_python(synthetic_run, tmp_path, monkeypatch):
    dest = bundle(synthetic_run, tmp_path)
    monkeypatch.setenv("CHAT_API_KEY", "synthetic-secret-sentinel")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "synthetic-secret-sentinel")
    captured = {}
    def launch(command, **kwargs):
        captured.update(kwargs)
        assert "-I" in command
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(replay._base.subprocess, "run", launch)
    assert replay.replay(dest) == 0
    assert captured["env"]["CHAT_API_KEY"] == "offline-no-credential"
    assert "AWS_SECRET_ACCESS_KEY" not in captured["env"]
    assert "synthetic-secret-sentinel" not in json.dumps(captured["env"])


def test_compact_worker_boundary_denies_network_and_child_processes():
    path = Path(replay.__file__).resolve()
    code = f"""
import importlib.util, socket, subprocess, sys
spec=importlib.util.spec_from_file_location('compact_adapter', {str(path)!r})
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
deny=m._base.deny_network()
for operation in (socket.socket, lambda:socket.getaddrinfo('invalid.invalid',80), lambda:subprocess.run([sys.executable,'-c','pass']),deny):
    try: operation()
    except RuntimeError: continue
    raise AssertionError('offline boundary bypassed')
print('all-denied')
"""
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "all-denied"
