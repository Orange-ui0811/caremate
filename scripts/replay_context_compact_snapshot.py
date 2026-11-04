"""Strict offline replay adapter for the frozen compact-context experiment.

Reuse the existing replay helper without changing it or any experiment source.
The bundle carries this adapter and an exactly hashed copy of that helper. Only
the compact runner and its one frozen design document extend the old allowlist.
No model, network or subprocess work is permitted inside the replay worker.

  python -m scripts.replay_context_compact_snapshot --output RUN --capture-support SUPPORT
  python -m scripts.replay_context_compact_snapshot --output RUN --support SUPPORT --destination BUNDLE
  python BUNDLE/replay.py --bundle BUNDLE
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
ADAPTER_VERSION = "context-compact-offline-replay-v1"
EXPERIMENT_VERSION = "context-compact-eval-v1"
DESIGN_SOURCE = "docs/experiments/context-compact-validation-20261027.md"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# In a portable bundle, verify the copied base helper before executing it.
# A repository invocation loads the unchanged, reviewed sibling source.
if (HERE / "replay_support.py").exists():
    _support_path = HERE / "replay_support.py"
    _info = json.loads((HERE / "bundle_manifest.json").read_text(encoding="utf-8"))
    if _support_path.is_symlink() or _sha(_support_path) != _info.get("support_helper_sha256"):
        raise ValueError("bundled replay support helper changed")
else:
    _support_path = HERE / "replay_context_snapshot.py"
_spec = importlib.util.spec_from_file_location("_compact_replay_support", _support_path)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
_original_source_name = _base.source_name
_original_create_bundle = _base.create_bundle
_original_verify_bundle = _base.verify_bundle


def source_name(name):
    # This document is frozen evidence, never executable import support.
    if name == DESIGN_SOURCE:
        return name
    return _original_source_name(name)


def verified_sources(out):
    manifest = _base.read(out / "manifest.json")
    if manifest.get("version") != EXPERIMENT_VERSION:
        raise ValueError("unsupported frozen compact experiment")
    for name, expected in manifest["sources"].items():
        path = out / "source_snapshot" / source_name(name)
        if path.is_symlink() or _base.sha(path) != expected:
            raise ValueError("frozen source differs: " + name)
    if set(manifest["config"]) - _base.PUBLIC_CONFIG:
        raise ValueError("manifest contains unexpected configuration fields")
    return manifest


def _import_auditor(name):
    # The base worker selects its old V1 fallback for a version ending in v1.
    # Explicitly route that sole entry point to the new frozen runner; all its
    # private V2/V1 imports continue to use their original implementations.
    if name != "scripts.eval_context_state":
        raise ValueError("unexpected replay auditor entry point: " + name)
    return importlib.import_module("scripts.eval_context_compact")


def create_bundle(out, support, dest):
    _original_create_bundle(out, support, dest)
    shutil.copyfile(_support_path, dest / "replay_support.py")
    shutil.copyfile(Path(__file__), dest / "replay.py")
    info = _base.read(dest / "bundle_manifest.json")
    info.update(version=ADAPTER_VERSION, helper_sha256=_base.sha(dest / "replay.py"),
                support_helper_sha256=_base.sha(dest / "replay_support.py"),
                adapter_scope="Compact runner dispatch plus one frozen Markdown design document; original worker isolation, source/input hashes and strict replay are unchanged.")
    _base.write(dest / "bundle_manifest.json", info)


def verify_bundle(bundle):
    info = _base.read(bundle / "bundle_manifest.json")
    support = bundle / "replay_support.py"
    if info.get("version") != ADAPTER_VERSION or support.is_symlink() or _base.sha(support) != info.get("support_helper_sha256"):
        raise ValueError("bundled replay support helper differs")
    return _original_verify_bundle(bundle)


_base.VERSION = ADAPTER_VERSION
_base.source_name = source_name
_base.verified_sources = verified_sources
_base.create_bundle = create_bundle
_base.verify_bundle = verify_bundle
# Avoid monkeypatching Python's shared importlib module. The private helper only
# uses metadata and this explicit audit entry point through this facade.
_base.importlib = SimpleNamespace(metadata=importlib.metadata, import_module=_import_auditor)

capture_support = _base.capture_support
replay = _base.replay
worker = _base.worker


if __name__ == "__main__":
    raise SystemExit(_base.main())
