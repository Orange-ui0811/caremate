"""Replay a completed context-state experiment from an isolated source bundle.

Only this helper is new: frozen app/runner files and original evidence stay intact.
Capture missing import support while it is still available, then create a bundle:
  python -m scripts.replay_context_snapshot --output RUN --capture-support SUPPORT
  python -m scripts.replay_context_snapshot --output RUN --support SUPPORT --destination BUNDLE
Later, after the working project changes, use the self-contained copy:
  python BUNDLE/replay.py --bundle BUNDLE

All model/network entry points are denied in the audit subprocess. No .env,
credentials, database or model-pool cache is copied or inherited. Supplementary
support source is explicitly dated, not represented as pre-run frozen evidence.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VERSION = "context-offline-replay-v1"
FOLDERS = ("calls", "attempts", "results", "maintenance", "trajectories")
PUBLIC_CONFIG = {"chat_model", "chat_base_url", "chat_thinking", "chat_reasoning_effort",
                 "summary_model", "chat_failover_enabled", "summary_trigger_messages",
                 "context_window_turns", "context_window_max_tokens"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_name(name):
    p = PurePosixPath(name)
    if ("\\" in name or p.is_absolute() or ".." in p.parts or ":" in name
            or p.suffix != ".py" or p.parts[0] not in {"app", "scripts", "tests"}):
        raise ValueError("unsafe source path")
    return p.as_posix()


def verified_sources(out):
    manifest = read(out / "manifest.json")
    if manifest.get("version") not in {"context-state-eval-v1", "context-state-eval-v2"}:
        raise ValueError("unsupported frozen experiment")
    for name, expected in manifest["sources"].items():
        name = source_name(name)
        path = out / "source_snapshot" / name
        if path.is_symlink() or sha(path) != expected:
            raise ValueError("frozen source differs: " + name)
    if set(manifest["config"]) - PUBLIC_CONFIG:
        raise ValueError("manifest contains unexpected configuration fields")
    return manifest


def import_support(out, manifest, project_root):
    """Conservative AST import closure; only local Python modules, never config data."""
    frozen = set(manifest["sources"])
    pending = [n for n in frozen if n.startswith(("app/", "scripts/"))]
    seen, supplemental = set(), {}

    def existing(name):
        name = source_name(name)
        path = out / "source_snapshot" / name if name in frozen else project_root / name
        return path if path.is_file() and not path.is_symlink() else None

    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        path = existing(name)
        if path is None:
            raise ValueError("support source unavailable: " + name)
        if name not in frozen:
            supplemental[name] = path
        parts = PurePosixPath(name).parts[:-1]
        for i in range(1, len(parts) + 1):
            init = "/".join(parts[:i]) + "/__init__.py"
            if existing(init) is not None:
                pending.append(init)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    prefix = list(parts[:len(parts) - node.level + 1])
                    module = ".".join(prefix + ([module] if module else []))
                modules = [module] + [module + "." + a.name for a in node.names]
            for module in modules:
                if not module.startswith(("app.", "scripts.")):
                    continue
                stem = module.replace(".", "/")
                for candidate in (stem + ".py", stem + "/__init__.py"):
                    if existing(candidate) is not None:
                        pending.append(candidate)
                        break
    return supplemental


def capture_support(out, dest, *, project_root=ROOT):
    if out == dest or out in dest.parents:
        raise ValueError("support must be outside original evidence")
    manifest = verified_sources(out)
    sources = import_support(out, manifest, project_root)
    dest.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, path in sorted(sources.items()):
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        hashes[name] = sha(target)
    result = {"version": VERSION, "captured_at": datetime.now(timezone.utc).isoformat(),
              "source_manifest_sha256": sha(out / "manifest.json"), "sources": hashes,
              "limitation": "Supplementary import support captured after experiment preparation; not pre-run frozen evidence or Git history. Original manifest source hashes remain unchanged.",
              "packages_at_capture": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}
    write(dest / "support_manifest.json", result)
    return result


@contextmanager
def completed_evidence(out):
    """Refuse an active writer; read-lock without changing the original lock file."""
    report = read(out / "audited_report.json")
    if not report.get("complete"):
        raise ValueError("only completed evidence may be copied")
    lock = out / "writer.lock"
    handle = lock.open("rb") if lock.exists() else None
    locked = False
    try:
        if handle:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBRLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            locked = True
        yield
    finally:
        if handle:
            if locked:
                if os.name == "nt":
                    import msvcrt
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


def inventory(out):
    paths = [out / n for n in ("manifest.json", "dataset.json", "preflight_evidence.json")]
    paths += [p for folder in FOLDERS for p in (out / folder).glob("*.json")]
    return {p.relative_to(out).as_posix(): sha(p) for p in paths}


def create_bundle(out, support, dest):
    if out == dest or out in dest.parents:
        raise ValueError("replay must be outside original evidence")
    manifest = verified_sources(out)
    supplemental = read(support / "support_manifest.json")
    if supplemental["source_manifest_sha256"] != sha(out / "manifest.json"):
        raise ValueError("support snapshot belongs to another manifest")
    overlap = set(manifest["sources"]) & set(supplemental["sources"])
    if overlap:
        raise ValueError("support must not override frozen source")
    for name, expected in supplemental["sources"].items():
        if sha(support / source_name(name)) != expected:
            raise ValueError("support source differs: " + name)
    with completed_evidence(out):
        before = inventory(out)
        dest.mkdir(parents=True, exist_ok=False)
        for name in before:
            target = dest / "evidence" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((out / name).read_bytes())
        shutil.copyfile(out / "audited_report.json", dest / "reference_audited_report.json")
        for name in manifest["sources"]:
            data = (out / "source_snapshot" / name).read_bytes()
            for root in (dest / "code", dest / "evidence/source_snapshot"):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        for name in supplemental["sources"]:
            target = dest / "code" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((support / name).read_bytes())
        if inventory(out) != before or inventory(dest / "evidence") != before:
            raise ValueError("evidence changed while copying")
    shutil.copyfile(Path(__file__), dest / "replay.py")
    (dest / "temporary").mkdir()
    write(dest / "bundle_manifest.json", {"version": VERSION,
        "source_manifest_sha256": sha(out / "manifest.json"), "artifacts": before,
        "frozen_sources": manifest["sources"], "support": supplemental,
        "helper_sha256": sha(dest / "replay.py")})


def verify_bundle(bundle):
    info = read(bundle / "bundle_manifest.json")
    for name, expected in info["artifacts"].items():
        if sha(bundle / "evidence" / name) != expected:
            raise ValueError("copied evidence differs: " + name)
    if inventory(bundle / "evidence") != info["artifacts"]:
        raise ValueError("copied evidence file set differs")
    for name, expected in {**info["frozen_sources"], **info["support"]["sources"]}.items():
        if sha(bundle / "code" / source_name(name)) != expected:
            raise ValueError("bundled source differs: " + name)
    verified_sources(bundle / "evidence")
    if sha(bundle / "replay.py") != info["helper_sha256"]:
        raise ValueError("bundled helper changed")
    return info


def deny_network():
    import socket

    def denied(*_args, **_kwargs):
        raise RuntimeError("offline replay forbids network/model calls")

    def audit(event, args):
        if event in {"socket.__new__", "socket.connect", "socket.sendto", "socket.getaddrinfo",
                     "socket.gethostbyname", "socket.gethostbyaddr", "subprocess.Popen", "os.system"}:
            raise RuntimeError("offline replay forbids network/subprocess access: " + event)

    sys.addaudithook(audit)
    for name in ("create_connection", "getaddrinfo", "gethostbyname", "gethostbyaddr"):
        setattr(socket, name, denied)
    return denied


def value_differences(expected, actual, path=""):
    """Explain mismatches without relaxing the frozen audit or replacing results."""
    if type(expected) is not type(actual):
        return [{"path": path, "expected": expected, "actual": actual}]
    if isinstance(expected, dict):
        return [row for key in sorted(set(expected) | set(actual)) for row in value_differences(
            expected.get(key), actual.get(key), path + "/" + str(key))]
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return [{"path": path + "/length", "expected": len(expected), "actual": len(actual)}]
        return [row for i, (a, b) in enumerate(zip(expected, actual))
                for row in value_differences(a, b, path + "/" + str(i))]
    return [] if expected == actual else [{"path": path, "expected": expected, "actual": actual}]


def diagnostic_order_equal(expected, actual):
    """Two score diagnostics are multisets; never normalize inputs or pass flags.

    identifier_mentions iterates a Python set. It makes these diagnostic arrays
    order-dependent across interpreters, although all their consumers use only
    presence/filtering. Sorting preserves duplicate entries and every field.
    The original strict audit remains untouched and separately reported.
    """
    def normalized(value):
        value = json.loads(json.dumps(value, ensure_ascii=False))
        for key in ("nonuse_mentions", "unexpected_asserted_orders"):
            entries = value.get("score", {}).get(key)
            if isinstance(entries, list):
                value["score"][key] = sorted(entries, key=lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True))
        return value
    return normalized(expected) == normalized(actual)


def replay_differences(runner, out, issues):
    wanted = {item.removeprefix("replay: ") for item in issues if item.startswith("replay: ")}
    differences = {}
    manifest = read(out / "manifest.json")
    for case in read(out / "dataset.json")["cases"]:
        for arm in manifest["arms"]:
            if not any(k.startswith(case["id"] + "-") and k.endswith("-" + arm) for k in wanted):
                continue
            generator, value = runner.workflow(case, arm, manifest), None
            try:
                while True:
                    item = generator.send(value)
                    value = None
                    if item[0] == "call":
                        value = read(out / "calls" / (item[1] + ".json"))
                    elif item[2] in wanted:
                        expected = read(out / item[1] / (item[2] + ".json"))
                        differences[item[2]] = {"differences": value_differences(expected, item[3]),
                            "diagnostic_list_order_only": item[1] == "results" and diagnostic_order_equal(expected, item[3])}
            except StopIteration:
                pass
    return differences


def worker(bundle):
    """Run only in a fresh interpreter with an allowlisted environment and -I."""
    info = verify_bundle(bundle)
    report = {"version": VERSION, "network_denied": True, "model_calls": 0,
              "support_limit": info["support"]["limitation"], "complete": False, "audit_passed": False}
    try:
        # CPython 3.13 on Windows may run the local `ver` command when a library
        # first asks for platform.machine(). Cache this stdlib-only OS metadata
        # before denying subprocess creation; no project code is imported yet.
        import platform
        platform.uname()
        platform.processor()
        denied = deny_network()
        sys.dont_write_bytecode = True
        code = (bundle / "code").resolve()
        os.chdir(code)
        sys.path.insert(0, str(code))
        manifest = read(bundle / "evidence/manifest.json")
        mismatches = {p: {"expected": v, "installed": importlib.metadata.version(p)}
                      for p, v in manifest["packages"].items() if importlib.metadata.version(p) != v}
        report["package_mismatches"] = mismatches
        if mismatches:
            raise ValueError("frozen package versions differ")
        from app.config import settings
        for key, value in manifest["config"].items():
            setattr(settings, key, value)
        report["public_config_restored"] = sorted(manifest["config"])
        from app.core import llm
        llm.get_chat_model = denied
        llm.structured = denied
        module_name = "scripts.eval_context_state_v2" if manifest["version"].endswith("v2") else "scripts.eval_context_state"
        runner = importlib.import_module(module_name)

        async def denied_call(*args, **kwargs):
            return denied()

        for module in list(sys.modules.values()):
            if getattr(module, "__name__", "").startswith("scripts.") and hasattr(module, "Caller"):
                module.Caller.call = denied_call
        result = runner.audit(bundle / "evidence")
        report.update(complete=result["complete"], audit_passed=result["audit_passed"],
                      counts=result["counts"], issues=result["issues"], missing=result["missing"])
        differences = replay_differences(runner, bundle / "evidence", result["issues"]) if result["issues"] else {}
        write(bundle / "replay_differences.json", differences)
        report["strict_audit_passed"] = result["audit_passed"]
        report["diagnostic_order_replay_passed"] = (result["complete"] and not result["missing"]
            and len(differences) == len(result["issues"])
            and all(d["diagnostic_list_order_only"] for d in differences.values()))
        report["diagnostic_order_cases"] = sorted(differences)
        report["diagnostic_order_scope"] = "Only score.nonuse_mentions and score.unexpected_asserted_orders are compared as duplicate-preserving multisets. Every other field, prompt/context sequence, contract, raw output, score flag and usage remains exact. Original strict audit is not modified."
        reference = read(bundle / "reference_audited_report.json")
        report["matches_original_audited_report"] = result == reference
        report["differing_report_keys"] = sorted(k for k in set(result) | set(reference) if result.get(k) != reference.get(k))
        report["all_metrics_match_original"] = all(result.get(k) == reference.get(k)
            for k in set(result) | set(reference) if k not in {"audit_passed", "issues"})
        combined = {**info["frozen_sources"], **info["support"]["sources"]}
        imports = {}
        for name, module in list(sys.modules.items()):
            if not name.startswith(("app.", "scripts.")) or not getattr(module, "__file__", None):
                continue
            filename = Path(module.__file__).resolve()
            rel = filename.relative_to(code).as_posix()
            if rel not in combined or sha(filename) != combined[rel]:
                raise ValueError("unbundled local module loaded: " + name)
            imports[name] = rel
        report["local_imports"] = imports
        verify_bundle(bundle)
        report["copied_inputs_unchanged"] = True
    except Exception as exc:
        import traceback
        report["audit_passed"] = False
        report["diagnostic_order_replay_passed"] = False
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:700]}
        (bundle / "replay-error.log").write_text(traceback.format_exc(), encoding="utf-8")
    write(bundle / "replay_report.json", report)
    print(json.dumps({k: report.get(k) for k in ("complete", "strict_audit_passed", "diagnostic_order_replay_passed", "all_metrics_match_original", "error")}, ensure_ascii=False))
    return 0 if report.get("diagnostic_order_replay_passed") and report.get("all_metrics_match_original") else 1


def replay(bundle):
    verify_bundle(bundle)
    env = {k: os.environ[k] for k in ("SystemRoot", "WINDIR", "COMSPEC") if k in os.environ}
    env.update(CHAT_MODEL="offline-model", CHAT_BASE_URL="http://invalid.invalid/v1",
               CHAT_API_KEY="offline-no-credential", EMBED_API_KEY="offline-no-credential",
               RERANK_API_KEY="offline-no-credential", DATABASE_URL="sqlite+aiosqlite:///:memory:",
               TEST_DATABASE_URL="sqlite+aiosqlite:///:memory:", MODEL_POOL_ENABLED="false",
               LANGFUSE_PUBLIC_KEY="", LANGFUSE_SECRET_KEY="", LANGCHAIN_TRACING_V2="false",
               PYTHONIOENCODING="utf-8", PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1",
               TEMP=str(bundle / "temporary"), TMP=str(bundle / "temporary"))
    result = subprocess.run([sys.executable, "-I", str(bundle / "replay.py"), "--_worker", str(bundle)],
                            cwd=bundle / "code", env=env, timeout=180, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    (bundle / "replay-process.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.stdout:
        print(result.stdout.strip())
    if result.returncode and not (bundle / "replay_report.json").exists():
        print("Offline subprocess failed; see replay-process.log")
    return result.returncode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    p.add_argument("--capture-support", type=Path)
    p.add_argument("--support", type=Path)
    p.add_argument("--destination", type=Path)
    p.add_argument("--bundle", type=Path)
    p.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args._worker:
        return worker(args._worker.resolve())
    if args.bundle:
        return replay(args.bundle.resolve())
    if not args.output:
        p.error("--output or --bundle is required")
    if args.capture_support:
        result = capture_support(args.output.resolve(), args.capture_support.resolve())
        print(json.dumps({"support_files": len(result["sources"]), "captured_at": result["captured_at"]}))
        return 0
    if not args.support or not args.destination:
        p.error("--support and --destination are required for a new replay bundle")
    create_bundle(args.output.resolve(), args.support.resolve(), args.destination.resolve())
    return replay(args.destination.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
