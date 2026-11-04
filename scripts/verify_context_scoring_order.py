"""Offline proof that deterministic diagnostic order does not change scoring.

Compiles the exact pure score functions without importing app configuration or
model/database clients. The only permitted scorer source change is ids ->
sorted(ids). Original experiment files, including all call receipts, are hashed
before/after and never written. The report must be outside the experiment.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from scripts.replay_context_snapshot import (completed_evidence, deny_network,
    diagnostic_order_equal, read, sha, value_differences, verified_sources, write)

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def all_file_hashes(out):
    # The separately held read lock makes its own coordination byte inaccessible
    # through another Windows handle. It is not experiment evidence.
    return {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*'))
            if p.is_file() and p.name != 'writer.lock'}


def load_pure_score(scoring_text, legacy_text):
    scope = {'re': re}
    legacy = ast.parse(legacy_text)
    functions = [n for n in legacy.body if isinstance(n, ast.FunctionDef) and n.name == 'score_probe']
    if len(functions) != 1:
        raise ValueError('frozen legacy score function unavailable')
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<frozen-legacy-score>', 'exec'), scope)
    scope['legacy_score'] = scope['score_probe']
    tree = ast.parse(scoring_text)
    nodes = [n for n in tree.body if isinstance(n, (ast.Assign, ast.FunctionDef))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pure-context-score>', 'exec'), scope)
    return scope['score'], scope['VERSION']


def boolean_values(value, path=''):
    if isinstance(value, bool):
        return {path: value}
    if isinstance(value, dict):
        return {p: v for key, item in value.items() for p, v in boolean_values(item, path + '/' + key).items()}
    if isinstance(value, list):
        return {p: v for i, item in enumerate(value) for p, v in boolean_values(item, path + '/' + str(i)).items()}
    return {}


def verify(out, report_path):
    if out == report_path or out in report_path.parents:
        raise ValueError('verification report must be outside original experiment')
    manifest = verified_sources(out)
    name = 'scripts/context_state_scoring.py'
    old_text = (out / 'source_snapshot' / name).read_text(encoding='utf-8')
    new_text = (ROOT / name).read_text(encoding='utf-8')
    if old_text.count('for value in ids:') != 1 or new_text != old_text.replace('for value in ids:', 'for value in sorted(ids):'):
        raise ValueError('scorer change is not exactly the declared diagnostic sorting fix')
    legacy = (out / 'source_snapshot/scripts/eval_context_comparison.py').read_text(encoding='utf-8')
    old_score, old_version = load_pure_score(old_text, legacy)
    new_score, new_version = load_pure_score(new_text, legacy)
    if old_version != new_version or new_version != manifest['scoring_version']:
        raise ValueError('score version changed')
    deny_network()
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'score_version': new_version,
              'source_change': 'Only identifier_mentions: for value in ids -> for value in sorted(ids)',
              'frozen_scoring_sha256': manifest['sources'][name], 'fixed_scoring_sha256': sha(ROOT / name),
              'source_manifest_sha256': sha(out / 'manifest.json'), 'real_model_calls': 0,
              'method': 'Exact pure scorer/legacy function AST compiled locally; no app configuration, model or database modules imported.',
              'allowed_difference': ['score.nonuse_mentions list order', 'score.unexpected_asserted_orders list order']}
    with completed_evidence(out):
        before = all_file_hashes(out)
        cases = {c['id']: c for c in read(out / 'dataset.json')['cases']}
        expected = {f"{case['id']}-{step['id']}-{arm}" for case in cases.values()
                    for step in case['steps'] for arm in manifest['arms']}
        files = list((out / 'results').glob('*.json'))
        issues, changes = [], {}
        counts = defaultdict(lambda: {'n': 0, 'task_passed': 0, 'target_passed': 0})
        booleans_checked = 0
        if {p.stem for p in files} != expected:
            issues.append('result inventory differs from declared case/step/arm combinations')
        for path in sorted(files):
            result = read(path)
            prior_hash = digest(result)
            case = cases[result['case_id']]
            step = next(s for s in case['steps'] if s['id'] == result['probe_id'])
            old, new, stored = old_score(case, step, result), new_score(case, step, result), result['score']
            for label, value in (('frozen-score', old), ('fixed-score', new)):
                if not diagnostic_order_equal({'score': stored}, {'score': value}):
                    issues.append(label + ' changed non-diagnostic content: ' + path.stem)
            # Diagnostic lists may reorder nonuse flags along with their entries;
            # compare all remaining booleans by exact paths, not just task_pass.
            stable = lambda score: {k: v for k, v in score.items() if k not in ('nonuse_mentions', 'unexpected_asserted_orders')}
            expected_bool = boolean_values(stable(stored))
            booleans_checked += len(expected_bool)
            if expected_bool != boolean_values(stable(new)):
                issues.append('boolean score changed: ' + path.stem)
            if stored != new:
                changes[path.stem] = value_differences({'score': stored}, {'score': new})
            if prior_hash != digest(result):
                issues.append('score function mutated original result in memory: ' + path.stem)
            summary = counts[result['arm']]
            summary['n'] += 1
            summary['task_passed'] += int(new['task_pass'])
            summary['target_passed'] += int(new['target_pass'])
        original_report = read(out / 'audited_report.json')
        for arm, values in counts.items():
            if any(values[k] != original_report['strategies'][arm][k] for k in values):
                issues.append('arm task/target aggregate changed: ' + arm)
        tokens = dict.fromkeys(('input_tokens', 'output_tokens', 'total_tokens'), 0)
        call_inputs, usage_receipts = {}, {}
        for path in sorted((out / 'calls').glob('*.json')):
            call = read(path)
            if call['input_sha256'] != digest(call['contract']):
                issues.append('stored call input hash invalid: ' + path.stem)
            call_inputs[path.stem] = call['input_sha256']
            usage_receipts[path.stem] = [a.get('usage') for a in call.get('attempts', [])]
            for usage in usage_receipts[path.stem]:
                for key in tokens:
                    tokens[key] += (usage or {}).get(key, 0) or 0
        expected_tokens = {key: sum(g['usage'][key] for g in original_report['strategies'].values()) for key in tokens}
        if tokens != expected_tokens:
            issues.append('all-call tokens differ from original audited report')
        unchanged = all_file_hashes(out) == before
        if not unchanged:
            issues.append('original experiment file bytes changed')
        report.update(passed=not issues, issues=issues, results_checked=len(files),
                      expected_results=len(expected), boolean_values_checked=booleans_checked,
                      call_inputs_checked=len(call_inputs), all_call_input_hashes=digest(call_inputs),
                      all_usage_receipts_hash=digest(usage_receipts), tokens_including_maintenance=tokens,
                      tokens_match_original_audit=tokens == expected_tokens,
                      original_files_checked=len(before), original_files_unchanged=unchanged,
                      inventory_excludes=['writer.lock coordination byte (read-lock held during verification)'],
                      original_file_inventory_sha256=digest(before), arm_counts=dict(counts),
                      diagnostic_order_changes=changes, diagnostic_order_change_count=len(changes))
    write(report_path, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--report', required=True, type=Path)
    args = p.parse_args()
    result = verify(args.output.resolve(), args.report.resolve())
    print(json.dumps({k: result[k] for k in ('passed', 'results_checked', 'boolean_values_checked',
        'call_inputs_checked', 'diagnostic_order_change_count', 'original_files_unchanged', 'issues')}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
