"""Combine predeclared identical-protocol repetitions without best-run selection.

Each conversation stays one bootstrap unit. Repetitions are averaged within it;
neither repeated model outputs nor correlated follow-ups inflate the sample size.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts import eval_context_state as evaluation

MATCH_FIELDS = ('version', 'arms', 'mode', 'split', 'families', 'dataset_sha256',
                'context_budget', 'input_limit_estimate', 'max_output', 'mode_body',
                'summary_trigger_tokens', 'summary_target_tokens', 'summary_keep_turns',
                'scoring_version', 'config', 'sources', 'case_concurrency', 'caller_concurrency')


def combine(manifests, reports):
    if len(manifests) < 2 or len(manifests) != len(reports):
        raise ValueError('at least two corresponding complete repetitions required')
    first = manifests[0]
    labels = [m.get('repeat_id') for m in manifests]
    if all(labels) and len(set(labels)) != len(labels):
        raise ValueError('repeat_id values must be unique')
    namespaces = [m.get('request_id_namespace') for m in manifests]
    if all(namespaces) and len(set(namespaces)) != len(namespaces):
        raise ValueError('model-pool request namespaces must be unique across repetitions')
    if any(any(m.get(k) != first.get(k) for k in MATCH_FIELDS) for m in manifests[1:]):
        raise ValueError('repetitions must share data, code, model and evaluation protocol')
    if any(not r.get('complete') or not r.get('audit_passed') for r in reports):
        raise ValueError('incomplete or unaudited repetitions cannot be silently excluded')
    arms, combined = first['arms'], {}
    for arm in arms:
        per_conversation = {}
        runs = [r['strategies'][arm] for r in reports]
        expected_ids = set(runs[0]['per_conversation'])
        if any(set(r['per_conversation']) != expected_ids for r in runs):
            raise ValueError('conversation set mismatch')
        for cid in sorted(expected_ids):
            steps = [r['per_conversation'][cid] for r in runs]
            if any(len(s) != len(steps[0]) for s in steps):
                raise ValueError('repetition has missing steps')
            per_conversation[cid] = [passed for s in steps for passed in s]
        combined[arm] = {
            'n_conversations': len(expected_ids), 'repetitions': len(runs),
            'per_run': [{'n': r['n'], 'passed': r['task_passed'], 'target_passed': r['target_passed'],
                         'transport_failed': r['transport_failed'], 'protocol_failed': r['protocol_failed'],
                         'budget_blocked': r['budget_blocked'], 'usage': r['usage']} for r in runs],
            'mean_task_rate': sum(r['task_passed'] / r['n'] for r in runs) / len(runs),
            'mean_usage_per_run': {k: sum(r['usage'][k] for r in runs) / len(runs) for k in ('input_tokens', 'output_tokens', 'total_tokens')},
            'per_conversation': per_conversation}
    pairs = {}
    for name in reports[0].get('paired_comparisons', {}):
        a, b = name.split('->')
        pairs[name] = evaluation.paired_difference(combined[a]['per_conversation'], combined[b]['per_conversation'])
    return {'version': 'context-state-repetitions-v1', 'repetitions': len(reports),
            'dataset_sha256': first['dataset_sha256'], 'strategies': combined, 'paired_comparisons': pairs,
            'scope': 'All supplied identical-protocol runs included; model repetitions averaged within conversation. CI unit remains conversation, not individual turns or repetitions.',
            'limitations': 'Internally constructed synthetic data; rule scores, not independently human-reviewed accuracy. Same-template dependence and provider caching remain limits.'}


def aggregate(paths, output):
    paths = [Path(p).resolve() for p in paths]
    if len(paths) != len(set(paths)):
        raise ValueError('duplicate run directory')
    if len({p.name for p in paths}) != len(paths):
        raise ValueError('output directory names must differ to isolate model-pool request IDs')
    manifests, reports = [], []
    for path in paths:
        with evaluation.writer_lock(path):
            reports.append(evaluation.audit(path))
        manifests.append(evaluation.read(path / 'manifest.json'))
    report = combine(manifests, reports)
    report['runs'] = [{'path': str(path), 'manifest_sha256': hashlib.sha256((path / 'manifest.json').read_bytes()).hexdigest(),
                       'report_sha256': hashlib.sha256((path / 'audited_report.json').read_bytes()).hexdigest()} for path in paths]
    evaluation.save_immutable(Path(output), report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--runs', nargs='+', required=True); p.add_argument('--output', required=True)
    args = p.parse_args()
    result = aggregate(args.runs, args.output)
    print(json.dumps({'repetitions': result['repetitions'], 'paired_comparisons': result['paired_comparisons']}, ensure_ascii=False))
