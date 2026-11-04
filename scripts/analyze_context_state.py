"""Read-only progress, failure triage and paired evidence for frozen state evals.

No model calls, cache writes, rescoring or execution of the experiment runner.
Outputs go only to a separate analysis directory. Automated labels are triage
hints, not independent human semantic judgments or a replacement score.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from scripts.context_state_analysis_stats import build_stats, markdown, reconcile_attempt_usage


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same_raw_subset(left, right):
    """Do not interpret two absent component-ablation fields as equal windows."""
    a, b = left['context_manifest'], right['context_manifest']
    for key in ('common_raw_ids', 'selected_message_ids'):
        if key in a and key in b:
            return a[key] == b[key]
    return None


def write_separate(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def triage(case, step, result):
    score = result['score']
    category = 'task_rule_failure'
    if score['transport_failure']:
        category = 'provider_or_transport_failure'
    elif score['protocol_failure']:
        category = 'output_protocol_failure'
    elif score['budget_blocked']:
        category = 'context_budget_blocked'
    elif not score['target_pass']:
        category = 'selected_object_or_action_mismatch'
    elif score['fabricated_identifiers']:
        category = 'identifier_rule_flag_requires_review'
    elif not score['answer_rule_pass']:
        category = 'answer_rule_flag_requires_review'
    context = result.get('context_text', '')
    source_visibility = [{'text': text, 'literal_visible': text in context} for text in step.get('gold_source_texts', [])]
    return {'category_hint': category, 'query': step['query'], 'expected': step['expected'],
            'coref': result.get('coref', {}).get('value'), 'plan': result.get('plan', {}).get('value'),
            'reply': result.get('reply', {}).get('value'), 'tool_result': result.get('tool_result'),
            'stage_status': {s: result.get(s, {}).get('status') for s in ('coref', 'plan', 'reply')},
            'source_visibility': source_visibility, 'score': score,
            'candidate': result.get('snapshot', {}).get('candidate'),
            'active_constraints': result.get('snapshot', {}).get('active_constraints', []),
            'context_manifest': result.get('context_manifest'),
            'interpretation_limits': ['Literal source absence is not semantic absence; summaries may paraphrase.',
                                      'Literal presence does not prove correct relationship interpretation.',
                                      'The category is a deterministic triage hint, not a human semantic grade.']}


def analyze(output):
    output = Path(output).resolve(); manifest = read(output / 'manifest.json'); data = read(output / 'dataset.json')
    root = Path(__file__).resolve().parents[1]
    mismatches = []
    for name, digest in manifest['sources'].items():
        for kind, path in (('current', root / name), ('snapshot', output / 'source_snapshot' / name)):
            if not path.exists() or sha(path) != digest:
                mismatches.append({'kind': kind, 'source': name})
    cases = {c['id']: c for c in data['cases']}
    steps = {(c['id'], s['id']): s for c in data['cases'] for s in c['steps']}
    expected = {f'{c["id"]}-{s["id"]}-{a}' for c in data['cases'] for s in c['steps'] for a in manifest['arms']}
    expected_maintenance = {f'{c["id"]}-{a}' for c in data['cases'] for a in manifest['arms']}
    results = {p.stem: read(p) for p in (output / 'results').glob('*.json')}
    calls_by_key = {p.stem: read(p) for p in (output / 'calls').glob('*.json')}
    calls = list(calls_by_key.values())
    protocol_details = []
    for key, call in sorted(calls_by_key.items()):
        if call['status'] not in ('parse_error', 'empty', 'truncated'):
            continue
        attempt = call['attempts'][-1] if call.get('attempts') else {}
        protocol_details.append({'key': key, 'stage': call['contract']['stage'],
                                 'status': call['status'], 'tool_choice': call['contract'].get('tool_choice'),
                                 'attempt_count': len(call.get('attempts', [])),
                                 'transport_ok': attempt.get('transport_ok'),
                                 'finish_reason': attempt.get('finish_reason'),
                                 'tool_calls': attempt.get('tool_calls', []),
                                 'raw_content': attempt.get('raw_content', ''),
                                 'source_sha256': sha(output / 'calls' / (key + '.json'))})
    existing_maintenance = {p.stem for p in (output / 'maintenance').glob('*.json')}
    maintenance = {p.stem: read(p) for p in (output / 'maintenance').glob('*.json')}
    existing_trajectories = {p.stem for p in (output / 'trajectories').glob('*.json')}
    audit_path = output / 'audited_report.json'
    audit = read(audit_path) if audit_path.exists() else None
    finished = (set(results) == expected and existing_maintenance == expected_maintenance and
                existing_trajectories == expected_maintenance and bool(audit and audit['complete'] and audit['audit_passed']))
    failures, categories = {}, Counter()
    for key, result in sorted(results.items()):
        if result['score']['task_pass']:
            continue
        item = triage(cases[result['case_id']], steps[result['case_id'], result['probe_id']], result)
        item['result_sha256'] = sha(output / 'results' / (key + '.json'))
        failures[key] = item; categories[item['category_hint']] += 1
    pairs = {}
    for a, b in (('W', 'P'), ('P', 'Q'), ('Q', 'S'), ('A', 'B'), ('A', 'R'), ('B', 'R'), ('R', 'M'), ('C', 'R')):
        if a not in manifest['arms'] or b not in manifest['arms']:
            continue
        rows = []
        for (cid, sid), step in steps.items():
            left, right = results.get(f'{cid}-{sid}-{a}'), results.get(f'{cid}-{sid}-{b}')
            if not left or not right:
                continue
            if left['score']['task_pass'] == right['score']['task_pass'] and left['score']['target_pass'] == right['score']['target_pass']:
                continue
            rows.append({'case_id': cid, 'probe_id': sid, 'query': step['query'],
                         'left_task': left['score']['task_pass'], 'right_task': right['score']['task_pass'],
                         'left_target': left['score']['target_pass'], 'right_target': right['score']['target_pass'],
                         'left_status': {s: left[s]['status'] for s in ('coref', 'plan', 'reply')},
                         'right_status': {s: right[s]['status'] for s in ('coref', 'plan', 'reply')},
                         'same_raw_subset': same_raw_subset(left, right),
                         'same_prior_history': left['history_sha256'] == right['history_sha256'],
                         'left_plan': left['plan']['value'], 'right_plan': right['plan']['value']})
        pairs[a + '->' + b] = rows
    stats = build_stats(manifest, cases, results, calls_by_key, maintenance)
    attempts = {p.stem: read(p) for p in (output / 'attempts').glob('*.json')}
    stats['attempt_usage_reconciliation'] = reconcile_attempt_usage(calls_by_key, attempts, stats['costs']['per_arm'])
    return {'generated_at': datetime.now(timezone.utc).isoformat(), 'finished_and_audited': finished,
            'scope': 'Read-only deterministic triage; no model calls, rescoring, or independent human semantic review.',
            'source_count': len(manifest['sources']), 'source_mismatches': mismatches,
            'counts': {'results': len(results), 'expected_results': len(expected), 'calls': len(calls),
                       'maintenance': len(existing_maintenance), 'expected_maintenance': len(expected_maintenance),
                       'trajectories': len(existing_trajectories)},
            'missing_results': sorted(expected - set(results)), 'unexpected_results': sorted(set(results) - expected),
            'call_status': dict(Counter(c['status'] for c in calls)),
            'protocol_failure_details': protocol_details,
            'stage_status': {s: dict(Counter(c['status'] for c in calls if c['contract']['stage'] == s)) for s in ('coref', 'plan', 'reply', 'summary')},
            'failure_categories': dict(categories), 'failures': failures, 'paired_differences': pairs,
            'extended_statistics': stats,
            'audit_report_sha256': sha(audit_path) if audit_path.exists() else None,
            'manifest_sha256': sha(output / 'manifest.json'), 'analysis_source_sha256': sha(__file__),
            'analysis_helper_sha256': sha(Path(__file__).with_name('context_state_analysis_stats.py')),
            'interpretation_limits': ['Different carried replies can cause later history divergence in trajectory mode; common-window attribution applies only when visible histories match.',
                                      'New state and recall selectors are deterministic but their usefulness still depends on extraction correctness.',
                                      'This file does not decide whether a rule flag is a true semantic failure.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    parser.add_argument('--name', default='PROGRESS_TRIAGE.json')
    args = parser.parse_args()
    if Path(args.name).name != args.name or not args.name.endswith('.json'):
        raise ValueError('analysis filename must be a plain .json filename')
    report = analyze(args.output)
    destination = Path(args.output) / 'analysis' / args.name
    write_separate(destination, report)
    destination.with_suffix('.md').write_text(markdown(report['extended_statistics']), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('finished_and_audited', 'source_mismatches', 'counts', 'call_status', 'failure_categories')}, ensure_ascii=False))
