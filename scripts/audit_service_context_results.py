"""Independent post-run accounting and failure material, without rescoring rules.

No model calls, frozen source edits, cache edits or manual pass overrides.
"""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sqlite3


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def audit(out):
    manifest = read(out/'manifest.json')
    report = read(out/'audited_report.json')
    if not report['complete'] or not report['audit_passed']:
        raise ValueError('Complete passing causal audit required')
    cases = read(out/'dataset.json')['cases']
    arms = manifest['arms']
    calls = {p.stem: read(p) for p in (out/'calls').glob('*.json')}
    results = {p.stem: read(p) for p in (out/'results').glob('*.json')}
    attempts = [a for r in calls.values() for a in r['attempts']]
    expected_results = {f'{c["id"]}-{p["id"]}-{arm}' for c in cases for p in c['probes'] for arm in arms}
    checks = {'result_keys': set(results) == expected_results,
              'maintenance_count': len(list((out/'maintenance').glob('*.json'))) == len(cases)*len(arms),
              'receipt_count': len(list((out/'attempts').glob('*.json'))) == len(attempts)}
    failures = []
    for arm in arms:
        for cohort in ('all', 'regression', 'pressure'):
            selected = [c for c in cases if cohort == 'all' or c['cohort'] == cohort]
            rr = [results[f'{c["id"]}-{p["id"]}-{arm}'] for c in selected for p in c['probes']]
            prefixes = tuple(c['id']+'-' for c in selected)
            cc = [r for k, r in calls.items() if k.startswith(prefixes) and '-'+arm+'-' in k]
            aa = [a for r in cc for a in r['attempts']]
            stats = report['strategies'][arm][cohort]
            checks[f'score_{arm}_{cohort}'] = sum(r['score']['task_rule_v3'] for r in rr) == stats['passed']
            for field in ('input_tokens', 'output_tokens', 'total_tokens'):
                checks[f'{field}_{arm}_{cohort}'] = sum(a.get('usage', {}).get(field, 0) or 0 for a in aa) == stats['usage'][field]
    for c in cases:
        for p in c['probes']:
            for arm in arms:
                key = f'{c["id"]}-{p["id"]}-{arm}'; r = results[key]
                expected_tool = None
                if r['plan']['ok']:
                    plan = r['plan']['value']
                    if plan['action'] == 'logistics':
                        expected_tool = c['tool_data']['orders'].get(plan['order_id'], {'error': '订单号不存在，请确认'})
                    elif plan['action'] == 'care':
                        care = c['tool_data']['manuals'].get(plan['sku'])
                        expected_tool = {'sku': plan['sku'], 'care': care} if care else {'error': '该型号无说明，请确认'}
                checks['tool_'+key] = r['tool_result'] == expected_tool
                if not r['score']['task_rule_v3']:
                    failures.append({'key': key, 'family': c['kind'], 'cohort': c['cohort'],
                        'query': p['query'], 'expected': p['expected'], 'score': r['score'],
                        'coref': r['coref']['value'], 'plan': r['plan']['value'], 'reply': r['reply']['value']})
    paired = {}
    for baseline in ('C_off', 'S_off'):
        paired[baseline] = {}
        for cohort in ('all', 'regression', 'pressure'):
            pairs = [(f'{c["id"]}-{p["id"]}', results[f'{c["id"]}-{p["id"]}-{baseline}'],
                      results[f'{c["id"]}-{p["id"]}-N_off'])
                     for c in cases if cohort == 'all' or c['cohort'] == cohort for p in c['probes']]
            valid = [t for t in pairs if all(r[s]['ok'] for r in t[1:] for s in ('coref', 'plan', 'reply'))]
            paired[baseline][cohort] = {
                'old_fail_new_pass': [k for k,a,b in pairs if not a['score']['task_rule_v3'] and b['score']['task_rule_v3']],
                'old_pass_new_fail': [k for k,a,b in pairs if a['score']['task_rule_v3'] and not b['score']['task_rule_v3']],
                'common_valid_posthoc': {'n': len(valid), 'old': sum(a['score']['task_rule_v3'] for _,a,b in valid),
                                        'new': sum(b['score']['task_rule_v3'] for _,a,b in valid)}}
    # Request names use the inherited runner prefix. Date-named earlier runs may
    # share names; created_at bounds this run. Transport does not replay results;
    # replay is only in the production chat API, which this experiment bypasses.
    since = datetime.fromisoformat(manifest['created_at']).timestamp()
    with sqlite3.connect('file:data/model_pool.sqlite3?mode=ro', uri=True) as con:
        con.row_factory = sqlite3.Row
        pool = [dict(r) for r in con.execute(
            'SELECT request_id,attempt,started,finished,outcome,http_status FROM attempts WHERE request_id LIKE ? AND started>=? ORDER BY started',
            ('think-'+out.name+'-%', since))]
    expected_ids = {f'think-{out.name}-{key}-{i}' for key, r in calls.items() for i in range(len(r['attempts']))}
    checks['pool_requests_match'] = {r['request_id'] for r in pool} == expected_ids
    checks['pool_finished'] = all(r['finished'] is not None and r['outcome'] != 'RUNNING' for r in pool)
    checks['no_old_attempts_reused'] = all(datetime.fromisoformat(a['started_at']).timestamp() >= since for a in attempts)
    summary = {'passed': all(checks.values()), 'checks': checks,
        'counts': {'calls': len(calls), 'attempts': len(attempts), 'results': len(results), 'maintenance': len(cases)*len(arms)},
        'attempt_status': dict(Counter(a['status'] for a in attempts)),
        'final_call_status': dict(Counter(r['status'] for r in calls.values())),
        'pool_status': dict(Counter(r['outcome'] for r in pool)), 'pool_extra_attempts': len(pool)-len(attempts),
        'missing_usage_attempts': sum(not a.get('usage') for a in attempts),
        'raw_tool_markup_replies': [k for k,r in results.items() if 'DSML' in str(r['reply']['value'])],
        'pairs': paired, 'failure_review_material': failures,
        'caveats': ['Post-hoc common-valid subset can be selection biased.',
                    'Frozen rule results are not independent human semantic labels.',
                    'Pressure histories target known boundaries, not blind generalization.',
                    'Pool rows are restricted to this manifest start time; old runs may share trace names.']}
    save(out/'POOL_ATTEMPTS.json', pool)
    save(out/'SUPPLEMENTARY_AUDIT.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('checks','failure_review_material')}, ensure_ascii=False, indent=2))
    if not summary['passed']:
        print('FAILED_CHECKS', [k for k,v in checks.items() if not v])
        raise SystemExit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True)
    audit(Path(p.parse_args().output).resolve())
