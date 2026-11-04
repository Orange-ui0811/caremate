"""Independent accounting for matched-context runs; never calls a model.

This checks saved receipts, fixtures, source snapshots and the production pool
ledger. It does not turn the frozen rule score into a human semantic judgment.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return path.read_text(encoding='utf-8')


def load(path):
    return json.loads(read(path))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def audit(out, *, pool_db=None, check_pool=True):
    out = Path(out)
    manifest = load(out/'manifest.json')
    report = load(out/'audited_report.json')
    cases = load(out/'dataset.json')['cases']
    arms = manifest['arms']
    calls = {p.stem: load(p) for p in (out/'calls').glob('*.json')}
    results = {p.stem: load(p) for p in (out/'results').glob('*.json')}
    expected = {f'{c["id"]}-{p["id"]}-{a}' for c in cases for p in c['probes'] for a in arms}
    attempts = [a for r in calls.values() for a in r['attempts']]
    checks = {
        'complete_causal_audit': bool(report.get('complete') and report.get('audit_passed')),
        'result_keys': set(results) == expected,
        'maintenance_count': len(list((out/'maintenance').glob('*.json'))) == len(cases)*len(arms),
        'attempt_count': len(list((out/'attempts').glob('*.json'))) == len(attempts),
    }
    for name, expected_hash in manifest['sources'].items():
        checks['snapshot_'+name] = hashlib.sha256((out/'source_snapshot'/name).read_bytes()).hexdigest() == expected_hash
    for key, r in calls.items():
        for i, a in enumerate(r['attempts'], 1):
            path = out/'attempts'/f'{key}-{i}.json'
            checks[f'attempt_{key}_{i}'] = path.exists() and load(path) == a
        if r['attempts']:
            last = r['attempts'][-1]
            checks['final_'+key] = r['status'] == last['status'] and r['value'] == last['value'] and r['ok'] == (last['status']=='ok')
    cohorts = sorted({c['cohort'] for c in cases})
    for arm in arms:
        for cohort in ['all', *cohorts]:
            selected = [c for c in cases if cohort == 'all' or c['cohort']==cohort]
            result_keys = [f'{c["id"]}-{p["id"]}-{arm}' for c in selected for p in c['probes']]
            rr = [results[k] for k in result_keys if k in results]
            prefixes = tuple(c['id']+'-' for c in selected)
            cc = [r for k, r in calls.items() if k.startswith(prefixes) and '-'+arm+'-' in k]
            aa = [a for r in cc for a in r['attempts']]
            stats = report['strategies'][arm][cohort]
            checks[f'n_{arm}_{cohort}'] = len(rr)==stats['n']==len(result_keys)
            checks[f'score_{arm}_{cohort}'] = sum(r['score']['task_rule_v3'] for r in rr)==stats['passed']
            checks[f'target_{arm}_{cohort}'] = sum(r['score']['target_pass'] for r in rr)==stats['target_passed']
            for field in ('input_tokens','output_tokens','total_tokens'):
                checks[f'{field}_{arm}_{cohort}'] = sum(a.get('usage',{}).get(field,0) or 0 for a in aa)==stats['usage'][field]
    failures = []
    source_visibility = []
    for c in cases:
        for p in c['probes']:
            for arm in arms:
                key = f'{c["id"]}-{p["id"]}-{arm}'
                if key not in results:
                    continue
                r = results[key]
                for stage in ('coref','plan','reply'):
                    checks[f'result_receipt_{key}_{stage}'] = r[stage] == calls.get(key+'-'+stage)
                expected_tool = None
                if r['plan']['ok']:
                    plan = r['plan']['value']
                    if plan['action']=='logistics':
                        expected_tool = c['tool_data']['orders'].get(plan['order_id'], {'error':'订单号不存在，请确认'})
                    elif plan['action']=='care':
                        care = c['tool_data']['manuals'].get(plan['sku'])
                        expected_tool = {'sku':plan['sku'],'care':care} if care else {'error':'该型号无说明，请确认'}
                checks['tool_'+key] = r['tool_result'] == expected_tool
                gold = p.get('gold_evidence', [])
                for e in gold:
                    valid = 0 < e['seq'] <= 2*p['at_turn'] and c['history'][e['seq']-1]['content'][e['start']:e['end']] == e['text']
                    checks[f'gold_{c["id"]}_{p["id"]}_{e["seq"]}_{e["start"]}'] = valid
                stage_visibility = {}
                for stage in ('coref','plan','reply'):
                    prompt = '\n'.join(str(m.get('content','')) for m in r[stage]['contract']['messages'])
                    stage_visibility[stage] = {'exact_quotes_present':sum(e['text'] in prompt for e in gold),'gold_quotes':len(gold)}
                source_visibility.append({'key':key, 'stages':stage_visibility})
                if not r['score']['task_rule_v3']:
                    failures.append({'key':key,'family':c['kind'],'cohort':c['cohort'],
                        'query':p['query'],'expected':p['expected'],'score':r['score'],
                        'coref':r['coref']['value'],'plan':r['plan']['value'],'reply':r['reply']['value']})
    since = datetime.fromisoformat(manifest['created_at']).timestamp()
    checks['fresh_attempts'] = all(datetime.fromisoformat(a['started_at']).timestamp() >= since for a in attempts)
    pool = []
    if check_pool:
        if pool_db is None:
            from app.config import settings
            pool_db = ROOT / settings.model_pool_db_path
        with sqlite3.connect(Path(pool_db).resolve().as_uri()+'?mode=ro', uri=True) as con:
            con.row_factory = sqlite3.Row
            pool = [dict(r) for r in con.execute(
                'SELECT request_id,attempt,started,finished,outcome,http_status FROM attempts WHERE request_id LIKE ? AND started>=? ORDER BY started',
                ('think-'+out.name+'-%', since))]
        expected_ids = {f'think-{out.name}-{k}-{i}' for k,r in calls.items() for i in range(len(r['attempts']))}
        checks['pool_requests_match'] = {r['request_id'] for r in pool} == expected_ids
        checks['pool_finished'] = all(r['finished'] is not None and r['outcome']!='RUNNING' for r in pool)
    paired = {}
    for a, b in [('W_matched','M_matched'),('W_matched','S_matched'),('S_matched','R_matched'),('C_current','R_matched')]:
        if a not in arms or b not in arms:
            continue
        pairs = [(f'{c["id"]}-{p["id"]}', results.get(f'{c["id"]}-{p["id"]}-{a}'), results.get(f'{c["id"]}-{p["id"]}-{b}')) for c in cases for p in c['probes']]
        pairs = [(k,x,y) for k,x,y in pairs if x is not None and y is not None]
        paired[a+'__'+b] = {'first_fail_second_pass':[k for k,x,y in pairs if not x['score']['task_rule_v3'] and y['score']['task_rule_v3']],
            'first_pass_second_fail':[k for k,x,y in pairs if x['score']['task_rule_v3'] and not y['score']['task_rule_v3']]}
    summary = {'passed':all(checks.values()), 'checks':checks,
        'counts':{'calls':len(calls),'attempts':len(attempts),'results':len(results),'maintenance':len(list((out/'maintenance').glob('*.json')))},
        'final_call_status':dict(Counter(r['status'] for r in calls.values())),
        'attempt_status':dict(Counter(a['status'] for a in attempts)),
        'missing_usage_attempts':sum(not a.get('usage') for a in attempts),
        'pool_checked':check_pool,'pool_status':dict(Counter(r['outcome'] for r in pool)),
        'pool_extra_attempts':len(pool)-len(attempts) if check_pool else None,
        'pairs':paired,'failure_review_material':failures,'exact_source_visibility':source_visibility,
        'caveats':['Exact quote visibility is not semantic memory retention: summaries can paraphrase.',
                   'Rule scores are not independent human semantic labels.',
                   'C_current is an operational reference, not an equal-budget causal control.',
                   'Transfer cases are synthetic pre-frozen variants, not an external blind benchmark.',
                   'Current history and two probes only; given history generation is excluded.']}
    save(out/'POOL_ATTEMPTS.json',pool)
    save(out/'SUPPLEMENTARY_AUDIT.json',summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    result = audit(Path(args.output).resolve())
    print(json.dumps({k:v for k,v in result.items() if k not in ('checks','failure_review_material','exact_source_visibility')},ensure_ascii=False,indent=2))
    if not result['passed']:
        print('FAILED_CHECKS',[k for k,v in result['checks'].items() if not v])
        raise SystemExit(1)
