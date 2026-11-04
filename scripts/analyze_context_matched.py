"""Read-only analysis of a completed, audited matched-context experiment.

This module is not imported by the experiment runner. It never calls a model,
rescales a score, or changes calls/results/manifest. Only ANALYSIS.json and
ANALYSIS.md are written, after both audits and accounting checks pass.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


STAGES = ('coref', 'plan', 'reply')
USAGE_FIELDS = ('input_tokens', 'output_tokens', 'total_tokens')
MATCHED_ARMS = ('W_matched', 'M_matched', 'S_matched', 'R_matched')
LABELS = {'C_current': '当前策略参照', 'W_matched': '原文', 'M_matched': '摘要＋原文',
          'S_matched': '原文＋来源状态', 'R_matched': '原文＋来源状态＋回查'}
PAIRS = (('W_matched', 'S_matched'), ('S_matched', 'R_matched'), ('C_current', 'W_matched'))
CAVEATS = [
    '两次追问是同一段给定历史的两个检查点；第一问的模型回复没有写入第二问历史。它们不是独立样本，也不是完整交互式多轮轨迹。',
    'W/M/S/R 使用同一估算方法和对话包上限；上限相同不代表实际输入量相同。C 是现实参照，不能用其差异作同预算单组件归因。',
    'R 的回查证据占用固定对话包额度，会置换近期原文；S 相对 W 还增加状态提示头与支持原句。差异对应完整策略组合。',
    '摘要维护开销按发生时间分配到紧随其后的检查点；每次维护只计一次。Token 含全部维护和两次追问，不含给定历史各轮生成。',
    'API 延迟为各阶段实际尝试等待之和，不含节流、排队、重试退避、上下文组装和后台维护；不是线上首字延迟。',
    '逐字可见性只检查标注原句是否出现在实际提示中。摘要改述可能保留同一信息；原句不在不等于信息丢失，原句在也不保证模型用对。',
    '延用已冻结的规则分数和目标分数，不修订否定引用误判，不把它们称为人工语义准确率。',
    '样本小且来自八个已知场景族；只报告以会话为单位的配对描述，不把任务数当独立统计样本，不据此宣称泛化最优。',
    '缺少 usage 字段的尝试费用未知；汇总是已报告用量，不能按零成本解释。',
]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def estimate(text):
    """Exact copy of the frozen experiment's declared heuristic, not a tokenizer."""
    cjk = sum('\u3400' <= c <= '\u9fff' for c in text)
    return math.ceil(cjk * 1.2 + (len(text) - cjk) / 3) + 8


def verify_estimator(snapshot):
    """Compare function bodies without importing experiment/model modules."""
    import ast
    tree = ast.parse(Path(snapshot).read_text(encoding='utf-8-sig'))
    candidate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'estimate')
    own = ast.parse(Path(__file__).read_text(encoding='utf-8'))
    reference = next(n for n in own.body if isinstance(n, ast.FunctionDef) and n.name == 'estimate')

    def body(function):
        statements = function.body
        if statements and isinstance(statements[0], ast.Expr) and isinstance(statements[0].value, ast.Constant) and isinstance(statements[0].value.value, str):
            statements = statements[1:]
        return [ast.dump(n, include_attributes=False) for n in statements]

    if body(candidate) != body(reference):
        raise ValueError('frozen estimator differs; analysis must use the matching estimator')


def percentile(values, q):
    return sorted(values)[max(0, math.ceil(len(values) * q) - 1)] if values else None


def usage(records):
    attempts = [a for r in records for a in r.get('attempts', [])]
    result = {field: sum(a.get('usage', {}).get(field, 0) or 0 for a in attempts) for field in USAGE_FIELDS}
    result['unknown_attempts_by_field'] = {field: sum(a.get('usage', {}).get(field) is None for a in attempts) for field in USAGE_FIELDS}
    result['attempts'] = len(attempts)
    result['complete_usage'] = not any(result['unknown_attempts_by_field'].values())
    return result


def latency(values):
    return {'count': len(values), 'sum_seconds': sum(values),
            'mean_seconds': statistics.mean(values) if values else None,
            'p50_seconds': percentile(values, .5), 'p95_seconds': percentile(values, .95)}


def collect(out, manifest, cases, calls, results, report):
    rows, assigned = [], set()
    arms = manifest['arms']
    for case in cases:
        case_turns = len(case['history']) // 2
        probes = sorted(case['probes'], key=lambda p: p['at_turn'])
        for arm in arms:
            events = read(out / 'maintenance' / f'{case["id"]}-{arm}.json')
            previous = 0
            used_events = set()
            for probe in probes:
                key = f'{case["id"]}-{probe["id"]}-{arm}'
                result = results[key]
                visible = case['history'][:2 * probe['at_turn']]
                full_estimate = sum(estimate(r['content']) for r in visible) + estimate(probe['query'])
                stage_keys = [key + '-' + stage for stage in STAGES]
                summary_keys = []
                for event in events:
                    if previous < event['at_message'] <= 2 * probe['at_turn']:
                        summary_key = event['key']
                        if summary_key in used_events:
                            raise ValueError('duplicate maintenance assignment ' + summary_key)
                        used_events.add(summary_key)
                        summary_keys.append(summary_key)
                previous = 2 * probe['at_turn']
                owned = stage_keys + summary_keys
                if assigned.intersection(owned):
                    raise ValueError('call counted more than once')
                assigned.update(owned)
                for stage, stage_key in zip(STAGES, stage_keys):
                    if result[stage] != calls[stage_key]:
                        raise ValueError('result/call receipt mismatch ' + stage_key)
                if any(calls[k]['contract']['stage'] != 'summary' for k in summary_keys):
                    raise ValueError('non-summary maintenance call')
                gold = probe.get('gold_evidence', [])
                raw_ids = [m['id'] for m in result['body'] if isinstance(m.get('id'), str)
                           and m['id'].startswith('db-') and m['id'][3:].isdigit()
                           and int(m['id'][3:]) <= 2 * probe['at_turn']]
                visibility = {}
                for stage, stage_key in zip(STAGES, stage_keys):
                    prompt = '\n'.join(str(m.get('content', '')) for m in calls[stage_key]['contract']['messages'])
                    quoted = []
                    for e in gold:
                        if not (0 < e['seq'] <= len(visible)) or visible[e['seq']-1]['content'][e['start']:e['end']] != e['text']:
                            raise ValueError('invalid gold source span ' + key)
                        quoted.append({'ref': f"m{e['seq']}:{e['start']}-{e['end']}",
                                       'characters': len(e['text']), 'exactly_visible': e['text'] in prompt})
                    visibility[stage] = {'quotes': quoted, 'visible_count': sum(x['exactly_visible'] for x in quoted),
                                         'gold_count': len(quoted), 'input_sha256': calls[stage_key]['input_sha256'],
                                         'call_status': calls[stage_key]['status']}
                rows.append({'key': key, 'case_id': case['id'], 'family': case['kind'], 'cohort': case['cohort'],
                    'history_turns': case_turns, 'checkpoint_turn': probe['at_turn'], 'probe_id': probe['id'], 'arm': arm,
                    'full_visible_history_estimate': full_estimate,
                    'full_history_over_budget': full_estimate > manifest['context_budget'],
                    'task_pass': bool(result['score']['task_rule_v3']), 'target_pass': bool(result['score']['target_pass']),
                    'coref_pass': bool(result['score']['coref_pass']), 'score': result['score'],
                    'query': probe['query'], 'expected': probe['expected'],
                    'actual_plan': result['plan']['value'], 'reply': result['reply']['value'],
                    'calls': owned, 'summary_calls': summary_keys, 'probe_calls': stage_keys,
                    'raw_message_ids': raw_ids, 'gold_quote_visibility': visibility,
                    'stage_status': {s: result[s]['status'] for s in STAGES},
                    'api_seconds': sum(result[s]['seconds'] for s in STAGES)})
            if used_events != {e['key'] for e in events}:
                raise ValueError('unassigned maintenance calls for ' + case['id'] + '-' + arm)
    if assigned != set(calls):
        raise ValueError('call set does not match assigned probe/maintenance receipts')
    for arm in arms:
        selected = [r for r in rows if r['arm'] == arm]
        if len(selected) != report['strategies'][arm]['all']['n']:
            raise ValueError('result count differs from audit report')
        observed = summarize(selected, calls)
        expected = report['strategies'][arm]['all']
        if observed['task_passed'] != expected['passed'] or observed['target_passed'] != expected['target_passed']:
            raise ValueError('frozen scores differ from audit report')
        if any(observed['usage'][k] != expected['usage'][k] for k in USAGE_FIELDS):
            raise ValueError('usage differs from audit report')
    return rows


def summarize(rows, calls):
    keys = [key for row in rows for key in row['calls']]
    summary_keys = [key for row in rows for key in row['summary_calls']]
    probe_keys = [key for row in rows for key in row['probe_calls']]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate cost in aggregation')
    cases = defaultdict(list)
    for row in rows:
        cases[row['case_id']].append(row['api_seconds'])
    task_passed = sum(r['task_pass'] for r in rows)
    return {'tasks': len(rows), 'cases': len(cases), 'task_passed': task_passed,
        'task_rate': task_passed / len(rows) if rows else None,
        'target_passed': sum(r['target_pass'] for r in rows), 'coref_passed': sum(r['coref_pass'] for r in rows),
        'usage': usage([calls[k] for k in keys]), 'probe_usage': usage([calls[k] for k in probe_keys]),
        'summary': {'calls': len(summary_keys), 'failed_calls': sum(not calls[k]['ok'] for k in summary_keys),
                    'usage': usage([calls[k] for k in summary_keys]),
                    'api_seconds': sum(calls[k]['seconds'] for k in summary_keys)},
        'probe_api_latency': latency([r['api_seconds'] for r in rows]),
        'two_checkpoint_api_latency': latency([sum(values) for values in cases.values() if len(values) == 2]),
        'call_status': dict(Counter(calls[k]['status'] for k in keys)),
        'failed_stage_status': dict(Counter(status for r in rows for status in r['stage_status'].values() if status != 'ok')),
        'failed_probe_counts': {status: sum(status in r['stage_status'].values() for r in rows)
                                for status in ('transport_error', 'parse_error', 'budget_blocked', 'truncated', 'empty', 'validation_error')},
        'case_ids': sorted(cases)}


def groups(rows, calls, arms):
    def by(predicate):
        selected = [r for r in rows if predicate(r)]
        return {arm: summarize([r for r in selected if r['arm'] == arm], calls) for arm in arms}

    lengths = sorted({24, 32, 64, *(r['history_turns'] for r in rows)})
    cohorts = sorted({r['cohort'] for r in rows})
    return {'all': by(lambda r: True),
        'cohort': {value: by(lambda r, v=value: r['cohort'] == v) for value in cohorts},
        'history_turns': {str(value): by(lambda r, v=value: r['history_turns'] == v) for value in lengths},
        'checkpoint_turns': {str(value): by(lambda r, v=value: r['checkpoint_turn'] == v)
                             for value in sorted({r['checkpoint_turn'] for r in rows})},
        'full_history_budget': {label: by(lambda r, v=value: r['full_history_over_budget'] == v)
                                for label, value in (('within', False), ('over', True))},
        'cohort_and_history_turns': {f'{cohort}/{length}': by(lambda r, c=cohort, n=length: r['cohort'] == c and r['history_turns'] == n)
                                    for cohort in cohorts for length in lengths}}


def compare(rows):
    index = {(r['case_id'], r['probe_id'], r['arm']): r for r in rows}
    checkpoints = sorted({(r['case_id'], r['probe_id']) for r in rows})
    comparisons = {}
    for first, second in PAIRS:
        details, case_deltas = [], defaultdict(list)
        for cid, pid in checkpoints:
            a, b = index[(cid, pid, first)], index[(cid, pid, second)]
            delta = int(b['task_pass']) - int(a['task_pass'])
            case_deltas[cid].append(delta)
            details.append({'case_id': cid, 'probe_id': pid, 'family': a['family'], 'cohort': a['cohort'],
                'history_turns': a['history_turns'], 'full_history_over_budget': a['full_history_over_budget'],
                'first_task_pass': a['task_pass'], 'second_task_pass': b['task_pass'],
                'first_target_pass': a['target_pass'], 'second_target_pass': b['target_pass'],
                'task_delta_second_minus_first': delta,
                'outcome': 'second_only_pass' if delta == 1 else 'first_only_pass' if delta == -1 else 'both_pass' if a['task_pass'] else 'both_fail'})
        per_case = [{'case_id': cid, 'paired_checkpoints': len(values),
                     'task_rate_delta_second_minus_first': statistics.mean(values)} for cid, values in sorted(case_deltas.items())]
        examples, target_differences, nonsemantic_differences = [], 0, []
        # Deterministic examples where the chosen object/action actually differs in correctness.
        for detail in details:
            if detail['first_target_pass'] == detail['second_target_pass']:
                continue
            target_differences += 1
            cid, pid = detail['case_id'], detail['probe_id']
            a, b = index[(cid, pid, first)], index[(cid, pid, second)]
            if a['stage_status']['plan'] != 'ok' or b['stage_status']['plan'] != 'ok':
                nonsemantic_differences.append({'case_id': cid, 'probe_id': pid,
                    'first_plan_status': a['stage_status']['plan'], 'second_plan_status': b['stage_status']['plan']})
                continue
            examples.append({'case_id': cid, 'probe_id': pid, 'query': a['query'], 'expected': a['expected'],
                'first': {'arm': first, 'target_pass': a['target_pass'], 'task_pass': a['task_pass'],
                          'plan': a['actual_plan'], 'stage_status': a['stage_status'], 'reply': a['reply']},
                'second': {'arm': second, 'target_pass': b['target_pass'], 'task_pass': b['task_pass'],
                           'plan': b['actual_plan'], 'stage_status': b['stage_status'], 'reply': b['reply']}})
        comparisons[first + '__' + second] = {'first': first, 'second': second,
            'outcome_counts': dict(Counter(r['outcome'] for r in details)), 'tasks': details, 'per_case': per_case,
            'mean_case_task_rate_delta_second_minus_first': statistics.mean(r['task_rate_delta_second_minus_first'] for r in per_case) if per_case else None,
            'target_examples': examples[:4], 'target_difference_tasks': target_differences,
            'nonsemantic_target_differences': nonsemantic_differences,
            'interpretation': '会话级描述性配对差值；不把每个检查点视为独立样本，无显著性或泛化最优声明。'}
    return comparisons


def build_analysis(out):
    out = Path(out)
    required = ('manifest.json', 'dataset.json', 'audited_report.json', 'SUPPLEMENTARY_AUDIT.json')
    missing = [name for name in required if not (out / name).exists()]
    if missing:
        raise ValueError('incomplete run; analysis not written: missing ' + ', '.join(missing))
    manifest, data, report, supplementary = [read(out / name) for name in required]
    if not report.get('complete') or not report.get('audit_passed') or not supplementary.get('passed'):
        raise ValueError('incomplete or failed audits; analysis not written')
    if set(manifest['arms']) != {'C_current', *MATCHED_ARMS}:
        raise ValueError('this analyzer requires the declared five-arm experiment')
    if digest(data) != manifest['dataset_sha256']:
        raise ValueError('dataset differs from frozen manifest')
    estimator_path = out / 'source_snapshot' / 'app/core/context_policy.py'
    if digest_file(estimator_path) != manifest['sources']['app/core/context_policy.py']:
        raise ValueError('estimator snapshot hash mismatch')
    verify_estimator(estimator_path)
    cases = data['cases']
    expected = {f'{c["id"]}-{p["id"]}-{a}' for c in cases for p in c['probes'] for a in manifest['arms']}
    results = {path.stem: read(path) for path in (out / 'results').glob('*.json')}
    if set(results) != expected:
        raise ValueError('incomplete/unexpected result set; analysis not written')
    calls = {path.stem: read(path) for path in (out / 'calls').glob('*.json')}
    rows = collect(out, manifest, cases, calls, results, report)
    return {'analysis_version': 'context-matched-analysis-v1',
        'inputs': {name: digest_file(out / name) for name in required},
        'analysis_sources': {'scripts/analyze_context_matched.py': digest_file(__file__),
                            **({'tests/test_context_matched_analysis.py': digest_file(Path(__file__).resolve().parents[1] / 'tests/test_context_matched_analysis.py')}
                               if (Path(__file__).resolve().parents[1] / 'tests/test_context_matched_analysis.py').exists() else {})},
        'estimator': {'source_sha256': manifest['sources']['app/core/context_policy.py'],
                      'definition': 'ceil(CJK_characters*1.2 + other_characters/3) + 8 per message',
                      'budget': manifest['context_budget'],
                      'full_history_measure': 'Only the history prefix visible at each checkpoint, plus that checkpoint question; excludes system/tool envelope. Uses the same experiment estimate.'},
        'counts': {'cases': len(cases), 'tasks_per_arm': sum(len(c['probes']) for c in cases),
                   'results': len(rows), 'calls': len(calls)},
        'groups': groups(rows, calls, manifest['arms']), 'comparisons': compare(rows),
        'matched_context_visibility': [{k: row[k] for k in ('key', 'case_id', 'probe_id', 'cohort', 'family', 'history_turns',
            'checkpoint_turn', 'arm', 'full_visible_history_estimate', 'full_history_over_budget', 'raw_message_ids', 'gold_quote_visibility')}
            for row in rows if row['arm'] in MATCHED_ARMS],
        'checkpoint_records': rows,
        'caveats': CAVEATS}


def number(value, digits=2):
    return '-' if value is None else f'{value:.{digits}f}'


def token_display(value, field):
    prefix = '≥' if value['unknown_attempts_by_field'][field] else ''
    return prefix + f'{value[field]:,}'


def markdown(analysis):
    lines = ['# 统一预算上下文对照分层分析', '',
             f"数据：{analysis['counts']['cases']} 条合成会话，每组 {analysis['counts']['tasks_per_arm']} 个检查点。主审计及独立审计已通过。", '',
             '**两次追问是给定历史的两个检查点，第一问模型回复未写入第二问历史。**', '']
    sections = [('总体', analysis['groups']['all'])]
    sections += [('样本组：' + k, v) for k, v in analysis['groups']['cohort'].items()]
    sections += [('完整会话长度：' + k + ' 轮', v) for k, v in analysis['groups']['history_turns'].items()]
    sections += [('检查点实际可见历史：' + k + ' 轮', v) for k, v in analysis['groups']['checkpoint_turns'].items()]
    sections += [('检查点完整历史估算：' + ('超过' if k == 'over' else '未超过') + str(analysis['estimator']['budget']), v)
                 for k, v in analysis['groups']['full_history_budget'].items()]
    for title, arms in sections:
        lines += ['## ' + title, '',
            '|策略|任务规则通过|对象/动作正确|会话数|输入含维护|输出含维护|总Token含维护|摘要输入/输出|单检查点API P50/P95秒|传输/解析/预算失败任务|',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for arm, stats in arms.items():
            u, su, lag = stats['usage'], stats['summary']['usage'], stats['probe_api_latency']
            failures = stats['failed_probe_counts']
            lines.append(f"|{LABELS[arm]}|{stats['task_passed']}/{stats['tasks']}|{stats['target_passed']}/{stats['tasks']}|{stats['cases']}|{token_display(u, 'input_tokens')}|{token_display(u, 'output_tokens')}|{token_display(u, 'total_tokens')}|{token_display(su, 'input_tokens')}/{token_display(su, 'output_tokens')}|{number(lag['p50_seconds'])}/{number(lag['p95_seconds'])}|{failures['transport_error']}/{failures['parse_error']}/{failures['budget_blocked']}|")
        if not any(s['tasks'] for s in arms.values()):
            lines += ['', '本分层没有样本，不能据此判断性能。']
        lines.append('')
    lines += ['## 两个检查点API等待合计', '', '|策略|完整两检查点会话数|均值秒|P50/P95秒|', '|---|---:|---:|---:|']
    for arm, stats in analysis['groups']['all'].items():
        lag = stats['two_checkpoint_api_latency']
        lines.append(f"|{LABELS[arm]}|{lag['count']}|{number(lag['mean_seconds'])}|{number(lag['p50_seconds'])}/{number(lag['p95_seconds'])}|")
    lines += ['', '## 配对任务差异', '']
    for pair in analysis['comparisons'].values():
        first, second = pair['first'], pair['second']
        lines += [f"### {LABELS[first]} → {LABELS[second]}", '',
            f"会话级平均任务通过率差值（后者减前者）：{number(100 * pair['mean_case_task_rate_delta_second_minus_first'])} 个百分点。仅作描述。", '',
            '|会话/检查点|分层|前者任务/对象|后者任务/对象|任务结果|', '|---|---|---|---|---|']
        for item in pair['tasks']:
            lines.append(f"|{item['case_id']}/{item['probe_id']}|{item['cohort']}/{item['history_turns']}轮|{int(item['first_task_pass'])}/{int(item['first_target_pass'])}|{int(item['second_task_pass'])}/{int(item['second_target_pass'])}|{item['outcome']}|")
        lines.append('')
        for ex in pair['target_examples']:
            lines += [f"- 对象/动作差异实例 `{ex['case_id']}/{ex['probe_id']}`：{ex['query']}",
                '  标准目标：`' + json.dumps(ex['expected'], ensure_ascii=False) + '`；',
                '  前者计划：`' + json.dumps(ex['first']['plan'], ensure_ascii=False) + '`；',
                '  后者计划：`' + json.dumps(ex['second']['plan'], ensure_ascii=False) + '`。']
        if not pair['target_examples']:
            lines.append('没有双方规划均成功、但对象/动作正确性不同的实例；任务规则结果仍可能不同。')
        if pair['nonsemantic_target_differences']:
            lines.append(f"另有 {len(pair['nonsemantic_target_differences'])} 个目标分数差异涉及规划阶段协议/接口/预算失败，单列在 JSON，不能直接当成记忆选错对象。")
        lines.append('')
    lines += ['## 原文与标注证据可见性', '',
        '每个 matched 组的原文消息 ID、每条 gold 原句在 coref/plan/reply 实际提示中的逐字可见性，完整保存在 ANALYSIS.json 的 matched_context_visibility。',
        '摘要改述不计为逐字可见，因此此字段不是语义记忆准确率；消息 ID 是本评测的历史位置标记。', '',
        '## 口径与限制', '']
    lines += ['- ' + caveat for caveat in analysis['caveats']]
    return '\n'.join(lines) + '\n'


def self_test():
    assert estimate('中文abc') == math.ceil(2*1.2 + 3/3) + 8
    good = {'status': 'ok', 'ok': True, 'seconds': 1.0,
            'attempts': [{'usage': {'input_tokens': 10, 'output_tokens': 2, 'total_tokens': 12}}]}
    missing = {'status': 'transport_error', 'ok': False, 'seconds': 2.0, 'attempts': [{'usage': {}}]}
    measured = usage([good, missing])
    assert measured['input_tokens'] == 10 and not measured['complete_usage']
    assert measured['unknown_attempts_by_field'] == dict.fromkeys(USAGE_FIELDS, 1)
    calls = {'p1': good, 'p2': good, 'maintenance': good}
    rows = [{'case_id': 'one', 'task_pass': True, 'target_pass': True, 'coref_pass': True,
             'calls': ['p1', 'maintenance'], 'probe_calls': ['p1'], 'summary_calls': ['maintenance'],
             'api_seconds': 3.0, 'stage_status': {'coref': 'ok'}},
            {'case_id': 'one', 'task_pass': False, 'target_pass': True, 'coref_pass': True,
             'calls': ['p2'], 'probe_calls': ['p2'], 'summary_calls': [],
             'api_seconds': 4.0, 'stage_status': {'coref': 'ok'}}]
    stats = summarize(rows, calls)
    assert stats['usage']['input_tokens'] == 30
    assert stats['summary']['usage']['input_tokens'] == 10
    assert stats['two_checkpoint_api_latency']['sum_seconds'] == 7.0
    assert stats['cases'] == 1 and stats['tasks'] == 2 and stats['task_rate'] == .5
    try:
        summarize(rows + rows, calls)
    except ValueError:
        pass
    else:
        raise AssertionError('duplicate cost must fail')
    print('SELF_TEST_OK: estimator, missing usage, maintenance accounting, paired checkpoints, duplicate guard')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', help='completed experiment directory')
    parser.add_argument('--dry-run', action='store_true', help='validate/compute without writing analysis files')
    parser.add_argument('--self-test', action='store_true', help='run minimal in-memory checks, no model or files')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif not args.output:
        parser.error('--output is required unless --self-test is used')
    else:
        directory = Path(args.output).resolve()
        try:
            value = build_analysis(directory)
            document = markdown(value)
        except (ValueError, KeyError, FileNotFoundError) as error:
            print('ANALYSIS_NOT_WRITTEN:', str(error))
            raise SystemExit(2)
        if not args.dry_run:
            (directory / 'ANALYSIS.json').write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            (directory / 'ANALYSIS.md').write_text(document, encoding='utf-8')
        print(json.dumps({'valid': True, 'written': not args.dry_run, 'counts': value['counts']}, ensure_ascii=False))
