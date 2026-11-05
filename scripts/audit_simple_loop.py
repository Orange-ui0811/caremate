"""Receipt-based review, without changing or replacing frozen raw trajectories."""
import collections
import hashlib
import json
import statistics
from pathlib import Path

OUT = Path('E:/Mewhelp/data/verification/multi-intent-simple-loop/20261105-v1')


def audit(d):
    goals = dict(d.get('goals', {}))
    corrections = []
    case = d['case']
    logs = [a for a in d.get('audit', []) if a['status'] == '成功']
    # The frozen requirement concerns retrieval input, not a particular node.
    faq = [a for a in logs if a['tool'] == 'query_faq']
    if 'knowledge' in goals and not goals['knowledge'] and any(
        '智能猫砂盆' in a['args'].get('keyword', '') for a in faq
    ):
        goals['knowledge'] = True
        corrections.append('商品名已传给 query_faq；原脚本只统计 retrieve_knowledge，漏记合法工具路径。')
    # A hit on either historical order does not resolve the singular ambiguity.
    if case['id'] == 'S6' and d.get('selections', 0) != case['selections']:
        for k in ('order', 'logistics'):
            if goals.get(k):
                goals[k] = False
                corrections.append(k + '虽查到数据，但未完成必要选单，不能计为正确对象的任务完成。')
    if case['id'] == 'S5' and goals.get('refund') and not any(
        a['tool'] == 'query_order' and a['args'].get('order_id') == '2002' for a in logs
    ):
        goals['refund'] = False
        corrections.append('“取消”等词不等于独立退款完成；缺少所选2002的本轮回执。')
    passed = bool(goals) and all(goals.values()) and all(d.get('checks', {}).values()) and not d.get('error')
    tickets = d.get('tickets', [])
    confirms = collections.Counter(
        x.get('preview', {}).get('ticket_type', x.get('form'))
        for x in d.get('decisions', []) if x.get('confirmed') is True
    )
    duplicate_tickets = len(tickets) - len({x['number'] for x in tickets})
    excess_tickets = sum((collections.Counter(t['type'] for t in tickets) - collections.Counter(case['tickets'])).values())
    unconfirmed = sum(max(0, n - confirms[k]) for k, n in collections.Counter(t['type'] for t in tickets).items())
    # For the baseline form path, normalize its form name to the business type.
    if confirms.get('refund') or confirms.get('complaint'):
        confirms['退款'] += confirms['refund']; confirms['投诉'] += confirms['complaint']
        unconfirmed = sum(max(0, n - confirms[k]) for k, n in collections.Counter(t['type'] for t in tickets).items())
    final = d.get('steps', [{}])[-1].get('state', {})
    return {
        'case': case['id'], 'rep': d['rep'], 'raw_pass': d.get('passed', False),
        'audited_pass': passed, 'raw_goals': d.get('goals', {}), 'goals': goals,
        'corrections': corrections, 'checks': d.get('checks', {}),
        'selections': d.get('selections', 0), 'required_selections': case['selections'],
        'extra_selections': max(0, d.get('selections', 0) - case['selections']),
        'missing_selections': max(0, case['selections'] - d.get('selections', 0)),
        'duplicate_order_calls': d.get('duplicate_order_calls', 0),
        'tickets': len(tickets), 'duplicate_ticket_numbers': duplicate_tickets,
        'excess_tickets': excess_tickets, 'unconfirmed_tickets': unconfirmed,
        'error': d.get('error'),
        'knowledge_gates': [t.get('evidence_gate') for t in final.get('task_plan', {}).values() if t['kind'] == 'knowledge'],
    }


def summarize(folder):
    files = sorted(folder.glob('r*/S*.json'))
    ds = [json.loads(p.read_text(encoding='utf-8')) for p in files]
    rows = [dict(audit(d), file=str(p.relative_to(OUT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for d, p in zip(ds, files)]
    http = [h for d in ds for h in d.get('http', [])]
    usage = collections.Counter()
    for h in http:
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            usage[key] += h.get('usage', {}).get(key, 0)
    by_model = collections.Counter(h.get('request', {}).get('model', 'unknown') for h in http)
    elapsed = [d['elapsed_seconds'] for d in ds if 'elapsed_seconds' in d]
    return {
        'n': len(ds), 'raw_pass': sum(r['raw_pass'] for r in rows),
        'audited_pass': sum(r['audited_pass'] for r in rows),
        'goals_done': sum(sum(r['goals'].values()) for r in rows),
        'goals_total': sum(len(r['goals']) for r in rows),
        'business_goals_done': sum(sum(v for k, v in r['goals'].items() if k != 'knowledge') for r in rows),
        'business_goals_total': sum(sum(k != 'knowledge' for k in r['goals']) for r in rows),
        **{k: sum(r[k] for r in rows) for k in ('selections', 'required_selections', 'extra_selections',
          'missing_selections', 'duplicate_order_calls', 'tickets', 'duplicate_ticket_numbers', 'excess_tickets', 'unconfirmed_tickets')},
        'terminal_errors': sum(bool(r['error']) for r in rows),
        'http_errors': sum(bool(h.get('error')) or h.get('status', 200) >= 400 for h in http),
        'requests_by_endpoint': dict(collections.Counter(h['endpoint'] for h in http)),
        'requests_by_model': dict(by_model), 'reported_token_usage': dict(usage),
        'requests_without_usage': sum('usage' not in h for h in http),
        'seconds_mean': statistics.mean(elapsed) if elapsed else None,
        'seconds_median': statistics.median(elapsed) if elapsed else None,
        'per_case': {f'S{i}': sum(r['audited_pass'] for r in rows if r['case'] == f'S{i}') for i in range(1, 9)},
        'rows': rows,
    }


def main():
    groups = {'single_frozen': 'formal1/single', 'simple_initial': 'formal1/simple', 'simple_replyfix': 'replyfix/simple'}
    result = {k: summarize(OUT / p) for k, p in groups.items()}
    (OUT / 'AUDIT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    visible = []
    for label, folder in groups.items():
        for p in sorted((OUT / folder).glob('r*/S*.json')):
            d = json.loads(p.read_text(encoding='utf-8'))
            visible.append(f'===== {label} {p.parent.name} {p.stem} =====\n问题：{d["case"]["raw"]}')
            for i, step in enumerate(d.get('steps', [])):
                visible.append(f'--- 展示 {i+1} ---\n{step["state"].get("answer", "")}')
    (OUT / 'VISIBLE_REPLIES.txt').write_text('\n\n'.join(visible), encoding='utf-8')
    print(json.dumps({k: {x: y for x, y in v.items() if x != 'rows'} for k, v in result.items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
