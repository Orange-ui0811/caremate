"""Versioned deterministic checks, not a substitute for semantic human review."""
from __future__ import annotations

import re

from scripts.eval_context_comparison import score_probe as legacy_score

VERSION = 'context-state-rule-v1'
IDENTIFIER = re.compile(r'(?:订单(?:号)?|编号|号码|订单\s*[为是：:])\s*([A-Za-z0-9_-]*\d[A-Za-z0-9_-]*)')
NEGATIVE = re.compile(r'(?:未|没有|并未|不再|不|别|不要|无需|勿|先别|暂不)(?:实际)?(?:查询|查|处理|使用|采用|提交|选择)|(?:作废|失效|报错|抄错|不适用|排除|不作为)')


def identifier_mentions(text, known=()):
    """IDs of arbitrary digit length; dates/tracking strings are not bare IDs.

    Only explicit non-use clauses are exempted. Negation is a diagnostic rule,
    not a full Chinese semantic parser. Tool arguments remain authoritative.
    """
    mentions = []
    for sentence in re.split(r'[。！？；;\n]', text or ''):
        ids = {m.group(1) for m in IDENTIFIER.finditer(sentence)}
        ids |= {x for x in known if re.search(r'(?<![A-Za-z0-9])' + re.escape(x) + r'(?![A-Za-z0-9])', sentence)}
        for value in sorted(ids):
            # Keep a local clause: a denial about A must not excuse an asserted B.
            for m in re.finditer(re.escape(value), sentence):
                left = max(sentence.rfind('，', 0, m.start()), sentence.rfind(',', 0, m.start())) + 1
                ends = [i for i in (sentence.find('，', m.end()), sentence.find(',', m.end())) if i >= 0]
                clause = sentence[left:min(ends) if ends else len(sentence)]
                mentions.append({'value': value, 'nonuse': bool(NEGATIVE.search(clause)), 'clause': clause})
                break
    return mentions


def score(case, step, result):
    expected = step['expected']; plan_record = result.get('plan', {})
    plan = plan_record.get('value') or {}; reply = result.get('reply', {}).get('value') or ''
    rewrite = result.get('coref', {}).get('value') or ''
    stages = [result.get(s, {}) for s in ('coref', 'plan', 'reply')]
    statuses = [s.get('status', 'missing') for s in stages]
    transport = any(s in ('transport_error', 'validation_error') for s in statuses)
    protocol = any(s in ('parse_error', 'empty', 'truncated') for s in statuses)
    blocked = 'budget_blocked' in statuses
    known = set(case['all_orders'])
    # Fixture inventory is not evidence. A future valid fixture ID is still a
    # hallucination if no visible user source has supplied it at this point.
    visible_user_text = '\n'.join(r['content'] for r in result.get('visible_rows', case['history']) if r['role'] == 'user') + '\n' + step['query']
    observed = {oid for oid in known if re.search(r'(?<![A-Za-z0-9])' + re.escape(oid) + r'(?![A-Za-z0-9])', visible_user_text)}
    mentions = identifier_mentions(reply, known)
    rewritten = identifier_mentions(rewrite, known)
    fabricated = sorted({m['value'] for m in mentions + rewritten if m['value'] not in observed})
    if plan.get('order_id') and plan['order_id'] not in observed:
        fabricated.append(plan['order_id'])
    action = expected['action']; action_ok = plan.get('action') == action
    target = action_ok and (action != 'logistics' or plan.get('order_id') == expected['order_id']) and (action != 'care' or plan.get('sku', '').upper() == expected['sku'].upper())
    target = target and (action == 'logistics' or not plan.get('order_id')) and (action == 'care' or not plan.get('sku'))
    data = result.get('tool_result') or {}
    if action == 'logistics':
        if data.get('error'):
            answer = bool(re.search(r'无法|失败|不可用|未能|暂时不能|未核实|稍后', reply)) and not re.search(r'SF\d+', reply)
        else:
            answer = expected['order_id'] in reply and data.get('tracking_no', '\x00') in reply
        unwanted = [m for m in mentions if m['value'] in known and m['value'] != expected['order_id'] and not m['nonuse']]
        answer = answer and not unwanted
    elif action == 'care':
        answer = expected['sku'].upper() in reply.upper() and '断电' in reply and '软布' in reply
        unwanted = []
    elif action == 'clarify':
        answer = bool(re.search(r'哪|具体|提供|补充|确认|型号|订单号|无法确定|不能确定', reply)) and not any(not m['nonuse'] and m['value'] not in known for m in mentions)
        unwanted = []
    else:
        groups = expected.get('required_any_groups', [])
        answer = bool(reply.strip()) and all(any(x in reply for x in group) for group in groups)
        answer = answer and not any(x in reply for x in expected.get('forbidden_claims', []))
        unwanted = []
    all_ok = all(s.get('ok') for s in stages)
    evidence = result.get('context_text', '')
    gold = step.get('gold_source_texts', [])
    literal = [s in evidence for s in gold]
    probe = {'expected': expected, 'query': step['query'], 'at_turn': max(1, len(result.get('visible_rows', case['history'])) // 2)}
    old_case = {**case, 'history': result.get('visible_rows', case['history'])}
    legacy = legacy_score(old_case, probe, result)
    return {'version': VERSION, 'task_pass': bool(all_ok and target and answer and not fabricated),
            'action_pass': bool(action_ok), 'target_pass': bool(target), 'answer_rule_pass': bool(answer),
            'transport_failure': transport, 'protocol_failure': protocol, 'budget_blocked': blocked,
            'fabricated_identifiers': sorted(set(fabricated)), 'nonuse_mentions': [m for m in mentions if m['nonuse']],
            'unexpected_asserted_orders': unwanted, 'source_literal_visibility': literal,
            'source_literal_recall': sum(literal) / len(literal) if literal else None,
            'legacy': legacy, 'limitations': 'Deterministic task checks only; negative-reference and recall phrase checks require semantic spot review.'}
