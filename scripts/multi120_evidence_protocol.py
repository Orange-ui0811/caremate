"""Public interaction reconstruction and conservative evidence validation, v2."""
import json
import re


def norm(text):
    return ''.join(str(text).split())


def reply_delta(answer, previous=''):
    return answer[len(previous):].lstrip() if previous and answer.startswith(previous) else answer


def cooperative_order_reply(case, step, previous, actor):
    """Only respond to a NEW solicitation, using public actions/receipts, never task_plan."""
    if step.get('interrupt') or any(a['kind'] == 'supply_missing_order' for a in actor):
        return None
    goals = [g for g in case['goals'] if g['kind'] in ('refund', 'order', 'logistics') and not g['explicit_order']]
    if len(goals) != 1:
        return None
    goal = goals[0]
    if any(a['kind'] == 'select_order' and a.get('value') == goal['order_id'] for a in actor):
        return None
    if any(a.get('goal_id') == goal['id'] and a['kind'] == 'form_submit' and a.get('status') == 200 for a in actor):
        return None
    latest = reply_delta(step.get('answer', ''), previous)
    ask = r'(?:提供|告诉|告知|补充|需要|确认|选择).{0,24}(?:订单|单号)|(?:订单号).{0,12}(?:多少|什么)'
    if not re.search(ask, latest):
        return None
    return f"这项{goal['kind']}对应的订单号是{goal['order_id']}。其他原有事项继续，不增加新事项。"


def interaction_events(r):
    """Attach actions to actual public cards, including cancellations, not to the end."""
    events = []
    steps = r.get('steps', [])
    actors = r.get('actor', [])
    assigned = {}
    intr_steps = {(s.get('interrupt') or {}).get('interrupt_id'): i for i, s in enumerate(steps) if s.get('interrupt')}
    for index, act in enumerate(actors):
        if act.get('interrupt_id') in intr_steps:
            assigned[index] = intr_steps[act['interrupt_id']]
        elif act['kind'] in ('user_message', 'restart_sqlite_connection'):
            assigned[index] = 0 if act['kind'] == 'user_message' else next((i for i, s in enumerate(steps) if s.get('interrupt')), 0)
        elif act['kind'] == 'supply_missing_order':
            dialogue = r.get('dialogue', [])
            pos = next((i for i, m in enumerate(dialogue) if m['role'] == 'user' and m.get('content') == act['text']), None)
            prior = next((m.get('content', '').strip() for m in reversed(dialogue[:pos]) if m['role'] == 'assistant'), '') if pos is not None else ''
            found = next((i for i, s in enumerate(steps) if not s.get('interrupt') and prior and s.get('answer', '').strip().endswith(prior)), None)
            if found is not None:
                assigned[index] = found
        elif act['kind'] in ('cancel_form', 'form_submit', 'duplicate_form_submit'):
            # v4 records cancel_form immediately before its interrupt-linked task_form.
            following = next((a for a in actors[index + 1:] if a.get('interrupt_id')), None)
            if act['kind'] == 'cancel_form' and following and following.get('interrupt_id') in intr_steps:
                assigned[index] = intr_steps[following['interrupt_id']]
            else:
                goal = next((g for g in r['case']['goals'] if g['id'] == act.get('goal_id')), None)
                ticket = (act.get('response') or {}).get('ticket_no')
                for i, s in enumerate(steps):
                    forms = s.get('actions', [])
                    if goal and any((f['type'] == 'refund_form' and goal['kind'] == 'refund' and str(f.get('draft', {}).get('order_id')) == goal['order_id']) or (f['type'] == 'create_ticket' and goal['kind'] == 'complaint') for f in forms):
                        assigned[index] = i
                        break
                    if ticket and any(t['ticket_no'] == ticket for t in s.get('tickets_before_next_actor', [])):
                        assigned[index] = max(0, i - 1)
                        break
        elif act['kind'] in ('unsupported_cancel_selector', 'unresolvable_selector', 'unsupported_interrupt'):
            assigned[index] = max(0, len(steps) - 1)
    previous = ''
    for i, step in enumerate(steps):
        answer = step.get('answer', '')
        delta = reply_delta(answer, previous)
        previous = answer
        if delta:
            events.append({'event_id': f'a{i}', 'role': 'assistant', 'content': delta})
        events.append({'event_id': f's{i}', 'role': 'server_snapshot', 'tickets_before_next_actor': step.get('tickets_before_next_actor', []), 'authorized_count': step.get('authorized_count', 0)})
        if step.get('interrupt') or step.get('actions'):
            events.append({'event_id': f'card{i}', 'role': 'assistant_card', 'interrupt': step.get('interrupt'), 'actions': step.get('actions', [])})
        for k, act in enumerate(actors):
            if assigned.get(k) == i:
                events.append({'event_id': f'u{k}', 'role': 'user' if act['kind'] in ('user_message', 'supply_missing_order') else 'user_action', 'action': act})
    unplaced = [{'actor_index': i, 'action': a} for i, a in enumerate(actors) if i not in assigned]
    if r.get('error'):
        events.append({'event_id': 'execution_error', 'role': 'server_error', 'error_type': r.get('error_type'), 'content': r['error']})
    return events, unplaced


def selectors(r):
    result, seen = [], set()
    for i, step in enumerate(r.get('steps', [])):
        intr = step.get('interrupt') or {}
        if intr.get('type') == 'select_order':
            key = intr.get('interrupt_id') or f'step:{i}'
            if key not in seen:
                seen.add(key)
                result.append({'step': i, 'interrupt_id': key, 'public_payload': intr})
        elif any(a.get('type') == 'select_order' for a in step.get('actions', [])):
            result.append({'step': i, 'public_payload': next(a for a in step['actions'] if a.get('type') == 'select_order')})
    return {'observed_selector_n': len(result), 'unmapped_selector_n': sum(a['kind'] == 'unresolvable_selector' for a in r.get('actor', [])), 'events': result}


def validate_labels(data, events, source_ids, goal_ids):
    """A real quote is necessary but is not itself proof of a factual violation."""
    texts = {e['event_id']: e['content'] for e in events if e['role'] == 'assistant'}
    refs = set(source_ids) | {e['event_id'] for e in events}
    if sorted(x['goal_id'] for x in data['items']) != sorted(goal_ids):
        raise ValueError('goal IDs mismatch')
    def quoted(e):
        return bool(e['quote'].strip()) and e['event_id'] in texts and norm(e['quote']) in norm(texts[e['event_id']])
    for item in data['items']:
        item['evidence_valid'] = not item['response_ok'] or bool(item['evidence']) and all(quoted(e) for e in item['evidence'])
    for claim in data['claims']:
        claim['quote_valid'] = quoted(claim['evidence'])
        claim['sources_valid'] = bool(claim['source_refs']) and all(ref in refs for ref in claim['source_refs'])
        claim['decision_consistent'] = not (claim['decision'] == 'violation' and re.search(r'不构成.{0,12}(?:风险|违规|错误)|故不列为|不存在.{0,8}(?:风险|违规)', claim['reason']))
        claim['confirmed_violation'] = claim['decision'] == 'violation' and claim['quote_valid'] and claim['sources_valid'] and claim['decision_consistent']
        claim['unresolved'] = claim['decision'] == 'uncertain' or claim['decision'] == 'violation' and not claim['confirmed_violation']
    return data
