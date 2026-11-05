"""Conversation-scoped order referents, never operation consent or cached facts."""
import re

from app.core import order_binding as binding
from app.tools import business

REFERENCE = re.compile(r'刚(?:刚|才).{0,8}(?:订单|那单|这单)|之前.{0,6}(?:订单|选的)|上(?:一|个|笔).{0,3}订单|这(?:一)?(?:笔订单|个订单|单)|该订单|那(?:笔订单|个订单|单)')
PRONOUN = re.compile(r'它|该商品|这(?:件|个)(?:商品|产品)?')
NEW_OBJECT = re.compile(r'另一|另外|其他|换一(?:笔|个|单)|不是这|不指这')
AFTERSALES = re.compile(r'退货|退款|换货|保修|维修|售后|质量|漏电|故障|清洗')
ISSUE = re.compile(r'漏电|触电|冒烟|起火|坏|故障|破损|质量|少发|漏发|发错|不能用|损坏')
PRODUCT = re.compile(r'猫砂盆|饮水机|猫爬架|猫粮|猫窝|猫砂|玩具')
PROVEN = {'simple_context', 'context_v2', 'user_selection', 'task_clause', 'adjacent_order_subject', 'adjacent_order_subject_action',
          'explicit_same_message_reference', 'explicit_parallel_orders',
          'conversation_reference', 'selected_reference_group'}


def _supported(tasks, uid):
    return [t for t in tasks.values() if (t.get('kind') in binding.ORDER_KINDS or
            (t.get('kind') in {'knowledge','return_advice'} and t.get('plan_version') in {2,3}))
            and t.get('binding', {}).get('method') in PROVEN
            and t.get('binding', {}).get('status') == 'resolved'
            and business.owns_order(uid, t.get('order_id') or '')]


def restore(previous, plans, uid, cid):
    """Upgrade old conversations from owned plan/selection records, not assistant prose."""
    old = previous.get('business_context') or {}
    if old.get('user_id') != uid or old.get('conversation_id') != cid:
        old = {}
    supported = _supported(previous.get('task_plan', {}), uid)
    ordered = sorted(plans, key=lambda p: p.get('origin', 0), reverse=True)
    if not supported:
        supported = next((items for p in ordered if (items := _supported(p.get('tasks', {}), uid))), [])
    ids = list(dict.fromkeys(t['order_id'] for t in supported)) or [o for o in old.get('order_ids', []) if business.owns_order(uid, o)]
    issues = dict(old.get('user_reports', {}))
    # The original refund task carries the user's product/fault report. A later
    # generic status query must not erase it. Cancellation does not erase identity.
    for p in reversed(ordered):
        for t in _supported(p.get('tasks', {}), uid):
            if t['kind'] == 'refund' and (t['order_id'] not in issues or ISSUE.search(t.get('source', ''))):
                issues[t['order_id']] = {'text': t.get('source', ''), 'message_id': p['origin']}
    sources = dict(old.get('order_sources', {}))
    for p in reversed(ordered):
        for t in _supported(p.get('tasks', {}), uid):
            sources[t['order_id']] = {'kind':'verified_task','message_id':p['origin'],
                                      'task_id':t.get('task_id'),'method':t['binding']['method']}
    return {'user_id': uid, 'conversation_id': cid, 'order_ids': ids, 'order_sources':sources,
            'recent': bool(_supported(previous.get('task_plan', {}), uid)) or any(
                t.get('context_order_id') in ids for t in previous.get('task_plan', {}).values()),
            'user_reports': {o: issues[o] for o in ids if o in issues}}


def bind_tasks(raw, items, context, uid):
    """Resolve only a positively linked clause group; new/unrelated objects reset it."""
    cs = binding.clauses(raw)
    focus = context.get('order_ids', [])
    prior = focus[0] if len(focus) == 1 and business.owns_order(uid, focus[0]) else None
    product = business.order_snapshot(prior).get('product', '') if prior else ''
    groups = {}; target = None; group = None
    for i, c in enumerate(cs):
        ids = binding.numbers(c.text)
        if NEW_OBJECT.search(c.text):
            target = None; group = None
        if len(ids) == 1 and ('订单' in c.text or any(binding.has_action(k, c.text) for k in binding.ORDER_KINDS)):
            target = ids[0]; group = f'clause:{i}'
        elif len(ids) > 1:
            target = None; group = None
        elif REFERENCE.search(c.text) and not NEW_OBJECT.search(c.text):
            if not group:
                target = prior; group = f'reference:{i}'
        elif (not group and context.get('recent') and not NEW_OBJECT.search(c.text)
              and (PRONOUN.search(c.text) or AFTERSALES.search(c.text))
              and not any(m.group() not in product for m in PRODUCT.finditer(c.text))):
            # Implicit bare aftersales follow-ups are knowledge context only.
            target = prior; group = f'{"pronoun" if PRONOUN.search(c.text) else "implicit"}:{i}'
        groups[i] = (target, group)
    out = []
    for original in items:
        t = dict(original)
        _, indices = binding.source_clauses(raw, t['source'])
        if t['kind'] == 'knowledge':
            indices = [i for i in indices if AFTERSALES.search(cs[i].text)]
        elif t['kind'] in binding.ORDER_KINDS:
            indices = [i for i in indices if binding.has_action(t['kind'], cs[i].text)
                       or (t['kind'] == 'order' and '订单' in cs[i].text)]
            # A separate refund request is not automatically about a logistics
            # sibling. A write target needs its own explicit referential phrase.
            if t['kind'] == 'refund' and not t.get('order_id'):
                indices = [i for i in indices if REFERENCE.search(cs[i].text) or PRONOUN.search(cs[i].text)]
        else:
            indices = []
        choices = {groups[i] for i in indices if groups[i][1]}
        if len(choices) == 1:
            oid, refgroup = next(iter(choices))
            implicit = refgroup.startswith('implicit:')
            if not implicit or t['kind'] == 'knowledge':
                if t['kind'] == 'knowledge' and oid:
                    known_product = business.order_snapshot(oid).get('product', '') if business.owns_order(uid, oid) else ''
                    if any(m.group() not in known_product for i in indices for m in PRODUCT.finditer(cs[i].text)):
                        out.append(t)
                        continue
                t['reference_group'] = refgroup
                if oid and business.owns_order(uid, oid):
                    if t['kind'] in binding.ORDER_KINDS and not t.get('order_id') and t.get('binding', {}).get('status') != 'ambiguous':
                        t.update(order_id=oid, binding={'status':'resolved', 'method':'conversation_reference',
                            'order_ids':[oid], 'reference_group':refgroup, 'source':t['source']})
                    if t['kind'] == 'knowledge':
                        t['context_order_id'] = oid
        out.append(t)
    return out


def selected(state, tasks, current_key, oid):
    """Share one user selection only among tasks linked to the same raw referent."""
    task = tasks[current_key]; group = task.get('reference_group')
    if group:
        for key, other in tasks.items():
            if key == current_key or other.get('reference_group') != group:
                continue
            if state.get('task_plan_version')==2 and other['status'] in {'DONE','FAILED','CANCELLED','SUPERSEDED'}:
                continue
            if other['kind'] in binding.ORDER_KINDS and not other.get('order_id'):
                other.update(order_id=oid, binding={'status':'resolved','method':'selected_reference_group','order_ids':[oid]})
            elif other['kind'] == 'knowledge':
                other['context_order_id'] = oid
    old = state.get('business_context') or {}
    reports = dict(old.get('user_reports', {}))
    if task['kind'] == 'refund' and (oid not in reports or ISSUE.search(task.get('source', ''))):
        reports[oid] = {'text':task.get('source', ''), 'message_id':state['current_message_id']}
    ids = list(dict.fromkeys(t['order_id'] for t in _supported(tasks, state['user_id'])))
    return {'user_id':state['user_id'], 'conversation_id':state['conversation_id'],
            'order_ids':ids, 'recent':True, 'user_reports':reports}


def knowledge_context(state, task):
    """Fetch current owned order facts; historical claims stay explicitly attributed."""
    oid = task.get('context_order_id')
    if not oid or not business.owns_order(state['user_id'], oid):
        return None
    order = (task.get('task_context') or {}).get('order_facts') or business.order_snapshot(oid)
    report = (state.get('business_context') or {}).get('user_reports', {}).get(oid, {})
    return {'order_id':oid, 'product':order.get('product'), 'status':order.get('status'),
            'user_reported_issue':report.get('text', ''), 'source_message_id':report.get('message_id')}


def policy_query(question, context):
    """Retrieve governing policy separately from deciding a particular order's eligibility."""
    if context and re.search(r'(可以|可不可以|能不能|能否|可否|能|允许|支持).*退(?:货|款)?', question):
        quality = '质量问题' if ISSUE.search(context.get('user_reported_issue', '') + question) else ''
        return f'{quality}商品的退货退款申请条件、流程与运费规则是什么？'
    return f"{context.get('product') or ''} {question}" if context else question
