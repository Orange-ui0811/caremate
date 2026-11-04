"""Programmatically constructed fictional cases; no human review or blind claim.

Development and reserved cases use separate wording templates.  Reserved cases
are an internal holdout, not an independently authored or external benchmark.
Only history/query enter a context policy; expected values and source labels are
strictly evaluator data.  A shared fixture backend is queried by model arguments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FAMILIES = ('early_order', 'early_product', 'correction', 'task_switch',
            'relation_chain', 'ambiguous', 'constraint_scope', 'partial_retraction',
            'promise_update', 'tool_failure', 'missing_fact', 'old_constraint')

BACKGROUND = (
    ('包装上的图示可以代替说明书吗？', '图示用于快速提示，具体使用步骤和限制仍需查阅对应产品说明。'),
    ('我在整理收货记录，应保留什么材料？', '可以保留包装、清单和可见情况的照片。材料只能用于后续核实，不代表任何处理已获批准。'),
    ('为什么咨询和提交服务申请要分开？', '咨询用于了解条件，提交可能创建业务记录，是否完成要以实际系统结果为准。'),
    ('安装时怎样记录配件？', '按清单记录名称、数量和外观，发现不清楚的地方先核实，不自行猜测功能。'),
)


def _pair(text, answer='收到，作为本次会话的背景记录。'):
    return [{'role': 'user', 'content': text}, {'role': 'assistant', 'content': answer}]


def _expected(action, order='', sku='', *, required=(), forbidden=()):
    return {'action': action, 'order_id': order, 'sku': sku, 'date': '',
            'required_any_groups': [list(g) for g in required], 'forbidden_claims': list(forbidden)}


def _step(index, query, expected, setup=(), evidence=()):
    rows = []
    for text in setup:
        rows.extend(_pair(text))
    return {'id': f'p{index}', 'query': query, 'setup': rows, 'expected': expected,
            'gold_source_texts': list(evidence), 'canonical_reply': '收到，我们继续核实当前问题。'}


def make_case(family, split, variant=0):
    """The three variants are fixed before calls, never selected by score."""
    ix = FAMILIES.index(family)
    base = (8200 if split == 'development' else 920000) + ix * 20 + variant * 5
    a, b, c, d = [str(base + n) for n in range(4)]
    sa, sb = f'MH-CAM{ix + 2}', f'MH-W{ix + 30}'
    first = (f'订单{a}买的是摄像头，型号{sa}。订单{b}买的是饮水机，型号{sb}。'
             if split == 'development' else
             f'我有两份购买凭证：先发的这份写着{a}，对应摄像设备{sa}；另一份订单号为{b}，设备型号是{sb}。')
    rows = _pair(first)
    long = variant == 1
    # Token length is reported from actual serialized prompts. This label is
    # only a predeclared construction stratum, not a claim of exceeding a cap.
    turns = 8 if split == 'development' else (44 if long else 20)
    for n in range(turns):
        q, answer = BACKGROUND[(n + ix) % len(BACKGROUND)]
        rows += _pair(q, answer * (4 if long else 1))
    dev = split == 'development'
    logistics = lambda oid: _expected('logistics', oid)
    care = lambda sku: _expected('care', sku=sku)
    clarify = lambda: _expected('clarify')
    recall = lambda *groups: _expected('recall', required=groups)
    steps = []
    if family == 'early_order':
        steps = [_step(1, '查最先登记那笔的物流。' if dev else '回到我一开始给出的那份凭证，它的包裹到哪里了？', logistics(a), evidence=(first,)),
                 _step(2, f'这次只查订单{b}。它到哪了？', logistics(b)),
                 _step(3, '再查最开始那笔的物流，不是刚才那笔。', logistics(a), evidence=(first,))]
    elif family == 'early_product':
        steps = [_step(1, '最早提到的设备如何清洁？' if dev else '先发的购买凭证上的那件设备，平时该怎么擦洗？', care(sa), evidence=(first,)),
                 _step(2, f'现在改问型号{sb}的清洁办法。', care(sb)),
                 _step(3, '回到最初那件摄像设备，它的清洁方法是什么？', care(sa), evidence=(first,))]
    elif family == 'correction':
        correction = f'{a}报错，应为{c}。' if dev else f'开头那张凭证我抄错了编号，实际应填{c}；原来写的{a}不适用于本次查询。'
        final = f'正确订单号是{d}，前面的号码作废。' if dev else f'再次对照原件，这次咨询最终以{d}为准，刚才的{c}也抄错了。'
        steps = [_step(1, '按更正后的号码查物流。', logistics(c), (correction,), (correction,)),
                 _step(2, '最新核对的那笔到哪里了？', logistics(d), (final,), (final,)),
                 _step(3, '我刚刚确认的这笔订单，再查询一次物流。', logistics(d), evidence=(final,))]
    elif family == 'task_switch':
        one, two = f'本次只查订单{a}。', (f'现在处理订单{b}，第一单先放一边。' if dev else f'之前那张凭证暂告一段落，接下来咨询的配送对象换成{b}。')
        steps = [_step(1, '当前这单到哪里了？', logistics(a), (one,), (one,)),
                 _step(2, '现在说的这个包裹到哪里了？', logistics(b), (two,), (two,)),
                 _step(3, f'回到订单{a}，查询它的物流。', logistics(a))]
    elif family == 'relation_chain':
        rel = f'订单{a}放在家里，订单{b}送到办公室；办公室那件由小林保管。'
        alias = '之后我说同事那件，就是小林保管的设备。' if dev else '研发工位上的设备现在归小林照看，我提到同事的设备就是在说它。'
        rows[2:2] = _pair(rel) + _pair(alias)
        steps = [_step(1, '同事那件设备对应的订单到哪里了？', logistics(b), evidence=(rel, alias)),
                 _step(2, '留在家里那件，物流到哪里了？', logistics(a), evidence=(rel,)),
                 _step(3, '小林保管的设备应该怎么清洁？', care(sb), evidence=(first, rel, alias))]
    elif family == 'ambiguous':
        unknown = '两笔订单我都还没选定，别默认最近一笔。' if dev else '上述两份凭证都只是背景，我尚未决定这次要查询哪份，请等我说明。'
        steps = [_step(1, '查一下那单物流。', clarify(), (unknown,), (unknown,)),
                 _step(2, f'我明确选择订单{a}，请查物流。', logistics(a)),
                 _step(3, '刚才那个选择撤回；两笔都可能查，我还没选定订单。那单到哪了？', clarify())]
    elif family == 'constraint_scope':
        rule = f'先别给订单{a}退款，只允许查询；订单{b}的事情另说。'
        steps = [_step(1, '我对第一笔的操作限制是什么？', recall(('不退款', '别退款', '不提交', '先别', '仅查询', '只允许查询')), (rule,), (rule,)),
                 _step(2, f'现在只查订单{b}的物流。', logistics(b)),
                 _step(3, f'我对订单{a}之前的限制还有效吗？', recall(('有效', '仍', '没有撤回', '未撤回'), ('退款', '提交')), evidence=(rule,))]
    elif family == 'partial_retraction':
        rule = f'订单{a}先别提交退款；订单{b}暂时不要查物流。'
        change = f'撤回对订单{b}不查物流的限制，现在可以查询；订单{a}仍然先别退款。'
        steps = [_step(1, f'目前可以给订单{a}提交退款吗？', recall(('不能', '不可以', '先别', '不要', '未确认')), (rule,), (rule,)),
                 _step(2, f'查订单{b}的物流。', logistics(b), (change,), (change,)),
                 _step(3, f'刚刚放开查询后，订单{a}的退款限制也取消了吗？', recall(('没有', '未取消', '仍', '不变', '不能')), evidence=(rule, change))]
    elif family == 'promise_update':
        promise = f'专员书面承诺订单{a}的附件2026-11-03到货，只是承诺，还没收到。'
        change = f'专员把订单{a}附件承诺到货日期更正为2026-11-08，旧日期不再适用，尚未确认送达。'
        e1, e2 = recall(('2026-11-03',)), recall(('2026-11-08',))
        e1['date'], e2['date'] = '2026-11-03', '2026-11-08'
        steps = [_step(1, '专员目前承诺哪天到货？用YYYY-MM-DD。', e1, (promise,), (promise,)),
                 _step(2, '现在有效的承诺日期是哪天？用YYYY-MM-DD。', e2, (change,), (change,)),
                 _step(3, '有承诺日期，就能说附件已经送达了吗？', recall(('不能', '不代表', '尚未', '还没有')), evidence=(promise, change))]
    elif family == 'tool_failure':
        rows += _pair(f'之前客服口头说订单{a}已经送达。', '那只是此前的表述，当前状态需要查询。')
        steps = [_step(1, f'现在实际查一下订单{a}的物流。', logistics(a)),
                 _step(2, f'再查询订单{a}，以这次系统结果为准。', logistics(a)),
                 _step(3, '如果刚才查询失败，可以确认包裹已签收吗？', recall(('不能', '无法', '不可以')))]
    elif family == 'missing_fact':
        steps = [_step(1, '我那台加湿器如何清洁？我还没告诉你型号。', clarify()),
                 _step(2, '要查物流，但不是前面两笔；新订单号我还没发。现在能查吗？', clarify()),
                 _step(3, f'新问题已说明：我问型号{sa}如何清洁。', care(sa))]
    else:
        rule = f'订单{a}只做咨询，先别提交退款，这个限制直到我明确撤回。'
        rows[2:2] = _pair(rule)
        steps = [_step(1, f'现在可以直接给订单{a}提交退款吗？', recall(('不能', '先别', '不可以', '未撤回')), evidence=(rule,)),
                 _step(2, f'本次只查订单{b}的物流。', logistics(b)),
                 _step(3, f'聊过另一单以后，订单{a}最早的限制是什么？', recall(('退款', '提交'), ('先别', '不能', '不提交', '不要', '仅咨询', '只做咨询')), evidence=(rule,))]
    tools = {'orders': {oid: {'order_id': oid, 'tracking_no': 'SF' + oid + '88', 'status': '运输中', 'last_event': '包裹已到分拨中心'} for oid in (a, b, c, d)},
             'manuals': {sa: '断电后用软布清洁表面，不浸泡机身。', sb: '断电后用软布擦拭外壳，可拆部件按说明清洗。'}}
    if family == 'tool_failure':
        tools['orders'][a] = {'order_id': a, 'error': '物流服务暂时不可用，当前状态未核实'}
    return {'id': f'{split[:3]}-{ix + 1:02d}-{variant}', 'split': split, 'kind': family,
            'history': rows, 'steps': steps, 'tool_data': tools, 'all_orders': [a, b, c, d],
            'all_skus': [sa, sb], 'length_stratum': 'long' if long else 'short',
            'origin': 'programmatically constructed fictional template; no independent human review'}


def dataset():
    cases = [make_case(f, 'development') for f in FAMILIES]
    cases += [make_case(f, 'reserved', v) for f in FAMILIES for v in range(2)]
    return {'version': 'context-state-cases-v1', 'cases': cases,
            'scope': '12 development and 24 internally reserved synthetic conversations across 12 families; 3 scripted user decisions each. Not independently authored, human-reviewed, or external blind data.',
            'metadata_policy': 'Expected values, gold_source_texts and fixture data never enter memory builders. Fixtures receive only model-selected arguments.',
            'trajectory_policy': 'Scripted user turns; actual prior assistant replies are appended in trajectory mode. No reactive user simulator and no production write operations.'}


def validate_dataset(value):
    seen = set()
    for case in value['cases']:
        if case['id'] in seen:
            raise ValueError('duplicate case id')
        seen.add(case['id'])
        if case['split'] not in ('development', 'reserved') or case['kind'] not in FAMILIES:
            raise ValueError('unknown split/family')
        rows = list(case['history'])
        for step in case['steps']:
            rows += step['setup']
            if not step['query'] or not step['expected']['action']:
                raise ValueError('missing question or expected action')
            if any(not any(t in r['content'] for r in rows) for t in step['gold_source_texts']):
                raise ValueError('source label outside visible history')
            rows += _pair(step['query'], step['canonical_reply'])


def export(path):
    path = Path(path)
    if path.exists():
        raise ValueError('dataset output already exists')
    data = dataset(); validate_dataset(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(data, ensure_ascii=False, indent=2) + '\n').encode()
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True)
    print(export(p.parse_args().output))
