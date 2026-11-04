"""Frozen regression comparing per-node context budgets with the V2 reader.

No new task scorer or reader shortcuts: every probe still performs coreference,
structured planning, and reply generation. C/M use the unmodified old workflow;
CW/CS share a token estimator and per-stage limits. Existing reserved cases are
now a previously inspected regression set, not an unseen holdout.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.config import settings
from app.core import summarizer
from app.core.context_compact import assemble_compact_context
from app.core.context_policy import estimate
from app.core.context_state import build_snapshot
from app.core.prompts import COREF_REWRITE_PROMPT
from scripts.context_state_analysis_stats import (
    family_cluster_difference, group_calls, reconcile_attempt_usage, usage,
)

# A distinct module instance avoids mutating either public V1 or V2 imports.
_spec = importlib.util.spec_from_file_location(
    'scripts._context_compact_v2_private', Path(__file__).with_name('eval_context_state_v2.py'))
_v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v2)
_impl = _v2._impl
_original_workflow = _impl.workflow
_original_audit = _v2._original_audit
transport = _v2.transport
VERSION = 'context-compact-eval-v1'
ARMS = ('C', 'M', 'CW', 'CS')
STAGE_BUDGETS = {'coref': 1800, 'intent': 1000, 'plan': 4200, 'answer': 4200}
LABELS = {'C': '当前摘要与滑窗参照（未匹配预算）',
          'M': '原来源状态与分段摘要（大包复用）',
          'CW': '分节点紧凑原文窗口与分段摘要',
          'CS': '分节点紧凑原文、来源状态与回查'}
PAIRS = (('C', 'M'), ('C', 'CW'), ('C', 'CS'), ('M', 'CW'), ('M', 'CS'), ('CW', 'CS'))
_impl.ARMS = ARMS
_impl.LABELS = LABELS
_impl.SOURCES = list(dict.fromkeys([*_impl.SOURCES,
    'app/core/context_compact.py', 'scripts/eval_context_compact.py',
    'scripts/context_state_analysis_stats.py', 'tests/test_context_compact.py',
    'tests/test_eval_context_compact.py', 'docs/experiments/context-compact-validation-20261027.md']))


def _compact_packet(snapshot, query, arm, stage, segments, contract):
    view = assemble_compact_context(
        snapshot, query, stage=stage,
        policy={'CW': 'compact_window', 'CS': 'compact_source'}[arm],
        budget=contract['stage_budgets'][stage], summaries=segments)
    return {'body': view['messages'], 'coref_history': view['history'],
            'manifest': view['manifest'], 'blocked': view['blocked'], 'reason': view['reason']}


def _serial_packet(packet):
    return {**{k: v for k, v in packet.items() if k != 'body'},
            'body': [m.model_dump(mode='json') for m in packet['body']]}


def workflow(case, arm, contract):
    if arm in ('C', 'M'):
        yield from _original_workflow(case, arm, contract)
        return
    if arm not in ('CW', 'CS'):
        raise ValueError('unknown compact arm')
    rows, events, transcript, segments, attempted_segments = list(case['history']), [], [], [], set()
    ingested = 0
    for step in case['steps']:
        rows += step['setup']
        # Identical M maintenance semantics and prompts, with separate model
        # calls per arm; stochastic summary differences remain part of the run.
        for end in range(ingested + 2, len(rows) + 1, 2):
            plan = _impl.summary_plan(_impl.rows_with_ids(rows[:end]), segments, contract)
            if plan is None:
                continue
            span = (plan['from_message_id'], plan['to_message_id'])
            if span in attempted_segments:
                continue
            attempted_segments.add(span)
            key = f'{case["id"]}-{arm}-source-summary-{span[0]}-{span[1]}'
            target = contract.get('summary_target_tokens', 1000)
            dialog = '\n'.join(r['role'] + ': ' + r['content'] for r in plan['rows'])
            prompt = summarizer._SOURCE_SUMMARY_PROMPT.format_messages(dialog=dialog, target=target)
            record = yield ('call', key, 'off', 'summary', prompt, _impl.Summary)
            text = (record['value'] or {}).get('summary', '') if record['ok'] else ''
            accepted = bool(record['ok'] and text.strip() and estimate(text) <= target)
            events.append({'key': key, 'at_message': end, 'from_message_id': span[0], 'to_message_id': span[1],
                'source_hash': plan['source_hash'], 'source_refs': plan['source_refs'],
                'status': record['status'], 'accepted': accepted, 'target_tokens': target,
                'summary_estimate': estimate(text) if text else 0,
                'rejection_reason': '' if accepted else ('over_target' if record['ok'] and text else record['status'])})
            if accepted:
                segments.append({'id': key, 'conversation_id': case['id'],
                    'from_message_id': span[0], 'to_message_id': span[1],
                    'source_hash': plan['source_hash'], 'source_refs': plan['source_refs'],
                    'summary': text.strip(), 'model': settings.chat_model, 'prompt_version': 'source-background-v1'})
        ingested = len(rows)
        snapshot = build_snapshot(_impl.rows_with_ids(rows + [{'role': 'user', 'content': step['query']}]), case['id'])
        # All selections use the original current query. Rewritten-query based
        # retrieval would be a separate intervention and is not introduced here.
        packets = {stage: _compact_packet(snapshot, step['query'], arm, stage, segments, contract)
                   for stage in ('coref', 'plan', 'answer')}
        errors = {stage: p['reason'] if p['blocked'] else None for stage, p in packets.items()}
        key = f'{case["id"]}-{step["id"]}-{arm}'
        coref = yield ('call', key + '-coref', 'off', 'coref',
            COREF_REWRITE_PROMPT.format_messages(query=step['query'], history=packets['coref']['coref_history'] or '(无)'),
            None, errors['coref'])
        instruction = HumanMessage('本轮原话：' + step['query'] + '\n指代改写结果（需用历史核实）：' +
                                   (coref['value'] if coref['ok'] else step['query']))
        plan = yield ('call', key + '-plan', 'off', 'plan',
            [SystemMessage(_impl.READER_SYSTEM), *packets['plan']['body'], instruction], _impl.ServicePlan, errors['plan'])
        data = _impl.tool_result(case, plan['value']) if plan['ok'] else None
        packet = packets['answer']
        prompt = [SystemMessage(_impl.REPLY_SYSTEM), *packet['body'], instruction]
        tool_call = None
        if data is not None:
            p = plan['value']
            name = 'query_logistics' if p['action'] == 'logistics' else 'query_faq'
            args = {'order_id': p['order_id']} if p['action'] == 'logistics' else {'query': p['sku'] + ' 清洁方法'}
            tool_call = {'id': key + '-tool', 'name': name, 'args': args, 'type': 'tool_call'}
            prompt += [AIMessage('', tool_calls=[tool_call]),
                       ToolMessage(json.dumps(data, ensure_ascii=False), tool_call_id=tool_call['id'])]
        else:
            prompt += [HumanMessage('本轮计划：' + json.dumps(plan['value'], ensure_ascii=False) + '\n请核实历史后作答。')]
        reply = yield ('call', key + '-reply', 'off', 'reply', prompt, None, errors['answer'])
        result = {'case_id': case['id'], 'split': case['split'], 'family': case['kind'], 'arm': arm,
            'probe_id': step['id'], 'query': step['query'], 'visible_rows': _impl.rows_with_ids(rows),
            'coref': coref, 'plan': plan, 'reply': reply, 'tool_result': data, 'tool_call': tool_call,
            'body': [m.model_dump(mode='json') for m in packet['body']],
            'context_text': '\n'.join(m.content for m in packet['body']), 'context_manifest': packet['manifest'],
            'stage_contexts': {stage: _serial_packet(p) for stage, p in packets.items()},
            'snapshot': snapshot, 'summary_segments': list(segments),
            'history_sha256': _impl.digest(rows), 'mode': contract['mode']}
        result['score'] = _impl.score(case, step, result)
        yield ('save', 'results', key, result)
        carried = reply['value'] if reply['ok'] else '[上次回答未成功：' + reply['status'] + ']'
        if contract['mode'] == 'checkpoints':
            carried = step['canonical_reply']
        rows += [{'role': 'user', 'content': step['query']}, {'role': 'assistant', 'content': carried}]
        transcript.append({'query': step['query'], 'assistant': carried, 'call_key': key,
                           'tool_call': tool_call, 'tool_result': data})
    yield ('save', 'maintenance', f'{case["id"]}-{arm}', events)
    yield ('save', 'trajectories', f'{case["id"]}-{arm}', transcript)


def prepare(out, **kwargs):
    kwargs.setdefault('arms', ARMS)
    kwargs.setdefault('summary_trigger', 12000)
    kwargs.setdefault('summary_target', 1000)
    if kwargs.get('budget', 9800) != 9800:
        raise ValueError('reference M context budget is fixed at 9800')
    if kwargs['summary_trigger'] != 12000 or kwargs['summary_target'] != 1000:
        raise ValueError('source summary trigger/target are fixed at 12000/1000')
    manifest = _v2.prepare(out, **kwargs)
    manifest.update(version=VERSION, stage_budgets=dict(STAGE_BUDGETS),
        request_id_namespace='context-compact-v1-' + out.name,
        estimator='app.core.context_policy.estimate; same heuristic for M/CW/CS; not provider tokenizer',
        summary_admission=('Existing per-segment cap min(1400, stage_budget//4) is unchanged. '
                           'At coref 1800 this is 450 estimated units; a valid maintained summary may be omitted '
                           'from coref while admitted to plan/answer. This budget-induced effect is part of the treatment.'),
        dataset_role='Previously inspected synthetic regression set; reserved is only the original split identifier, not an unseen holdout.',
        intervention='Per-stage compact context projection only; no coref skipping, no plan/reply merging, no reader or score changes.',
        fairness=('C is an unmatched legacy system reference. M is the original 9800-unit source+summary package reused across stages. '
                  'CW and CS use identical per-stage ceilings and estimator; CW deliberately omits source state and recall. '
                  'M/CW/CS use identical summary trigger, target, prefix and maintenance prompts, with independent summary calls. '
                  'Actual generated summaries and prior replies may differ; this run estimates whole policy outcomes, not isolated semantic effects. '
                  'A smaller ceiling is not equal actual provider Token cost. All stage inputs and usage receipts are retained.'),
        summary_policy=('M/CW/CS independently replay the same fixed-prefix summary maintenance with trigger 12000, target 1000 and eight recent turns retained. '
                        'Reject over-target summaries without semantic retries. Report each arm\'s maintenance costs and use separately. '
                        'C retains its legacy rolling-summary schedule.'))
    _impl.write_json(out / 'manifest.json', manifest)
    return manifest


def check(out, *, offline=False):
    m = _v2._original_check(out, offline=offline)
    if (m['version'] != VERSION or m.get('transport_version') != transport.VERSION or
            m.get('plan_tool_choice') != _v2.FORCED_PLAN or m.get('stage_budgets') != STAGE_BUDGETS or
            m['context_budget'] != 9800 or m.get('summary_trigger_tokens') != 12000 or
            m.get('summary_target_tokens') != 1000 or m.get('summary_keep_turns') != 8 or
            set(m['arms']) - set(ARMS)):
        raise ValueError('not the frozen compact context protocol')
    if not .25 <= m['start_interval'] <= 5:
        raise ValueError('invalid frozen transport interval')
    _v2.cases.validate_dataset(_impl.read(out / 'dataset.json'))
    return m


def audit(out):
    report = _original_audit(out)
    m = _impl.check(out, offline=True)
    cases = {c['id']: c for c in _impl.read(out / 'dataset.json')['cases']}
    results = {p.stem: _impl.read(p) for p in (out / 'results').glob('*.json')}
    calls = {p.stem: _impl.read(p) for p in (out / 'calls').glob('*.json')}
    attempts = {p.stem: _impl.read(p) for p in (out / 'attempts').glob('*.json')}
    grouped, unassigned = group_calls(cases, m['arms'], calls)
    costs = {}
    for arm in m['arms']:
        selected = [r for (cid, a), records in grouped.items() if a == arm for r in records]
        costs[arm] = {'all': usage(selected),
            'stages': {s: usage(r for r in selected if r['contract']['stage'] == s)
                       for s in ('coref', 'plan', 'reply', 'summary')}}
    reconciliation = reconcile_attempt_usage(calls, attempts, costs)
    if unassigned or not reconciliation['all_receipts_accounted']:
        report['issues'].append('unique attempt cost reconciliation')
    for key, result in results.items():
        if result['arm'] not in ('CW', 'CS'):
            continue
        for stage, packet in result['stage_contexts'].items():
            if not packet['blocked'] and sum(estimate(r['content']) for r in packet['body']) > m['stage_budgets'][stage]:
                report['issues'].append('stage context cap: ' + key + '/' + stage)
    report.update(experiment_version=VERSION, transport_version=transport.VERSION,
        plan_tool_choice=_v2.FORCED_PLAN, stage_budgets=m['stage_budgets'],
        dataset_role=m.get('dataset_role', 'Previously inspected regression set'),
        cost_breakdown=costs, cost_reconciliation=reconciliation,
        family_paired_comparisons={a + '->' + b: family_cluster_difference(cases, results, a, b)
                                  for a, b in PAIRS if a in m['arms'] and b in m['arms']},
        paired_comparisons={a + '->' + b: _impl.paired_difference(
            report['strategies'][a]['per_conversation'], report['strategies'][b]['per_conversation'])
            for a, b in PAIRS if a in m['arms'] and b in m['arms']})
    report['audit_passed'] = report['complete'] and not report['issues']
    _impl.write_json(out / 'audited_report.json', report)
    lines = ['# 分节点紧凑上下文回归评测', '',
             f'完整：{report["complete"]}；离线审计：{report["audit_passed"]}；样本：{len(cases)}条；模式：{m["mode"]}。', '',
             '|方案|任务规则通过|对象动作通过|接口/协议/预算失败|输入含维护|输出含维护|总Token|摘要调用/拒收/入模检查点|',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm, g in report['strategies'].items():
        u = g['usage']
        used = g['summary_used_checkpoints'] if arm != 'C' else '—'
        lines.append(f'|{LABELS[arm]}|{g["task_passed"]}/{g["n"]}|{g["target_passed"]}/{g["n"]}|'
                     f'{g["transport_failed"]}/{g["protocol_failed"]}/{g["budget_blocked"]}|'
                     f'{u["input_tokens"]:,}|{u["output_tokens"]:,}|{u["total_tokens"]:,}|'
                     f'{g["summary_calls"]}/{g["summary_rejected"]}/{used}|')
    lines += ['', '|方案|指代Token|规划Token|回答Token|摘要Token|', '|---|---:|---:|---:|---:|']
    for arm, cost in costs.items():
        lines.append('|' + arm + '|' + '|'.join(f'{cost["stages"][s]["total_tokens"]:,}'
                     for s in ('coref', 'plan', 'reply', 'summary')) + '|')
    lines += ['', '|配对（后者减前者）|场景族数|通过率差/百分点|场景族bootstrap描述性95%区间|', '|---|---:|---:|---|']
    for name, pair in report['family_paired_comparisons'].items():
        diff = pair['mean_difference_pp']
        interval = pair['descriptive_95_percentile_interval_pp']
        shown_diff = f'{diff:.2f}' if diff is not None else '—'
        lines.append(f'|{name}|{pair["family_clusters"]}|{shown_diff}|' +
                     (f'[{interval[0]:.2f}, {interval[1]:.2f}]' if interval else '不足两个场景族') + '|')
    lines += ['', '本轮只比较每环节的历史装配和预算，不跳过指代、不合并规划与回复，不改变评分规则。',
              'M历史包上限9800；CW/CS指代1800、规划与回答4200，均使用同一启发式估算；完整请求仍受16000上限约束，输出上限8192。C保持原实现，其预算未匹配。',
              'M/CW/CS摘要维护均为12000触发、1000目标、保留近期8轮；摘要分别调用，随机输出和连续轨迹差异均保留。CW省略来源状态与回查，属于消融基线。',
              '保留原摘要入模规则min(1400,阶段预算/4)：指代1800对应单段450上限，维护成功的摘要也可能不进入指代环节，而进入规划与回答；这是本次预算联动的一部分。',
              '24条原reserved会话已用于开发分析，现为已见回归集，不是未见留出集；12个场景族的长版本使用重复背景施加预算压力，不代表真实流量。',
              '公共reader使用只读工具夹具，不是线上完整Workflow/RAG/SSE。成本含初始历史维护与三个追问，不含生成给定历史；不能直接宣称线上节省或按API等待推导首字延迟。',
              '每次调用及重试分别核算。缓存读取是输入Token的子集，不重复累加，不把Token直接当供应商费用。失败和拒收摘要同样保留并计费。',
              '任务通过是固定规则的通过率，不等于人工语义正确率。统计区间按场景族重采样，保留同族会话与追问相关性；小样本不证明普遍非劣。',
              '', f'唯一attempt费用对账：{reconciliation["all_receipts_accounted"]}；审计问题：{len(report["issues"])}；缺失记录：{len(report["missing"])}。']
    (out / 'AUDITED_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report


_impl.workflow = workflow
_impl.check = check
_impl.audit = audit
run = _impl.run
writer_lock = _impl.writer_lock


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True)
    action = p.add_mutually_exclusive_group()
    action.add_argument('--prepare', action='store_true'); action.add_argument('--audit', action='store_true')
    p.add_argument('--split', choices=('development', 'reserved'), default='reserved')
    p.add_argument('--limit', type=int); p.add_argument('--families')
    p.add_argument('--arms', default=','.join(ARMS))
    p.add_argument('--mode', choices=('trajectory', 'checkpoints'), default='trajectory')
    p.add_argument('--verified-preflight', required=False)
    p.add_argument('--case-concurrency', type=int, default=8); p.add_argument('--caller-concurrency', type=int, default=8)
    p.add_argument('--start-interval', type=float, default=.5); p.add_argument('--repeat-id')
    args = p.parse_args(); out = Path(args.output).resolve()
    if args.prepare:
        manifest = prepare(out, split=args.split, limit=args.limit,
            families=tuple(args.families.split(',')) if args.families else None,
            arms=tuple(args.arms.split(',')), mode=args.mode,
            case_concurrency=args.case_concurrency, caller_concurrency=args.caller_concurrency,
            start_interval=args.start_interval, repeat_id=args.repeat_id, preflight=args.verified_preflight)
        print(json.dumps({'prepared': True, 'version': manifest['version'], 'arms': manifest['arms'],
                          'dataset_sha256': manifest['dataset_sha256']}, ensure_ascii=False))
    elif args.audit:
        with writer_lock(out):
            print(json.dumps(audit(out), ensure_ascii=False))
    else:
        with writer_lock(out):
            asyncio.run(run(out))


if __name__ == '__main__':
    main()
