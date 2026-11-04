"""New experiment version: forced plan transport and declared long stress stratum.

V1 source and outputs remain immutable. A private module instance reuses the
causal workflow/audit without changing the imported V1 module's globals.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from scripts import context_state_cases_v2 as cases
from scripts import context_state_transport_v2 as transport

_spec = importlib.util.spec_from_file_location('scripts._context_state_v2_private', Path(__file__).with_name('eval_context_state.py'))
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
_impl.transport = transport
_impl.dataset = cases.dataset
_impl.validate_dataset = cases.validate_dataset
_impl.SOURCES = list(dict.fromkeys([*_impl.SOURCES, 'scripts/context_state_cases_v2.py',
    'scripts/context_state_transport_v2.py', 'scripts/eval_context_state_v2.py',
    'tests/test_context_state_v2.py']))
_original_prepare, _original_check, _original_audit = _impl.prepare, _impl.check, _impl.audit
ARMS = _impl.ARMS
VERSION = 'context-state-eval-v2'
FORCED_PLAN = {'type': 'function', 'function': {'name': 'ServicePlan'}}


def prepare(out, **kwargs):
    start_interval = kwargs.pop('start_interval', .5)
    if not .25 <= start_interval <= 5:
        raise ValueError('start interval must be between 0.25 and 5 seconds')
    proof_path = kwargs.get('preflight')
    if not proof_path:
        raise ValueError('V2 verified preflight required')
    proof_calls = [_impl.read(p) for p in (Path(proof_path) / 'calls').glob('*.json')]
    plans = [r for r in proof_calls if r['contract']['stage'] == 'plan']
    if len(plans) != 1 or plans[0]['contract'].get('tool_choice') != FORCED_PLAN:
        raise ValueError('preflight must verify forced ServicePlan protocol; V1 proof is insufficient')
    if any(r['contract'].get('transport_version') != transport.VERSION for r in proof_calls):
        raise ValueError('V2 transport proof required')
    kwargs.setdefault('summary_trigger', 12000)
    if kwargs['summary_trigger'] is None:
        kwargs['summary_trigger'] = 12000
    manifest = _original_prepare(out, **kwargs)
    value = _impl.read(out / 'dataset.json')
    manifest.update(version=VERSION, transport_version=transport.VERSION,
                    plan_tool_choice=FORCED_PLAN,
                    start_interval=start_interval,
                    request_id_namespace='context-state-v2-' + out.name,
                    development_protocol_change='Uniformly force ServicePlan for every arm. V1 auto had 19 natural-language plan outputs among 108 checkpoints; no V1 failures are replaced or retried.',
                    pressure_construction={c['id']: c['pressure_construction'] for c in value['cases'] if 'pressure_construction' in c})
    manifest['fairness'] = ('W/P/Q/S use a common raw selection for an identical history input and Q/S identical source quotes. '
                           'In trajectory mode prior actual replies may diverge after p1; matching message IDs alone do not establish identical text. '
                           'Subsequent trajectory differences estimate total policy effects, not isolated direct component effects. '
                           'Use checkpoints or p1 for fixed-history ablation. A/B/R share assembly and differ by recall off/always/conditional. '
                           'R/M differ by budget-triggered summaries; when no summary was generated their difference is not a summary effect. '
                           'C remains an unmatched legacy reference; S/R is a bundled comparison.')
    _impl.write_json(out / 'manifest.json', manifest)
    return manifest


def check(out, *, offline=False):
    manifest = _original_check(out, offline=offline)
    if manifest['version'] != VERSION or manifest.get('transport_version') != transport.VERSION or manifest.get('plan_tool_choice') != FORCED_PLAN:
        raise ValueError('not a frozen V2 protocol')
    if not .25 <= manifest['start_interval'] <= 5:
        raise ValueError('invalid frozen transport interval')
    cases.validate_dataset(_impl.read(out / 'dataset.json'))
    return manifest


def audit(out):
    report = _original_audit(out)
    report['experiment_version'] = VERSION
    report['transport_version'] = transport.VERSION
    report['plan_tool_choice'] = FORCED_PLAN
    _impl.write_json(out / 'audited_report.json', report)
    path = out / 'AUDITED_REPORT.md'
    text = path.read_text(encoding='utf-8')
    text = text.replace('W/P/Q/S原文子集一致；', 'W/P/Q/S仅在相同历史输入下共享原文选择；trajectory回填后需核对历史内容，不能仅凭相同消息ID做组件归因；')
    text += ('\nV2 全部组统一强制 ServicePlan 函数输出；V1 原始失败和评分均未修改。'
             '预声明 long 组只扩展既有背景回复，历史估算至少 15,000；摘要触发默认 12,000。'
             '长背景包含重复材料，属于预算压力构造，不代表真实客服流量分布。\n')
    path.write_text(text, encoding='utf-8')
    return report


_impl.check = check
_impl.audit = audit
workflow = _impl.workflow
summary_plan = _impl.summary_plan
run = _impl.run
writer_lock = _impl.writer_lock


async def preflight(out):
    if out.exists():
        raise ValueError('new preflight directory required')
    out.mkdir(parents=True)
    caller = transport.Caller(out)
    requests = [
        ('coref', [SystemMessage('只补全指代，输出一个完整句子。'), HumanMessage('历史：用户明确选择订单712345。当前：查它的物流。')], None),
        ('plan', [SystemMessage('按用户明确指定的订单生成只读物流查询计划。'), HumanMessage('查询订单712345的物流。')], _impl.ServicePlan),
        ('summary', [SystemMessage('将对话总结到summary字段。'), HumanMessage('用户询问包装是否破损；客服建议保留包装照片，尚未创建任何业务申请。')], _impl.Summary)]
    records = []
    try:
        for stage, messages, schema in requests:
            records.append(await caller.call('preflight-' + stage, 'off', stage, messages, schema))
    finally:
        from app.core.model_transport import close_clients
        await close_clients()
    verified = (all(r['ok'] for r in records) and records[1]['contract']['tool_choice'] == FORCED_PLAN and
                all(a.get('usage', {}).get('output_token_details', {}).get('reasoning') == 0 for r in records for a in r['attempts']))
    report = {'verified': verified, 'chat_model': settings.chat_model, 'chat_base_url': settings.chat_base_url,
              'transport_version': transport.VERSION, 'plan_tool_choice': FORCED_PLAN,
              'status': {r['contract']['stage']: r['status'] for r in records}}
    _impl.write_json(out / 'preflight_report.json', report)
    return report


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True)
    action = p.add_mutually_exclusive_group()
    action.add_argument('--prepare', action='store_true'); action.add_argument('--audit', action='store_true')
    action.add_argument('--preflight', action='store_true')
    p.add_argument('--split', choices=('development', 'reserved'), default='development')
    p.add_argument('--limit', type=int); p.add_argument('--families')
    p.add_argument('--arms', default=','.join(ARMS))
    p.add_argument('--mode', choices=('trajectory', 'checkpoints'), default='trajectory')
    p.add_argument('--budget', type=int, default=9800); p.add_argument('--verified-preflight')
    p.add_argument('--case-concurrency', type=int, default=4); p.add_argument('--caller-concurrency', type=int, default=4)
    p.add_argument('--start-interval', type=float, default=.5, help='Frozen minimum seconds between background requests')
    p.add_argument('--summary-trigger', type=int, default=12000); p.add_argument('--summary-target', type=int, default=1000)
    p.add_argument('--repeat-id')
    args = p.parse_args(); out = Path(args.output).resolve()
    if args.preflight:
        print(json.dumps(asyncio.run(preflight(out)), ensure_ascii=False))
    elif args.prepare:
        manifest = prepare(out, split=args.split, limit=args.limit,
                           families=tuple(args.families.split(',')) if args.families else None,
                           arms=tuple(args.arms.split(',')), mode=args.mode, budget=args.budget,
                           case_concurrency=args.case_concurrency, caller_concurrency=args.caller_concurrency,
                           start_interval=args.start_interval,
                           summary_trigger=args.summary_trigger, summary_target=args.summary_target,
                           repeat_id=args.repeat_id, preflight=args.verified_preflight)
        print(json.dumps({'prepared': True, 'version': manifest['version'], 'dataset_sha256': manifest['dataset_sha256']}, ensure_ascii=False))
    elif args.audit:
        with writer_lock(out):
            print(json.dumps(audit(out), ensure_ascii=False))
    else:
        with writer_lock(out):
            asyncio.run(run(out))


if __name__ == '__main__':
    main()
