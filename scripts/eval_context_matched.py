"""Five-arm, frozen, matched-budget offline context comparison.

C is an unchanged production-context reference, not a matched-budget arm.
W/M/S/R share one estimator and one context cap; actual inputs may differ.
Only synthetic histories and read-only tool fixtures are sent to the provider.
"""
import argparse
import asyncio
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.config import settings
from app.core import summarizer
from app.core.context_matched import CONTEXT_BUDGET, EVIDENCE_BUDGET, packet_cost, raw_packet, state_packet, validate_rows
from app.core.context_policy import estimate
from app.core.model_pool import call_context, get_store, providers
from app.core.prompts import COREF_REWRITE_PROMPT, SUMMARY_PROMPT
from app.core.service_context import ContextBudgetError, ServiceMemory
from app.graph.nodes import _agent_messages, _history_text
from scripts import eval_context_thinking as base
from scripts.context_benchmark_strategies import raw_messages
from scripts.eval_context_policy_v3 import ROOT, Summary, ServicePlan, READER_SYSTEM, REPLY_SYSTEM, config_snapshot, digest, percentile, score, stamp, tool_result, write_json, writer_lock

ARMS = ('C_current', 'W_matched', 'M_matched', 'S_matched', 'R_matched')
LABELS = {'C_current': '当前摘要+3000窗口（现实参照）', 'W_matched': '统一预算原文窗口',
          'M_matched': '统一预算摘要+原文', 'S_matched': '统一预算原文+来源状态',
          'R_matched': '统一预算原文+来源状态+历史回查'}
INPUT_LIMIT = 16000
PURPOSE = {'coref': 'coreference', 'summary': 'conversation_summary', 'plan': 'answer_generation', 'reply': 'answer_generation'}
SOURCES = list(dict.fromkeys([*base.SOURCES, 'app/core/service_context.py',
    'app/core/context_matched.py', 'scripts/eval_context_matched.py',
    'scripts/context_matched_cases.py', 'tests/test_context_matched.py',
    'tests/test_context_matched_cases.py', 'scripts/audit_context_matched_results.py',
    'tests/test_context_matched_audit.py']))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def route_snapshot():
    # Fingerprints are hashes, never provider credentials.
    return sorted(({'base_url': p.base_url, 'model': p.model, 'fallback': p.fallback,
                    'fingerprint': p.fingerprint} for p in providers('chat', settings.chat_model)),
                  key=lambda p: (p['base_url'], p['model'], p['fingerprint']))


def input_estimate(messages, contract):
    value = packet_cost(messages)
    value += sum(estimate(json.dumps(m.tool_calls, ensure_ascii=False))
                 for m in messages if isinstance(m, AIMessage) and m.tool_calls)
    if contract['schema']:
        value += estimate(json.dumps(contract['schema'], ensure_ascii=False))
    return value


def contract_for(mode, stage, messages, schema=None, local_error=None):
    wired, contract = base.wire_contract(mode, stage, messages, schema)
    if local_error:
        contract = {**contract, 'local_context_error': local_error}
    return wired, contract


class Caller(base.Caller):
    async def call(self, key, mode, stage, messages, schema=None, local_error=None):
        wired, contract = contract_for(mode, stage, messages, schema, local_error)
        approx = input_estimate(wired, contract)
        path = self.out / 'calls' / (key + '.json')
        if local_error or approx > INPUT_LIMIT:
            result = {'contract': contract, 'input_sha256': digest(contract), 'attempts': [],
                      'status': 'budget_blocked', 'ok': False, 'value': None, 'finish_reason': None,
                      'seconds': 0, 'queue_seconds': 0, 'approx_input_tokens': approx, 'calls': [],
                      'local_error': local_error or 'complete input estimate exceeds 16000'}
            if path.exists():
                if read(path) != result:
                    raise ValueError('blocked call changed ' + key)
            else:
                write_json(path, result)
            return result
        with call_context(purpose=PURPOSE[stage]):
            result = await super().call(key, mode, stage, messages, schema)
        # The inherited caller uses the same content/schema counter. Include the
        # actual tool-call arguments as well, and persist that richer estimate.
        if result['approx_input_tokens'] != approx:
            result = {**result, 'approx_input_tokens': approx}
            write_json(path, result)
        return result


def workflow(case, arm):
    """Replayable causal generator; assembly receives only the visible prefix."""
    if arm not in ARMS:
        raise ValueError('unknown arm')
    cid = case['id']
    summary, boundary, ingested, events = '', 0, 0, []
    memory = ServiceMemory(cid)
    for probe in case['probes']:
        rows = case['history'][:2 * probe['at_turn']]
        validate_rows(rows)
        if arm in ('S_matched', 'R_matched'):
            memory.advance(rows, cid)
        if arm in ('C_current', 'M_matched'):
            for end in range(ingested + 2, len(rows) + 1, 2):
                if end - boundary < settings.summary_trigger_messages:
                    continue
                objects = [SimpleNamespace(id=i, role=r['role']) for i, r in enumerate(rows[:end], 1)]
                new = summarizer.compute_boundary(objects, settings.context_window_turns)
                if not new or new <= boundary:
                    continue
                dialog = '\n'.join(('用户:' if r['role'] == 'user' else '客服:') + r['content'] for r in rows[boundary:new])
                key = f'{cid}-{arm}-summary-{end}'
                record = yield ('call', key, 'off', 'summary', SUMMARY_PROMPT.format_messages(old_summary=summary or '(无)', dialog=dialog), Summary)
                events.append({'at_message': end, 'old_boundary': boundary, 'new_boundary': new,
                               'key': key, 'status': record['status']})
                if record['ok']:
                    summary, boundary = record['value']['summary'], new
        ingested = len(rows)
        local_error = None
        try:
            if arm == 'C_current':
                state = {'messages': raw_messages(rows) + [HumanMessage(probe['query'], id=f'db-{len(rows)+1}')],
                         'summary': summary, 'summary_upto_msg_id': boundary}
                context = {'body': _agent_messages(state)[1:], 'coref_history': _history_text(state),
                           'summary': summary, 'boundary': boundary}
            elif arm in ('W_matched', 'M_matched'):
                context = raw_packet(rows, probe['query'], summary=summary, boundary=boundary)
            else:
                context = state_packet(memory, probe['query'], recall=arm == 'R_matched')
        except ContextBudgetError:
            local_error = 'essential context exceeds 9800; no truncation or provider call'
            context = {'body': raw_messages(rows) + [HumanMessage(probe['query'])],
                       'coref_history': '\n'.join(r['role'] + ':' + r['content'] for r in rows),
                       'assembly_error': local_error, 'budget': CONTEXT_BUDGET}
        key = f'{cid}-{probe["id"]}-{arm}'
        cr = yield ('call', key + '-coref', 'off', 'coref', COREF_REWRITE_PROMPT.format_messages(
            query=probe['query'], history=context['coref_history'] or '(无)'), None, local_error)
        instruction = HumanMessage('本轮原话：' + probe['query'] + '\n指代改写结果（需用历史核实）：' + (cr['value'] if cr['ok'] else probe['query']))
        plan = yield ('call', key + '-plan', 'off', 'plan', [SystemMessage(READER_SYSTEM), *context['body'], instruction], ServicePlan, local_error)
        tool = tool_result(case, plan['value']) if plan['ok'] else None
        prompt = [SystemMessage(REPLY_SYSTEM), *context['body'], instruction]
        if tool is not None:
            p = plan['value']
            name = 'query_logistics' if p['action'] == 'logistics' else 'query_faq'
            args = {'order_id': p['order_id']} if p['action'] == 'logistics' else {'query': p['sku'] + ' 清洁方法'}
            prompt += [AIMessage('', tool_calls=[{'id': 'eval-readonly', 'name': name, 'args': args, 'type': 'tool_call'}]),
                       ToolMessage(json.dumps(tool, ensure_ascii=False), tool_call_id='eval-readonly')]
        else:
            prompt.append(HumanMessage('本轮计划：' + json.dumps(plan['value'], ensure_ascii=False) + '\n请核实历史后作答。'))
        reply = yield ('call', key + '-reply', 'off', 'reply', prompt, None, local_error)
        result = {'case_id': cid, 'cohort': case['cohort'], 'family': case['kind'],
                  'probe_id': probe['id'], 'at_turn': probe['at_turn'], 'coref': cr, 'plan': plan, 'reply': reply,
                  'tool_result': tool, 'body': [m.model_dump(mode='json') for m in context['body']],
                  'context_text': '\n'.join(m.content for m in context['body']),
                  'context': {k: v for k, v in context.items() if k != 'body'},
                  'state': memory.snapshot() if arm in ('S_matched', 'R_matched') else None}
        result['context_sha256'] = digest(result['body'])
        result['score'] = score(case, probe, result)
        yield ('save', 'results', key, result)
    yield ('save', 'maintenance', f'{cid}-{arm}', events)


def validate_dataset(data):
    ids = set()
    for case in data['cases']:
        if case['id'] in ids:
            raise ValueError('duplicate case')
        ids.add(case['id'])
        validate_rows(case['history'])
        if case['cohort'] not in ('regression', 'transfer') or not case['kind']:
            raise ValueError('cohort/family required')
        positions = [p['at_turn'] for p in case['probes']]
        if len(positions) != 2 or positions != sorted(set(positions)) or positions[0] < 1 or positions[-1] * 2 > len(case['history']):
            raise ValueError('two ordered causal probes required')


def prepare(out, dataset_path, preflight_path):
    if out.exists():
        raise ValueError('new output directory required')
    if not dataset_path:
        raise ValueError('--dataset required for preparation')
    routes = route_snapshot()
    if not routes or settings.chat_failover_enabled is not False or any(p['fallback'] or p['model'] != settings.chat_model or p['base_url'].rstrip('/') != settings.chat_base_url.rstrip('/') for p in routes):
        raise ValueError('fixed model without failover required')
    if settings.summary_model and settings.summary_model != settings.chat_model:
        raise ValueError('same summary model required')
    if not preflight_path:
        raise ValueError('--verified-preflight required')
    proof_dir = Path(preflight_path)
    proof = {'report': read(proof_dir / 'preflight_report.json'),
             'calls': [read(p) for p in sorted((proof_dir / 'calls').glob('*.json'))]}
    if not proof['report'].get('verified') or len(proof['calls']) != 3:
        raise ValueError('successful synthetic preflight required')
    if proof['report'].get('chat_base_url') != settings.chat_base_url or proof['report'].get('chat_model') != settings.chat_model:
        raise ValueError('preflight endpoint/model mismatch')
    if {r['contract']['stage'] for r in proof['calls']} != {'coref', 'plan', 'summary'}:
        raise ValueError('preflight stages changed')
    for r in proof['calls']:
        if not r['ok'] or r['contract']['model'] != settings.chat_model or r['contract']['extra_body'] != base.mode_body('off'):
            raise ValueError('preflight protocol mismatch')
        if any(a.get('usage', {}).get('output_token_details', {}).get('reasoning') != 0 for a in r['attempts']):
            raise ValueError('preflight nonthinking not verified')
    data = read(Path(dataset_path)); validate_dataset(data)
    sources = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    out.mkdir(parents=True)
    write_json(out / 'dataset.json', data)
    write_json(out / 'preflight_evidence.json', proof)
    for name in sources:
        target = out / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    write_json(out / 'manifest.json', {'created_at': stamp(), 'version': 'context-matched-v1',
        'sources': sources, 'dataset_sha256': digest(data), 'config': config_snapshot(),
        'routes': routes,
        'preflight_sha256': digest(proof),
        'pool_policy': get_store().policy().model_dump(), 'arms': list(ARMS),
        'context_budget': CONTEXT_BUDGET, 'evidence_budget_R': EVIDENCE_BUDGET,
        'input_limit_estimate': INPUT_LIMIT, 'max_output': base.MAX_OUTPUT,
        'mode_body': base.mode_body('off'), 'concurrency': base.CONCURRENCY, 'start_interval': base.INTERVAL,
        'temperatures': {'coref': .3, 'summary': .3, 'plan': 0, 'reply': 0},
        'packages': {p: importlib.metadata.version(p) for p in ('langchain-core', 'langchain-openai', 'openai', 'pydantic', 'httpx')},
        'scope': 'Frozen synthetic regression and transfer variants; two correlated probes; common read-only reader, not full production RAG/SSE/refund execution or blind generalization.',
        'fairness': 'W/M/S/R share estimator, total context cap and stage visibility. Equal ceilings are not equal actual inputs. C retains production window and coref view, only a reality reference. S/R preserve the same narrow extractor and last-two-controls rules; only optional ranked retrieval differs. Tool projection is disabled in all arms.',
        'summary': 'C/M independently regenerate summaries with identical prompt and scheduling; stochastic summaries are not shared or output-matched. All maintenance is charged to its arm.',
        'cost': 'All historical maintenance and two probes; excludes generating provided historical turns. API waiting excludes queue, pacing, retries backoff, background maintenance; not online TTFT.',
        'retry': 'One additional attempt for retryable transport errors; no semantic/format retries. Existing saved calls reused only within this exact run.'})


def check(out):
    manifest = read(out / 'manifest.json')
    if manifest['config'] != config_snapshot() or manifest['pool_policy'] != get_store().policy().model_dump():
        raise ValueError('configuration changed')
    if manifest['routes'] != route_snapshot():
        raise ValueError('provider routes changed')
    if manifest['dataset_sha256'] != digest(read(out / 'dataset.json')):
        raise ValueError('dataset changed')
    if manifest['preflight_sha256'] != digest(read(out / 'preflight_evidence.json')):
        raise ValueError('preflight evidence changed')
    constants = {'context_budget': CONTEXT_BUDGET, 'evidence_budget_R': EVIDENCE_BUDGET,
                 'input_limit_estimate': INPUT_LIMIT, 'max_output': base.MAX_OUTPUT,
                 'mode_body': base.mode_body('off'), 'concurrency': base.CONCURRENCY, 'start_interval': base.INTERVAL}
    if any(manifest[k] != value for k, value in constants.items()) or manifest['arms'] != list(ARMS):
        raise ValueError('experiment contract changed')
    for name, sha in manifest['sources'].items():
        for path in (ROOT / name, out / 'source_snapshot' / name):
            if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                raise ValueError('source changed ' + name)


async def run(out):
    check(out); caller = Caller(out); case_gate = asyncio.Semaphore(4)
    async def one(case):
        async with case_gate:
            arms = list(ARMS); random.Random('matched-' + case['id']).shuffle(arms)
            async def one_arm(arm):
                generator = workflow(case, arm); value = None
                try:
                    while True:
                        item = generator.send(value); value = None
                        if item[0] == 'call':
                            value = await caller.call(*item[1:])
                        else:
                            _, folder, key, result = item
                            write_json(out / folder / (key + '.json'), result)
                            if folder == 'results':
                                print(key, 'task', result['score']['task_rule_v3'], 'target', result['score']['target_pass'], flush=True)
                except StopIteration:
                    pass
            await asyncio.gather(*(one_arm(arm) for arm in arms))
    cases = list(read(out / 'dataset.json')['cases']); random.Random(20261026).shuffle(cases)
    try:
        await asyncio.gather(*(one(case) for case in cases))
    finally:
        from app.core.model_transport import close_clients
        await close_clients()
    report = audit(out)
    print('FINISHED', json.dumps({k: report[k] for k in ('complete', 'audit_passed', 'nonthinking_verified', 'missing', 'issues')}, ensure_ascii=False), flush=True)


def metrics(entries, records):
    attempts = [a for r in records for a in r['attempts']]
    latencies = [sum(r[s]['seconds'] for s in ('coref', 'plan', 'reply')) for r in entries]
    summaries = [r for r in records if r['contract']['stage'] == 'summary']
    summary_attempts = [a for r in summaries for a in r['attempts']]
    usage = lambda aa: {k: sum(a.get('usage', {}).get(k, 0) or 0 for a in aa) for k in ('input_tokens', 'output_tokens', 'total_tokens')}
    return {'n': len(entries), 'passed': sum(r['score']['task_rule_v3'] for r in entries),
        'legacy_passed': sum(r['score']['task_pass'] for r in entries),
        'target_passed': sum(r['score']['target_pass'] for r in entries),
        'coref_passed': sum(r['score']['coref_pass'] for r in entries), 'usage': usage(attempts),
        'probe_usage': usage([a for r in records if r['contract']['stage'] != 'summary' for a in r['attempts']]),
        'reasoning_tokens': sum(a.get('usage', {}).get('output_token_details', {}).get('reasoning', 0) or 0 for a in attempts),
        'p50_api_seconds': percentile(latencies, .5), 'p95_api_seconds': percentile(latencies, .95),
        'transport_failed_probes': sum(r['score']['transport_failure'] for r in entries),
        'parse_failed_probes': sum(any(r[s]['status'] == 'parse_error' for s in ('coref', 'plan', 'reply')) for r in entries),
        'budget_blocked_probes': sum(any(r[s]['status'] == 'budget_blocked' for s in ('coref', 'plan', 'reply')) for r in entries),
        'truncated_probes': sum(r['score']['truncated'] for r in entries),
        'missing_usage_attempts': sum(not a.get('usage') for a in attempts),
        'successful_attempts_without_reasoning_usage': sum(a.get('transport_ok') and 'reasoning' not in a.get('usage', {}).get('output_token_details', {}) for a in attempts),
        'call_status': dict(Counter(r['status'] for r in records)),
        'attempt_status': dict(Counter(a['status'] for a in attempts)),
        'models': dict(Counter(a.get('model') for a in attempts)),
        'summary': {'calls': len(summaries), 'failed_calls': sum(not r['ok'] for r in summaries),
                    'status': dict(Counter(r['status'] for r in summaries)),
                    'input': usage(summary_attempts)['input_tokens'], 'output': usage(summary_attempts)['output_tokens'],
                    'usage': usage(summary_attempts)},
        'stages': {stage: {'calls': sum(r['contract']['stage'] == stage for r in records),
            'status': dict(Counter(r['status'] for r in records if r['contract']['stage'] == stage)),
            'usage': usage([a for r in records if r['contract']['stage'] == stage for a in r['attempts']]),
            'p50_api_seconds': percentile([r['seconds'] for r in records if r['contract']['stage'] == stage], .5),
            'p95_api_seconds': percentile([r['seconds'] for r in records if r['contract']['stage'] == stage], .95)}
            for stage in ('coref', 'plan', 'reply', 'summary')}}


def audit(out):
    check(out)
    cases = read(out / 'dataset.json')['cases']; issues, missing = [], []
    expected_calls, expected_attempts = set(), set()
    expected_files = {'results': set(), 'maintenance': set()}
    entries = {a: [] for a in ARMS}; records = {a: [] for a in ARMS}
    for case in cases:
        for arm in ARMS:
            generator = workflow(case, arm); value = None
            try:
                while True:
                    item = generator.send(value); value = None
                    if item[0] == 'call':
                        _, key, mode, stage, messages, schema, *rest = item
                        expected_calls.add(key)
                        path = out / 'calls' / (key + '.json')
                        if not path.exists():
                            missing.append(key); break
                        r = read(path)
                        local_error = rest[0] if rest else None
                        wired, contract = contract_for(mode, stage, messages, schema, local_error)
                        approx = input_estimate(wired, contract)
                        if r['contract'] != contract or r['input_sha256'] != digest(contract):
                            issues.append('causal contract ' + key)
                        if r['approx_input_tokens'] != approx:
                            issues.append('input estimate ' + key)
                        for i, attempt in enumerate(r['attempts'], 1):
                            attempt_key = key + f'-{i}'; expected_attempts.add(attempt_key)
                            ap = out / 'attempts' / (attempt_key + '.json')
                            if not ap.exists():
                                missing.append(attempt_key)
                            elif read(ap) != attempt or attempt['input_sha256'] != digest(contract):
                                issues.append('attempt receipt ' + key)
                            if i > 1 and not r['attempts'][i-2].get('retryable'):
                                issues.append('semantic retry ' + key)
                        blocked = bool(local_error) or approx > INPUT_LIMIT
                        if blocked != (r['status'] == 'budget_blocked'):
                            issues.append('budget policy ' + key)
                        if blocked:
                            if r['attempts'] or r['ok']:
                                issues.append('blocked provider call ' + key)
                        elif not r['attempts'] or r['value'] != r['attempts'][-1]['value']:
                            issues.append('final receipt ' + key)
                        elif r['status'] != r['attempts'][-1]['status'] or r['ok'] != (r['status'] == 'ok'):
                            issues.append('status receipt ' + key)
                        records[arm].append((case, r)); value = r
                    else:
                        _, folder, key, result = item
                        expected_files[folder].add(key)
                        path = out / folder / (key + '.json')
                        if not path.exists():
                            missing.append(key); continue
                        saved = read(path)
                        if saved != result:
                            issues.append('causal replay ' + key)
                        if folder == 'results':
                            entries[arm].append((case, result))
                            if arm != 'C_current' and not result['context'].get('assembly_error') and sum(estimate(m['content']) for m in result['body']) > CONTEXT_BUDGET:
                                issues.append('context budget ' + key)
                            if result['state']:
                                for e in result['state']['evidence']:
                                    if e['seq'] > 2 * result['at_turn'] or case['history'][e['seq']-1]['content'][e['start']:e['end']] != e['text']:
                                        issues.append('source span ' + key)
            except StopIteration:
                pass
    # Incomplete causal runs cannot enumerate later summary keys; existing files
    # are still reported as unexpected rather than used to fill missing results.
    if expected_calls != {p.stem for p in (out / 'calls').glob('*.json')}:
        issues.append('unexpected call set')
    if expected_attempts != {p.stem for p in (out / 'attempts').glob('*.json')}:
        issues.append('unexpected attempt set')
    for folder, expected in expected_files.items():
        if expected != {p.stem for p in (out / folder).glob('*.json')}:
            issues.append('unexpected ' + folder + ' set')
    strategies = {}
    for arm in ARMS:
        strategies[arm] = {}
        for cohort in ('all', 'regression', 'transfer'):
            strategies[arm][cohort] = metrics([r for c, r in entries[arm] if cohort == 'all' or c['cohort'] == cohort],
                [r for c, r in records[arm] if cohort == 'all' or c['cohort'] == cohort])
        strategies[arm]['families'] = {kind: metrics([r for c, r in entries[arm] if c['kind'] == kind],
            [r for c, r in records[arm] if c['kind'] == kind]) for kind in sorted({c['kind'] for c in cases})}
    complete = not missing and all(len(entries[a]) == sum(len(c['probes']) for c in cases) for a in ARMS)
    report = {'complete': complete, 'audit_passed': complete and not issues, 'issues': issues, 'missing': missing,
        'nonthinking_verified': complete and all(strategies[a]['all']['call_status'].get('ok', 0) > 0 and strategies[a]['all']['reasoning_tokens'] == 0 and
            strategies[a]['all']['successful_attempts_without_reasoning_usage'] == 0 for a in ARMS),
        'strategies': strategies, 'counts': {'cases': len(cases), 'results': sum(map(len, entries.values())),
            'calls': len(expected_calls), 'attempts': len(expected_attempts), 'maintenance': len(expected_files['maintenance'])},
        'scope': 'Equal context ceilings and estimator for W/M/S/R, not equal observed tokens. C is reference only. Regression and predeclared transfer are not blind or production accuracy.'}
    write_json(out / 'audited_report.json', report)
    lines = ['# 统一预算上下文对照', '', f'记录完整：{complete}；离线因果回放审计：{report["audit_passed"]}；非思考用量验证：{report["nonthinking_verified"]}。', '']
    for cohort, label in (('regression', '已知回归'), ('transfer', '预冻结表达迁移'), ('all', '合计')):
        lines += ['## ' + label, '', '|方案|任务规则通过|对象/动作正确|输入含维护|输出含维护|摘要失败/调用|接口失败追问|解析失败追问|预算阻断追问|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for arm in ARMS:
            s = strategies[arm][cohort]
            lines.append(f'|{LABELS[arm]}|{s["passed"]}/{s["n"]}|{s["target_passed"]}/{s["n"]}|{s["usage"]["input_tokens"]:,}|{s["usage"]["output_tokens"]:,}|{s["summary"]["failed_calls"]}/{s["summary"]["calls"]}|{s["transport_failed_probes"]}|{s["parse_failed_probes"]}|{s["budget_blocked_probes"]}|')
        lines.append('')
    lines += ['W/M/S/R 对话包同为9800估算Token，含原文、状态/摘要及问题；完整请求含系统词、工具定义及参数，共用16000估算上限、8192输出上限。额度相同不代表实际用量相同。',
        'C保留当前3000窗口与6条消息指代视图，仅作现实参照，不用于同预算组件因果归因。C/M摘要独立生成，有采样差异。',
        'S/R复用同一窄规则提取器及最近两条限制；R仅额外开放按原有词项规则回查。没有偷偷新增泛化规则或保护全部有效约束。',
        '只读公共评测reader；未验证线上RAG、SSE或退款写操作。迁移样本预先冻结，但不是完全盲测；每条会话两次追问相关。',
        '原规则评分仍含已知否定引用误判，保留legacy与v3，不能称为人工语义准确率。完整/审计通过仅表示记录齐全和复算一致，不表示模型全部成功。',
        'Token包含历史维护与两次追问，不含给定历史逐轮生成。缺失usage的调用用量未知，不按零成本解释。API等待不含排队、节流、退避、后台维护，不是线上首字延迟。']
    (out / 'AUDITED_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True); parser.add_argument('--dataset'); parser.add_argument('--verified-preflight')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare', action='store_true'); mode.add_argument('--audit', action='store_true')
    args = parser.parse_args(); output = Path(args.output).resolve()
    if args.prepare:
        prepare(output, args.dataset, args.verified_preflight)
    elif args.audit:
        with writer_lock(output):
            print(json.dumps(audit(output), ensure_ascii=False))
    else:
        with writer_lock(output):
            asyncio.run(run(output))
