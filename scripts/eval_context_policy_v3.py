"""Frozen four-arm regression experiment. No production graph/database mutation.

Prepare -> test -> run -> offline audit. Cached endpoint attempts are immutable;
only transport exceptions get one immediate retry. Empty/truncated/wrong answers
are retained. Historical user/assistant turns are supplied, not generated.
"""
import argparse
import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import random
import re
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from pydantic import BaseModel
import httpx
from openai import APIConnectionError, APIStatusError

from app.config import settings
from app.core import llm, summarizer
from app.core.context_policy import EvidenceState, assemble, estimate
from app.core.model_pool import call_context, get_store, providers, PoolBusy
from app.core.prompts import COREF_REWRITE_PROMPT, SUMMARY_PROMPT
from app.graph.nodes import _agent_messages, _history_text
from scripts.context_benchmark_strategies import raw_messages
from scripts.eval_context_comparison import (ServicePlan, READER_SYSTEM, REPLY_SYSTEM,
                                            score_probe, tool_result, write_json)

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('C_current', 'W_raw8k', 'S_state8k', 'R_recall8k')
LABELS = {'C_current': '当前摘要+3000近似窗口', 'W_raw8k': '8000估算原文窗口',
          'S_state8k': '原文窗口+来源状态', 'R_recall8k': '原文窗口+来源状态+按需回查'}
SOURCES = ['scripts/eval_context_policy_v3.py', 'app/core/context_policy.py',
           'scripts/eval_context_comparison.py', 'scripts/context_benchmark_strategies.py',
           'app/core/llm.py', 'app/core/model_pool.py', 'app/core/model_transport.py',
           'app/core/prompts.py', 'app/core/memory.py', 'app/core/summarizer.py',
           'app/graph/nodes.py', 'app/config.py']
OLD_DATA = ROOT / 'data/ch07/context-comparison/20261016-gitee-v1/dataset.json'
LIMITS = {'coref': 512, 'plan': 1024, 'reply': 1536, 'summary': 1024}
EXTRA_BODY = {'thinking': {'type': 'disabled'}, 'enable_thinking': False}
CONCURRENCY = 8


class Summary(BaseModel):
    summary: str


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def config_snapshot():
    keys = ('chat_model', 'chat_base_url', 'chat_thinking', 'chat_reasoning_effort', 'summary_model',
            'chat_failover_enabled', 'summary_trigger_messages', 'context_window_turns',
            'context_window_max_tokens')
    return {k: getattr(settings, k) for k in keys}


def prepare(out):
    if (out / 'manifest.json').exists():
        raise ValueError('existing run: use --run/--audit, never overwrite manifest')
    dataset = json.loads(OLD_DATA.read_text(encoding='utf-8'))
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'dataset.json', dataset)
    sources = {}
    for name in SOURCES:
        data = (ROOT / name).read_bytes()
        dest = out / 'source_snapshot' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        sources[name] = hashlib.sha256(data).hexdigest()
    policy = get_store().policy().model_dump()
    routes = [{'host': p.base_url, 'model': p.model, 'fingerprint': p.fingerprint,
               'fallback': p.fallback} for p in providers('chat', settings.chat_model)]
    if any(p['fallback'] or p['model'] != settings.chat_model for p in routes):
        raise ValueError('fixed-model experiment requires no configured fallback')
    write_json(out / 'manifest.json', {
        'created_at': stamp(), 'version': 'context-policy-v3', 'arms': ARMS,
        'dataset_sha256': digest(dataset), 'sources': sources, 'config': config_snapshot(),
        'pool_policy': policy, 'routes': routes, 'limits': LIMITS, 'concurrency': CONCURRENCY,
        'timeout_seconds': 100, 'temperatures': {'coref': .3, 'summary': .3, 'plan': 0, 'reply': 0},
        'scoring': 'legacy strict plus explicit obsolete-reference exception, frozen before calls',
        'scope': '48 known synthetic regression histories; 8 families; 2 causal probes; not holdout/production',
        'cost_scope': 'all chronological memory maintenance and two probes, not generation of every history turn',
        'baseline': 'production summary prompt/schedule/assembler; uniform output caps and endpoint runner in every arm',
        'protocol': 'tool_choice=auto plus schema validation; format instruction merged into initial system message; enable_thinking=false; both earlier diagnostic stages archived',
        'extra_body': EXTRA_BODY,
        'prior_diagnostic': ['data/ch07/context-policy/20261023-gitee-v3/STAGE_STATUS.json', 'data/ch07/context-policy/20261023-gitee-v3-nonthinking/STAGE_STATUS.json'],
        'state': 'deterministic source ledger and narrow literal candidate slots; no model state extraction or action authorization',
        'budgets': 'W/S/R window 8000 heuristic tokens; state and recall each 1800; whole prompt plus schema must fit 16000 heuristic tokens',
        'retry': 'one immediate retry for transport exceptions only; retain every attempt; no semantic retries'})


@contextmanager
def writer_lock(out):
    import msvcrt
    path = out / 'writer.lock'
    with path.open('a+b') as f:
        if f.tell() == 0:
            f.write(b'0'); f.flush()
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def sanitized_reply(text):
    """Allow explicit obsolete/unused references, not arbitrary negative clauses.

    Does not change the frozen legacy score. The separate relaxed metric is still
    a rule metric, not human semantic accuracy. Return the exceptions for review.
    """
    removed = []
    number = r'(?:\d{4}-\d{2}-\d{2}|\d{4,})'
    patterns = [rf'(?:作废|失效)(?:的)?(?:订单|号码|日期|订单号)?\s*({number})',
                rf'({number})(?:的)?(?:号码|订单|日期)?\s*(?:已作废|已失效|未使用|不再使用|不用于查询)',
                rf'(?:不要使用|不使用|未使用)(?:订单|号码|日期|订单号)?\s*({number})']
    cleaned = text
    for pattern in patterns:
        def replace(m):
            removed.append(m.group(1))
            return m.group().replace(m.group(1), '[明确作废值]')
        cleaned = re.sub(pattern, replace, cleaned)
    return cleaned, removed


def score(case, probe, result):
    legacy = score_probe(case, probe, result)
    reply = result['reply'].get('value') or ''
    cleaned, excluded = sanitized_reply(reply)
    adjusted = {**result, 'reply': {**result['reply'], 'value': cleaned}}
    revised = score_probe(case, probe, adjusted)
    # Removing an explicitly obsolete occurrence must not erase a correct current
    # identifier elsewhere. Strict success stays success.
    new_pass = legacy['task_pass'] or revised['task_pass']
    stage_status = {s: result[s].get('status') for s in ('coref', 'plan', 'reply')}
    return {**legacy, 'task_rule_v3': bool(new_pass), 'excluded_obsolete_mentions': excluded,
            'empty_reply': not reply.strip(), 'stage_status': stage_status,
            'transport_failure': any(result[s].get('status') == 'transport_error' for s in stage_status),
            'truncated': any(result[s].get('finish_reason') == 'length' for s in stage_status)}


class Caller:
    def __init__(self, out):
        self.out = out
        self.sem = asyncio.Semaphore(CONCURRENCY)

    async def call(self, key, stage, messages, schema=None, model=None):
        if schema:
            instruction = '请调用 '+schema.__name__+' 工具返回结构化结果，不要用正文代替工具参数。'
            if not messages or not isinstance(messages[0], SystemMessage):
                raise ValueError('structured stage requires initial system message')
            messages = [SystemMessage(messages[0].content+'\n'+instruction), *messages[1:]]
        wire = [m.model_dump(mode='json') for m in messages]
        contract = {'messages': wire, 'stage': stage, 'model': model or settings.chat_model,
                    'max_tokens': LIMITS[stage], 'schema': schema.model_json_schema() if schema else None,
                    'tool_choice': 'auto' if schema else None, 'extra_body': EXTRA_BODY}
        sha = digest(contract)
        path = self.out / 'calls' / (key + '.json')
        if path.exists():
            old = json.loads(path.read_text(encoding='utf-8'))
            if old['input_sha256'] != sha:
                raise ValueError('cached call contract changed: ' + key)
            return old
        approximate = sum(estimate(m.content) for m in messages) + (estimate(json.dumps(contract['schema'],ensure_ascii=False)) if schema else 0)
        if stage != 'summary' and approximate > 16000:
            raise ValueError('whole input estimate exceeds frozen application budget')
        queued = time.monotonic()
        async with self.sem:
            queue_seconds = time.monotonic()-queued
            attempts = []
            for attempt in range(2):
                attempt_path = self.out/'attempts'/(key+f'-{attempt+1}.json')
                if attempt_path.exists():
                    saved=json.loads(attempt_path.read_text(encoding='utf-8'))
                    if saved.get('input_sha256')!=sha:raise ValueError('attempt input changed: '+key)
                    attempts.append(saved)
                    if saved.get('retryable') and attempt==0:continue
                    break
                started = time.monotonic()
                aid = f'ctxv3-{self.out.name}-{key}-{attempt}'
                base = {'attempt': attempt+1, 'request_id': aid, 'started_at': stamp(), 'input_sha256': sha}
                try:
                    with call_context(request_id=aid, priority='background'):
                        m = llm.get_chat_model(streaming=True, model=model, temperature=.3 if stage in ('coref','summary') else 0)
                        m.max_tokens = LIMITS[stage]
                        m.extra_body = {**(getattr(m,'extra_body',None) or {}), **EXTRA_BODY}
                        chain = m.bind_tools([schema],tool_choice='auto') if schema else m
                        raw = await asyncio.wait_for(chain.ainvoke(messages), 100)
                    parsed = None
                    if schema:
                        matches=[t for t in raw.tool_calls if t['name']==schema.__name__]
                        if len(matches)==1:
                            try: parsed=schema.model_validate(matches[0]['args'])
                            except ValueError: pass
                    value = parsed.model_dump() if parsed is not None else (raw.content if not schema else None)
                    finish = raw.response_metadata.get('finish_reason')
                    status = 'ok'
                    if finish == 'length': status = 'truncated'
                    elif schema and value is None: status = 'parse_error'
                    elif not schema and (not isinstance(value,str) or not value.strip()): status = 'empty'
                    elif stage == 'summary' and value is not None and not value['summary'].strip(): status = 'empty'
                    record = {**base, 'status': status, 'transport_ok': True, 'value': value,
                              'finish_reason': finish, 'model': raw.response_metadata.get('model_name'),
                              'usage': raw.usage_metadata or {}, 'tool_calls': raw.tool_calls,
                              'raw_content': raw.content, 'seconds': round(time.monotonic()-started,4)}
                    attempts.append(record)
                    write_json(self.out/'attempts'/(key+f'-{attempt+1}.json'), record)
                    break  # Never retry empty, truncated, invalid JSON or a wrong answer.
                except Exception as exc:
                    is_transport = isinstance(exc, (APIConnectionError, APIStatusError, httpx.TransportError, TimeoutError, PoolBusy))
                    retryable = isinstance(exc, (APIConnectionError, httpx.TransportError, TimeoutError, PoolBusy)) or (isinstance(exc,APIStatusError) and (exc.status_code in (408,429) or exc.status_code >= 500))
                    record = {**base, 'status': 'transport_error' if is_transport else 'validation_error', 'transport_ok': False,
                              'value': None, 'error_type': type(exc).__name__, 'usage': {},
                              'retryable': retryable,
                              'seconds': round(time.monotonic()-started,4)}
                    attempts.append(record)
                    write_json(self.out/'attempts'/(key+f'-{attempt+1}.json'), record)
                    if attempt == 0 and retryable:
                        await asyncio.sleep(1)
                    else:
                        break
            final = attempts[-1]
            result = {'input_sha256': sha, 'contract': contract, 'attempts': attempts,
                      'status': final['status'], 'ok': final['status'] == 'ok', 'value': final['value'],
                      'finish_reason': final.get('finish_reason'), 'seconds': sum(a['seconds'] for a in attempts),
                      'queue_seconds': round(queue_seconds,4), 'approx_input_tokens': approximate,
                      'calls': [{'input_tokens': a['usage'].get('input_tokens'),
                                 'output_tokens': a['usage'].get('output_tokens')} for a in attempts if a.get('transport_ok')]}
            write_json(path, result)
            return result


async def evaluate_case(case, out, caller):
    cid = case['id']
    summary, boundary, ingested = '', 0, 0
    state = EvidenceState()
    maintenance = []
    probes_out = {arm: [] for arm in ARMS}
    order = list(ARMS)
    random.Random(cid).shuffle(order)
    for probe in case['probes']:
        rows = case['history'][:2*probe['at_turn']]
        t0 = time.perf_counter()
        state.advance(rows)
        state_cpu = time.perf_counter()-t0
        # Current baseline: same rolling boundaries, original prompt, bounded output.
        for end in range(ingested+2, len(rows)+1, 2):
            if end-boundary < settings.summary_trigger_messages:
                continue
            objects = [SimpleNamespace(id=i,role=r['role']) for i,r in enumerate(rows[:end],1)]
            new_boundary = summarizer.compute_boundary(objects, settings.context_window_turns)
            if not new_boundary or new_boundary <= boundary: continue
            dialog = '\n'.join(f"{'用户' if r['role']=='user' else '客服'}:{r['content']}" for r in rows[boundary:new_boundary])
            messages = SUMMARY_PROMPT.format_messages(old_summary=summary or '(无)', dialog=dialog)
            record = await caller.call(f'{cid}-summary-{end}', 'summary', messages, Summary,
                                       model=settings.summary_model or None)
            maintenance.append({'at_message': end, 'old_boundary': boundary, 'new_boundary': new_boundary,
                                'call_key': f'{cid}-summary-{end}', 'status': record['status']})
            if record['ok'] and record['value']['summary'].strip():
                summary, boundary = record['value']['summary'], new_boundary
        ingested = len(rows)
        write_json(out/'states'/f'{cid}-{probe["id"]}.json', state.snapshot())
        for arm in order:
            key = f'{cid}-{probe["id"]}-{arm}'
            t0 = time.perf_counter()
            if arm == 'C_current':
                st = {'messages': raw_messages(rows)+[HumanMessage(probe['query'],id=f'db-{len(rows)+1}')],
                      'summary': summary, 'summary_upto_msg_id': boundary}
                context = {'body': _agent_messages(st)[1:], 'coref_history': _history_text(st),
                           'retrieved': [], 'source_ids': [], 'summary': summary, 'boundary': boundary}
            else:
                context = assemble(rows, probe['query'], state if arm!='W_raw8k' else None, arm=='R_recall8k')
            cpu_seconds = time.perf_counter()-t0 + (state_cpu if arm in ('S_state8k','R_recall8k') else 0)
            start = time.monotonic()
            rewrite = await caller.call(key+'-coref','coref', COREF_REWRITE_PROMPT.format_messages(query=probe['query'],history=context['coref_history'] or '(无)'))
            effective = rewrite['value'] if rewrite['ok'] else probe['query']
            instructions = HumanMessage('本轮原话：'+probe['query']+'\n指代改写结果（需用历史核实）：'+effective)
            plan = await caller.call(key+'-plan','plan',[SystemMessage(READER_SYSTEM),*context['body'],instructions],ServicePlan)
            data = tool_result(case,plan['value']) if plan['ok'] else None
            answer_prompt = [SystemMessage(REPLY_SYSTEM),*context['body'],instructions]
            if data is not None:
                p = plan['value']
                name = 'query_logistics' if p['action']=='logistics' else 'query_faq'
                args = {'order_id':p['order_id']} if p['action']=='logistics' else {'query':p['sku']+' 清洁方法'}
                answer_prompt += [AIMessage('',tool_calls=[{'id':'eval-readonly','name':name,'args':args,'type':'tool_call'}]),
                                  ToolMessage(json.dumps(data,ensure_ascii=False),tool_call_id='eval-readonly')]
            else:
                answer_prompt.append(HumanMessage('本轮计划：'+json.dumps(plan['value'],ensure_ascii=False)+'\n请核实历史后作答。'))
            reply = await caller.call(key+'-reply','reply',answer_prompt)
            result = {'probe_id':probe['id'],'at_turn':probe['at_turn'],'coref':rewrite,'plan':plan,'reply':reply,
                      'tool_result':data,'context_text':'\n'.join(m.content for m in context['body']),
                      'context':{k:v for k,v in context.items() if k!='body'},
                      'body': [m.model_dump(mode='json') for m in context['body']],
                      'assembly_cpu_seconds':cpu_seconds,'observed_wall_seconds':time.monotonic()-start,
                      'context_sha256':digest([m.model_dump(mode='json') for m in context['body']])}
            result['score']=score(case,probe,result)
            write_json(out/'results'/(key+'.json'),result)
            probes_out[arm].append(result['score']['task_rule_v3'])
            print(cid,probe['id'],arm,'target',result['score']['target_pass'],'task',result['score']['task_rule_v3'],flush=True)
        write_json(out/'maintenance'/(cid+'.json'),maintenance)
    print('CASE_DONE',cid,{k:sum(v) for k,v in probes_out.items()},flush=True)


def percentile(values,q):
    return sorted(values)[max(0,math.ceil(len(values)*q)-1)] if values else None


def audit(out):
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    dataset=json.loads((out/'dataset.json').read_text(encoding='utf-8'))
    issues=[]
    if digest(dataset)!=manifest['dataset_sha256']: issues.append('dataset hash mismatch')
    for name,expected in manifest['sources'].items():
        if hashlib.sha256((out/'source_snapshot'/name).read_bytes()).hexdigest()!=expected: issues.append('snapshot mismatch '+name)
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected: issues.append('current source differs; use frozen sources '+name)
    stats={}; scores={}; missing=[]
    for arm in ARMS:
        entries=[]; records=[]; by_kind={}; by_case={}
        for case in dataset['cases']:
            case_scores=[]
            for probe in case['probes']:
                key=f'{case["id"]}-{probe["id"]}-{arm}'
                path=out/'results'/(key+'.json')
                if not path.exists(): missing.append(key);continue
                r=json.loads(path.read_text(encoding='utf-8'))
                sc=score(case,probe,r)
                if sc!=r['score']: issues.append('score mismatch '+key)
                if r['at_turn']!=probe['at_turn']:issues.append('causal boundary '+key)
                if digest(r['body']) != r['context_sha256']:issues.append('context body hash '+key)
                visible=case['history'][:2*probe['at_turn']]
                saved_state=json.loads((out/'states'/f'{case["id"]}-{probe["id"]}.json').read_text(encoding='utf-8'))
                regenerated=EvidenceState();regenerated.advance(visible)
                if regenerated.snapshot()!=saved_state:issues.append('state causal replay '+key)
                if arm!='C_current':
                    rebuilt=assemble(visible,probe['query'],regenerated if arm!='W_raw8k' else None,arm=='R_recall8k')
                    if digest([m.model_dump(mode='json') for m in rebuilt['body']])!=r['context_sha256']:issues.append('context causal replay '+key)
                for event in r['context'].get('retrieved',[]):
                    if max(event['source_ids'])>2*probe['at_turn']:issues.append('future retrieval '+key)
                for stage in ('coref','plan','reply'):
                    rec=r[stage]
                    if digest(rec['contract'])!=rec['input_sha256']:issues.append('call contract hash '+key+'-'+stage)
                    cached=json.loads((out/'calls'/(key+'-'+stage+'.json')).read_text(encoding='utf-8'))
                    if rec!=cached:issues.append('result/call disagreement '+key+'-'+stage)
                    records.append(rec)
                entries.append(r);case_scores.append(sc['task_rule_v3'])
                by_kind.setdefault(case['kind'],[]).append(sc['task_rule_v3'])
            if len(case_scores)==2:by_case[case['id']]=sum(case_scores)/2
        if arm=='C_current':
            for path in (out/'calls').glob('*-summary-*.json'): records.append(json.loads(path.read_text(encoding='utf-8')))
        attempts=[a for r in records for a in r['attempts']]
        usage={k:sum(a.get('usage',{}).get(k,0) or 0 for a in attempts) for k in ('input_tokens','output_tokens','total_tokens')}
        latency=[sum(r[s]['seconds'] for s in ('coref','plan','reply')) for r in entries]
        stats[arm]={'label':LABELS[arm],'n':len(entries),'passed':sum(r['score']['task_rule_v3'] for r in entries),
                    'legacy_passed':sum(r['score']['task_pass'] for r in entries),
                    'target_passed':sum(r['score']['target_pass'] for r in entries),
                    'coref_passed':sum(r['score']['coref_pass'] for r in entries),
                    'empty_replies':sum(r['score']['empty_reply'] for r in entries),
                    'truncated_probes':sum(r['score']['truncated'] for r in entries),
                    'transport_failed_probes':sum(r['score']['transport_failure'] for r in entries),
                    'failed_maintenance':sum(r['status']!='ok' for r in records if r['contract']['stage']=='summary'),
                    'transport_error_attempts':sum(not a['transport_ok'] for a in attempts),
                    'usage':usage,'maintenance_input':sum(a['usage'].get('input_tokens',0) or 0 for r in records if r['contract']['stage']=='summary' for a in r['attempts']),
                    'p50_seconds':percentile(latency,.5),'p95_seconds':percentile(latency,.95),
                    'stages':{s:sum(r[s]['status']!='ok' for r in entries) for s in ('coref','plan','reply')},
                    'by_kind':{k:{'passed':sum(v),'n':len(v)} for k,v in by_kind.items()}}
        scores[arm]=by_case
    for arm in ARMS:
        pairs=[scores[arm][c]-scores['C_current'][c] for c in sorted(scores[arm].keys()&scores['C_current'].keys())]
        rng=random.Random(20261023)
        draws=[sum(rng.choice(pairs) for _ in pairs)/len(pairs) for _ in range(4000)] if pairs else []
        stats[arm]['difference_vs_current']=statistics.mean(pairs) if pairs else None
        stats[arm]['paired_conversation_bootstrap_95ci']=[percentile(draws,.025),percentile(draws,.975)]
    report={'complete':not missing,'audit_passed':not missing and not issues,'issues':issues,'missing':missing,
            'strategies':stats,'manifest':manifest,'limit':'Known synthetic regression set; no production acceptance or unseen generalization claim. Token usage covers maintenance and 2 probes; transport failures may have unreported billed usage.'}
    write_json(out/'audited_report.json',report)
    lines=['# 上下文策略四组回归测试','',f'完成：{report["complete"]}；审计通过：{report["audit_passed"]}。',
           '', '|方案|任务规则通过|旧严格规则|对象/动作正确|输入含维护|输出|P50/P95秒|接口失败追问|空回答|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm,r in stats.items():
        lines.append(f'|{r["label"]}|{r["passed"]}/{r["n"]}|{r["legacy_passed"]}|{r["target_passed"]}|{r["usage"]["input_tokens"]:,}|{r["usage"]["output_tokens"]:,}|{r["p50_seconds"]}/{r["p95_seconds"]}|{r["transport_failed_probes"]}|{r["empty_replies"]}|')
    lines += ['', '48条既有合成回归会话，8场景族，各2次追问；不代表新场景泛化或完整线上客服。',
              '四组统一设置输出上限：指代512、规划1024、回复1536、摘要1024；非思考模式。',
              '输入含全部滚动摘要维护；历史轮次文本是给定的，未包含每轮在线生成成本。状态维护为本地确定性来源索引，无额外LLM调用。',
              '延迟为三阶段API等待累计，排除实验并发队列与后台维护；不是线上首字延迟。',
              '只对传输异常立即重试一次，空回答、截断和语义错误保留。所有尝试保存在attempts/。',
              '新规则只豁免明确作废/未使用编号引用，旧规则同时报告；依然是规则评分，不是人工语义准确率。',
              '尚未部署到正式客服。运行结束后应同时检查残留接口、空输出与预算截断。']
    (out/'AUDITED_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


async def run(out):
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    for name,expected in manifest['sources'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise ValueError('source changed '+name)
    if config_snapshot()!=manifest['config']:raise ValueError('model/config changed; start a new run')
    if get_store().policy().model_dump()!=manifest['pool_policy']:raise ValueError('pool policy changed; start new run')
    if settings.chat_thinking!='disabled' or settings.chat_failover_enabled is not False:raise ValueError('require fixed non-thinking model')
    data=json.loads((out/'dataset.json').read_text(encoding='utf-8'))
    if digest(data)!=manifest['dataset_sha256']:raise ValueError('dataset changed')
    caller=Caller(out)
    gate=asyncio.Semaphore(CONCURRENCY)
    async def one(case):
        async with gate:await evaluate_case(case,out,caller)
    cases=list(data['cases']);random.Random(20261023).shuffle(cases)
    try:
        await asyncio.gather(*(one(c) for c in cases))
    finally:
        from app.core.model_transport import close_clients
        await close_clients()
    report=audit(out)
    print('FINISHED',json.dumps({'audit_passed':report['audit_passed'],'n':{a:r['n'] for a,r in report['strategies'].items()}}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--audit',action='store_true')
    args=parser.parse_args()
    output=Path(args.output).resolve()
    if args.prepare:prepare(output)
    elif args.audit:print(json.dumps({'audit_passed':audit(output)['audit_passed']}))
    else:
        with writer_lock(output):asyncio.run(run(output))
