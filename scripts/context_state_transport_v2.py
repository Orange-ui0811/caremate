"""Independent transport: force ServicePlan for every arm, preserve all receipts.

No parsing/semantic retries or forced fallback on an unsupported tool_choice.
All remaining provider controls match V1. Old runner/transport files untouched.
"""
from __future__ import annotations

import asyncio
import time

import httpx
from openai import APIConnectionError, APIStatusError

from app.core import llm
from app.core.model_pool import call_context, PoolBusy
from scripts import eval_context_matched as previous
from scripts.eval_context_policy_v3 import digest, stamp, write_json

base = previous.base
INPUT_LIMIT = previous.INPUT_LIMIT
PURPOSE = previous.PURPOSE
route_snapshot = previous.route_snapshot
input_estimate = previous.input_estimate
read = previous.read
SOURCES = list(dict.fromkeys([*previous.SOURCES, 'scripts/context_state_transport_v2.py']))
VERSION = 'forced-service-plan-v2'


def contract_for(mode, stage, messages, schema=None, local_error=None):
    wired, contract = previous.contract_for(mode, stage, messages, schema, local_error)
    if stage == 'plan':
        if schema is None or schema.__name__ != 'ServicePlan':
            raise ValueError('plan requires ServicePlan schema')
        contract['tool_choice'] = {'type': 'function', 'function': {'name': schema.__name__}}
    contract['transport_version'] = VERSION
    return wired, contract


def parse_response(raw, stage, schema):
    value = raw.content
    if schema:
        matches = [t for t in raw.tool_calls if t['name'] == schema.__name__]
        value = None
        if len(matches) == 1:
            try:
                value = schema.model_validate(matches[0]['args']).model_dump()
            except ValueError:
                pass
    finish = raw.response_metadata.get('finish_reason')
    status = 'ok'
    if finish == 'length':
        status = 'truncated'
    elif schema and value is None:
        status = 'parse_error'
    elif not schema and (not isinstance(value, str) or not value.strip()):
        status = 'empty'
    elif stage == 'summary' and not value['summary'].strip():
        status = 'empty'
    return value, finish, status


class Caller:
    def __init__(self, out):
        self.out = out
        self.gate = asyncio.Semaphore(base.CONCURRENCY)
        self.pace = asyncio.Lock()
        self.next_start = 0
        manifest = out / 'manifest.json'
        self.interval = read(manifest)['start_interval'] if manifest.exists() else base.INTERVAL

    async def call(self, key, mode, stage, messages, schema=None, local_error=None):
        messages, contract = contract_for(mode, stage, messages, schema, local_error)
        sha = digest(contract); path = self.out / 'calls' / (key + '.json')
        approx = input_estimate(messages, contract)
        if path.exists():
            cached = read(path)
            if cached['input_sha256'] != sha:
                raise ValueError('cached input changed ' + key)
            return cached
        if local_error or approx > INPUT_LIMIT:
            result = {'contract': contract, 'input_sha256': sha, 'attempts': [],
                      'status': 'budget_blocked', 'ok': False, 'value': None, 'finish_reason': None,
                      'seconds': 0, 'queue_seconds': 0, 'approx_input_tokens': approx, 'calls': [],
                      'local_error': local_error or 'complete input estimate exceeds 16000'}
            write_json(path, result)
            return result
        queued = time.monotonic()
        async with self.gate:
            queue_seconds = time.monotonic() - queued; attempts = []
            for index in range(2):
                ap = self.out / 'attempts' / (key + f'-{index + 1}.json')
                if ap.exists():
                    attempt = read(ap)
                    if attempt['input_sha256'] != sha:
                        raise ValueError('attempt changed')
                    attempts.append(attempt)
                    if index == 0 and attempt.get('retryable'):
                        continue
                    break
                async with self.pace:
                    await asyncio.sleep(max(0, self.next_start - time.monotonic()))
                    self.next_start = time.monotonic() + self.interval
                start = time.monotonic()
                attempt = {'input_sha256': sha, 'attempt': index + 1, 'started_at': stamp()}
                try:
                    with call_context(request_id=f'context-state-v2-{self.out.name}-{key}-{index}',
                                      priority='background', purpose=PURPOSE[stage]):
                        model = llm.get_chat_model(streaming=True, temperature=contract['temperature'])
                        model.max_tokens = base.MAX_OUTPUT; model.reasoning_effort = None
                        model.extra_body = base.mode_body(mode)
                        chain = model.bind_tools([schema], tool_choice=contract['tool_choice']) if schema else model
                        raw = await asyncio.wait_for(chain.ainvoke(messages), 100)
                    value, finish, status = parse_response(raw, stage, schema)
                    attempt.update(status=status, transport_ok=True, value=value, finish_reason=finish,
                                   model=raw.response_metadata.get('model_name'), usage=raw.usage_metadata or {},
                                   raw_content=raw.content, tool_calls=raw.tool_calls,
                                   seconds=round(time.monotonic() - start, 4))
                except Exception as exc:
                    is_transport = isinstance(exc, (APIConnectionError, APIStatusError, httpx.TransportError, TimeoutError, PoolBusy))
                    code = getattr(exc, 'status_code', None)
                    retryable = (isinstance(exc, (APIConnectionError, httpx.TransportError, TimeoutError, PoolBusy)) or
                                 isinstance(exc, APIStatusError) and (code in (408, 429) or code >= 500))
                    attempt.update(status='transport_error' if is_transport else 'validation_error', transport_ok=False,
                                   value=None, usage={}, error_type=type(exc).__name__, http_status=code,
                                   retryable=retryable, seconds=round(time.monotonic() - start, 4))
                write_json(ap, attempt); attempts.append(attempt)
                if index == 0 and attempt.get('retryable'):
                    await asyncio.sleep(4)
                else:
                    break
            last = attempts[-1]
            result = {'contract': contract, 'input_sha256': sha, 'attempts': attempts, 'status': last['status'],
                      'ok': last['status'] == 'ok', 'value': last['value'], 'finish_reason': last.get('finish_reason'),
                      'seconds': sum(a['seconds'] for a in attempts), 'queue_seconds': queue_seconds,
                      'approx_input_tokens': approx,
                      'calls': [{'input_tokens': a['usage'].get('input_tokens'), 'output_tokens': a['usage'].get('output_tokens')}
                                for a in attempts if a['transport_ok']]}
            write_json(path, result)
            return result
