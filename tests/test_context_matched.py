import copy
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.context_matched import packet_cost, raw_packet, state_packet, validate_rows
from app.core.service_context import ContextBudgetError, ServiceMemory
from scripts import eval_context_matched as run
from scripts.context_benchmark_cases import make_suite


def history():
    rows = [{'role': 'user', 'content': '办公室的订单610010，最早承诺2026-10-01送达。'},
            {'role': 'assistant', 'content': '已收到。'}]
    rows += [{'role': role, 'content': '普通包装背景。' * 80}
             for _ in range(8) for role in ('user', 'assistant')]
    rows += [{'role': 'user', 'content': '这次只查订单610011。先别退款。'},
             {'role': 'assistant', 'content': '我会先查询。'}]
    return rows


def test_matched_packets_share_counter_cap_and_stage_evidence():
    rows = history(); query = '办公室那笔最早承诺哪天送达？'
    memory = ServiceMemory('test'); memory.advance(rows, 'test')
    packets = [raw_packet(rows, query, budget=2400),
               raw_packet(rows, query, summary='办公室订单610010。', boundary=2, budget=2400),
               state_packet(memory, query, recall=False, budget=2400),
               state_packet(memory, query, recall=True, budget=2400)]
    for packet in packets:
        assert packet['approx_context_tokens'] == packet_cost(packet['body']) <= 2400
        assert packet['body'][-1].content == query
        for message in packet['body'][:-1]:
            assert message.content in packet['coref_history']
    state, recall = packets[-2:]
    assert state['state'] == recall['state']
    assert state['protected_refs'] == recall['protected_refs']
    assert set(state['evidence_refs']) == set(state['protected_refs'])
    assert set(recall['evidence_refs']) > set(state['evidence_refs'])
    assert any('610010' in e['text'] for e in memory.snapshot()['evidence']
               if f"m{e['seq']}:{e['start']}-{e['end']}" in recall['evidence_refs'])
    # Local note creation never changes the caller's source journal.
    assert rows == history()


def test_original_turns_not_partially_cut_and_malformed_histories_rejected():
    rows = [{'role': 'user', 'content': '短问题'}, {'role': 'assistant', 'content': '短回答'},
            {'role': 'user', 'content': '长' * 4000}, {'role': 'assistant', 'content': '长' * 4000}]
    result = raw_packet(rows, '继续', budget=100)
    assert [m.id for m in result['body'][:-1]] == ['db-1', 'db-2']
    assert result['omitted_turns'] == [[3, 4]]
    with pytest.raises(ValueError):
        validate_rows(rows[:-1])
    with pytest.raises(ValueError):
        raw_packet([rows[0], rows[0]], '继续')
    with pytest.raises(ContextBudgetError):
        raw_packet([], '问题' * 100, budget=10)
    with pytest.raises(ContextBudgetError):
        raw_packet([], '问题', summary='长摘要' * 100, budget=10)


async def test_full_input_guard_counts_tool_arguments_and_does_not_call_provider(tmp_path, monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError('blocked requests must not call a model')
    monkeypatch.setattr(run.base.llm, 'get_chat_model', forbidden)
    caller = run.Caller(tmp_path)
    messages = [SystemMessage('test'), AIMessage('', tool_calls=[{
        'id': 'x', 'name': 'query_faq', 'args': {'query': '长' * 15000}, 'type': 'tool_call'}])]
    result = await caller.call('oversize', 'off', 'reply', messages)
    assert result['status'] == 'budget_blocked'
    assert result['approx_input_tokens'] > run.INPUT_LIMIT and not result['attempts']
    assert result == await caller.call('oversize', 'off', 'reply', messages)
    result = await caller.call('context-block', 'off', 'reply', [HumanMessage('small')],
                               local_error='essential context exceeds 9800')
    assert result['status'] == 'budget_blocked' and result['approx_input_tokens'] < run.INPUT_LIMIT


class FakeModel:
    schema = None
    extra_body = None
    invocations = 0

    def bind_tools(self, schemas, tool_choice):
        self.schema = schemas[0]
        return self

    async def ainvoke(self, messages):
        type(self).invocations += 1
        assert self.extra_body == run.base.mode_body('off')
        usage = {'input_tokens': 10, 'output_tokens': 20, 'total_tokens': 30,
                 'output_token_details': {'reasoning': 0}}
        metadata = {'finish_reason': 'stop', 'model_name': 'fake'}
        if self.schema:
            value = {'summary': '用户需要查询订单610010。'} if self.schema == run.Summary else {
                'action': 'logistics', 'order_id': '610010', 'sku': '', 'answer': ''}
            return AIMessage('', tool_calls=[{'id': 'fake', 'name': self.schema.__name__,
                'args': value, 'type': 'tool_call'}], usage_metadata=usage, response_metadata=metadata)
        return AIMessage('订单610010，运单SF61001088，运输中。', usage_metadata=usage, response_metadata=metadata)


async def test_five_arm_causal_replay_cost_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(run.base.llm, 'get_chat_model', lambda **kwargs: FakeModel())
    monkeypatch.setattr(run.base, 'INTERVAL', 0)
    monkeypatch.setattr(run, 'check', lambda out: None)
    case = copy.deepcopy(make_suite()['cases'][0]); case['cohort'] = 'regression'
    run.write_json(tmp_path / 'dataset.json', {'cases': [case]})
    await run.run(tmp_path)
    report = run.read(tmp_path / 'audited_report.json')
    assert report['complete'] and report['audit_passed'] and report['nonthinking_verified']
    assert report['counts']['results'] == 10 and report['counts']['maintenance'] == 5
    for arm in run.ARMS:
        stats = report['strategies'][arm]['all']
        assert stats['n'] == 2
        assert stats['usage']['input_tokens'] == stats['probe_usage']['input_tokens'] + stats['summary']['input']
        assert bool(stats['summary']['calls']) == (arm in ('C_current', 'M_matched'))
    count = FakeModel.invocations
    await run.run(tmp_path)
    assert FakeModel.invocations == count
    assert run.read(tmp_path / 'audited_report.json') == report
    path = tmp_path / 'results' / f'{case["id"]}-p1-R_matched.json'
    result = run.read(path); result['state']['evidence'][0]['start'] += 1
    run.write_json(path, result)
    assert not run.audit(tmp_path)['audit_passed']


async def test_parse_failure_is_retained_without_semantic_retry(tmp_path, monkeypatch):
    class Invalid(FakeModel):
        async def ainvoke(self, messages):
            self.schema = None
            return await super().ainvoke(messages)
    monkeypatch.setattr(run.base.llm, 'get_chat_model', lambda **kwargs: Invalid())
    monkeypatch.setattr(run.base, 'INTERVAL', 0)
    result = await run.Caller(tmp_path).call('bad-plan', 'off', 'plan', [SystemMessage('test')], run.ServicePlan)
    assert result['status'] == 'parse_error'
    assert len(result['attempts']) == 1 and result['attempts'][0]['transport_ok']
    assert result['attempts'][0]['raw_content']


def test_query_and_case_metadata_are_not_future_information():
    case = copy.deepcopy(make_suite()['cases'][0]); case['cohort'] = 'regression'
    # Independent state streams only ingest source rows visible before each probe.
    generator = run.workflow(case, 'R_matched')
    first = next(generator)
    assert first[0:2] == ('call', case['id'] + '-p1-R_matched-coref')
    changed = copy.deepcopy(case)
    cutoff = 2 * case['probes'][0]['at_turn']
    for row in changed['history'][cutoff:]:
        row['content'] = 'future secret 999999'
    changed['probes'][0]['expected'] = {'action': 'logistics', 'order_id': '999999'}
    second = next(run.workflow(changed, 'R_matched'))
    assert [m.model_dump() for m in first[4]] == [m.model_dump() for m in second[4]]


def test_prepare_freezes_preflight_data_sources_and_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(run.settings, 'chat_failover_enabled', False)
    monkeypatch.setattr(run.settings, 'summary_model', '')
    monkeypatch.setattr(run, 'providers', lambda *args: [SimpleNamespace(
        base_url=run.settings.chat_base_url, model=run.settings.chat_model,
        fallback=False, fingerprint='test-not-a-credential')])
    monkeypatch.setattr(run, 'SOURCES', ['app/core/context_matched.py'])
    case = copy.deepcopy(make_suite()['cases'][0]); case['cohort'] = 'regression'
    dataset = tmp_path / 'source-dataset.json'
    run.write_json(dataset, {'cases': [case]})
    proof = tmp_path / 'preflight'
    run.write_json(proof / 'preflight_report.json', {'verified': True,
        'chat_base_url': run.settings.chat_base_url, 'chat_model': run.settings.chat_model})
    for stage in ('coref', 'summary', 'plan'):
        run.write_json(proof / 'calls' / (stage + '.json'), {'ok': True,
            'contract': {'stage': stage, 'model': run.settings.chat_model, 'extra_body': run.base.mode_body('off')},
            'attempts': [{'usage': {'output_token_details': {'reasoning': 0}}}]})
    output = tmp_path / 'run'
    run.prepare(output, dataset, proof)
    run.check(output)
    with pytest.raises(ValueError, match='new output'):
        run.prepare(output, dataset, proof)
    manifest = run.read(output / 'manifest.json')
    assert manifest['context_budget'] == 9800 and manifest['input_limit_estimate'] == 16000
    assert manifest['summary'].startswith('C/M independently')
    saved = run.read(output / 'dataset.json'); saved['cases'][0]['history'][0]['content'] += 'tamper'
    run.write_json(output / 'dataset.json', saved)
    with pytest.raises(ValueError, match='dataset changed'):
        run.check(output)
    run.write_json(output / 'dataset.json', run.read(dataset))
    snapshot = output / 'source_snapshot/app/core/context_matched.py'
    snapshot.write_text('tamper', encoding='utf-8')
    with pytest.raises(ValueError, match='source changed'):
        run.check(output)
