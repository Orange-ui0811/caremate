from copy import deepcopy
from pathlib import Path

import pytest

from langchain_core.messages import AIMessage, HumanMessage

from app.core.context_state import build_snapshot
from scripts import eval_context_state as run
from scripts.context_state_cases import make_case


def test_component_raw_subset_is_identical_and_gold_never_selected():
    rows = [{'role': 'user', 'content': '本次只查订单8200。'}, {'role': 'assistant', 'content': '收到。'}]
    rows += [{'role': role, 'content': '背景文字。' * 30} for _ in range(8) for role in ('user', 'assistant')]
    rows += [{'role': 'user', 'content': '订单8200先别退款。'}]
    snapshot = build_snapshot(run.rows_with_ids(rows), 'scope')
    packets = run.component_packets(snapshot, '查物流。', 2200)
    assert len({tuple(p['manifest']['common_raw_ids']) for p in packets.values()}) == 1
    assert all(run.cost(p['body']) <= 2200 for p in packets.values())
    assert '任务状态' not in packets['Q']['coref_history']
    assert '任务状态' in packets['S']['coref_history']
    assert packets['Q']['manifest']['protected_refs'] == packets['S']['manifest']['protected_refs']


def test_current_question_not_gold_changes_policy():
    case = make_case('early_order', 'development')
    changed = deepcopy(case); changed['steps'][0]['expected']['order_id'] = 'hidden-9999'
    changed['steps'][0]['gold_source_texts'] = ['hidden annotation']
    contract = {'mode': 'trajectory', 'context_budget': 9800}
    first = next(run.workflow(case, 'S', contract))
    second = next(run.workflow(changed, 'S', contract))
    assert [m.model_dump() for m in first[4]] == [m.model_dump() for m in second[4]]
    assert 'hidden' not in str(first)


class FakeModel:
    count = 0
    schema = None

    def bind_tools(self, schemas, tool_choice):
        self.schema = schemas[0]
        return self

    async def ainvoke(self, messages):
        FakeModel.count += 1
        usage = {'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15,
                 'output_token_details': {'reasoning': 0}}
        metadata = {'finish_reason': 'stop', 'model_name': 'fake'}
        if self.schema:
            value = {'summary': '订单8200和8201。'} if self.schema is run.Summary else {'action': 'logistics', 'order_id': '8200', 'sku': '', 'answer': ''}
            return AIMessage('', tool_calls=[{'id': 'fake', 'name': self.schema.__name__, 'args': value, 'type': 'tool_call'}], usage_metadata=usage, response_metadata=metadata)
        return AIMessage('订单8200，运单SF820088。独特回填标记。', usage_metadata=usage, response_metadata=metadata)


async def test_causal_audit_resume_and_actual_reply_carry_forward(tmp_path, monkeypatch):
    case = make_case('early_order', 'development')
    contract = {'arms': ['C', 'W', 'P', 'Q', 'S', 'R'], 'mode': 'trajectory', 'split': 'development', 'context_budget': 9800,
                'fairness': 'test', 'cost_scope': 'test', 'data_scope': 'test', 'trajectory_scope': 'test'}
    monkeypatch.setattr(run, 'check', lambda out, **kwargs: contract)
    monkeypatch.setattr(run.transport.base, 'INTERVAL', 0)
    monkeypatch.setattr(run.transport.base.llm, 'get_chat_model', lambda **kwargs: FakeModel())
    run.write_json(tmp_path / 'dataset.json', {'cases': [case]})
    await run.run(tmp_path)
    report = run.read(tmp_path / 'audited_report.json')
    assert report['complete'] and report['audit_passed'] and report['nonthinking_verified']
    assert report['counts']['results'] == 18 and report['counts']['trajectories'] == 6
    r = run.read(tmp_path / 'results' / (case['id'] + '-p2-S.json'))
    assert any('独特回填标记' in x['content'] for x in r['visible_rows'])
    calls = FakeModel.count
    await run.run(tmp_path)
    assert FakeModel.count == calls
    path = tmp_path / 'results' / (case['id'] + '-p2-S.json')
    r['snapshot']['watermark'] += 1; run.write_json(path, r)
    assert not run.audit(tmp_path)['audit_passed']


def test_trajectory_and_checkpoint_are_explicitly_different():
    case = make_case('early_order', 'development')
    def get_second_history(mode):
        g = run.workflow(case, 'W', {'mode': mode, 'context_budget': 9800}); value = None
        for _ in range(20):
            item = g.send(value); value = None
            if item[0] == 'call':
                if item[1].endswith('p2-W-coref'):
                    return str(item[4])
                value = {'ok': True, 'status': 'ok', 'calls': [1], 'value': {'action': 'clarify', 'order_id': '', 'sku': '', 'answer': ''} if item[3] == 'plan' else 'ACTUAL_CARRIED_REPLY'}
        raise AssertionError('second query not found')
    assert 'ACTUAL_CARRIED_REPLY' in get_second_history('trajectory')
    assert 'ACTUAL_CARRIED_REPLY' not in get_second_history('checkpoints')


def test_existing_records_are_not_overwritten(tmp_path):
    path = tmp_path / 'r.json'
    run.save_immutable(path, {'status': 'parse_error'})
    run.save_immutable(path, {'status': 'parse_error'})
    with pytest.raises(ValueError, match='differs'):
        run.save_immutable(path, {'status': 'ok'})
    assert run.read(path)['status'] == 'parse_error'


def test_prepare_snapshots_without_provider_calls_and_rejects_data_edits(tmp_path, monkeypatch):
    root = tmp_path / 'source'; root.mkdir(); (root / 'fixture.py').write_text('immutable = 1', encoding='utf-8')
    monkeypatch.setattr(run, 'ROOT', root); monkeypatch.setattr(run, 'SOURCES', ['fixture.py'])
    monkeypatch.setattr(run.settings, 'chat_failover_enabled', False)
    monkeypatch.setattr(run.settings, 'summary_model', '')
    monkeypatch.setattr(run.transport, 'route_snapshot', lambda: [{'base_url': run.settings.chat_base_url, 'model': run.settings.chat_model, 'fallback': False, 'fingerprint': 'fictional'}])
    proof = tmp_path / 'preflight'
    run.write_json(proof / 'preflight_report.json', {'verified': True, 'chat_model': run.settings.chat_model, 'chat_base_url': run.settings.chat_base_url})
    for stage in ('coref', 'plan', 'summary'):
        run.write_json(proof / 'calls' / (stage + '.json'), {'ok': True, 'contract': {'stage': stage, 'model': run.settings.chat_model, 'extra_body': run.transport.base.mode_body('off')},
            'attempts': [{'usage': {'output_token_details': {'reasoning': 0}}}]})
    out = tmp_path / 'result'
    manifest = run.prepare(out, limit=1, families=('correction', 'old_constraint'), arms=('C', 'S'),
                           case_concurrency=3, caller_concurrency=2, preflight=proof)
    assert run.check(out)['dataset_sha256'] == manifest['dataset_sha256']
    assert manifest['case_concurrency'] == 3 and manifest['caller_concurrency'] == 2
    assert run.read(out / 'dataset.json')['cases'][0]['kind'] == 'correction'
    assert (out / 'source_snapshot' / 'fixture.py').read_text() == 'immutable = 1'
    data = run.read(out / 'dataset.json'); data['cases'][0]['steps'][0]['query'] = 'changed'
    run.write_json(out / 'dataset.json', data)
    with pytest.raises(ValueError, match='dataset.json changed'):
        run.check(out)


def test_recall_arms_share_actual_assembler_and_only_toggle_recall():
    case = make_case('early_order', 'development')
    contract = {'mode': 'trajectory', 'context_budget': 9800}
    records = {}
    for arm in ('A', 'B', 'R'):
        g = run.workflow(case, arm, contract); value = None
        while True:
            item = g.send(value); value = None
            if item[0] == 'save' and item[1] == 'results':
                records[arm] = item[3]; break
            if item[0] == 'call':
                value = {'ok': True, 'status': 'ok', 'calls': [1], 'value': {'action': 'clarify', 'order_id': '', 'sku': '', 'answer': ''} if item[3] == 'plan' else '请确认订单号。'}
    assert len({run.digest(v['snapshot']) for v in records.values()}) == 1
    assert {a: v['context_manifest']['recall_policy'] for a, v in records.items()} == {'A': 'off', 'B': 'always', 'R': 'conditional'}


def test_segment_prefix_keeps_eight_turns_and_hash_matches_sources():
    rows = run.rows_with_ids(make_case('early_order', 'reserved', 1)['history'])
    plan = run.summary_plan(rows, [], {'context_budget': 9800, 'summary_trigger_tokens': 500, 'summary_keep_turns': 8})
    assert plan['rows'][-1]['role'] == 'assistant'
    assert len([r for r in rows if r['id'] > plan['to_message_id'] and r['role'] == 'user']) >= 8
    assert plan['source_hash'] == run.source_digest(plan['rows'])
    assert plan['source_refs'][0]['message_id'] == plan['from_message_id']


def test_oversize_summary_is_archived_but_not_applied():
    case = make_case('early_order', 'development')
    contract = {'mode': 'trajectory', 'context_budget': 1200, 'summary_trigger_tokens': 200,
                'summary_keep_turns': 8, 'summary_target_tokens': 128}
    g = run.workflow(case, 'M', contract); value = None; saved = []; calls = []
    try:
        while True:
            item = g.send(value); value = None
            if item[0] == 'call':
                calls.append(item)
                v = {'summary': '未压缩背景。' * 100} if item[3] == 'summary' else {'action': 'clarify', 'order_id': '', 'sku': '', 'answer': ''} if item[3] == 'plan' else '请确认订单号。'
                value = {'ok': True, 'status': 'ok', 'calls': [1], 'value': v}
            else:
                saved.append(item)
    except StopIteration:
        pass
    events = next(x[3] for x in saved if x[1] == 'maintenance')
    assert events and all(not e['accepted'] and e['rejection_reason'] == 'over_target' for e in events)
    assert all(not x[3]['summary_segments'] for x in saved if x[1] == 'results')
    keys = [x[1] for x in calls if x[3] == 'summary']
    assert len(keys) == len(set(keys))


def test_paired_ci_resamples_conversations_not_individual_probes():
    first = {'a': [False, False, True], 'b': [False, True, True]}
    second = {'a': [True, True, True], 'b': [True, True, True]}
    value = run.paired_difference(first, second)
    assert value['n_conversations'] == 2
    assert value['mean_difference_pp'] == pytest.approx(50)
    assert value['ci95_pp'] == pytest.approx([100 / 3, 200 / 3], abs=.001)
    assert value['wins'] == 2 and value['losses'] == 0
    assert value == run.paired_difference(first, second)


async def test_actual_summary_arm_replays_sources_maintenance_and_usage(tmp_path, monkeypatch):
    case = make_case('early_order', 'reserved', 1)
    contract = {'arms': ['R', 'M'], 'mode': 'trajectory', 'split': 'reserved', 'context_budget': 1200,
                'summary_trigger_tokens': 3500, 'summary_target_tokens': 1000, 'summary_keep_turns': 8,
                'fairness': 'test', 'cost_scope': 'test', 'data_scope': 'test', 'trajectory_scope': 'test'}
    monkeypatch.setattr(run, 'check', lambda out, **kwargs: contract)
    monkeypatch.setattr(run.transport.base, 'INTERVAL', 0)
    monkeypatch.setattr(run.transport.base.llm, 'get_chat_model', lambda **kwargs: FakeModel())
    run.write_json(tmp_path / 'dataset.json', {'cases': [case]})
    await run.run(tmp_path)
    report = run.read(tmp_path / 'audited_report.json')
    assert report['complete'] and report['audit_passed']
    stats = report['strategies']['M']
    assert stats['summary_calls'] > 0 and stats['summary_accepted_segments'] > 0
    assert stats['summary_used_checkpoints'] > 0
    assert stats['usage']['total_tokens'] > report['strategies']['R']['usage']['total_tokens']
    assert report['paired_comparisons']['R->M']['n_conversations'] == 1
