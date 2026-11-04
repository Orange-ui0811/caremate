from copy import deepcopy

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.context_policy import estimate
from scripts import context_state_cases as v1_cases
from scripts import context_state_cases_v2 as cases
from scripts import context_state_transport_v2 as transport
from scripts import eval_context_state as v1
from scripts import eval_context_state_v2 as run


def test_v2_only_extends_reserved_long_background_and_leaves_gold_fixed():
    old = {c['id']: c for c in v1_cases.dataset()['cases']}
    value = cases.dataset(); cases.validate_dataset(value)
    assert len(value['cases']) == 36
    extended = [c for c in value['cases'] if 'pressure_construction' in c]
    assert len(extended) == 12 and len({c['kind'] for c in extended}) == 12
    assert all(len(c['steps']) == 3 for c in value['cases'])
    for case in value['cases']:
        assert case['steps'] == old[case['id']]['steps']
        assert case['tool_data'] == old[case['id']]['tool_data']
        if case in extended:
            assert sum(estimate(r['content']) for r in case['history']) >= 15000
        else:
            assert case == old[case['id']]
    changed = deepcopy(value); changed['cases'][-1]['steps'][0]['query'] += ' changed'
    with pytest.raises(ValueError, match='non-background field'):
        cases.validate_dataset(changed)


def test_v2_private_reuse_does_not_mutate_v1_globals():
    assert v1.transport.__name__ == 'scripts.eval_context_matched'
    assert v1.dataset is v1_cases.dataset
    assert run._impl.transport is transport
    assert run._impl.dataset is cases.dataset


def test_forced_tool_choice_is_frozen_only_for_plan():
    messages = [SystemMessage('Plan'), HumanMessage('Query')]
    _, plan = transport.contract_for('off', 'plan', messages, v1.ServicePlan)
    assert plan['tool_choice'] == run.FORCED_PLAN
    assert plan['max_tokens'] == 8192 and plan['extra_body'] == transport.base.mode_body('off')
    _, summary = transport.contract_for('off', 'summary', messages, v1.Summary)
    assert summary['tool_choice'] == 'auto'
    with pytest.raises(ValueError, match='requires ServicePlan'):
        transport.contract_for('off', 'plan', messages)


class FakeModel:
    count = 0
    choices = []
    malformed = False
    schema = None

    def bind_tools(self, schemas, tool_choice):
        self.schema = schemas[0]
        FakeModel.choices.append(tool_choice)
        return self

    async def ainvoke(self, messages):
        FakeModel.count += 1
        metadata = {'finish_reason': 'stop', 'model_name': 'fake'}
        usage = {'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15, 'output_token_details': {'reasoning': 0}}
        if self.schema and not FakeModel.malformed:
            args = {'summary': '背景：两笔订单，尚未完成任何申请。'} if self.schema is v1.Summary else {'action': 'clarify', 'order_id': '', 'sku': '', 'answer': '请确认订单。'}
            return AIMessage('', tool_calls=[{'name': self.schema.__name__, 'args': args, 'id': 'fake', 'type': 'tool_call'}], response_metadata=metadata, usage_metadata=usage)
        return AIMessage('请确认订单。保持真实回填。', response_metadata=metadata, usage_metadata=usage)


async def test_forced_plan_sends_choice_preserves_parse_failures_and_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(transport.llm, 'get_chat_model', lambda **kwargs: FakeModel())
    monkeypatch.setattr(transport.base, 'INTERVAL', 0)
    monkeypatch.setattr(FakeModel, 'malformed', True)
    FakeModel.count = 0; FakeModel.choices = []
    caller = transport.Caller(tmp_path)
    messages = [SystemMessage('Plan'), HumanMessage('查订单8200')]
    result = await caller.call('failed-plan', 'off', 'plan', messages, v1.ServicePlan)
    assert result['status'] == 'parse_error' and len(result['attempts']) == 1
    assert FakeModel.choices == [run.FORCED_PLAN]
    assert result['attempts'][0]['raw_content'] == '请确认订单。保持真实回填。'
    assert await caller.call('failed-plan', 'off', 'plan', messages, v1.ServicePlan) == result
    assert FakeModel.count == 1


def test_long_stratum_really_triggers_summary_and_retains_recent_eight_turns():
    case = next(c for c in cases.dataset()['cases'] if c['length_stratum'] == 'long')
    rows = v1.rows_with_ids(case['history'])
    plan = run.summary_plan(rows, [], {'context_budget': 9800, 'summary_trigger_tokens': 12000, 'summary_keep_turns': 8})
    assert plan is not None
    assert len([r for r in rows if r['id'] > plan['to_message_id'] and r['role'] == 'user']) >= 8
    assert plan['source_hash'] == v1.source_digest(plan['rows'])


async def test_v2_causal_fake_model_run_and_audit_preserve_v1_module(tmp_path, monkeypatch):
    case = v1_cases.make_case('ambiguous', 'development')
    contract = {'arms': ['C', 'P', 'S', 'R', 'M'], 'mode': 'trajectory', 'split': 'development',
                'context_budget': 9800, 'summary_trigger_tokens': 12000, 'summary_target_tokens': 1000,
                'fairness': 'test', 'cost_scope': 'test', 'data_scope': 'test', 'trajectory_scope': 'test'}
    monkeypatch.setattr(run._impl, 'check', lambda out, **kwargs: contract)
    monkeypatch.setattr(transport.llm, 'get_chat_model', lambda **kwargs: FakeModel())
    monkeypatch.setattr(transport.base, 'INTERVAL', 0)
    monkeypatch.setattr(FakeModel, 'malformed', False)
    v1.write_json(tmp_path / 'dataset.json', {'cases': [case]})
    await run.run(tmp_path)
    report = v1.read(tmp_path / 'audited_report.json')
    assert report['complete'] and report['audit_passed'] and report['nonthinking_verified']
    assert report['experiment_version'] == run.VERSION
    assert report['counts']['results'] == 15
    assert v1.transport.__name__ == 'scripts.eval_context_matched'
    assert all(v1.read(p)['contract']['tool_choice'] == run.FORCED_PLAN for p in (tmp_path / 'calls').glob('*-plan.json'))
    before = FakeModel.count
    await run.run(tmp_path)
    assert FakeModel.count == before


async def test_v2_long_summary_maintenance_replays_with_full_cost(tmp_path, monkeypatch):
    case = next(c for c in cases.dataset()['cases'] if c['length_stratum'] == 'long')
    contract = {'arms': ['R', 'M'], 'mode': 'trajectory', 'split': 'reserved',
                'context_budget': 9800, 'summary_trigger_tokens': 12000, 'summary_target_tokens': 1000,
                'summary_keep_turns': 8,
                'fairness': 'test', 'cost_scope': 'test', 'data_scope': 'test', 'trajectory_scope': 'test'}
    monkeypatch.setattr(run._impl, 'check', lambda out, **kwargs: contract)
    monkeypatch.setattr(transport.llm, 'get_chat_model', lambda **kwargs: FakeModel())
    monkeypatch.setattr(transport.base, 'INTERVAL', 0)
    monkeypatch.setattr(FakeModel, 'malformed', False)
    v1.write_json(tmp_path / 'dataset.json', {'cases': [case]})
    await run.run(tmp_path)
    report = v1.read(tmp_path / 'audited_report.json')
    assert report['complete'] and report['audit_passed']
    assert report['strategies']['M']['summary_calls'] >= 1
    assert report['strategies']['M']['summary_accepted_segments'] >= 1
    assert report['strategies']['M']['summary_used_checkpoints'] >= 1
    assert report['strategies']['M']['summary_usage']['total_tokens'] > 0
    assert report['strategies']['R']['summary_calls'] == 0


def test_v2_prepare_requires_new_protocol_proof_and_freezes_default_trigger(tmp_path, monkeypatch):
    root = tmp_path / 'source'; root.mkdir(); (root / 'fixture.py').write_text('frozen=1', encoding='utf-8')
    monkeypatch.setattr(run._impl, 'ROOT', root); monkeypatch.setattr(run._impl, 'SOURCES', ['fixture.py'])
    monkeypatch.setattr(run.settings, 'chat_failover_enabled', False)
    monkeypatch.setattr(run.settings, 'summary_model', '')
    monkeypatch.setattr(transport, 'route_snapshot', lambda: [{'base_url': run.settings.chat_base_url, 'model': run.settings.chat_model, 'fallback': False, 'fingerprint': 'fictional'}])
    proof = tmp_path / 'proof'
    v1.write_json(proof / 'preflight_report.json', {'verified': True, 'chat_model': run.settings.chat_model, 'chat_base_url': run.settings.chat_base_url})
    for stage in ('coref', 'plan', 'summary'):
        contract = {'stage': stage, 'model': run.settings.chat_model, 'extra_body': transport.base.mode_body('off'),
                    'transport_version': transport.VERSION, 'tool_choice': run.FORCED_PLAN if stage == 'plan' else None}
        v1.write_json(proof / 'calls' / (stage + '.json'), {'ok': True, 'contract': contract,
                      'attempts': [{'usage': {'output_token_details': {'reasoning': 0}}}]})
    out = tmp_path / 'run'
    manifest = run.prepare(out, preflight=proof, split='reserved', limit=2, arms=('R', 'M'))
    assert manifest['summary_trigger_tokens'] == 12000
    assert manifest['start_interval'] == .5 and transport.Caller(out).interval == .5
    assert manifest['version'] == run.VERSION and len(manifest['pressure_construction']) == 1
    assert run.check(out)['dataset_sha256'] == manifest['dataset_sha256']
    path = proof / 'calls' / 'plan.json'; saved = v1.read(path); saved['contract']['tool_choice'] = 'auto'; v1.write_json(path, saved)
    with pytest.raises(ValueError, match='V1 proof is insufficient'):
        run.prepare(tmp_path / 'rejected', preflight=proof)
