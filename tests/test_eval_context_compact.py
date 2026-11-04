from copy import deepcopy

import pytest
from langchain_core.messages import AIMessage

from scripts import context_state_cases_v2 as cases
from scripts import context_state_transport_v2 as transport
from scripts import eval_context_compact as run
from scripts import eval_context_state as v1
from scripts import eval_context_state_v2 as v2


def contract(arms=run.ARMS):
    return {'arms': list(arms), 'mode': 'trajectory', 'split': 'reserved',
            'stage_budgets': dict(run.STAGE_BUDGETS), 'context_budget': 9800,
            'summary_trigger_tokens': 12000, 'summary_target_tokens': 1000, 'summary_keep_turns': 8,
            'fairness': 'test', 'cost_scope': 'test', 'data_scope': 'test', 'trajectory_scope': 'test'}


class FakeModel:
    count = 0
    schema = None

    def bind_tools(self, schemas, tool_choice):
        self.schema = schemas[0]
        return self

    async def ainvoke(self, messages):
        FakeModel.count += 1
        usage = {'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15,
                 'input_token_details': {'cache_read': 2}, 'output_token_details': {'reasoning': 0}}
        metadata = {'finish_reason': 'stop', 'model_name': 'fake'}
        if self.schema:
            value = ({'summary': '用户有多笔订单。尚未申请退款或提交任何业务申请。'}
                     if self.schema is v1.Summary else
                     {'action': 'clarify', 'order_id': '', 'sku': '', 'answer': '请确认订单。'})
            return AIMessage('', tool_calls=[{'name': self.schema.__name__, 'args': value,
                             'id': 'fake', 'type': 'tool_call'}],
                             usage_metadata=usage, response_metadata=metadata)
        return AIMessage('请确认订单。实际回复标记。', usage_metadata=usage, response_metadata=metadata)


def test_private_wrapper_leaves_original_versions_and_workflow_unchanged():
    assert v1.ARMS == ('C', 'W', 'P', 'Q', 'S', 'A', 'B', 'R', 'M')
    assert v1.transport.__name__ == 'scripts.eval_context_matched'
    assert v2.ARMS == v1.ARMS
    assert v2._impl.workflow is v2.workflow
    assert run._impl is not v2._impl
    assert run._impl.workflow is run.workflow


def test_compact_arms_select_without_annotations_and_use_same_three_calls():
    case = next(c for c in cases.dataset()['cases'] if c['split'] == 'reserved' and c['length_stratum'] == 'short')
    changed = deepcopy(case)
    changed['steps'][0]['expected']['order_id'] = 'GOLD_ONLY_99999'
    changed['steps'][0]['gold_source_texts'] = ['GOLD_ONLY_99999']
    for arm in ('CW', 'CS'):
        first = next(run.workflow(case, arm, contract()))
        second = next(run.workflow(changed, arm, contract()))
        assert first[0] == 'call' and first[3] == 'coref'
        assert [m.model_dump() for m in first[4]] == [m.model_dump() for m in second[4]]
        assert 'GOLD_ONLY' not in str(first)


async def test_compact_full_fake_run_has_three_stages_cost_reconciliation_and_causal_audit(tmp_path, monkeypatch):
    case = next(c for c in cases.dataset()['cases'] if c['split'] == 'reserved' and c['length_stratum'] == 'short')
    monkeypatch.setattr(run._impl, 'check', lambda out, **kwargs: contract())
    monkeypatch.setattr(transport.llm, 'get_chat_model', lambda **kwargs: FakeModel())
    monkeypatch.setattr(transport.base, 'INTERVAL', 0)
    v1.write_json(tmp_path / 'dataset.json', {'cases': [case]})
    await run.run(tmp_path)
    report = v1.read(tmp_path / 'audited_report.json')
    assert report['complete'] and report['audit_passed'] and report['nonthinking_verified']
    assert report['experiment_version'] == run.VERSION
    assert report['counts']['results'] == 12
    assert report['cost_reconciliation']['all_receipts_accounted']
    assert report['cost_reconciliation']['unique_attempt_file_usage']['total_tokens'] == sum(
        g['usage']['total_tokens'] for g in report['strategies'].values())
    for arm in ('CW', 'CS'):
        records = [v1.read(p) for p in (tmp_path / 'results').glob('*-' + arm + '.json')]
        assert len(records) == 3
        assert all(set(r['stage_contexts']) == {'coref', 'plan', 'answer'} for r in records)
        assert all(report['cost_breakdown'][arm]['stages'][s]['attempt_count'] == 3 for s in ('coref', 'plan', 'reply'))
        assert any('实际回复标记' in str(r['visible_rows']) for r in records)
    before = FakeModel.count
    await run.run(tmp_path)
    assert FakeModel.count == before
    file = next((tmp_path / 'results').glob('*-CS.json'))
    edited = v1.read(file); edited['stage_contexts']['coref']['coref_history'] += ' tampered'
    v1.write_json(file, edited)
    assert not run.audit(tmp_path)['audit_passed']


async def test_all_three_source_variants_maintain_identical_initial_summary_contract(tmp_path, monkeypatch):
    case = next(c for c in cases.dataset()['cases'] if c['length_stratum'] == 'long')
    frozen = contract(('M', 'CW', 'CS'))
    monkeypatch.setattr(run._impl, 'check', lambda out, **kwargs: frozen)
    monkeypatch.setattr(transport.llm, 'get_chat_model', lambda **kwargs: FakeModel())
    monkeypatch.setattr(transport.base, 'INTERVAL', 0)
    v1.write_json(tmp_path / 'dataset.json', {'cases': [case]})
    await run.run(tmp_path)
    report = v1.read(tmp_path / 'audited_report.json')
    assert report['audit_passed']
    summary_calls = [v1.read(p) for p in (tmp_path / 'calls').glob('*-source-summary-*.json')]
    assert len(summary_calls) == 3
    assert len({r['input_sha256'] for r in summary_calls}) == 1
    for arm in ('M', 'CW', 'CS'):
        assert report['strategies'][arm]['summary_calls'] == 1
        assert report['strategies'][arm]['summary_accepted_segments'] == 1
        assert report['cost_breakdown'][arm]['stages']['summary']['total_tokens'] > 0


def test_prepare_freezes_regression_role_stage_budgets_and_original_scoring(tmp_path, monkeypatch):
    root = tmp_path / 'source'; root.mkdir()
    (root / 'fixture.py').write_text('frozen=1', encoding='utf-8')
    monkeypatch.setattr(run._impl, 'ROOT', root)
    monkeypatch.setattr(run._impl, 'SOURCES', ['fixture.py'])
    monkeypatch.setattr(run.settings, 'chat_failover_enabled', False)
    monkeypatch.setattr(run.settings, 'summary_model', '')
    monkeypatch.setattr(transport, 'route_snapshot', lambda: [{'base_url': run.settings.chat_base_url,
        'model': run.settings.chat_model, 'fallback': False, 'fingerprint': 'fictional'}])
    proof = tmp_path / 'proof'
    v1.write_json(proof / 'preflight_report.json', {'verified': True,
        'chat_model': run.settings.chat_model, 'chat_base_url': run.settings.chat_base_url})
    for stage in ('coref', 'plan', 'summary'):
        call_contract = {'stage': stage, 'model': run.settings.chat_model,
            'extra_body': transport.base.mode_body('off'), 'transport_version': transport.VERSION,
            'tool_choice': v2.FORCED_PLAN if stage == 'plan' else None}
        v1.write_json(proof / 'calls' / (stage + '.json'), {'ok': True, 'contract': call_contract,
            'attempts': [{'usage': {'output_token_details': {'reasoning': 0}}}]})
    out = tmp_path / 'run'
    m = run.prepare(out, split='reserved', limit=2, preflight=proof)
    assert m['version'] == run.VERSION and m['arms'] == list(run.ARMS)
    assert m['stage_budgets'] == run.STAGE_BUDGETS
    assert m['scoring_version'] == v1.SCORE_VERSION
    assert 'not an unseen holdout' in m['dataset_role']
    assert run.check(out)['dataset_sha256'] == m['dataset_sha256']
    m['stage_budgets']['coref'] = 9999
    v1.write_json(out / 'manifest.json', m)
    with pytest.raises(ValueError, match='frozen compact'):
        run.check(out)
