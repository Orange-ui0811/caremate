"""Analysis-only fixtures; never imported by the frozen experiment runner."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import analyze_context_matched as analysis


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


@pytest.fixture
def finished_run(tmp_path):
    arms = ['C_current', *analysis.MATCHED_ARMS]
    history = [{'role': 'user', 'content': '订单730010是第一单。'}, {'role': 'assistant', 'content': '已登记。'},
               {'role': 'user', 'content': '普通包装说明。' * 2000}, {'role': 'assistant', 'content': '只是背景。'}]
    case = {'id': 'analysis-case', 'kind': 'early_order', 'cohort': 'transfer', 'turns': 2, 'history': history,
            'probes': [{'id': f'p{i}', 'at_turn': i, 'query': '查询最早订单的物流。',
                        'expected': {'action': 'logistics', 'order_id': '730010'},
                        'gold_evidence': [{'seq': 1, 'start': 0, 'end': len(history[0]['content']), 'text': history[0]['content']}]}
                       for i in (1, 2)]}
    data = {'cases': [case]}
    source = Path(__file__).resolve().parents[1] / 'app/core/context_policy.py'
    snapshot = tmp_path / 'source_snapshot/app/core/context_policy.py'
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(source.read_bytes())
    manifest = {'arms': arms, 'context_budget': 9800, 'dataset_sha256': analysis.digest(data),
                'sources': {'app/core/context_policy.py': hashlib.sha256(source.read_bytes()).hexdigest()}}
    report = {'complete': True, 'audit_passed': True, 'strategies': {}}

    def receipt(stage, value, quote=True):
        return {'contract': {'stage': stage, 'messages': [{'role': 'user', 'content': history[0]['content'] if quote else '已改述的第一单信息。'}]},
                'input_sha256': 'test-contract', 'status': 'ok', 'ok': True, 'value': value, 'seconds': 1.0,
                'attempts': [{'usage': {'input_tokens': 10, 'output_tokens': 2, 'total_tokens': 12}}]}

    for arm in arms:
        has_summary = arm in ('C_current', 'M_matched')
        events = []
        if has_summary:
            key = f'{case["id"]}-{arm}-summary-2'
            put(tmp_path / 'calls' / (key + '.json'), receipt('summary', {'summary': '第一单730010。'}))
            events.append({'key': key, 'at_message': 2})
        put(tmp_path / 'maintenance' / f'{case["id"]}-{arm}.json', events)
        for probe in case['probes']:
            key = f'{case["id"]}-{probe["id"]}-{arm}'
            result = {'body': [{'id': 'db-1', 'content': history[0]['content']}, {'id': None, 'content': probe['query']}],
                      'score': {'task_rule_v3': probe['id'] == 'p1', 'target_pass': True, 'coref_pass': True}}
            for stage in analysis.STAGES:
                value = {'action': 'logistics', 'order_id': '730010'} if stage == 'plan' else '订单730010。'
                r = receipt(stage, value, quote=arm != 'M_matched')
                result[stage] = r
                put(tmp_path / 'calls' / (key + '-' + stage + '.json'), r)
            put(tmp_path / 'results' / (key + '.json'), result)
        report['strategies'][arm] = {'all': {'n': 2, 'passed': 1, 'target_passed': 2,
                                            'usage': {'input_tokens': 60 + 10 * has_summary,
                                                      'output_tokens': 12 + 2 * has_summary,
                                                      'total_tokens': 72 + 12 * has_summary}}}
    put(tmp_path / 'manifest.json', manifest)
    put(tmp_path / 'dataset.json', data)
    put(tmp_path / 'audited_report.json', report)
    put(tmp_path / 'SUPPLEMENTARY_AUDIT.json', {'passed': True})
    return tmp_path


def test_read_only_complete_analysis_accounts_summary_once_and_records_visibility(finished_run):
    before = {str(p.relative_to(finished_run)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in finished_run.rglob('*') if p.is_file()}
    value = analysis.build_analysis(finished_run)
    text = analysis.markdown(value)
    assert value['counts']['results'] == 10
    assert value['groups']['all']['M_matched']['usage']['input_tokens'] == 70
    assert value['groups']['full_history_budget']['within']['M_matched']['summary']['usage']['input_tokens'] == 10
    assert value['groups']['full_history_budget']['over']['M_matched']['summary']['usage']['input_tokens'] == 0
    assert value['groups']['checkpoint_turns']['1']['W_matched']['tasks'] == 1
    assert value['groups']['history_turns']['64']['W_matched']['tasks'] == 0
    assert value['groups']['all']['W_matched']['two_checkpoint_api_latency']['sum_seconds'] == 6
    for row in value['matched_context_visibility']:
        assert row['raw_message_ids'] == ['db-1']
        expected_count = 0 if row['arm'] == 'M_matched' else 1
        assert row['gold_quote_visibility']['coref']['visible_count'] == expected_count
    assert all(pair['mean_case_task_rate_delta_second_minus_first'] == 0 for pair in value['comparisons'].values())
    assert '第一问模型回复未写入第二问历史' in text
    assert '本分层没有样本' in text
    after = {str(p.relative_to(finished_run)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in finished_run.rglob('*') if p.is_file()}
    assert before == after


def test_incomplete_audit_is_not_analyzed(finished_run):
    put(finished_run / 'audited_report.json', {'complete': False, 'audit_passed': False})
    with pytest.raises(ValueError, match='incomplete or failed'):
        analysis.build_analysis(finished_run)
    assert not (finished_run / 'ANALYSIS.json').exists()


def test_changed_dataset_is_rejected(finished_run):
    data = analysis.read(finished_run / 'dataset.json')
    data['cases'][0]['history'][0]['content'] = 'changed history'
    put(finished_run / 'dataset.json', data)
    with pytest.raises(ValueError, match='dataset differs'):
        analysis.build_analysis(finished_run)


def test_missing_usage_is_visible_as_unknown():
    incomplete = {'attempts': [{'usage': {'input_tokens': 5}}]}
    result = analysis.usage([incomplete])
    assert result['input_tokens'] == 5
    assert result['unknown_attempts_by_field']['output_tokens'] == 1
    assert analysis.token_display(result, 'output_tokens') == '≥0'
    assert not result['complete_usage']


def test_estimator_mismatch_rejected(finished_run):
    snapshot = finished_run / 'source_snapshot/app/core/context_policy.py'
    changed = snapshot.read_text(encoding='utf-8').replace('cjk * 1.2', 'cjk * 1.1')
    snapshot.write_text(changed, encoding='utf-8')
    with pytest.raises(ValueError, match='estimator differs'):
        analysis.verify_estimator(snapshot)
