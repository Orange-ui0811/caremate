from copy import deepcopy

import pytest

from scripts.aggregate_context_state import combine


def fixture():
    manifest = {'arms': ['A', 'R'], 'dataset_sha256': 'fixed'}
    def stats(values):
        return {'n': 6, 'task_passed': sum(sum(v) for v in values.values()), 'target_passed': 6,
                'transport_failed': 0, 'protocol_failed': 0, 'budget_blocked': 0,
                'usage': {'input_tokens': 100, 'output_tokens': 10, 'total_tokens': 110},
                'per_conversation': values}
    report = {'complete': True, 'audit_passed': True,
              'strategies': {'A': stats({'a': [False, False, True], 'b': [False, True, True]}),
                             'R': stats({'a': [True, True, True], 'b': [True, True, True]})},
              'paired_comparisons': {'A->R': {}}}
    return manifest, report


def test_repeated_steps_do_not_inflate_conversation_count():
    m, r = fixture(); out = combine([m, deepcopy(m)], [r, deepcopy(r)])
    assert out['paired_comparisons']['A->R']['n_conversations'] == 2
    assert out['paired_comparisons']['A->R']['mean_difference_pp'] == pytest.approx(50)
    assert out['strategies']['A']['repetitions'] == 2
    assert out['strategies']['A']['mean_usage_per_run']['total_tokens'] == 110


def test_changed_protocol_and_incomplete_run_are_rejected():
    m, r = fixture(); changed = deepcopy(m); changed['context_budget'] = 1
    with pytest.raises(ValueError, match='protocol'):
        combine([m, changed], [r, r])
    bad = deepcopy(r); bad['complete'] = False
    with pytest.raises(ValueError, match='incomplete'):
        combine([m, m], [r, bad])


def test_repetitions_cannot_share_call_pool_namespace():
    m, r = fixture()
    first = {**m, 'repeat_id': '1', 'request_id_namespace': 'think-same'}
    second = {**m, 'repeat_id': '2', 'request_id_namespace': 'think-same'}
    with pytest.raises(ValueError, match='namespaces'):
        combine([first, second], [r, r])
