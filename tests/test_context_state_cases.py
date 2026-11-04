from collections import Counter
from copy import deepcopy

import pytest

from scripts.context_state_cases import FAMILIES, dataset, export, validate_dataset


def test_cases_are_deterministic_split_labeled_and_causal(tmp_path):
    data = dataset(); validate_dataset(data)
    assert data == dataset()
    assert Counter(c['split'] for c in data['cases']) == {'development': 12, 'reserved': 24}
    assert set(c['kind'] for c in data['cases']) == set(FAMILIES)
    assert all(len(c['steps']) == 3 for c in data['cases'])
    assert len(set(c['id'] for c in data['cases'])) == 36
    assert 'human-reviewed' in data['scope']
    path = tmp_path / 'new' / 'dataset.json'
    assert len(export(path)) == 64
    with pytest.raises(ValueError):
        export(path)


def test_gold_cannot_reference_a_future_source():
    data = deepcopy(dataset())
    data['cases'][0]['steps'][0]['gold_source_texts'] = ['future information not supplied']
    with pytest.raises(ValueError, match='visible history'):
        validate_dataset(data)


def test_reserved_templates_and_pressure_are_not_just_id_changes():
    data = dataset()['cases']
    dev = next(c for c in data if c['id'] == 'dev-01-0')
    reserved = next(c for c in data if c['id'] == 'res-01-0')
    long = next(c for c in data if c['id'] == 'res-01-1')
    assert '购买凭证' in reserved['history'][0]['content']
    assert '购买凭证' not in dev['history'][0]['content']
    assert len(long['history']) > len(reserved['history']) > len(dev['history'])
