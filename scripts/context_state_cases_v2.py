"""V2 pressure stratum; V1 questions, fixtures and gold remain unchanged.

Only the predeclared reserved long variant's background assistant descriptions
are lengthened. These repetitions are synthetic context-pressure material, not
additional independent conversations or realistic user traffic.
"""
from __future__ import annotations

import hashlib
import json

from app.core.context_policy import estimate
from scripts import context_state_cases as original

FAMILIES = original.FAMILIES
TARGET_HISTORY_ESTIMATE = 15000


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def lengthen(case):
    before = _hash(case)
    original_estimate = sum(estimate(r['content']) for r in case['history'])
    background = dict(original.BACKGROUND)
    eligible = []
    for index in range(1, len(case['history'])):
        previous, row = case['history'][index - 1:index + 1]
        if previous['role'] == 'user' and previous['content'] in background and row['role'] == 'assistant':
            eligible.append((index, background[previous['content']]))
    if not eligible:
        raise ValueError('long variant lacks declared background')
    additions = 0
    while sum(estimate(r['content']) for r in case['history']) < TARGET_HISTORY_ESTIMATE:
        for index, text in eligible:
            case['history'][index]['content'] += text
        additions += 1
    case['pressure_construction'] = {
        'original_case_sha256': before,
        'original_history_estimate': original_estimate,
        'history_estimate': sum(estimate(r['content']) for r in case['history']),
        'target_estimate': TARGET_HISTORY_ESTIMATE,
        'background_rows_extended': [i + 1 for i, _ in eligible],
        'extra_repetitions_per_background_row': additions,
        'method': 'Append original background response text equally to existing background assistant rows; preserve all user text, facts, steps, fixtures and scoring labels.'}


def dataset():
    value = original.dataset()
    for case in value['cases']:
        if case['split'] == 'reserved' and case['length_stratum'] == 'long':
            lengthen(case)
    value['version'] = 'context-state-cases-v2'
    value['scope'] += (' V2 lengthens only the 12 predeclared reserved long variants to at least '
                       '15000 estimated history tokens using repetitive background; not natural-traffic length distribution.')
    value['v1_dataset_sha256'] = _hash(original.dataset())
    return value


def validate_dataset(value):
    original.validate_dataset(value)
    baseline = {c['id']: c for c in original.dataset()['cases']}
    for case in value['cases']:
        old = baseline[case['id']]
        changed = case['split'] == 'reserved' and case['length_stratum'] == 'long'
        if not changed:
            if case != old:
                raise ValueError('short/development case changed')
            continue
        for field in old:
            if field != 'history' and case[field] != old[field]:
                raise ValueError('non-background field changed: ' + field)
        allowed = set(case['pressure_construction']['background_rows_extended'])
        if len(case['history']) != len(old['history']):
            raise ValueError('history shape changed')
        for index, (row, prior) in enumerate(zip(case['history'], old['history']), 1):
            if index in allowed:
                if row['role'] != 'assistant' or not row['content'].startswith(prior['content']):
                    raise ValueError('background replacement is not append-only')
            elif row != prior:
                raise ValueError('non-background history changed')
        if sum(estimate(r['content']) for r in case['history']) < TARGET_HISTORY_ESTIMATE:
            raise ValueError('declared pressure case is too short')
