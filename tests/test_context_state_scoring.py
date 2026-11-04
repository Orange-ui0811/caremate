from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.context_state_cases import make_case
from scripts.context_state_scoring import identifier_mentions, score


def fixture():
    case = make_case('early_order', 'development'); step = case['steps'][0]
    oid = step['expected']['order_id']
    record = lambda value: {'ok': True, 'status': 'ok', 'value': value, 'calls': [{'input_tokens': 1}]}
    result = {'visible_rows': case['history'], 'context_text': '',
        'coref': record(f'查询订单{oid}物流'),
        'plan': record({'action': 'logistics', 'order_id': oid, 'sku': '', 'answer': ''}),
        'reply': record(f'订单{oid}的运单SF{oid}88运输中。'),
        'tool_result': case['tool_data']['orders'][oid]}
    return case, step, result


def test_explicit_unused_old_order_does_not_fail_but_actual_wrong_argument_does():
    case, step, result = fixture(); other = case['all_orders'][1]
    result['reply']['value'] += f'未查询订单{other}。'
    value = score(case, step, result)
    assert value['task_pass'] and value['nonuse_mentions']
    assert not value['legacy']['task_pass']
    result['plan']['value']['order_id'] = other
    assert not score(case, step, result)['target_pass']


def test_fabrication_checks_four_digit_ids_and_future_fixture_ids():
    case, step, result = fixture()
    result['coref']['value'] = '查询订单1001。'
    assert '1001' in score(case, step, result)['fabricated_identifiers']
    result['coref']['value'] = '查询订单' + case['all_orders'][3]
    assert case['all_orders'][3] in score(case, step, result)['fabricated_identifiers']


def test_negation_about_one_order_does_not_excuse_asserting_another():
    rows = identifier_mentions('未查订单1001，已查询订单1002。', ['1001', '1002'])
    assert {r['value']: r['nonuse'] for r in rows} == {'1001': True, '1002': False}


def test_protocol_and_transport_are_not_scored_as_memory_failures():
    case, step, result = fixture()
    result['plan'].update(ok=False, status='parse_error', value=None)
    value = score(case, step, result)
    assert value['protocol_failure'] and not value['transport_failure'] and not value['task_pass']
    result['plan']['status'] = 'transport_error'
    value = score(case, step, result)
    assert value['transport_failure'] and not value['protocol_failure']


def test_tool_failure_requires_honest_reply_not_invented_tracking():
    case, step, result = fixture(); result['tool_result'] = {'error': '暂不可用'}
    result['reply']['value'] = '物流查询暂时失败，无法核实，请稍后再试。'
    assert score(case, step, result)['task_pass']
    result['reply']['value'] += '运单SF123488。'
    assert not score(case, step, result)['answer_rule_pass']


def test_identifier_diagnostics_are_deterministic_across_hash_seeds(tmp_path):
    # Compile the actual pure extractor without importing app/model/database setup.
    # Distinct processes exercise Python set randomization rather than comparing
    # repeated calls in one interpreter (which would miss the original defect).
    source = Path(__file__).resolve().parents[1] / 'scripts/context_state_scoring.py'
    code = '''
import ast, json, re, sys
tree = ast.parse(open(sys.argv[1], encoding="utf-8").read())
nodes = [n for n in tree.body if isinstance(n, ast.Assign) or
         isinstance(n, ast.FunctionDef) and n.name == "identifier_mentions"]
scope = {"re": re}; exec(compile(ast.Module(body=nodes, type_ignores=[]), sys.argv[1], "exec"), scope)
text = "未查询订单8341，订单8340暂不提交退款，已查询订单8342。未查询订单8340。"
print(json.dumps(scope["identifier_mentions"](text, ["8342", "8340", "8341"]), ensure_ascii=True))
'''
    outputs = []
    for seed in ('0', '1', '7', '42', '20261027'):
        env = {key: os.environ[key] for key in ('SystemRoot', 'WINDIR') if key in os.environ}
        env.update(PYTHONHASHSEED=seed, PYTHONIOENCODING='utf-8')
        result = subprocess.run([sys.executable, '-c', code, str(source)], cwd=tmp_path,
                                env=env, capture_output=True, text=True, encoding='utf-8', timeout=20)
        assert result.returncode == 0, result.stderr
        outputs.append(json.loads(result.stdout))
    assert all(value == outputs[0] for value in outputs)
    assert [(r['value'], r['nonuse']) for r in outputs[0]] == [
        ('8340', True), ('8341', True), ('8342', False), ('8340', True)]
