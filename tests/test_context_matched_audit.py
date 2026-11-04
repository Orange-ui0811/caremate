import copy
import json

import pytest

from scripts.audit_context_matched_results import audit


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


def fixture_run(out):
    arm = 'W_matched'
    prefix = 'case-p1-'+arm
    tool = {'order_id':'1001','tracking_no':'SF100188','status':'运输中'}
    case = {'id':'case','cohort':'regression','kind':'early_order',
        'history':[{'role':'user','content':'订单1001'},{'role':'assistant','content':'好的'}],
        'tool_data':{'orders':{'1001':tool},'manuals':{}},
        'probes':[{'id':'p1','at_turn':1,'query':'那笔订单在哪里？',
            'expected':{'action':'logistics','order_id':'1001'},
            'gold_evidence':[{'seq':1,'start':0,'end':6,'text':'订单1001','role':'user'}]}]}
    put(out/'dataset.json',{'cases':[case]})
    put(out/'manifest.json',{'created_at':'2020-01-01T00:00:00+00:00','arms':[arm],'sources':{}})
    result = {'score':{'task_rule_v3':True,'target_pass':True},'tool_result':tool}
    for stage in ('coref','plan','reply'):
        value = {'action':'logistics','order_id':'1001','sku':''} if stage=='plan' else '订单1001'
        attempt = {'status':'ok','value':value,'started_at':'2020-01-01T00:00:01+00:00',
            'usage':{'input_tokens':10,'output_tokens':1,'total_tokens':11}}
        receipt = {'status':'ok','ok':True,'value':value,'attempts':[attempt],
            'contract':{'stage':stage,'messages':[{'content':'订单1001'}]}}
        put(out/'calls'/f'{prefix}-{stage}.json', receipt)
        put(out/'attempts'/f'{prefix}-{stage}-1.json', attempt)
        result[stage] = receipt
    put(out/'results'/f'{prefix}.json',result)
    put(out/'maintenance'/f'case-{arm}.json',[])
    stats = {'n':1,'passed':1,'target_passed':1,'usage':{'input_tokens':30,'output_tokens':3,'total_tokens':33}}
    report = {'complete':True,'audit_passed':True,'strategies':{arm:{'all':copy.deepcopy(stats),'regression':copy.deepcopy(stats)}}}
    put(out/'audited_report.json',report)
    return prefix


def test_independent_accounting_accepts_consistent_receipts(tmp_path):
    fixture_run(tmp_path)
    result = audit(tmp_path,check_pool=False)
    assert result['passed']
    assert result['counts']=={'calls':3,'attempts':3,'results':1,'maintenance':1}
    assert result['exact_source_visibility'][0]['stages']['coref']=={'exact_quotes_present':1,'gold_quotes':1}


@pytest.mark.parametrize('tamper', ['token_total','tool_result','raw_attempt','future_gold'])
def test_independent_audit_detects_corruption(tmp_path,tamper):
    prefix = fixture_run(tmp_path)
    if tamper=='token_total':
        path = tmp_path/'audited_report.json'
        value = json.loads(path.read_text(encoding='utf-8'))
        value['strategies']['W_matched']['all']['usage']['input_tokens'] = 0
    elif tamper=='tool_result':
        path = tmp_path/'results'/f'{prefix}.json'
        value = json.loads(path.read_text(encoding='utf-8'))
        value['tool_result']['order_id'] = '9999'
    elif tamper=='raw_attempt':
        path = tmp_path/'attempts'/f'{prefix}-plan-1.json'
        value = json.loads(path.read_text(encoding='utf-8'))
        value['value']['order_id'] = '9999'
    else:
        path = tmp_path/'dataset.json'
        value = json.loads(path.read_text(encoding='utf-8'))
        value['cases'][0]['probes'][0]['gold_evidence'][0]['seq'] = 3
    put(path,value)
    assert not audit(tmp_path,check_pool=False)['passed']
