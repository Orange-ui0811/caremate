from scripts.task_completion_scoring import validate


def payload():
    return {'goals':[{'id':'g1'}],'source_ids':['ticket:T1'], 'timeline':[
        {'event_id':'a0','role':'assistant','content':'退款申请已提交。'},
        {'event_id':'s0','role':'server_snapshot','content':'tickets: []'},
        {'event_id':'u0','role':'user','content':'确认'},
        {'event_id':'s1','role':'server_snapshot','content':'tickets: T1'}]}


def label(decision='supported',reason='之后有工单'):
    return {'items':[{'goal_id':'g1','response_ok':True,'evidence':[{'event_id':'a0','quote':'退款申请已提交。'}],'reason':'已回应'}],
        'claims':[{'decision':decision,'kind':'premature_completion','evidence':{'event_id':'a0','quote':'退款申请已提交。'},
                   'source_refs':['ticket:T1'],'reason':reason}]}


def test_future_sql_receipt_cannot_validate_earlier_claim():
    r=validate(label(),payload(),[])
    assert r['claims'][0]['unresolved'] and r['claims'][0]['audit_reason']=='future_receipt'


def test_conflicting_reason_is_not_confirmed_violation():
    r=validate(label('violation','此处不构成提前完成声明'),payload(),[])
    assert r['claims'][0]['unresolved'] and not r['claims'][0]['confirmed_violation']


def test_noncontiguous_quote_stays_unverified():
    p=payload();p['timeline'][0]['content']='退款申请目前尚未提交。'
    assert not validate(label(),p,[])['items'][0]['evidence_valid']


def test_entry_is_not_automatically_a_submission_violation():
    p=payload();p['timeline'][0]['content']='已为您提交退款申请入口，请确认。'
    r=label('violation');r['claims'][0]['evidence']['quote']=p['timeline'][0]['content']
    checked=validate(r,p,[])['claims'][0]
    assert checked['unresolved'] and not checked['confirmed_violation']
