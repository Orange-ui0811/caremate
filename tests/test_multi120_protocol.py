import copy
from collections import Counter

from scripts.multi120_cases import build_cases,original
from scripts.score_multi120 import hard_metrics,evaluate


def result(c):
    return {'case':c,'case_id':c['id'],'arm':'enabled','rep':1,'status':'ok','steps':[],
            'tickets':[],'actor':[],'authorized':[],'tool_audit':[]}


def test_exactly120_unique_scenarios_with_provenance_and_targets():
    cases=build_cases();src=original()
    assert len(cases)==len({c['id'] for c in cases})==len({c['query'] for c in cases})==120
    assert set(Counter(c['group'] for c in cases).values())=={15}
    assert len({c['family_id'] for c in cases})==40
    for c in cases:
        assert c['source_question_ids'] and all(i in src for i in c['source_question_ids'])
        for g in c['goals']:
            if g.get('order_id'):assert g['order_id'] in c['orders']
            if g['kind']=='knowledge':assert g['required_points'] and g['source']['id'] in c['source_question_ids']


def test_two_refunds_are_not_duplicates_but_second_same_order_is():
    c=next(c for c in build_cases() if c['group']=='binding' and len([g for g in c['goals'] if g['kind']=='refund'])==2)
    r=result(c)
    for g in c['goals']:
        r['tickets'].append({'ticket_no':g['id'],'ticket_type':'退款','description':'退款申请 订单号='+g['order_id']})
        r['authorized'].append({'goal_id':g['id'],'kind':'refund','order_id':g['order_id']})
    m=hard_metrics(r)
    assert all(m['write_terminal'].values()) and m['duplicate_ticket_n']==m['wrong_ticket_n']==0
    r['tickets'].append({**r['tickets'][0],'ticket_no':'duplicate'})
    assert hard_metrics(r)['duplicate_ticket_n']==1


def test_wrong_order_cannot_hide_in_original_user_quote():
    c=next(c for c in build_cases() if c['group']=='binding' and len(c['goals'])==3)
    r=result(c);correct=next(g['order_id'] for g in c['goals'] if g['kind']=='refund')
    wrong=next(g['order_id'] for g in c['goals'] if g['kind']=='logistics')
    r['tickets']=[{'ticket_no':'wrong','ticket_type':'退款','description':f'退款审核申请 订单号={wrong} 用户原述：我要给{correct}退款'}]
    assert hard_metrics(r)['wrong_refund_order_n']==1


def test_eventual_confirmation_does_not_authorize_an_earlier_write():
    c=next(c for c in build_cases() if any(g['kind']=='refund' for g in c['goals']))
    g=next(g for g in c['goals'] if g['kind']=='refund');r=result(c)
    ticket={'ticket_no':'early','ticket_type':'退款','description':'退款申请 订单号='+g['order_id']}
    r['tickets']=[ticket];r['authorized']=[{'goal_id':g['id'],'kind':'refund','order_id':g['order_id']}]
    r['steps']=[{'authorized_count':0,'tickets_before_next_actor':[ticket]}]
    assert hard_metrics(r)['unauthorized_ticket_n']==1


def test_visible_success_cannot_replace_verified_read():
    c=next(c for c in build_cases() if any(g['kind']=='order' for g in c['goals']));r=result(c)
    j={'status':'ok','items':[{'goal_id':g['id'],'response_ok':True} for g in c['goals']],'unsafe_claims':[]}
    assert not evaluate(r,j)['success']


def test_judge_failure_is_unknown_and_source_error_still_fails():
    c=next(c for c in build_cases() if all(g['kind']=='knowledge' for g in c['goals']))
    r=result(c);r['steps']=[{'answer':'有答复，但裁判不可用'}]
    assert evaluate(r,{'status':'judge_error'})['success'] is None
    r['status']='error'
    assert evaluate(r,{'status':'judge_error'})['success'] is False


def test_parallel_process_ticket_namespaces_do_not_collide_at_same_timestamp(monkeypatch):
    from datetime import datetime
    from scripts.eval_multi120 import ticket_sequence_namespace,ARMS
    from app.db import repository
    class FixedClock:
        @classmethod
        def now(cls):return datetime(2026,11,3,22,0,0)
    monkeypatch.setattr(repository,'datetime',FixedClock)
    monkeypatch.setattr(repository,'_TICKET_SEQ',0)
    first=repository._gen_ticket_no()
    repository._TICKET_SEQ=0
    assert repository._gen_ticket_no()==first  # Reproduce separate-process collision.
    identifiers=set()
    for rep in (1,2,3):
      for arm in ARMS:
       for case_index in range(120):
        repository._TICKET_SEQ=ticket_sequence_namespace(arm,case_index,rep)
        for _ in range(16):
            new=repository._gen_ticket_no()
            assert new not in identifiers
            identifiers.add(new)


def test_chronological_judge_sees_confirmation_between_preview_and_receipt():
    from scripts.score_multi120_chronological import timeline
    r={'case':{'followup':None},'steps':[
        {'answer':'尚未建单','interrupt':{'interrupt_id':'one'}},
        {'answer':'尚未建单\n\n工单已创建：T1','interrupt':None}],
        'actor':[{'kind':'task_actions','interrupt_id':'one','value':{'confirmed':True}}]}
    events=timeline(r)
    assert [x['role'] for x in events]==['assistant','user_action','assistant']
    assert events[2]['content']=='工单已创建：T1'


def test_bad_judge_quote_does_not_discard_other_valid_items_or_pass_itself():
    from scripts.score_multi120_chronological import validate_evidence
    events=[{'event_id':'a0','role':'assistant','content':'订单已签收，金额289元。'}]
    j={'items':[
        {'goal_id':'g1','response_ok':True,'evidence':[{'event_id':'a0','quote':'金额289元'}]},
        {'goal_id':'g2','response_ok':True,'evidence':[{'event_id':'a0','quote':'退款已到账'}]}], 'unsafe_claims':[]}
    checked=validate_evidence(j,events)
    assert checked['items'][0]['evidence_valid'] is True
    assert checked['items'][1]['evidence_valid'] is False
