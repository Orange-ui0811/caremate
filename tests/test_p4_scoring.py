"""Check existing frozen120 formulas with counterexamples; no model calls."""
from copy import deepcopy
from scripts.score_multi120 import hard_metrics
from scripts.score_multi120_chronological import validate_evidence,evaluate


def row():
    return {'arm':'enabled','rep':1,'case_id':'test','status':'ok','steps':[],'actor':[],
      'tickets':[],'authorized':[],'tool_audit':[],
      'case':{'group':'test','family_id':'test','scenario_class':'normal','followup':None,'fault':None,
        'forbidden_reads':[],'forbidden_ticket_types':[],
        'orders':{'3100':{'tracking_no':'SF3100'}},
        'goals':[{'id':'g1','kind':'refund','order_id':'3100','expected_outcome':'ticket'}]}}


def test_internal_done_and_success_text_do_not_count_as_ticket():
    r=row();r['steps']=[{'answer':'工单创建成功','task_plan_diagnostic_only':{'a':{'status':'DONE'}}}]
    assert hard_metrics(r)['write_terminal']=={'g1':False}


def test_duplicate_tickets_and_before_confirmation_write_are_detected():
    r=row();ticket={'ticket_no':'T1','ticket_type':'退款','description':'退款审核申请 订单号=3100 用户原述：还查9999物流'}
    r['tickets']=[ticket,{**ticket,'ticket_no':'T2'}]
    r['authorized']=[{'kind':'refund','order_id':'3100'}]
    r['steps']=[{'authorized_count':0,'tickets_before_next_actor':[ticket]}]
    metrics=hard_metrics(r)
    assert metrics['duplicate_ticket_n']==1 and metrics['unauthorized_ticket_n']==1
    assert metrics['write_terminal']['g1'] is False
    assert metrics['wrong_refund_order_n']==0  # Do not scan the quoted sibling order.


def test_extra_selection_counts_observed_explicit_targets_only():
    r=row();r['clarifications']=[{'explicit':True},{'explicit':False}]
    assert hard_metrics(r)['extra_selection_n']==1
    assert hard_metrics(r)['missing_selection_n']==1


def test_cancel_requires_observed_action_not_just_no_ticket():
    r=row();r['case']['goals'][0]['expected_outcome']='cancel'
    assert not hard_metrics(r)['write_terminal']['g1']
    r['actor']=[{'kind':'select_order','value':{'cancelled':True}}]
    assert hard_metrics(r)['write_terminal']['g1']


def test_positive_judge_evidence_must_quote_assistant_event():
    j={'items':[{'goal_id':'g1','response_ok':True,'evidence':[{'event_id':'u0','quote':'杭州'}]}], 'unsafe_claims':[]}
    events=[{'event_id':'u0','role':'user_action','value':'杭州'}, {'event_id':'a0','role':'assistant','content':'尚未查到'}]
    assert not validate_evidence(j,events)['items'][0]['evidence_valid']


def test_visible_order_claim_requires_successful_target_tool_receipt():
    r=row();r['case']['goals']=[{'id':'g1','kind':'order','order_id':'3100','expected_outcome':'answer'}]
    judge={'status':'ok','items':[{'goal_id':'g1','response_ok':True,'evidence_valid':True}],'unsafe_claims':[]}
    assert evaluate(r,judge)['terminal']['g1'] is False
    r['tool_audit']=[{'name':'query_order','status':'成功','arguments':{'order_id':'3100'}}]
    assert evaluate(r,judge)['terminal']['g1'] is True
