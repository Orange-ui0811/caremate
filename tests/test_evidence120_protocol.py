import json

from scripts.multi120_evidence_protocol import cooperative_order_reply, interaction_events, selectors, validate_labels
from scripts.rescore_multi120_evidence import evaluate


def raw():
    return {'arm':'enabled','rep':1,'case_id':'test','status':'ok','steps':[], 'actor':[],
            'tickets':[], 'authorized':[], 'tool_audit':[], 'dialogue':[],
            'case':{'group':'test','family_id':'test','scenario_class':'normal','followup':None,'fault':None,
                    'orders':{'3100':{'tracking_no':'SF3100'}},'logistics':{},'forbidden_reads':[], 'forbidden_ticket_types':[],
                    'goals':[{'id':'g1','kind':'refund','order_id':'3100','explicit_order':False,'expected_outcome':'ticket'}]}}


def test_stale_prompt_in_cumulative_answer_does_not_reinject_order():
    case=raw()['case'];before='请提供订单号。'
    assert cooperative_order_reply(case,{'answer':before+'\n\n工单已创建。'},before,[]) is None
    assert cooperative_order_reply(case,{'answer':before},'',[]) is not None
    assert cooperative_order_reply(case,{'answer':before},'',[{'kind':'select_order','value':'3100'}]) is None


def test_new_order_question_after_another_answer_still_can_be_answered():
    assert cooperative_order_reply(raw()['case'],{'answer':'政策已介绍。\n\n请提供退款订单号。'},'政策已介绍。',[]) is not None


def test_unmapped_selector_is_counted_without_fabricating_necessity():
    r=raw();r['steps']=[{'answer':'','interrupt':{'type':'select_order','interrupt_id':'one'}}]
    r['actor']=[{'kind':'unresolvable_selector'}]
    assert selectors(r)['observed_selector_n']==1
    assert selectors(r)['unmapped_selector_n']==1


def test_form_cancel_is_between_preview_and_next_reply():
    r=raw();r['steps']=[{'answer':'请核对','interrupt':{'type':'task_form','interrupt_id':'one'}},
                       {'answer':'请核对\n\n已取消','interrupt':None}]
    r['actor']=[{'kind':'cancel_form','goal_id':'g1'}, {'kind':'task_form','interrupt_id':'one','value':{'confirmed':False}}]
    events,unplaced=interaction_events(r)
    cancel=next(i for i,e in enumerate(events) if e.get('action',{}).get('kind')=='cancel_form')
    reply=next(i for i,e in enumerate(events) if e.get('content')=='已取消')
    assert cancel<reply and not unplaced


def test_supported_claim_never_fails_because_it_is_present_in_claims():
    events=[{'event_id':'a0','role':'assistant','content':'退款入口已打开，请确认。'}]
    data={'items':[{'goal_id':'g1','response_ok':True,'evidence':[{'event_id':'a0','quote':'退款入口已打开，请确认。'}]}],
          'claims':[{'decision':'supported','evidence':{'event_id':'a0','quote':'退款入口已打开，请确认。'},'source_refs':['card0'],'reason':'待确认入口不是已退款。'}]}
    d=validate_labels(data,events,['card0'],['g1'])
    assert not d['claims'][0]['confirmed_violation'] and not d['claims'][0]['unresolved']


def test_reason_saying_no_risk_cannot_create_confirmed_violation():
    d={'items':[], 'claims':[{'decision':'violation','evidence':{'event_id':'a0','quote':'已打开入口'},'source_refs':['card0'],'reason':'不构成风险。'}]}
    c=validate_labels(d,[{'event_id':'a0','role':'assistant','content':'已打开入口'}],['card0'],[])['claims'][0]
    assert not c['confirmed_violation'] and c['unresolved'] and not c['decision_consistent']


def test_real_quote_without_valid_counterevidence_is_unknown_not_violation():
    d={'items':[], 'claims':[{'decision':'violation','evidence':{'event_id':'a0','quote':'6元'},'source_refs':['kb:invented'],'reason':'收费错误。'}]}
    c=validate_labels(d,[{'event_id':'a0','role':'assistant','content':'6元'}],[],[])['claims'][0]
    assert not c['confirmed_violation'] and c['unresolved']


def test_rejected_text_cancellation_is_not_successful_cancellation():
    r=raw();r['case']['goals'][0]['expected_outcome']='cancelled';r['case']['followup']='取消退款'
    r.update(status='error',error_type='ResumeRejected',actor=[{'kind':'user_message','text':'取消退款'}])
    d={'timeline':[],'unplaced_actions':[],'gold_issues':{}}
    out=evaluate(r,{'status':'no_answer','items':[],'claims':[]},d)
    assert out['terminal']['g1'] is False and out['success'] is False


def test_conflicting_gold_keeps_denominator_and_unknown():
    r=raw();r['case']['goals']=[{'id':'g1','kind':'knowledge','expected_outcome':'answer'}]
    data={'timeline':[],'unplaced_actions':[],'gold_issues':{'g1':'6元与10元冲突'}}
    j={'status':'ok','items':[{'goal_id':'g1','response_ok':True,'evidence_valid':True}],'claims':[]}
    out=evaluate(r,j,data)
    assert out['terminal']=={'g1':None} and out['success'] is None


def test_rejudge_disables_both_configured_and_vendor_fallback(monkeypatch, tmp_path):
    from app.config import settings
    from scripts import rescore_multi120_evidence as scorer
    names=['chat_model','model_pool_db_path','model_pool_background_limit','model_pool_queue_timeout',
           'langfuse_public_key','chat_failover_enabled','model_pool_fallback_base_url',
           'model_pool_fallback_api_key','model_pool_fallback_model']
    for name in names:monkeypatch.setattr(settings,name,getattr(settings,name))
    monkeypatch.setattr(scorer,'OUT',tmp_path)
    (tmp_path/'protocol.json').write_text(json.dumps({'model_config':{'chat_model':'DeepSeek-V4-Flash'}}),encoding='utf-8')
    settings.model_pool_fallback_base_url='https://unused.invalid'
    settings.model_pool_fallback_api_key='test-only'
    settings.model_pool_fallback_model='different-model'
    scorer.wire()
    assert settings.chat_failover_enabled is False
    assert not settings.model_pool_fallback_base_url and not settings.model_pool_fallback_api_key
    assert not settings.model_pool_fallback_model


def test_driver_blockage_is_not_silently_blame_assigned_to_business():
    from scripts.audit_multi120_failures import classify
    r=raw();r['actor']=[{'kind':'unresolvable_selector'}]
    out=classify(r,{'timeline':[],'unplaced_actions':[],'gold_issues':{}},'')
    assert out['category']=='driver_blocked' and not out['confirmed_failure_by_existing_business_contract']


def test_caught_provider_error_still_has_external_provenance():
    from scripts.audit_multi120_failures import classify
    out=classify(raw(),{'timeline':[],'unplaced_actions':[],'gold_issues':{}},'',
                 [{'http_status':400,'outcome':'FAILED','model':'DeepSeek-V4-Flash'}])
    assert out['category']=='external_service_interference'


def test_missing_required_ticket_is_contract_failure_but_not_policy_judgment():
    from scripts.audit_multi120_failures import classify
    out=classify(raw(),{'timeline':[],'unplaced_actions':[],'gold_issues':{}},'')
    assert out['category']=='required_write_not_completed'
    assert out['write_outcomes'][0]['receipt_outcome'] is False
    assert out['semantic_verdict']=='not_determined_by_this_offline_audit'
