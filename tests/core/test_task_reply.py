import json
from app.core.task_reply import compose


def state():
    order={'order_id':'1001','product':'智能猫砂盆','status':'已付款','amount':1739,'tracking_no':'SF-test'}
    shipping={'tracking_no':'SF-test','status':'已揽件','current_city':'成都','trace':['成都分拨中心已发出','内部状态码:PICKED_UP']}
    receipt=lambda name,value:{'name':name,'content':json.dumps(value,ensure_ascii=False)}
    return {'task_plan':{'order':{'kind':'order','status':'DONE','order_id':'1001'},
                         'logistics':{'kind':'logistics','status':'DONE','order_id':'1001'}},
            'subflow_results':{'order':{'answer':'需要我查物流吗？','receipts':[receipt('query_order',order)]},
                'logistics':{'answer':'本次事项处理结果 DONE。还需要其他帮助吗？','receipts':[
                    receipt('query_order',order),receipt('query_logistics',shipping)]}}}


def test_combines_verified_order_logistics_without_model_closings_or_internal_trace():
    text,cites,seen=compose(state())
    assert text.count('**订单 1001')==1
    assert all(s in text for s in ['智能猫砂盆','1739','SF-test','成都','已揽件'])
    assert not any(s in text for s in ['需要我','其他帮助','本次事项','DONE','PICKED_UP','内部'])


def test_seen_results_are_not_repeated_after_confirmation():
    s=state();_,_,s['subflow_presented']=compose(s)
    assert compose(s)[0]==''
    s['task_plan']['refund']={'kind':'refund','status':'DONE','receipt':{'ticket_no':'T123'}}
    text,_,_=compose(s)
    assert 'T123' in text and '成都' not in text and '1739' not in text


def test_missing_or_mismatched_receipts_do_not_generate_business_facts():
    s=state();s['subflow_results']['logistics']['receipts'][-1]['content']='{"tracking_no":"OTHER","current_city":"上海"}'
    text,_,_=compose(s)
    assert '上海' not in text and '未能核实物流' in text
    s['subflow_results']={}
    assert '暂时无法核实' in compose(s)[0]


def test_two_knowledge_answers_have_global_citations_and_no_closings():
    s={'task_plan':{'a':{'kind':'knowledge','status':'DONE','citations':[{'n':1,'id':4,'answer':'甲政策'}]},
                    'b':{'kind':'knowledge','status':'DONE','citations':[{'n':1,'id':5,'answer':'乙政策'}]}},
       'subflow_results':{'a':{'answer':'甲政策[1]\n\n还有其他问题吗？'},'b':{'answer':'乙政策[1]'}}}
    text,cites,_=compose(s)
    assert '甲政策[1]' in text and '乙政策[2]' in text and '其他问题' not in text
    assert [c['id'] for c in cites]==[4,5]


def test_pending_cancel_and_unknown_submission_are_truthful():
    s={'task_plan':{'refund':{'kind':'refund','status':'WAITING_USER','step':'confirm_submission','order_id':'1001'},
                    'complaint':{'kind':'complaint','status':'CANCELLED'}}}
    text,_,_=compose(s)
    assert '尚未提交' in text and '未创建工单' in text and '已提交' not in text
    s['task_plan']['refund']['retry_submission']=True
    assert '暂时无法确认' in compose(s)[0]


def test_cancel_after_unknown_success_displays_actual_receipt():
    s={'task_plan':{'refund':{'kind':'refund','status':'DONE','retry_submission':True,'receipt':{'ticket_no':'T-original'}}}}
    text,_,_=compose(s)
    assert 'T-original' in text and '取消' not in text


def test_internal_knowledge_protocol_is_not_displayed():
    s={'task_plan':{'a':{'kind':'knowledge','status':'DONE'}},'subflow_results':{'a':{'answer':'task_id:123 DONE'}}}
    text,_,_=compose(s)
    assert 'task_id' not in text and 'DONE' not in text
