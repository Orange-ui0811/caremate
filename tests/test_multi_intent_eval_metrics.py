from scripts.judge_multi_intent import receipt_metrics


def row(**overrides):
    case={'query':'我要退款，还要投诉商家','cancel_order':False,'cancel_refund_submission':False,
          'cancel_complaint_submission':False,'forbidden_ticket_types':[],'forbid_logistics':False}
    return {'case':case,'tickets':[],'actor':[],'steps':[],'tool_audit':[],**overrides}


def test_wrong_order_ticket_is_not_refund_success():
    r=row(tickets=[{'ticket_type':'退款','description':'退款申请 订单号=2002 原因=质量问题'}])
    assert not receipt_metrics(r)['refund_terminal_goal']
    assert receipt_metrics(r)['wrong_refund_order_n']==1


def test_duplicate_correct_order_tickets_are_not_success():
    ticket={'ticket_type':'退款','description':'退款申请 订单号=1001 原因=质量问题'}
    m=receipt_metrics(row(tickets=[ticket,ticket]))
    assert not m['refund_terminal_goal'] and m['duplicate_ticket_n']==1


def test_success_text_or_internal_done_is_not_a_sql_receipt():
    r=row(steps=[{'answer':'退款成功，已全部处理','task_plan':{'refund':{'status':'DONE'}}}])
    assert not receipt_metrics(r)['refund_terminal_goal']


def test_cancel_must_be_observed_not_inferred_from_no_write():
    r=row();r['case']['cancel_order']=True
    assert not receipt_metrics(r)['refund_terminal_goal']
    r['actor']=[{'value':{'cancelled':True}}]
    assert receipt_metrics(r)['refund_terminal_goal']
