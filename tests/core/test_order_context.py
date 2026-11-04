import pytest

from app.core import order_context as ctx
from app.core.task_planner import normalize


@pytest.fixture(autouse=True)
def owned(monkeypatch):
    monkeypatch.setattr(ctx.business, 'owns_order', lambda uid, oid: uid == 'u' and oid in ('1001','2002'))
    monkeypatch.setattr(ctx.business, 'order_snapshot', lambda oid: {'order_id':oid,'product':'智能猫砂盆','status':'已付款'})


def context(ids=None, recent=True):
    return {'user_id':'u','conversation_id':7,'order_ids':ids if ids is not None else ['1001'],
            'recent':recent,'user_reports':{'1001':{'text':'猫砂盆漏电了，我要退款','message_id':1}}}


def items(raw):
    sources=raw.split('，')
    return normalize(raw,[{'kind':kind,'query':source,'source':source}
        for kind,source in zip(('order','logistics','knowledge'), sources)])


def test_132_followup_binds_all_three_tasks_to_verified_order():
    raw='刚刚的订单是哪个，它现在物流到哪了，可以退货吗'
    result=ctx.bind_tasks(raw,items(raw),context(),'u')
    assert [t.get('order_id') for t in result[:2]]==['1001','1001']
    assert result[2]['context_order_id']=='1001'
    facts=ctx.knowledge_context({'user_id':'u','business_context':context()}, result[2])
    assert facts['product']=='智能猫砂盆' and '漏电' in facts['user_reported_issue']


def test_one_selection_is_shared_only_inside_explicit_reference_group():
    raw='刚刚的订单是哪个，它现在物流到哪了，可以退货吗'
    tasks={str(i):t for i,t in enumerate(ctx.bind_tasks(raw,items(raw),context([]),'u'))}
    tasks['0'].update(order_id='1001',binding={'status':'resolved','method':'user_selection'})
    tasks['independent']={'kind':'refund','source':'另一笔退款','order_id':None}
    state={'user_id':'u','conversation_id':7,'current_message_id':9,'business_context':{}}
    ctx.selected(state,tasks,'0','1001')
    assert tasks['1']['order_id']=='1001' and tasks['2']['context_order_id']=='1001'
    assert tasks['independent']['order_id'] is None


@pytest.mark.parametrize('raw,kind',[
    ('我要退款','refund'),('另一笔订单查物流','logistics'),('换一笔订单查状态','order'),
    ('另外问一下，猫粮可以退货吗','knowledge')])
def test_new_or_unrelated_request_does_not_borrow_old_order(raw,kind):
    t=normalize(raw,[{'kind':kind,'source':raw,'query':raw}])
    result=ctx.bind_tasks(raw,t,context(),'u')[0]
    assert not result.get('order_id') and not result.get('context_order_id')


@pytest.mark.parametrize('ids', [[], ['1001','2002'], ['9999']])
def test_missing_ambiguous_or_foreign_focus_requires_clarification(ids):
    raw='刚刚的订单是哪个，它现在物流到哪了，可以退货吗'
    result=ctx.bind_tasks(raw,items(raw),context(ids),'u')
    assert all(not t.get('order_id') and not t.get('context_order_id') for t in result)


def test_explicit_current_order_and_new_object_override_history():
    raw='查订单2002状态，它现在物流到哪了，可以退货吗'
    result=ctx.bind_tasks(raw,items(raw),context(),'u')
    assert [t.get('order_id') for t in result[:2]]==['2002','2002']
    assert result[2]['context_order_id']=='2002'


def test_bare_pronoun_is_only_reused_for_recent_order_subject():
    raw='它现在物流到哪了'
    tasks=normalize(raw,[{'kind':'logistics','source':raw,'query':raw}])
    assert ctx.bind_tasks(raw,tasks,context(),'u')[0]['order_id']=='1001'
    assert ctx.bind_tasks(raw,tasks,context(recent=False),'u')[0]['order_id'] is None


def test_old_cancelled_plan_backfills_identity_and_issue_without_consent():
    task={'kind':'refund','order_id':'1001','source':'猫砂盆漏电了，我要退款','status':'CANCELLED',
          'binding':{'method':'user_selection','status':'resolved'}}
    previous={'task_plan':{'refund':task},'current_message_id':1}
    result=ctx.restore(previous,[{'origin':1,'tasks':{'refund':task}}],'u',7)
    assert result['order_ids']==['1001'] and '漏电' in result['user_reports']['1001']['text']
    assert 'confirmed' not in result and 'operation_id' not in result


def test_plain_assistant_or_unproven_model_order_does_not_create_focus():
    previous={'task_plan':{'order':{'kind':'order','order_id':'1001','source':'AI说1001'}}}
    assert ctx.restore(previous,[],'u',7)['order_ids']==[]


def test_logistics_order_is_not_borrowed_for_independent_refund():
    raw='查2002物流，我要退款，还要投诉商家'
    result=ctx.bind_tasks(raw,normalize(raw,[{'kind':'logistics','source':'查2002物流','query':'查2002物流'},
        {'kind':'refund','source':'我要退款','query':'我要退款'}]),context(),'u')
    assert result[0]['order_id']=='2002'
    assert result[1]['order_id'] is None and not result[1].get('reference_group')


def test_new_product_consultation_after_order_query_is_not_order_specific():
    raw='查订单1001状态，猫粮可以退货吗'
    result=ctx.bind_tasks(raw,normalize(raw,[{'kind':'knowledge','source':'猫粮可以退货吗','query':'猫粮可以退货吗'}]),context(),'u')
    assert not result[0].get('context_order_id')


def test_return_eligibility_retrieves_policy_without_replaying_historical_requests():
    facts={'product':'智能猫砂盆','user_reported_issue':'我刚买的猫砂盆漏电了，我要退款，而且我要投诉商家。'}
    query=ctx.policy_query('可以退货吗',facts)
    assert '质量问题' in query and '退货退款' in query
    assert '投诉' not in query and '我要退款' not in query
    assert ctx.policy_query('怎么清洗',facts)=='智能猫砂盆 怎么清洗'
