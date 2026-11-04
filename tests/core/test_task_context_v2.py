import json
import pytest
from langchain_core.messages import HumanMessage, AIMessage
from app.core import task_context as ctx, task_planner
from app.config import settings


@pytest.fixture(autouse=True)
def owned(monkeypatch):
    monkeypatch.setattr(ctx.business,'owns_order',lambda uid,oid:uid=='u' and oid in {'1001','2002'})


def context():
    return {'user_id':'u','candidates':[{'ref':'order:1001','order_id':'1001','source':{'kind':'user_selection'}}]}


def item(kind,raw,**extra):
    return task_planner.ContextTask(kind=kind,query=raw,source=raw,subject_ref='s1',reference_span=raw,
        candidate_ref='order:1001',binding_status='resolved',**extra).model_dump()


def test_colloquial_reads_and_manual_bind_without_action_keywords():
    raw='刚刚那个订单现在送到哪了，里面是什么商品，有没有使用说明'
    tasks=ctx.normalize(raw,[item(k,raw,**({'knowledge_scope':'product'} if k=='knowledge' else {}))
        for k in ['logistics','order','knowledge']],context())
    assert [t['order_id'] for t in tasks[:2]]==['1001','1001']
    assert tasks[2]['context_order_id']=='1001'
    assert {t['subject_ref'] for t in tasks}=={'s1'}


def test_unknown_foreign_candidate_and_invented_span_cannot_bind():
    raw='刚才那单送到哪了'
    t=item('logistics',raw);t['candidate_ref']='order:9999';t['order_id']='9999'
    out=ctx.normalize(raw,[t],context())[0]
    assert out['order_id'] is None and out['binding_validation']=='unsupported_candidate'
    t['reference_span']='不存在的原话'
    with pytest.raises(ValueError):ctx.normalize(raw,[t],context())


def test_independent_write_and_general_question_do_not_borrow_candidate():
    raw='我要退款'
    assert ctx.normalize(raw,[item('refund',raw)],context())[0]['order_id'] is None
    raw='平台运费规则是什么'
    t=ctx.normalize(raw,[item('knowledge',raw,knowledge_scope='general')],context())[0]
    assert not t.get('context_order_id') and not t['subject_ref']


def test_current_explicit_order_overrides_old_candidate_with_conflict_trace():
    raw='查订单2002物流'
    t=ctx.normalize(raw,[item('logistics',raw)],context())[0]
    assert t['order_id']=='2002' and t['binding_validation']=='current_order_conflict'


def test_ambiguous_shared_object_keeps_group_without_choosing_candidate():
    raw='那笔订单到哪了，里面是什么商品'
    values=[{**item(k,raw),'binding_status':'ambiguous','candidate_ref':None} for k in ['order','logistics']]
    result=ctx.normalize(raw,values,context())
    assert all(t['subject_ref']=='s1' and t['order_id'] is None for t in result)


def test_plural_unnamed_order_cannot_be_guessed_from_recency():
    raw='刚才这两笔中，那个订单到哪了'
    c=context();c['candidates'].append({'ref':'order:2002','order_id':'2002','source':{'kind':'user_selection'}})
    t=ctx.normalize(raw,[item('logistics',raw)],c)[0]
    assert t['order_id'] is None and t['subject_ref']=='s1'


def test_literal_ellipsis_reference_recovers_single_order_original_span():
    raw='不是1001，改查订单2002物流和商品信息'
    c=context();c['candidates'].append({'ref':'order:2002','order_id':'2002','source':{'kind':'current_message'}})
    t=item('order','和商品信息');t.update(candidate_ref='order:2002',reference_span='订单2002...商品信息')
    result=ctx.normalize(raw,[t],c)[0]
    assert result['order_id']=='2002' and result['reference_span']=='订单2002物流和商品信息'
    assert result['reference_span'] in raw and result['reference_alignment']['model_span']==t['reference_span']


def test_valid_local_order_reference_repairs_coordinated_read_projection():
    raw='不是1001，改查订单2002物流和商品信息'
    c=context();c['candidates'].append({'ref':'order:2002','order_id':'2002','source':{'kind':'current_message'}})
    t=item('order','改查订单2002物流和商品信息')
    t.update(candidate_ref='order:2002',reference_span='订单2002商品信息',order_reference='订单2002')
    result=ctx.normalize(raw,[t],c)[0]
    assert result['order_id']=='2002' and result['reference_span']=='订单2002'
    assert result['reference_alignment']['model_span']=='订单2002商品信息'


def test_coordinated_correction_also_overrides_stale_candidate():
    raw='不是1001，改查订单2002物流和商品信息'
    t=item('order','改查订单2002物流和商品信息');t['reference_span']='订单2002'
    result=ctx.normalize(raw,[t],context())[0]
    assert result['order_id']=='2002' and result['binding_validation']=='current_order_conflict'


@pytest.mark.parametrize('raw,span',[
    ('查订单1001物流，另查订单2002商品信息','订单1001...商品信息'),
    ('查订单1001和订单2002商品信息','订单1001...商品信息'),
    ('查订单1001商品信息','订单1001的商品信息')])
def test_paraphrase_or_cross_order_ellipsis_is_rejected(raw,span):
    t=item('order',raw);t['reference_span']=span
    with pytest.raises(ValueError):ctx.normalize(raw,[t],context())


def test_builder_uses_history_but_assistant_cannot_create_order_candidate(monkeypatch):
    state={'user_id':'u','conversation_id':7,'current_message_id':5,'task_origin':'刚刚那单送到哪了',
           'messages':[AIMessage('订单1001查询成功',id='db-1'),HumanMessage('收到',id='db-2'),
                       HumanMessage('刚刚那单送到哪了',id='db-5')]}
    c=ctx.build(state)
    assert c['candidates']==[]
    state['messages'][0]=HumanMessage('请查询订单1001状态',id='db-1')
    c=ctx.build(state)
    assert c['candidates'][0]['source']['message_id']==1


async def test_one_planner_invocation_receives_original_and_history(monkeypatch):
    raw='刚才那单送到哪了';calls=[]
    class Model:
        async def ainvoke(self,messages):
            calls.append(messages)
            return task_planner.ContextPlan(tasks=[task_planner.ContextTask.model_validate(item('logistics',raw))])
    monkeypatch.setattr(task_planner.llm,'structured',lambda *a,**kw:Model())
    c={**context(),'history':'用户此前选择1001'}
    out=await task_planner.extract(raw,context=c)
    assert len(calls)==1 and out[0]['order_id']=='1001'
    payload=json.loads(calls[0][-1]['content'])
    assert payload['current_request']==raw and payload['context']['history']==c['history']
