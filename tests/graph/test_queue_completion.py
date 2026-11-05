import copy
import json
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from app.config import settings
from app.core import intent
from app.db import repository, workflow
from app.graph import runtime, queue_contract
from app.graph.build import build_graph
from tests.graph.test_shared_agent_queue import setup, configure, call, start, resume
from tests.graph.test_subflow_tasks import wired, tickets


@pytest.fixture
async def corrected(setup,monkeypatch):
    monkeypatch.setattr(settings,'multi_intent_completion_enabled',True)
    return setup


def model(monkeypatch,corrected,items,replies):
    m=configure(monkeypatch,corrected,items,replies)
    async def classify(raw,query,history,candidates,**kw):
        corrected.append({'raw':raw,'feedback':kw.get('feedback')})
        return copy.deepcopy(items)
    monkeypatch.setattr(intent,'classify_tasks',classify)
    return m


def task(source,intent_name,oid=None,ref='same',action='consult'):
    return dict(source=source,query=source,intent=intent_name,order_id=oid,order_ref=ref,action=action)


async def test_nc13_selection_is_shared_and_logistics_finishes_before_refund(corrected,monkeypatch,tmp_path):
    items=[task('申请退款','退款退货',action='request'),task('查同一单物流','物流')]
    m=model(monkeypatch,corrected,items,[call('query_order',{'order_id':'1001'},'o'),
        call('query_logistics',{'tracking_no':'SF213502378238'},'l'),AIMessage('已揽件，成都。'),
        call('submit_refund',{'order_id':'1001','reason':'质量问题'},'r'),AIMessage('已退款，钱到账了。')])
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'one.sqlite')) as saver:
        out=await start(saver,monkeypatch,'忘记订单，申请退款，查同一单物流')
        assert out['interrupt']['type']=='select_order' and out['interrupt']['task_kind']=='logistics'
        assert out['state']['planning_error'] == ''
        assert out['state']['task_plan']['1']['status']=='blocked' and not m.seen
        with pytest.raises(runtime.ResumeRejected):await resume(out,'9999')
        out=await resume(out,'1001')
        assert out['interrupt']['type']=='task_form' and len(corrected)==1
        assert out['state']['task_plan']['1']['status']=='done'
        assert out['state']['task_plan']['0']['order_id']=='1001'
        assert '成都' in out['state']['answer'] and '钱到账' not in out['state']['answer']
        assert '尚未提交' in out['state']['answer']
        assert not await tickets()
        out=await resume(out,{'confirmed':False})
        assert out['state']['task_plan']['0']['status']=='cancelled'
        assert out['interrupt'] is None and len(corrected)==1
        assert '部分诉求暂未能准确理解' not in out['state']['answer']
        history=' '.join(x.content or '' for x in await repository.list_dialog_messages(out['conversation_id']))
        assert '钱到账' not in history and '已取消' in history


async def test_receipt_absence_gets_one_original_agent_retry_not_done(corrected,monkeypatch,tmp_path):
    model(monkeypatch,corrected,[task('订单1001物流','物流','1001')],
        [AIMessage('订单已送达。'),AIMessage('我查好了。')])
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'missing.sqlite')) as saver:
        out=await start(saver,monkeypatch,'订单1001物流')
        t=out['state']['task_plan']['0']
        assert t['status']=='failed' and t['repair_attempts']==1
        assert t['result']['completion_basis']=='missing_order_receipt'


async def test_empty_repair_keeps_original_agent_failure_explanation(corrected,monkeypatch,tmp_path):
    model(monkeypatch,corrected,[task('订单1001物流','物流','1001')],
        [AIMessage('物流服务暂时不可用，无法核实当前位置。'),AIMessage('')])
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'empty-repair.sqlite')) as saver:
        out=await start(saver,monkeypatch,'订单1001物流')
        assert out['state']['task_plan']['0']['status']=='failed'
        assert out['state']['answer']=='物流服务暂时不可用，无法核实当前位置。'
        history=await repository.list_dialog_messages(out['conversation_id'])
        assert history[-1].content==out['state']['answer']


async def test_fake_complaint_completion_retries_original_confirm_and_survives_restart(corrected,monkeypatch,tmp_path):
    model(monkeypatch,corrected,[task('登记商家推诿的投诉工单','人工',action='request')],
        [AIMessage('投诉已登记。'),call('create_ticket',{'ticket_type':'投诉','description':'商家推诿，请处理'},'c'),AIMessage('已登记。')])
    path=str(tmp_path/'ticket.sqlite')
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        out=await start(saver,monkeypatch,'登记商家推诿的投诉工单')
        assert out['interrupt']['type']=='confirm_ticket' and not await tickets()
        assert '已登记' not in out['state']['answer']
        cid=out['conversation_id'];n=len(await repository.list_dialog_messages(cid))
        with pytest.raises(runtime.ResumeRejected):await runtime.run_turn('alice','新请求',cid)
        assert len(await repository.list_dialog_messages(cid))==n
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        monkeypatch.setattr(runtime,'_graph',build_graph(saver))
        old=out;out=await resume(out,{'confirmed':True})
        assert len(await tickets())==1 and out['state']['task_plan']['0']['status']=='done'
        with pytest.raises(runtime.ResumeRejected):await resume(old,{'confirmed':True})
        assert len(await tickets())==1 and len(corrected)==1


async def test_stream_suppresses_unconfirmed_write_claims_and_history_matches(corrected,monkeypatch,tmp_path):
    model(monkeypatch,corrected,[task('登记商家推诿的投诉工单','人工',action='request')],
        [AIMessage('投诉已登记成功。'),AIMessage('已经成功。')])
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'stream.sqlite')) as saver:
        monkeypatch.setattr(runtime,'_graph',build_graph(saver))
        events=[e async for e in runtime.stream_turn('alice','登记商家推诿的投诉工单',None,multi_intent_enabled=True)]
        text=''.join(e.get('text','') for e in events)
        assert '尚未提交' in text and '成功' not in text
        assert not any(e['type']=='delta' and '登记' in e.get('text','') for e in events)
        cid=next(e['conversation_id'] for e in events if 'conversation_id' in e)
        history=''.join(x.content or '' for x in await repository.list_dialog_messages(cid) if x.role=='assistant')
        assert text==history and not await tickets()


async def test_completed_complaint_clears_old_form_in_stream_and_state(corrected,monkeypatch,tmp_path):
    model(monkeypatch,corrected,[task('登记商家推诿的投诉工单','投诉',action='request')],
        [call('create_ticket',{'ticket_type':'投诉','description':'商家推诿，请处理'},'c'),AIMessage('已登记。')])
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'stale-action.sqlite')) as saver:
        out=await start(saver,monkeypatch,'登记商家推诿的投诉工单')
        assert out['interrupt']['type']=='confirm_ticket' and not await tickets()
        events=[e async for e in runtime.stream_resume(out['conversation_id'],{'confirmed':True},
            user_id='alice',interrupt_id=out['interrupt']['interrupt_id'])]
        assert len(await tickets())==1
        assert not any(e['type']=='actions' for e in events)
        snapshot=await runtime.get_graph().aget_state({'configurable':{'thread_id':str(out['conversation_id'])}})
        assert snapshot.values['suggested_actions']==[]


async def test_bad_action_is_repaired_before_any_business_work(corrected,monkeypatch,tmp_path):
    good=task('给1001办退钱','退款退货','1001',action='request')
    bad={**good,'action':'consult','query':'订单1001如何申请退款'}
    m=model(monkeypatch,corrected,[good],[call('submit_refund',{'order_id':'1001','reason':'质量问题'},'r'),AIMessage('等待确认。')])
    seen=[]
    async def classify(*args,**kwargs):
        seen.append(kwargs);assert not m.seen
        return [bad if len(seen)==1 else good]
    monkeypatch.setattr(intent,'classify_tasks',classify)
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'action.sqlite')) as saver:
        out=await start(saver,monkeypatch,'给1001办退钱')
        assert len(seen)==2 and seen[1]['feedback']
        assert out['interrupt']['type']=='task_form'
        assert out['state']['task_plan']['0']['action']=='request'


async def test_bad_proposal_does_not_discard_independent_good_task(corrected,monkeypatch,tmp_path):
    items=[task('查订单1001','订单','1001'),task('不在原文的任务','退款退货','9999',action='request')]
    model(monkeypatch,corrected,items,[call('query_order',{'order_id':'1001'},'o'),AIMessage('智能猫砂盆。')])
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'partial.sqlite')) as saver:
        out=await start(saver,monkeypatch,'查订单1001')
        assert len(corrected)==2 and out['state']['task_plan']['0']['status']=='done'
        assert out['state']['task_plan']['invalid:0']['status']=='failed'
        assert '部分诉求' in out['state']['answer'] and not await tickets()


def test_formatting_normalization_never_invents_original_source():
    assert queue_contract.literal_span('甲，乙？','甲, 乙?')=='甲，乙？'
    assert queue_contract.literal_span('甲，乙？','甲，丙？') is None


def test_negation_and_foreign_order_are_not_authorization(monkeypatch):
    monkeypatch.setattr(queue_contract.business,'owns_order',lambda u,o:o=='1001')
    state={'task_origin':'不要给订单1001申请退款','user_id':'alice'}
    valid,errors=queue_contract.validate_proposals([task(state['task_origin'],'退款退货','1001',action='request')],state,{'1001'})
    assert not valid and errors
    valid,errors=queue_contract.validate_proposals([task(state['task_origin'],'订单','9999')],state,{'1001'})
    assert not valid and errors


@pytest.mark.parametrize('source,accepted',[
    ('订单1001和2002都因质量问题提交退款申请，分别等平台审核',True),
    ('别给订单1001申请退款',False),
    ('千万别再给订单1001申请退款',False),
])
def test_separate_applications_are_not_mistaken_for_negation(monkeypatch,source,accepted):
    monkeypatch.setattr(queue_contract.business,'owns_order',lambda u,o:o in {'1001','2002'})
    valid,errors=queue_contract.validate_proposals(
        [task(source,'退款退货','1001',action='request')],
        {'task_origin':source,'user_id':'alice'},{'1001','2002'})
    assert bool(valid) is accepted and bool(errors) is not accepted


@pytest.mark.parametrize('confirmed',[True,False])
async def test_new_contract_keeps_original_form_idempotency_and_cancel(corrected,monkeypatch,tmp_path,confirmed):
    from tests.graph.test_shared_agent_queue import test_original_refund_form_then_original_ticket_confirmation
    await test_original_refund_form_then_original_ticket_confirmation(corrected,monkeypatch,tmp_path,confirmed)


async def test_pending_old_v4_keeps_old_semantics_after_enabling_fix(corrected,monkeypatch,tmp_path):
    monkeypatch.setattr(settings,'multi_intent_completion_enabled',False)
    configure(monkeypatch,corrected,[{'intent':'退款退货','query':'申请退款','action':'request'}],
        [call('submit_refund',{'order_id':'1001','reason':'质量问题'},'r'),AIMessage('旧版回复。')])
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'legacy.sqlite')) as saver:
        out=await start(saver,monkeypatch,'申请退款')
        assert out['interrupt']['type']=='select_order'
        monkeypatch.setattr(settings,'multi_intent_completion_enabled',True)
        out=await resume(out,'1001')
        assert not out['state'].get('queue_contract_version')
        assert out['interrupt']['type']=='task_form' and '旧版回复' in out['state']['answer']
