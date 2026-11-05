import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.config import settings
from app.core import intent
from app.core.prompts import AGENT_SYSTEM
from app.db import repository, workflow
from app.graph import nodes, runtime, task_queue, simple_loop
from app.graph.build import build_graph
from app.tools import registry
from tests.graph.test_subflow_tasks import wired, tickets


class Model(BaseChatModel):
    replies: list
    seen: list = []
    bound: list = []
    gate: object = None

    @property
    def _llm_type(self):return 'controlled-stream'
    def bind_tools(self,tools,**kw):
        self.bound.append([t.name for t in tools]);return self
    def pop(self,messages):
        self.seen.append(messages)
        assert self.replies,'Unexpected extra model invocation'
        return self.replies.pop(0)
    def _generate(self,messages,**kwargs):
        return ChatResult(generations=[ChatGeneration(message=self.pop(messages))])
    async def _agenerate(self,messages,**kwargs):return self._generate(messages)
    async def _astream(self,messages,**kwargs):
        m=self.pop(messages)
        if m.tool_calls:
            yield ChatGenerationChunk(message=AIMessageChunk(content='',tool_call_chunks=[
                {'name':c['name'],'args':json.dumps(c['args']),'id':c['id'],'index':i} for i,c in enumerate(m.tool_calls)]))
        else:
            yield ChatGenerationChunk(message=AIMessageChunk(content=m.content[:2]))
            if self.gate is not None:await self.gate.wait()
            yield ChatGenerationChunk(message=AIMessageChunk(content=m.content[2:]))


def call(name,args,id):return AIMessage('',tool_calls=[{'name':name,'args':args,'id':id,'type':'tool_call'}])


@pytest.fixture
async def setup(wired,monkeypatch):
    monkeypatch.setattr(settings,'multi_intent_shared_agent_enabled',True)
    monkeypatch.setattr(settings,'context_fresh_business_evidence',False)
    async def coref(s):return {'resolved_query':nodes._user_text(s)}
    monkeypatch.setattr(nodes,'resolve_reference',coref)
    async def policy(s):return {'evidence':'[1] 质量问题可申请审核','citations':[{'id':1,'n':1,'question':'退货','answer':'可申请审核'}]}
    monkeypatch.setattr(nodes,'retrieve_policy',policy)
    @tool
    async def query_logistics(tracking_no:str)->dict:
        """Query logistics."""
        return {'tracking_no':tracking_no,'status':'已揽件','current_city':'成都'}
    @tool
    async def query_warranty(order_id:str)->dict:
        """Query warranty."""
        return {'order_id':order_id,'in_warranty':True}
    async def specs():return [*registry.builtin_specs(),registry.spec_from_langchain_tool(query_logistics,source='mcp'),registry.spec_from_langchain_tool(query_warranty,source='mcp')]
    monkeypatch.setattr(registry,'get_all_specs',specs)
    async def forbidden(*a,**kw):raise AssertionError('v4 must never enter the v3 business executor/renderer')
    for name in ('plan','prepare','agent','tools','execute','present'):monkeypatch.setattr(simple_loop,name,forbidden)
    return []


def configure(monkeypatch,setup,items,replies,gate=None):
    async def classify(raw,query,history,candidates):
        setup.append((raw,history,candidates))
        return [{'source':raw,'order_ref':'same','action':'consult',**t} for t in items]
    monkeypatch.setattr(intent,'classify_tasks',classify)
    model=Model(replies=replies,gate=gate)
    def get_model(**kw):
        assert kw=={'streaming':True},'Must keep original main_agent streaming model construction'
        return model
    monkeypatch.setattr(nodes,'get_chat_model',get_model)
    return model


async def start(saver,monkeypatch,text):
    monkeypatch.setattr(runtime,'_graph',build_graph(saver))
    return await runtime.run_turn('alice',text,None,multi_intent_enabled=True)


async def resume(out,value):
    return await runtime.resume_turn(out['conversation_id'],value,user_id='alice',interrupt_id=out['interrupt']['interrupt_id'])


async def test_real_agent_and_tools_preserve_registry_history_and_cache(setup,monkeypatch,tmp_path):
    items=[{'intent':'订单','query':'订单1001是什么商品','order_id':'1001'},
           {'intent':'物流','query':'订单1001送到哪了','order_id':'1001'}]
    model=configure(monkeypatch,setup,items,[call('query_order',{'order_id':'1001'},'o1'),AIMessage('商品是智能猫砂盆。'),
        call('query_order',{'order_id':'1001'},'o2'),call('query_logistics',{'tracking_no':'SF213502378238'},'l1'),AIMessage('**物流信息**\n- 成都，已揽件。')])
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'q.sqlite')) as saver:
        out=await start(saver,monkeypatch,'订单1001是什么商品，送到哪了')
        assert out['interrupt'] is None and len(setup)==1
        assert all(t['status']=='done' for t in out['state']['task_plan'].values())
        assert all(m[0].content.startswith(AGENT_SYSTEM+'\n\n## 当前队列任务边界') for m in model.seen)
        assert 'query_product' in model.bound[0] and 'query_logistics' not in model.bound[0]
        assert 'query_logistics' in model.bound[-1] and 'query_warranty' not in model.bound[-1]
        assert any(isinstance(m,ToolMessage) and m.name=='query_order' for m in model.seen[-1])
        assert not any(isinstance(m,HumanMessage) and m.content=='订单1001是什么商品，送到哪了' for m in model.seen[-1])
        assert any('订单1001是什么商品，送到哪了' in m.content for m in model.seen[-1] if m.id==nodes.TURN_CTX_ID)
        assert out['state']['messages'][0].content=='订单1001是什么商品，送到哪了'
        view=model.seen[-1]
        material=next(i for i,m in enumerate(view) if m.id==nodes.TURN_CTX_ID)
        first_reply=next(i for i,m in enumerate(view) if m.content=='商品是智能猫砂盆。')
        assert material>first_reply  # Current task materials follow completed siblings.
        assert '订单1001送到哪了' in view[material].content
        assert view[material+1].content=='订单1001送到哪了' and view[material+1].id is None
        from sqlalchemy import select
        from app.db.models import ToolAuditLog
        async with workflow.base.async_session() as s:
            rows=(await s.execute(select(ToolAuditLog))).scalars().all()
        assert [r.tool_name for r in rows]==['query_order','query_logistics']
        assert '智能猫砂盆' in out['state']['answer'] and '成都' in out['state']['answer']
        assert len([m for m in await repository.list_dialog_messages(out['conversation_id']) if m.role=='assistant'])==2


@pytest.mark.parametrize('confirmed',[True,False])
async def test_original_refund_form_then_original_ticket_confirmation(setup,monkeypatch,tmp_path,confirmed):
    from app.api.actions import router
    items=[{'intent':'退款退货','query':'申请退款','action':'request'},
           {'intent':'人工','query':'因商品漏电登记投诉工单','action':'request'}]
    model=configure(monkeypatch,setup,items,[call('submit_refund',{'order_id':'1001','reason':'质量问题'},'r'),AIMessage('请在退款表单确认原因。'),
        call('create_ticket',{'ticket_type':'投诉','description':'订单1001商品漏电，请平台处理'},'c'),AIMessage('投诉工单已提交。')])
    path=str(tmp_path/'form.sqlite')
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        out=await start(saver,monkeypatch,'商品漏电，我要申请退款，并登记投诉工单')
        assert out['interrupt']['type']=='select_order'
        with pytest.raises(runtime.ResumeRejected):await resume(out,'9999')
        out=await resume(out,'1001')
        assert out['interrupt']['type']=='task_form' and out['interrupt']['form']['type']=='refund_form'
        assert out['state']['task_plan']['1']['status']=='pending'
        with pytest.raises(runtime.ResumeRejected):await resume(out,{'confirmed':True})
        before=len(await repository.list_messages(out['conversation_id']))
        with pytest.raises(runtime.ResumeRejected):await runtime.run_turn('alice','另一个问题',out['conversation_id'])
        assert len(await repository.list_messages(out['conversation_id']))==before
        if confirmed:
            app=FastAPI();app.include_router(router)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as c:
                request={'user_id':'alice','conversation_id':out['conversation_id'],'order_id':'1001','reason':'质量问题','request_id':out['interrupt']['request_id']}
                bad=await c.post('/api/actions/create-refund',json={**request,'order_id':'2002'});assert bad.status_code==409
                first=await c.post('/api/actions/create-refund',json=request);second=await c.post('/api/actions/create-refund',json=request)
                assert first.status_code==second.status_code==200 and first.json()==second.json()
        old=out
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        monkeypatch.setattr(runtime,'_graph',build_graph(saver))
        out=await resume(old,{'confirmed':confirmed})
        assert out['interrupt']['type']=='confirm_ticket'  # Original agent_tools interrupt, not task_actions.
        assert out['state']['task_plan']['0']['status']==('done' if confirmed else 'cancelled')
        assert out['state']['task_plan']['1']['order_id']=='1001'
        if confirmed:
            material=next(m.content for m in model.seen[-1] if m.id==nodes.TURN_CTX_ID)
            assert 'ticket_no' in material and 'submitted_forms' in material
            assert '"task_intent": "退款退货"' in material and '"form_type": "refund_form"' in material
        else:
            material=next(m.content for m in model.seen[-1] if m.id==nodes.TURN_CTX_ID)
            assert '"cancelled_actions": [{"intent": "退款退货", "order_id": "1001"}]' in material
        with pytest.raises(runtime.ResumeRejected):await resume(old,{'confirmed':confirmed})
        out=await resume(out,{'confirmed':True})
        assert out['interrupt'] is None and len(setup)==1 and not model.replies
        assert sorted(t.ticket_type for t in await tickets())==(['投诉','退款'] if confirmed else ['投诉'])


async def test_body_stream_arrives_before_model_finishes(setup,monkeypatch,tmp_path):
    gate=asyncio.Event();first=asyncio.Event();seen=[]
    configure(monkeypatch,setup,[{'intent':'订单','query':'订单1001情况','order_id':'1001'}],[AIMessage('**订单信息**\n智能猫砂盆。')],gate)
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'stream.sqlite')) as saver:
        monkeypatch.setattr(runtime,'_graph',build_graph(saver))
        async def consume():
            async for ev in runtime.stream_turn('alice','查询订单1001',None,multi_intent_enabled=True):
                seen.append(ev)
                if ev['type']=='delta':first.set()
        task=asyncio.create_task(consume())
        try:
            await asyncio.wait_for(first.wait(),10)
            assert not task.done() and not any(e['type']=='done' for e in seen)
        finally:gate.set()
        await task
        assert ''.join(e['text'] for e in seen if e['type']=='delta')=='**订单信息**\n智能猫砂盆。'
        assert len([e for e in seen if e['type']=='delta'])>=2
        assert not any(e['type']=='task_event' for e in seen)


async def test_query_product_and_warranty_are_not_replaced_by_fixed_tool_steps(setup,monkeypatch,tmp_path):
    configure(monkeypatch,setup,[{'intent':'订单','query':'1001商品价格库存','order_id':'1001'},
        {'intent':'售后','query':'1001在保吗','order_id':'1001'}],
        [call('query_product',{'product_name':'智能猫砂盆'},'p'),AIMessage('查询到了商品库存。'),
         call('query_warranty',{'order_id':'1001'},'w'),AIMessage('保修查询显示在保。')])
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'tools.sqlite')) as saver:
        out=await start(saver,monkeypatch,'查询订单1001商品价格库存和是否在保')
        assert out['interrupt'] is None
        names={m.name for m in out['state']['messages'] if isinstance(m,ToolMessage)}
        assert {'query_product','query_warranty'}<=names


async def test_consultation_cannot_become_submission(setup,monkeypatch,tmp_path):
    configure(monkeypatch,setup,[{'intent':'退款退货','query':'1001能退吗','order_id':'1001'}],
        [call('submit_refund',{'order_id':'1001','reason':'质量问题'},'r'),AIMessage('是否可退需要审核。')])
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'consult.sqlite')) as saver:
        out=await start(saver,monkeypatch,'订单1001能退吗')
        assert out['interrupt'] is None and out['state']['suggested_actions']==[] and not await tickets()


def test_no_new_executor_or_answer_generator():
    import inspect
    source=inspect.getsource(task_queue)
    assert 'execute_tool_call(' not in source and 'get_chat_model(' not in source and 'workflow.submit(' not in source
    assert 'retrieve_knowledge(' not in source and 'retrieve_policy(' not in source


async def test_failed_tool_is_returned_to_original_agent_for_recovery(setup,monkeypatch,tmp_path):
    configure(monkeypatch,setup,[{'intent':'订单','query':'订单1001情况','order_id':'1001'}],
        [call('query_order',{'order_id':'9999'},'bad'),call('query_order',{'order_id':'1001'},'good'),AIMessage('已查到1001。')])
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'recovery.sqlite')) as saver:
        out=await start(saver,monkeypatch,'查询订单1001')
        assert any(isinstance(m,ToolMessage) and '订单不一致' in m.content for m in out['state']['messages'])
        assert out['state']['task_plan']['0']['order_id']=='1001'


async def test_original_knowledge_gate_fallback_then_remaining_task(setup,monkeypatch,tmp_path):
    configure(monkeypatch,setup,[{'intent':'商品咨询','query':'没有记录的商品说明'},
        {'intent':'人工','query':'请登记咨询工单','action':'request'}],
        [call('create_ticket',{'ticket_type':'咨询','description':'需要商品说明'},'c'),AIMessage('咨询工单已提交。')])
    async def weak(s):return {'evidence_strong':False,'fallback_source':'self_check','retrieved_snapshot':[]}
    monkeypatch.setattr(nodes,'retrieve_knowledge',weak)
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'weak.sqlite')) as saver:
        out=await start(saver,monkeypatch,'没有记录的商品说明，并请登记咨询工单')
        assert out['state']['task_plan']['0']['status']=='failed'
        assert out['interrupt']['type']=='confirm_ticket'
        out=await resume(out,{'confirmed':False})
        assert out['interrupt'] is None and not await tickets()
        from sqlalchemy import text
        async with workflow.base.async_session() as s:
            assert (await s.execute(text('SELECT COUNT(*) FROM low_confidence_questions'))).scalar()==1


async def test_previous_v3_card_resumes_after_shared_queue_enabled(wired,monkeypatch,tmp_path):
    from tests.graph import test_simple_loop as old
    monkeypatch.setattr(settings,'multi_intent_simple_enabled',True)
    monkeypatch.setattr(settings,'multi_intent_shared_agent_enabled',False)
    async def classify(s):return ([{'kind':'refund','query':s['task_origin'],'source':s['task_origin'],
        'status':'pending','order_ref':'s','order_id':None,'result':{}}],{},[])
    from app.core import simple_intents
    monkeypatch.setattr(simple_intents,'classify',classify)
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'v3.sqlite')) as saver:
        out=await start(saver,monkeypatch,'我要申请退款')
        assert out['state']['task_plan_version']==3 and out['interrupt']['type']=='select_order'
        monkeypatch.setattr(settings,'multi_intent_shared_agent_enabled',True)
        async def forbidden(*a,**kw):raise AssertionError('不能重新规划旧卡片')
        monkeypatch.setattr(intent,'classify_tasks',forbidden)
        out=await resume(out,{'cancelled':True})
        assert out['state']['task_plan_version']==3 and out['interrupt'] is None


async def test_write_response_loss_keeps_original_tool_receipt(setup,monkeypatch,tmp_path):
    configure(monkeypatch,setup,[{'intent':'人工','query':'登记订单1001漏电投诉工单','order_id':'1001','action':'request'}],
        [call('create_ticket',{'ticket_type':'投诉','description':'订单1001漏电'},'first'),
         call('create_ticket',{'ticket_type':'投诉','description':'订单1001漏电'},'retry'),AIMessage('工单已创建。')])
    original=workflow.submit;attempts=[]
    async def lost(*a,**kw):
        r=await original(*a,**kw);attempts.append(r)
        if len(attempts)==1:raise ConnectionError('response lost after commit')
        return r
    monkeypatch.setattr(workflow,'submit',lost)
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'lost.sqlite')) as saver:
        out=await start(saver,monkeypatch,'登记订单1001漏电投诉工单')
        out=await resume(out,{'confirmed':True})
        assert out['interrupt']['type']=='confirm_ticket'
        out=await resume(out,{'confirmed':True})
        assert out['interrupt'] is None and len(await tickets())==1
        assert attempts[0]['ticket_no']==attempts[1]['ticket_no']


@pytest.mark.parametrize('items',[
    [{'intent':'订单','query':'订单9999情况','order_id':'9999'}],
    [{'intent':'订单','query':'订单1001情况','order_id':'1001'},
     {'intent':'物流','query':'订单2002物流','order_id':'2002'}],
])
async def test_invalid_candidate_or_shared_reference_cannot_execute(setup,monkeypatch,tmp_path,items):
    model=configure(monkeypatch,setup,items,[])
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'invalid.sqlite')) as saver:
        out=await start(saver,monkeypatch,'查询订单1001的商品和订单2002的物流')
        assert out['state']['task_planning_error'] and not model.seen and not await tickets()


async def test_request_without_receipt_is_not_claimed_completed(setup,monkeypatch,tmp_path):
    configure(monkeypatch,setup,[{'intent':'人工','query':'登记投诉工单','action':'request'}],
        [AIMessage('请补充具体问题。')])
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'unfulfilled.sqlite')) as saver:
        out=await start(saver,monkeypatch,'帮我登记投诉工单')
        task=out['state']['task_plan']['0']
        assert task['status']=='failed' and task['result']['completion_basis']=='no_confirmed_write_receipt'
        assert not await tickets()


async def test_logistics_cannot_call_sibling_warranty_even_if_model_forges_call(setup,monkeypatch,tmp_path):
    model=configure(monkeypatch,setup,[{'intent':'物流','query':'订单1001物流','order_id':'1001'},
        {'intent':'售后','query':'订单1001是否在保','order_id':'1001'}],
        [call('query_warranty',{'order_id':'1001'},'early'),call('query_order',{'order_id':'1001'},'o'),
         call('query_logistics',{'tracking_no':'SF213502378238'},'l'),AIMessage('**物流**\n已揽件，成都。'),
         call('query_warranty',{'order_id':'1001'},'proper'),AIMessage('**保修**\n在保。')])
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'scope.sqlite')) as saver:
        out=await start(saver,monkeypatch,'订单1001送到哪了，是否在保')
        assert out['interrupt'] is None
        denied=next(m for m in out['state']['messages'] if isinstance(m,ToolMessage) and m.tool_call_id=='early')
        assert denied.status=='error'
        from sqlalchemy import select
        from app.db.models import ToolAuditLog
        async with workflow.base.async_session() as s:
            names=[r.tool_name for r in (await s.execute(select(ToolAuditLog))).scalars()]
        assert names==['query_order','query_logistics','query_warranty']
        first=next(m.content for m in model.seen[0] if m.id==nodes.TURN_CTX_ID)
        last=next(m.content for m in model.seen[-1] if m.id==nodes.TURN_CTX_ID)
        assert '"reply_phase": "partial"' in first and '"remaining_task_count": 1' in first
        assert '"reply_phase": "final"' in last and '"completed_intents": ["物流"]' in last
        assert all('query_warranty' not in b for b in model.bound[:4])
        assert 'query_warranty' in model.bound[-1]
