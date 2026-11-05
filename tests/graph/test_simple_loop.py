import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.config import settings
from app.core import simple_intents
from app.db import repository, workflow
from app.graph import nodes, runtime, simple_loop
from app.graph.build import build_graph
from app.tools import business
from tests.graph.test_subflow_tasks import wired, tickets


@pytest.fixture
def setup(wired,monkeypatch):
    monkeypatch.setattr(settings,'multi_intent_simple_enabled',True)
    calls=[];plans=[]
    async def coref(s):return {'resolved_query':s['task_origin']}
    monkeypatch.setattr(nodes,'resolve_reference',coref)
    async def policy(s):return {'evidence':'[1] 质量问题可申请核实','citations':[{'id':1,'n':1,'question':'退货条件','answer':'质量问题可申请核实','section_path':'退货','content_type':'policy'}]}
    monkeypatch.setattr(nodes,'retrieve_policy',policy)
    async def agent(s):
        tool=s['simple_tool'];return {'messages':[AIMessage('',tool_calls=[{'id':str(len(calls)),'name':tool['name'],'args':tool['args'],'type':'tool_call'}])]}
    monkeypatch.setattr(nodes,'main_agent',agent)
    async def specs():return []
    monkeypatch.setattr(simple_loop.registry,'get_all_specs',specs)
    async def execute(tc,cid,specs,**kwargs):
        calls.append((tc['name'],tc['args']))
        if tc['name']=='query_order':data=business.order_snapshot(tc['args']['order_id'])
        elif tc['name']=='query_logistics':data={'tracking_no':tc['args']['tracking_no'],'status':'已揽件','current_city':'成都'}
        else:data={'order_id':tc['args']['order_id'],'return_status':'审核中'}
        return SimpleNamespace(ok=True,tool_message=ToolMessage(json.dumps(data,ensure_ascii=False),name=tc['name'],tool_call_id=tc['id']))
    monkeypatch.setattr(simple_loop.engine,'execute_tool_call',execute)
    class Model:
        async def ainvoke(self,messages):return AIMessage('已取得当前查询结果；需要操作时请使用对应卡片。')
    monkeypatch.setattr(simple_loop,'get_chat_model',lambda:Model())
    return calls,plans


def classify(monkeypatch,setup,items):
    async def extract(state):
        setup[1].append(state['task_origin'])
        values=[simple_intents.Item(**{'source':state['task_origin'],'query':state['task_origin'],**t,'order_ref':t.get('order_ref') or 'none'}).model_dump() for t in items]
        context={'user_id':state['user_id'],'candidates':[{'order_id':'1001'},{'order_id':'2002'}]}
        return simple_intents.normalize(state['task_origin'],values,context),context,values
    monkeypatch.setattr(simple_intents,'classify',extract)


async def start(monkeypatch,saver,text):
    monkeypatch.setattr(runtime,'_graph',build_graph(saver))
    return await runtime.run_turn('alice',text,None,multi_intent_enabled=True)


async def resume(out,value):
    return await runtime.resume_turn(out['conversation_id'],value,user_id='alice',interrupt_id=out['interrupt']['interrupt_id'])


async def test_known_order_shared_receipts_and_advice_not_write(setup,monkeypatch,tmp_path):
    classify(monkeypatch,setup,[{'kind':k,'order_id':'1001','order_ref':'same','product_specific':k=='return_advice'}
                               for k in ['logistics','order','return_advice']])
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'read.sqlite')) as saver:
        out=await start(monkeypatch,saver,'刚刚那个订单送到哪了，它是什么商品，我可以退货吗')
        assert out['interrupt'] is None
        assert [x[0] for x in setup[0]]==['query_order','query_logistics','query_return_status']
        assert all(t['status']=='done' for t in out['state']['task_plan'].values())
        assert await tickets()==[] and len(setup[1])==1


async def test_sequential_select_confirm_restart_and_no_replan(setup,monkeypatch,tmp_path):
    classify(monkeypatch,setup,[{'kind':'refund','order_ref':'same'},{'kind':'complaint','order_ref':'same'}])
    path=str(tmp_path/'resume.sqlite')
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        out=await start(monkeypatch,saver,'我要申请退款，并登记投诉工单')
        assert out['interrupt']['type']=='select_order'
        before=len(await repository.list_messages(out['conversation_id']))
        with pytest.raises(runtime.ResumeRejected):await runtime.run_turn('alice','还有一个问题',out['conversation_id'],multi_intent_enabled=True)
        assert len(await repository.list_messages(out['conversation_id']))==before
        with pytest.raises(runtime.ResumeRejected):await resume(out,'9999')
        out=await resume(out,'1001');old=out
        assert out['interrupt']['actions'][0]['preview']['ticket_type']=='退款'
        assert next(t for t in out['state']['task_plan'].values() if t['kind']=='complaint')['status']=='pending'
        op=out['interrupt']['actions'][0]['operation_id']
        out=await resume(out,{'operation_id':op,'confirmed':True})
        assert out['interrupt']['actions'][0]['preview']['ticket_type']=='投诉'
        with pytest.raises(runtime.ResumeRejected):await resume(old,{'operation_id':op,'confirmed':True})
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        monkeypatch.setattr(runtime,'_graph',build_graph(saver))
        op=out['interrupt']['actions'][0]['operation_id']
        out=await resume(out,{'operation_id':op,'confirmed':True})
        assert out['interrupt'] is None and len(await tickets())==2 and len(setup[1])==1
        assert len(setup[0])==1


@pytest.mark.parametrize('cancel_at',['selection','confirmation'])
async def test_cancel_refund_continues_independent_complaint(setup,monkeypatch,tmp_path,cancel_at):
    classify(monkeypatch,setup,[{'kind':'refund','order_ref':'refund'},{'kind':'complaint','order_ref':''}])
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'cancel.sqlite')) as saver:
        out=await start(monkeypatch,saver,'我要申请退款，并登记投诉工单')
        if cancel_at=='selection':out=await resume(out,{'cancelled':True})
        else:
            out=await resume(out,'1001')
            out=await resume(out,{'operation_id':out['interrupt']['actions'][0]['operation_id'],'confirmed':False})
        assert out['interrupt']['actions'][0]['preview']['ticket_type']=='投诉'
        out=await resume(out,{'operation_id':out['interrupt']['actions'][0]['operation_id'],'confirmed':True})
        assert len(await tickets())==1 and (await tickets())[0].ticket_type=='投诉'


@pytest.mark.parametrize('retry',[True,False])
async def test_response_lost_preserves_same_operation(setup,monkeypatch,tmp_path,retry):
    classify(monkeypatch,setup,[{'kind':'refund','order_ref':'s','order_id':'1001'}])
    original=workflow.submit;receipts=[]
    async def lost(*args,**kw):
        r=await original(*args,**kw);receipts.append(r)
        if len(receipts)==1:raise ConnectionError('response lost after commit')
        return r
    monkeypatch.setattr(workflow,'submit',lost)
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'lost.sqlite')) as saver:
        out=await start(monkeypatch,saver,'对订单1001申请退款')
        op=out['interrupt']['actions'][0]['operation_id']
        out=await resume(out,{'operation_id':op,'confirmed':True})
        assert out['interrupt']['actions'][0]['operation_id']==op
        out=await resume(out,{'operation_id':op,'confirmed':retry})
        assert out['interrupt'] is None and len(await tickets())==1
        assert receipts[0]['ticket_no']==receipts[1]['ticket_no']


async def test_product_waits_then_retrieves_bound_product(setup,monkeypatch,tmp_path):
    classify(monkeypatch,setup,[{'kind':k,'order_ref':'s','product_specific':k=='knowledge'} for k in ['order','logistics','knowledge']])
    retrieved=[]
    async def retrieve(s):
        retrieved.append(s['resolved_query']);return {'evidence_strong':False,'fallback_source':'self_check','trace':{},'retrieved_snapshot':[]}
    monkeypatch.setattr(nodes,'retrieve_knowledge',retrieve)
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'kb.sqlite')) as saver:
        out=await start(monkeypatch,saver,'这笔订单是什么，送到哪了，有没有使用说明')
        assert not retrieved
        out=await resume(out,'1001')
        assert len(retrieved)==1 and '智能猫砂盆' in retrieved[0]
        assert out['interrupt'] is None
        assert sum(n=='query_order' for n,_ in setup[0])==1


def test_consultation_and_independent_order_are_separate():
    context={'user_id':'alice','candidates':[{'order_id':'1001'},{'order_id':'2002'}]}
    raw='刚刚那个订单现在送到哪了，它是什么商品，我可以退货吗'
    task={'kind':'refund','query':'我可以退货吗','source':'我可以退货吗','order_id':'1001','order_ref':'same'}
    result=simple_intents.normalize(raw,[task],context)[0]
    assert result['kind']=='return_advice' and result['order_id']=='1001'
    raw='查订单1001物流，另外一笔订单我要退款'
    task.update(source='另外一笔订单我要退款',query='我要退款',kind='refund')
    result=simple_intents.normalize(raw,[task],context)[0]
    assert result['order_id'] is None and result['order_ref'].startswith('independent:')


def test_ambiguous_single_referent_not_one_selector_per_guessed_candidate():
    raw='刚才说的两笔订单中，那笔订单现在送到哪了，它是什么商品'
    context={'user_id':'alice','candidates':[{'order_id':'1001'},{'order_id':'2002'}]}
    items=[dict(kind=k,query=k,source=raw,order_id=oid,order_ref='order:'+oid)
           for oid in ['1001','2002'] for k in ['order','logistics']]
    tasks=simple_intents.normalize(raw,items,context)
    assert len(tasks)==2 and len({t['order_ref'] for t in tasks})==1
    assert all(t['order_id'] is None for t in tasks)


def test_control_copy_never_confuses_order_selection_and_submission():
    t={'kind':'refund','order_id':'1001','status':'blocked','step':'confirm_submission'}
    text=simple_loop.control_reply(t)
    assert '确认或取消' in text and '选择' not in text and '尚未提交' in text
    t.update(status='done',receipt={'ticket_no':'T123'})
    text=simple_loop.control_reply(t)
    assert '已提交' in text and '确认' not in text


async def test_chat_409_keeps_card_and_watermark(setup,monkeypatch,tmp_path):
    import httpx
    from fastapi import FastAPI
    from app.api.chat import router
    classify(monkeypatch,setup,[{'kind':'refund','order_ref':'s'}])
    app=FastAPI();app.include_router(router)
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'api.sqlite')) as saver:
        out=await start(monkeypatch,saver,'我要申请退款')
        before=len(await repository.list_messages(out['conversation_id']))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
            response=await client.post('/api/chat',json={'user_id':'alice','conversation_id':out['conversation_id'],
                'message':'另外问一下物流','multi_intent_enabled':True,'request_id':'new-text-pending'})
            assert response.status_code==409 and response.json()['detail']['code']=='pending_task_card'
        assert len(await repository.list_messages(out['conversation_id']))==before
        out=await resume(out,'1001')
        assert out['interrupt']['type']=='task_actions'


async def test_new_turn_refreshes_order_not_history_cache(setup,monkeypatch,tmp_path):
    classify(monkeypatch,setup,[{'kind':'order','order_id':'1001','order_ref':'s'},{'kind':'logistics','order_id':'1001','order_ref':'s'}])
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'fresh.sqlite')) as saver:
        out=await start(monkeypatch,saver,'查询订单1001商品和物流')
        out=await runtime.run_turn('alice','刚刚订单1001现在送到哪了',out['conversation_id'],multi_intent_enabled=True)
        assert out['interrupt'] is None
        assert sum(n=='query_order' for n,_ in setup[0])==2
        assert sum(n=='query_logistics' for n,_ in setup[0])==2


async def test_v1_card_still_resumes_when_simple_enabled(wired,monkeypatch,tmp_path):
    from tests.graph.test_subflow_tasks import planner,invoke_new,resume as old_resume
    monkeypatch.setattr(settings,'multi_intent_context_v2_enabled',False)
    monkeypatch.setattr(settings,'multi_intent_simple_enabled',False)
    planner(monkeypatch,['order','logistics'])
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'old.sqlite')) as saver:
        graph=build_graph(saver)
        first=await invoke_new(graph,monkeypatch,'这笔订单是什么，物流到哪了')
        assert first['state']['task_plan_version']==1
        monkeypatch.setattr(settings,'multi_intent_simple_enabled',True)
        async def forbidden(*a,**kw):raise AssertionError('旧卡片不能走新规划')
        monkeypatch.setattr(simple_intents,'classify',forbidden)
        out,_=await old_resume(graph,monkeypatch,first['conversation_id'],'1001')
        assert out['state']['task_plan_version']==1 and out['interrupt'] is None


async def test_sse_reply_precedes_persisted_card_and_resume_continues(setup,monkeypatch,tmp_path):
    classify(monkeypatch,setup,[{'kind':'refund','order_ref':'s'},{'kind':'complaint','order_ref':'s'}])
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'sse.sqlite')) as saver:
        monkeypatch.setattr(runtime,'_graph',build_graph(saver))
        events=[ev async for ev in runtime.stream_turn('alice','商品漏电了，我要申请退款并登记投诉',None,multi_intent_enabled=True)]
        card=next(ev for ev in events if ev['type']=='interrupt')
        assert card['kind']=='select_order'
        assert next(i for i,e in enumerate(events) if e['type']=='task_event')<events.index(card)
        assert len([e for e in events if e['type']=='task_event' and '停止使用' in e.get('text','')])==1
        follow=[ev async for ev in runtime.stream_resume(card['conversation_id'],'1001',user_id='alice',interrupt_id=card['interrupt_id'])]
        assert any(e['type']=='interrupt' and e['kind']=='task_actions' for e in follow)
        assert not any(e['type']=='task_event' and '停止使用' in e.get('text','') for e in follow)
