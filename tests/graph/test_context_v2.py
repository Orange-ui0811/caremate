import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from app.config import settings
from app.core import task_context,task_planner
from app.graph import runtime,nodes
from app.graph.build import build_graph
from app.graph.task_dependencies import refresh
from tests.graph.test_subflow_tasks import wired,invoke_new,resume,tickets


def install(monkeypatch):
    monkeypatch.setattr(settings,'multi_intent_context_v2_enabled',True)
    calls=[]
    async def extract(raw,*,context=None):
        calls.append(context)
        return task_context.normalize(raw,[task_planner.ContextTask(kind=k,source=raw,query='有没有使用说明' if k=='knowledge' else raw,
            subject_ref='s',reference_span=raw,knowledge_scope='product' if k=='knowledge' else 'general').model_dump()
            for k in ['order','logistics','knowledge']],context)
    monkeypatch.setattr(task_planner,'extract',extract)
    return calls


async def test_wait_select_once_restart_and_then_retrieve(wired,tmp_path,monkeypatch):
    calls=install(monkeypatch);retrieved=[]
    async def retrieve(state):
        retrieved.append((state['resolved_query'],state.get('task_context')))
        return {'evidence_strong':False,'fallback_source':'self_check','trace':{},'retrieved_snapshot':[]}
    monkeypatch.setattr(nodes,'retrieve_knowledge',retrieve)
    path=str(tmp_path/'context.sqlite')
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        first=await invoke_new(build_graph(saver),monkeypatch,'这笔订单送到哪了，里面是什么商品，有没有使用说明')
        assert first['interrupt']['type']=='select_order' and not retrieved
        assert first['state']['task_plan']['knowledge']['status']=='WAITING_DEPENDENCY'
        cid=first['conversation_id']
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        out,_=await resume(build_graph(saver),monkeypatch,cid,'1001')
        assert out['interrupt'] is None and len(calls)==1
        assert wired==[('order','1001'),('logistics','1001')]
        assert len(retrieved)==1 and '智能猫砂盆' in retrieved[0][0]
        assert out['state']['task_plan']['knowledge']['evidence_gate']=='self_check'
        assert await tickets()==[]


async def test_cancellation_cancels_dependents_without_retrieval(wired,tmp_path,monkeypatch):
    install(monkeypatch)
    async def forbidden(*a):raise AssertionError('不能提前检索')
    monkeypatch.setattr(nodes,'retrieve_knowledge',forbidden)
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'cancel.sqlite')) as saver:
        graph=build_graph(saver)
        first=await invoke_new(graph,monkeypatch,'这笔订单送到哪了，里面是什么商品，有没有使用说明')
        out,_=await resume(graph,monkeypatch,first['conversation_id'],{'cancelled':True})
        assert all(t['status']=='CANCELLED' for t in out['state']['task_plan'].values())
        assert out['interrupt'] is None and wired==[]


def test_done_and_unrelated_tasks_are_not_reactivated():
    tasks={'a':{'kind':'order','status':'DONE','order_id':'1001','subject_ref':'s'},
           'b':{'kind':'knowledge','status':'WAITING_DEPENDENCY','knowledge_scope':'product','subject_ref':'s'},
           'c':{'kind':'refund','status':'PENDING','subject_ref':'another'}}
    refresh(tasks,'alice')
    assert tasks['a']['status']=='DONE' and tasks['b']['status']=='PENDING'
    assert tasks['b']['context_order_id']=='1001'
    assert tasks['c']['status']=='WAITING_USER' and not tasks['c'].get('order_id')


def test_rejected_product_binding_waits_instead_of_retrieving_without_object():
    tasks={'k':{'kind':'knowledge','status':'PENDING','knowledge_scope':'product','subject_ref':'',
                'binding_validation':'unsupported_candidate'}}
    refresh(tasks,'alice')
    assert tasks['k']['status']=='WAITING_USER' and tasks['k']['step']=='select_order'


async def test_enabling_v2_does_not_replan_existing_v1_interrupt(wired,tmp_path,monkeypatch):
    from tests.graph.test_subflow_tasks import planner
    monkeypatch.setattr(settings,'multi_intent_context_v2_enabled',False)
    planner(monkeypatch,['order','logistics'])
    path=str(tmp_path/'legacy.sqlite')
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        first=await invoke_new(build_graph(saver),monkeypatch,'这笔订单是什么，它物流到哪了')
        assert first['state']['task_plan_version']==1
        cid=first['conversation_id']
    monkeypatch.setattr(settings,'multi_intent_context_v2_enabled',True)
    async def forbidden(*a,**kw):raise AssertionError('恢复不能重新规划')
    monkeypatch.setattr(task_planner,'extract',forbidden)
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        out,_=await resume(build_graph(saver),monkeypatch,cid,'1001')
        assert out['state']['task_plan_version']==1 and out['interrupt'] is None
        assert wired==[('order','1001'),('logistics','1001')]
