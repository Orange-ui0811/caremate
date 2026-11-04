import json

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.graph import nodes, read_subflows as pilot


def inputs(query='订单1001是什么，它现在物流到哪了？', mid=1, ids=None):
    return {'query':query,'user_id':'alice','conversation_id':123,'current_message_id':mid,
            'business_context':{'user_id':'alice','conversation_id':123,'order_ids':ids or [],'recent':True}}


async def test_reuse_real_nodes_sqlite_resume_and_no_parent_history_write(tmp_path, monkeypatch):
    calls = []
    def no_append(*args, **kwargs):
        raise AssertionError('subflows must not append dialogue messages')
    monkeypatch.setattr(nodes.repository,'append_message',no_append)
    async def extract(raw):
        return [{'kind':k,'query':raw,'source':raw,'order_id':'1001',
                 'binding':{'status':'resolved','method':'task_clause'}} for k in ('order','logistics')]
    monkeypatch.setattr(pilot.task_planner, 'extract', extract)
    from app.tools import registry
    specs = registry.builtin_specs()
    # Real Agent/tool nodes, deterministic model and engine receipts.
    class Model:
        def bind_tools(self, tools):
            return self
        async def ainvoke(self, messages, config=None):
            text = ' '.join(str(m.content) for m in messages)
            tools = [m for m in messages if isinstance(m, ToolMessage)]
            if not tools:
                call = ('query_order', {'order_id':'1001'})
            elif '当前物流' in text and len(tools) == 1:
                call = ('query_logistics', {'tracking_no':'SF-test'})
            else:
                return AIMessage('订单1001查询结果已确认。')
            return AIMessage('', tool_calls=[{'name':call[0],'args':call[1],'id':f'call-{len(tools)}','type':'tool_call'}])
    monkeypatch.setattr(nodes,'get_chat_model',lambda **kw:Model())
    async def all_specs(): return specs
    monkeypatch.setattr(registry,'get_all_specs',all_specs)
    from types import SimpleNamespace
    async def engine(tc, *args, **kwargs):
        calls.append(tc['name'])
        return SimpleNamespace(tool_message=ToolMessage(json.dumps({'tracking_no':'SF-test','order_id':'1001'}),
            name=tc['name'],tool_call_id=tc['id']))
    monkeypatch.setattr(nodes.engine,'execute_tool_call',engine)
    path = str(tmp_path/'pilot.sqlite')
    config = {'configurable':{'thread_id':'123'}}
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        graph = pilot.build_pilot(saver, interrupt_after=['execute'])
        first = await graph.ainvoke(inputs(),config)
        assert [t['status'] for t in first['tasks']] == ['DONE','PENDING']
        assert calls == ['query_order']
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        graph = pilot.build_pilot(saver)
        out = await graph.ainvoke(None,config)
        assert [t['status'] for t in out['tasks']] == ['DONE','DONE']
        assert calls == ['query_order','query_order','query_logistics']
        assert 'messages' not in out  # Child transcripts never appended to dialogue.
        assert len(out['results']) == 2


async def test_missing_order_and_unsupported_are_explicit(monkeypatch):
    async def extract(raw):
        return [{'kind':'logistics','query':raw,'source':raw,'order_id':None},
                {'kind':'other','query':raw,'source':raw,'order_id':None}]
    monkeypatch.setattr(pilot.task_planner,'extract',extract)
    out = await pilot.build_pilot().ainvoke(inputs('查另一笔订单物流，并投诉商家'))
    assert [t['status'] for t in out['tasks']] == ['WAITING_USER','FAILED']
    assert not out['results']


async def test_unbound_task_does_not_block_bound_sibling(monkeypatch):
    async def extract(raw):
        return [{'kind':'order','query':'查订单1001','source':'查订单1001','order_id':'1001'},
                {'kind':'logistics','query':'另一笔物流','source':'另一笔订单物流','order_id':None}]
    monkeypatch.setattr(pilot.task_planner,'extract',extract)
    class Child:
        async def ainvoke(self, state):
            return {**state,'messages':[*state['messages'],
                ToolMessage('{"order_id":"1001"}',name='query_order',tool_call_id='r'),AIMessage('订单1001') ]}
    monkeypatch.setattr(pilot,'build_business_subflow',lambda:Child())
    out=await pilot.build_pilot().ainvoke(inputs('查订单1001，再查另一笔订单物流'))
    assert [t['status'] for t in out['tasks']] == ['DONE','WAITING_USER']
    assert len(out['results']) == 1


async def test_missing_receipts_cannot_be_marked_done(monkeypatch):
    async def extract(raw):
        return [{'kind':'order','query':raw,'source':raw,'order_id':'1001'}]
    monkeypatch.setattr(pilot.task_planner,'extract',extract)
    class Child:
        async def ainvoke(self, state):
            return {**state,'messages':[*state['messages'],AIMessage('已经查好了') ]}
    monkeypatch.setattr(pilot,'build_business_subflow',lambda:Child())
    out=await pilot.build_pilot().ainvoke(inputs('查订单1001'))
    assert out['tasks'][0]['status'] == 'FAILED'
    assert '已经查好了' not in out['results'][0]['answer']


async def test_unpaid_order_is_a_valid_no_logistics_result(monkeypatch):
    async def extract(raw):
        return [{'kind':'logistics','query':raw,'source':raw,'order_id':'2002'}]
    monkeypatch.setattr(pilot.task_planner,'extract',extract)
    class Child:
        async def ainvoke(self, state):
            return {**state,'messages':[*state['messages'],
                ToolMessage('{"order_id":"2002","status":"待付款"}',name='query_order',tool_call_id='r'),
                AIMessage('尚未付款，无物流进度')]}
    monkeypatch.setattr(pilot,'build_business_subflow',lambda:Child())
    out=await pilot.build_pilot().ainvoke(inputs('查订单2002物流'))
    assert out['tasks'][0]['status'] == 'DONE'
    assert out['results'][0]['outcome'] == 'UNPAID'


async def test_cross_round_reference_uses_existing_binder(monkeypatch):
    raw = '刚刚的订单是哪个，它现在物流到哪了'
    async def extract(text):
        return [{'kind':k,'query':text,'source':text,'order_id':None} for k in ('order','logistics')]
    monkeypatch.setattr(pilot.task_planner,'extract',extract)
    out = await pilot.plan(inputs(raw,ids=['1001']))
    assert [t['order_id'] for t in out['tasks']] == ['1001','1001']
    other = await pilot.plan(inputs(raw,ids=['1001','2002']))
    assert all(t['status'] == 'WAITING_USER' for t in other['tasks'])


async def test_scope_blocks_write_other_order_and_guessed_tracking_before_engine(monkeypatch):
    def forbidden(*a,**kw): raise AssertionError('must not execute')
    monkeypatch.setattr(nodes.engine,'execute_tool_call',forbidden)
    for name,args in [('create_ticket',{}),('submit_refund',{'order_id':'1001'}),
                      ('query_order',{'order_id':'2002'}),('query_logistics',{'tracking_no':'guess'})]:
        state = {'user_id':'alice','task_scope':{'order_id':'1001','allowed_tools':['query_order','query_logistics']},
                 'messages':[AIMessage('',tool_calls=[{'id':'x','name':name,'args':args,'type':'tool_call'}])]}
        out = await nodes.agent_tools(state)
        assert out['messages'][0].status == 'error'
