import json
from sqlalchemy import select
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from app.db import repository,workflow
from app.db.models import Message
from app.graph import runtime,subflow_tasks
from app.graph.build import build_graph
from tests.graph.test_subflow_tasks import wired,planner,invoke_new,resume


async def test_read_results_are_one_public_message_and_history_matches(wired,tmp_path,monkeypatch):
    planner(monkeypatch,['order','logistics'],oid='1001')
    async def read(state):
        task={**state['tasks'][0],'status':'DONE'}
        receipts=[{'name':'query_order','content':json.dumps({'order_id':'1001','status':'已付款','product':'猫砂盆','tracking_no':'SF-1'})}]
        if task['kind']=='logistics':receipts.append({'name':'query_logistics','content':json.dumps({'tracking_no':'SF-1','current_city':'成都','status':'已揽件'})})
        return {'tasks':[task],'results':[{'answer':'需要我继续帮您查询吗？','receipts':receipts}]}
    monkeypatch.setattr(subflow_tasks,'execute_read',read)
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'ui.sqlite')) as saver:
        out=await invoke_new(build_graph(saver),monkeypatch,'查询订单1001和物流')
        saved=await workflow.events(out['conversation_id'],'alice')
        assert len(saved)==1
        assert '成都' in saved[0]['text'] and '继续帮' not in saved[0]['text']
        assert out['state']['answer']==saved[0]['text']
        messages=await repository.list_dialog_messages(out['conversation_id'])
        assert len([m for m in messages if m.role=='assistant'])==1


async def test_only_latest_stage_returned_and_internal_previews_not_in_history(wired,tmp_path,monkeypatch):
    planner(monkeypatch,['refund','complaint'],oid='1001')
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'stages.sqlite')) as saver:
        graph=build_graph(saver)
        first=await invoke_new(graph,monkeypatch,'为订单1001退款并投诉')
        assert first['state']['answer'].count('尚未提交')==2
        refund=next(a for a in first['interrupt']['actions'] if a['preview']['ticket_type']=='退款')
        out,_=await resume(graph,monkeypatch,first['conversation_id'],{'operation_id':refund['operation_id'],'confirmed':True})
        assert '工单已提交' in out['state']['answer']
        assert '申请已准备' not in out['state']['answer']
        saved=await workflow.events(first['conversation_id'],'alice')
        assert len(saved)==2 and all(':reply:' in e['event_id'] for e in saved)
        async with workflow.base.async_session() as session:
            internal=(await session.execute(select(workflow.WorkflowRecord).where(workflow.WorkflowRecord.key.like('internal:%')))).scalars().all()
        assert internal and all('sequence' not in r.payload for r in internal)
        messages=await repository.list_dialog_messages(first['conversation_id'])
        assert len([m for m in messages if m.role=='assistant'])==2
        # Replaying presentation after a checkpoint gap returns the same row.
        state=out['state']; state['subflow_presented']=first['state']['subflow_presented']
        again=await subflow_tasks.present(state)
        assert again['task_events'][0]['message_id']==saved[-1]['message_id']


async def test_internal_event_scope_does_not_leak_to_parallel_request(wired):
    import asyncio
    cid=await repository.create_conversation('alice')
    async def internal():
        with workflow.internal_events():return await workflow.event(cid,'alice','hidden','内部调试 DONE')
    hidden,visible=await asyncio.gather(internal(),workflow.event(cid,'alice','public','您好'))
    assert hidden['internal'] and 'internal' not in visible
    assert [e['text'] for e in await workflow.events(cid,'alice')]==['您好']
    assert [m.content for m in await repository.list_dialog_messages(cid)]==['您好']


async def test_knowledge_wrappers_keep_gate_citations_and_one_public_reply(wired,tmp_path,monkeypatch):
    from langchain_core.messages import AIMessage
    from app.core import llm,task_planner
    from app.graph import nodes
    async def extract(raw):
        return [{'kind':kind,'query':query,'source':raw,'order_id':oid} for kind,query,oid in [
            ('order','查询订单1001','1001'),('knowledge','退货条件',None),('knowledge','运费规则',None)]]
    monkeypatch.setattr(task_planner,'extract',extract)
    async def retrieve(state):
        ident=10 if '退货' in state['resolved_query'] else 20
        return {'evidence_strong':True,'evidence_confidence':.9,'evidence':'经过检索的适用政策',
                'citations':[{'id':ident,'n':1,'answer':str(ident),'section_path':'政策'}]}
    monkeypatch.setattr(nodes,'retrieve_knowledge',retrieve)
    class Model:
        async def ainvoke(self,messages):
            return AIMessage(messages[-1]['content']+'依据政策[1]。\n\n还有其他问题吗？')
    monkeypatch.setattr(llm,'get_chat_model',lambda:Model())
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path/'citations.sqlite')) as saver:
        out=await invoke_new(build_graph(saver),monkeypatch,'查询订单1001，说明退货条件和运费规则')
        events=await workflow.events(out['conversation_id'],'alice')
        assert len(events)==1
        assert [c['id'] for c in events[0]['citations']]==[10,20]
        assert '[1]' in events[0]['text'] and '[2]' in events[0]['text']
        assert '还有其他问题' not in events[0]['text']
        # Evidence insufficiency stays a refusal when merging other successes.
        async def weak(state):return {'evidence_strong':False,'citations':[]}
        monkeypatch.setattr(nodes,'retrieve_knowledge',weak)
        followup=await runtime.run_turn('alice','查询订单1001，说明退货条件和运费规则',out['conversation_id'],multi_intent_enabled=True)
        assert '缺少充分依据' in followup['state']['answer']
        assert '依据政策' not in followup['state']['answer']
