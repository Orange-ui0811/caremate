"""Real-provider checks for the shared Agent path, isolated from live sessions."""
import argparse
import asyncio
import hashlib
import json
import re
from contextlib import AsyncExitStack
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from sqlalchemy import select,text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine,async_sessionmaker
from app.config import settings
from app.core import runtime_settings
from app.db import base,models,repository,workflow
from app.graph import runtime,nodes,simple_loop,task_queue
from app.graph.build import build_graph
from app.kb import milvus_client
from app.api.actions import create_refund_action,create_ticket_action
from app.schemas.actions import CreateRefundRequest,CreateTicketRequest
from scripts import eval_simple_loop as capture

OUT=Path('E:/Mewhelp/data/verification/multi-intent-shared-agent/20261105-v1')
CASES=[
 {'id':'R1','history':['请查询订单1001，这是我们接下来讨论的订单。'],'raw':'刚刚那个订单现在送到哪了，它是什么商品，我可以退货吗','intents':['订单','物流','退款退货'],'max_select':0},
 {'id':'R2','history':['请查询订单1001，这是我们接下来讨论的订单。'],'raw':'刚刚这单的商品漏电了，帮我申请退款，并登记投诉商家的工单，投诉内容是设备漏电存在安全隐患','intents':['退款退货','人工'],'max_select':0},
 {'id':'R3','history':[],'raw':'我收到的商品无法开机，帮我申请退款，另外请登记投诉工单，投诉内容是商品无法使用','intents':['退款退货','人工'],'max_select':1},
 {'id':'R4','history':[],'raw':'查询订单1001的物流，并对订单2002申请退款','intents':['物流','退款退货'],'max_select':0},
 {'id':'R5','history':[],'raw':'查询订单1001的物流，另外有一笔订单我要申请退款，退款订单我自己选择','intents':['物流','退款退货'],'max_select':1,'selection':'2002'},
 {'id':'R6','history':['请查询订单1001，这是我们接下来讨论的订单。'],'raw':'刚刚那笔订单现在送到哪了，它现在还在保修期吗','intents':['物流','售后'],'max_select':0},
 {'id':'R7','history':['请查询订单1001，这是我们接下来讨论的订单。'],'raw':'刚刚那个订单现在送到哪了，它是什么商品，有没有使用说明','intents':['订单','物流','商品咨询'],'max_select':0},
 {'id':'R8','history':[],'raw':'商品无法开机，我要申请退款，并登记投诉商家的工单，投诉内容是商品无法使用','intents':['退款退货','人工'],'max_select':1,'cancel_refund':True},
]


def save(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str),encoding='utf-8')


async def main(args):
    output=Path(args.out) if args.out else OUT
    folder=output/args.run;folder.mkdir(parents=True,exist_ok=True)
    if not (output/'cases.json').exists():save(output/'cases.json',CASES)
    assert json.loads((output/'cases.json').read_text(encoding='utf-8'))==CASES
    # Freeze quality rules before the first model call. Raw replies remain saved
    # for manual review; these deterministic checks do not prove all semantics.
    invitation=r'请问还|还有其他|需要我[^。\n]{0,35}[吗？]|需要的话[^。\n]{0,15}告诉|随时[^。\n]{0,15}(?:问|咨询|联系|告诉)|已排入队列|后续任务'
    topic_patterns={'R1':{'订单':'物流|揽件|分拨|退货|退款','物流':'退货|退款|保修'},
        'R2':{'退款退货':'投诉'},'R3':{'退款退货':'投诉'},
        'R6':{'物流':'保修|在保|维修','售后':'物流|揽件|分拨'},
        'R7':{'订单':'物流|揽件|分拨|说明|手册','物流':'说明|指南|手册'},'R8':{'退款退货':'投诉'}}
    rules={'invitation_pattern':invitation,'forbidden_sibling_topics':topic_patterns}
    if not (output/'quality-rules.json').exists():save(output/'quality-rules.json',rules)
    assert json.loads((output/'quality-rules.json').read_text(encoding='utf-8'))==rules
    expected_tickets={'R1':[],'R2':['投诉','退款'],'R3':['投诉','退款'],'R4':[],
                      'R5':[],'R6':[],'R7':[],'R8':['投诉']}
    if not (output/'expected-tickets.json').exists():save(output/'expected-tickets.json',expected_tickets)
    assert json.loads((output/'expected-tickets.json').read_text(encoding='utf-8'))==expected_tickets
    settings.multi_intent_shared_agent_enabled=True;settings.multi_intent_enabled=True
    settings.langfuse_public_key='';settings.model_pool_db_path=str(folder/'pool.sqlite')
    runtime_settings.SETTINGS_PATH=Path('E:/Mewhelp/data/runtime_settings.json')
    db='caremate_shared_'+args.run
    assert db.replace('_','').isalnum()
    url=make_url(settings.database_url)
    admin=create_async_engine(url.set(database=None),isolation_level='AUTOCOMMIT')
    async with admin.begin() as c:await c.execute(text(f'CREATE DATABASE IF NOT EXISTS {db} CHARACTER SET utf8mb4'))
    await admin.dispose();old=base.engine
    base.engine=create_async_engine(url.set(database=db),pool_pre_ping=True)
    base.async_session=async_sessionmaker(base.engine,expire_on_commit=False)
    async with base.engine.begin() as c:await c.run_sync(base.Base.metadata.create_all)
    await workflow.ensure_tables()
    def kb():
        c=milvus_client.get_client();assert c.has_collection('knowledge');c.load_collection('knowledge');milvus_client._ensured.add('knowledge');return milvus_client.count(c)
    save(folder/'manifest.json',{'database':db,'kb_count':await milvus_client.acall(kb),'history_recall':settings.context_legacy_recall_enabled,
        'context_policy':settings.context_policy,'confidence':runtime_settings.get_confidence_settings(),
        'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'app').rglob('*.py')}})
    capture.instrument()
    originals={name:getattr(nodes,name) for name in ['main_agent','agent_tools']}
    original_take=task_queue.take
    async def take(state):
        update=await original_take(state)
        capture.ACTIVE['current_task']=update.get('active_task')
        return update
    task_queue.take=take
    for name,fn in originals.items():
        async def wrapped(state,config=None,_fn=fn,_name=name):
            capture.ACTIVE.setdefault('node_visits',[]).append({'node':_name,'task':state.get('active_task'),'order_id':state.get('order_id'),
                'allowed_tools':sorted(task_queue.allowed_tools(state) or [])})
            return await _fn(state,config) if _name=='main_agent' else await _fn(state)
        setattr(nodes,name,wrapped)
    async def forbidden(*a,**kw):raise AssertionError('Old executor must not be entered')
    for name in ['plan','prepare','agent','tools','execute','present']:setattr(simple_loop,name,forbidden)
    async with AsyncExitStack() as stack:
        for case in CASES:
            if args.ids and case['id'] not in args.ids.split(','):continue
            target=folder/(case['id']+'.json')
            if target.exists():continue
            uid='shared-agent-synthetic';cid=await repository.create_conversation(uid)
            for line in case['history']:
                await repository.append_message(cid,'user',line)
                await repository.append_message(cid,'assistant','已确认订单1001。当前状态以接下来的实时查询为准。')
            capture.ACTIVE={'case':case,'conversation_id':cid,'http':[],'events':[],'retrieval':[],'steps':[],'decisions':[]}
            start=time.monotonic()
            saver=await stack.enter_async_context(AsyncSqliteSaver.from_conn_string(str(folder/(case['id']+'.sqlite'))))
            runtime._graph=build_graph(saver)
            async def collect(iterator):
                card=None
                async for ev in iterator:
                    capture.ACTIVE['events'].append({**ev,'seconds':time.monotonic()-start,'task':capture.ACTIVE.get('current_task')})
                    if ev['type']=='interrupt':card=ev
                snapshot=await runtime._graph.aget_state({'configurable':{'thread_id':str(cid)}})
                capture.ACTIVE['steps'].append(snapshot.values)
                return card
            try:
                card=await collect(runtime.stream_turn(uid,case['raw'],cid,multi_intent_enabled=True))
                for _ in range(8):
                    if not card:break
                    if card['kind']=='select_order':value=case.get('selection','1001')
                    elif card['kind']=='confirm_ticket':value={'confirmed':True}
                    elif card['kind']=='task_form':
                        form=card['form'];confirm=not(case.get('cancel_refund') and form['type']=='refund_form')
                        if confirm:
                            if form['type']=='refund_form':
                                receipt=await create_refund_action(CreateRefundRequest(user_id=uid,conversation_id=cid,
                                    request_id=card['request_id'],order_id=form['draft']['order_id'],reason='质量问题'))
                            else:receipt=await create_ticket_action(CreateTicketRequest(user_id=uid,conversation_id=cid,
                                    request_id=card['request_id'],ticket_type='投诉',description=case['raw']))
                            capture.ACTIVE.setdefault('form_receipts',[]).append(receipt.model_dump())
                        value={'confirmed':confirm}
                    else:raise AssertionError('Unexpected card '+card['kind'])
                    capture.ACTIVE['decisions'].append({'card':card,'value':value})
                    card=await collect(runtime.stream_resume(cid,value,user_id=uid,interrupt_id=card['interrupt_id']))
                final=capture.ACTIVE['steps'][-1]
                tasks=[t for t in final['task_plan'].values() if t.get('kind')!='safety']
                async with base.async_session() as s:
                    tickets=(await s.execute(select(models.Ticket).where(models.Ticket.conversation_id==cid))).scalars().all()
                    logs=(await s.execute(select(models.ToolAuditLog).where(models.ToolAuditLog.conversation_id==cid))).scalars().all()
                capture.ACTIVE['tickets']=[{'number':t.ticket_no,'type':t.ticket_type} for t in tickets]
                capture.ACTIVE['audit']=[{'name':t.tool_name,'args':t.arguments,'status':t.status} for t in logs]
                text_answer=''.join(e['text'] for e in capture.ACTIVE['events'] if e['type']=='delta')
                selections=sum(d['card']['kind']=='select_order' for d in capture.ACTIVE['decisions'])
                capture.ACTIVE['checks']={'original_agent_visited':any(v['node']=='main_agent' for v in capture.ACTIVE['node_visits']),
                    'expected_intents':set(case['intents'])<=set(t['intent'] for t in tasks),'no_extra_selection':selections<=case['max_select'],
                    'queue_terminal':card is None and all(t['status'] in {'done','cancelled','failed'} for t in tasks),
                    'ticket_receipts':sorted(t.ticket_type for t in tickets)==expected_tickets[case['id']],
                    'business_requests_finished':all(t['status']=='done' for t in tasks if t['intent']=='人工' and t['action']=='request'),
                    'body_deltas':sum(e['type']=='delta' for e in capture.ACTIVE['events'])>1,
                    'no_internal_summary':not any(x in text_answer for x in ['task_id','operation_id','本次事项处理结果'])}
                capture.ACTIVE['visible_answer']=text_answer
                quality=[]
                for key,t in final['task_plan'].items():
                    if not t.get('intent'):continue
                    body=''.join(e['text'] for e in capture.ACTIVE['events'] if e['type']=='delta' and e.get('task')==key)
                    pattern=topic_patterns.get(case['id'],{}).get(t['intent'])
                    quality.append({'task':key,'intent':t['intent'],'body':body,
                        'generic_invitations':re.findall(invitation,body),
                        'sibling_topics':re.findall(pattern,body) if pattern else []})
                capture.ACTIVE['quality']=quality
            except Exception as exc:
                import traceback
                capture.ACTIVE['error']=type(exc).__name__+':'+str(exc);capture.ACTIVE['traceback']=traceback.format_exc()
            capture.ACTIVE['seconds']=time.monotonic()-start
            save(target,capture.ACTIVE)
            print(json.dumps({'case':case['id'],'checks':capture.ACTIVE.get('checks'),'error':capture.ACTIVE.get('error')},ensure_ascii=False),flush=True)
    await base.engine.dispose();await old.dispose()
    from app.core.model_transport import close_clients
    await close_clients()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--ids');p.add_argument('--out')
    asyncio.run(main(p.parse_args()))
