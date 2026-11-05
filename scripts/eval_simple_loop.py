"""Frozen paired synthetic scenarios: original single route vs shared serial loop."""
import argparse
import asyncio
from contextlib import AsyncExitStack
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import httpx
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from sqlalchemy import select,text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine,async_sessionmaker
from app.config import settings
from app.core import simple_intents,runtime_settings
from app.db import base,models,repository,workflow
from app.graph import runtime,nodes,simple_loop
from app.graph.build import build_graph
from app.kb import milvus_client
from scripts.eval_context_v2 import Tee,record_response,save

OUT=Path('E:/Mewhelp/data/verification/multi-intent-simple-loop/20261105-v1')
ACTIVE={}


def instrument():
    original=httpx.AsyncClient.send
    async def send(client,request,**kwargs):
        if urlsplit(str(request.url)).hostname not in {'ai.gitee.com','api.moark.com'}:
            return await original(client,request,**kwargs)
        try:payload=json.loads(request.content)
        except ValueError:payload={}
        row={'endpoint':request.url.path,'request':payload};started=time.monotonic()
        ACTIVE.setdefault('http',[]).append(row);request.headers['accept-encoding']='identity'
        try:
            response=await original(client,request,**kwargs);row['status']=response.status_code
            if response.status_code>=400:row['error_body']=(await response.aread()).decode('utf-8',errors='replace')[:2000]
            if response.is_stream_consumed:record_response(row,response.content)
            else:response.stream=Tee(response.stream,row)
            return response
        except Exception as exc:row['error']=type(exc).__name__;raise
        finally:row['response_headers_seconds']=time.monotonic()-started
    httpx.AsyncClient.send=send
    classify=simple_intents.classify
    async def classified(state):
        row={'raw':state['task_origin'],'resolved':state.get('resolved_query')};ACTIVE.setdefault('planner',[]).append(row)
        result=await classify(state);row.update(tasks=result[0],context=result[1],proposals=result[2]);return result
    simple_intents.classify=classified
    for name in ('retrieve_knowledge','retrieve_policy'):
        original_fn=getattr(nodes,name)
        async def retrieved(state,_fn=original_fn,_name=name):
            row={'node':_name,'query':state.get('resolved_query'),'task_context':state.get('task_context')}
            ACTIVE.setdefault('retrieval',[]).append(row);result=await _fn(state);row['result']=result;return result
        setattr(nodes,name,retrieved)


async def seed(case,uid):
    cid=await repository.create_conversation(uid)
    for oid in case['focus']:
        msg=f'请确认订单{oid}，这是我们正在讨论的订单。'
        mid=await repository.append_message(cid,'user',msg)
        await repository.append_message(cid,'assistant',f'已经确认订单{oid}，商品信息以后续查询为准。')
        await workflow.save_plan(cid,uid,f'seed:{cid}:{oid}',mid,{'order':{'kind':'order','task_id':f'seed:{cid}:{oid}',
            'status':'DONE','order_id':oid,'source':msg,'binding':{'method':'user_selection','status':'resolved'}}})
    return cid


async def forms(out,case,uid):
    from app.api.actions import create_refund_action,create_ticket_action
    from app.schemas.actions import CreateRefundRequest,CreateTicketRequest
    seen=set()
    for a in out['state'].get('suggested_actions',[]):
        kind=a.get('type')
        if kind in seen:continue
        seen.add(kind)
        if kind=='refund_form':
            if case['action']=='cancel_refund':ACTIVE.setdefault('decisions',[]).append({'form':'refund','confirmed':False});continue
            oid=a.get('draft',{}).get('order_id')
            if not oid:continue
            receipt=await create_refund_action(CreateRefundRequest(user_id=uid,conversation_id=out['conversation_id'],
                order_id=oid,reason='质量问题',request_id=f"eval:{out['conversation_id']}:refund"))
            ACTIVE.setdefault('decisions',[]).append({'form':'refund','confirmed':True,'order_id':oid,'receipt':receipt.model_dump()})
        elif kind=='create_ticket' and '投诉' in case['raw']:
            receipt=await create_ticket_action(CreateTicketRequest(user_id=uid,conversation_id=out['conversation_id'],
                ticket_type='投诉',description=case['raw'],request_id=f"eval:{out['conversation_id']}:complaint"))
            ACTIVE.setdefault('decisions',[]).append({'form':'complaint','confirmed':True,'receipt':receipt.model_dump()})


async def run_case(case,args,folder,stack):
    global ACTIVE
    ACTIVE={'case':case,'version':args.version,'rep':args.rep,'http':[],'steps':[],'planner':[],'retrieval':[],'decisions':[]}
    started=time.monotonic();uid=f'simple-loop-synthetic-r{args.rep}'
    cid=await seed(case,uid);ACTIVE['conversation_id']=cid
    saver=await stack.enter_async_context(AsyncSqliteSaver.from_conn_string(str(folder/(case['id']+'.sqlite'))))
    runtime._graph=build_graph(saver)
    out=await runtime.run_turn(uid,case['raw'],cid,multi_intent_enabled=args.version=='simple')
    out['state']['answer']=nodes.resolve_answer(out['state'])
    ACTIVE['steps'].append(out);selections=0
    for _ in range(10):
        intr=out.get('interrupt')
        if not intr:break
        if intr['type']=='select_order':
            selections+=1;value='2002' if case['action']=='select_2002' else '1001'
        elif intr['type']=='task_actions':
            action=intr['actions'][0]
            value={'operation_id':action['operation_id'],'confirmed':not(case['action']=='cancel_refund' and action['preview']['ticket_type']=='退款')}
            ACTIVE['decisions'].append({**value,'preview':action['preview']})
        elif intr['type']=='confirm_ticket':
            value={'confirmed':not(case['action']=='cancel_refund' and intr.get('preview',{}).get('ticket_type')=='退款')}
            ACTIVE['decisions'].append({**value,'preview':intr.get('preview')})
        else:raise ValueError('Unexpected interrupt '+str(intr['type']))
        out=await runtime.resume_turn(cid,value,user_id=uid,interrupt_id=intr['interrupt_id'])
        out['state']['answer']=nodes.resolve_answer(out['state']);ACTIVE['steps'].append(out)
    if args.version=='single':await forms(out,case,uid)
    async with base.async_session() as s:
        tickets=(await s.execute(select(models.Ticket).where(models.Ticket.conversation_id==cid))).scalars().all()
        logs=(await s.execute(select(models.ToolAuditLog).where(models.ToolAuditLog.conversation_id==cid))).scalars().all()
    ACTIVE['tickets']=[{'number':t.ticket_no,'type':t.ticket_type,'description':t.description} for t in tickets]
    ACTIVE['audit']=[{'tool':t.tool_name,'args':t.arguments,'status':t.status,'result':t.result_summary} for t in logs]
    ACTIVE['selections']=selections;ACTIVE['extra_selections']=max(0,selections-case['selections'])
    answers='\n'.join(s['state'].get('answer','') for s in ACTIVE['steps'])
    tasks=out['state'].get('task_plan',{})
    order_calls=[a for a in ACTIVE['audit'] if a['tool']=='query_order' and a['status']=='成功']
    keys=[a['args'].get('order_id') for a in order_calls]
    ACTIVE['duplicate_order_calls']=len(keys)-len(set(keys))
    goals={}
    for kind in case['kinds']:
        if kind=='order':goals[kind]=bool(order_calls) and '智能猫砂盆' in answers
        elif kind=='logistics':goals[kind]=any(a['tool']=='query_logistics' and a['status']=='成功' for a in ACTIVE['audit']) and '成都' in answers
        elif kind=='return_advice':goals[kind]=any(r['node']=='retrieve_policy' for r in ACTIVE['retrieval']) and bool(re.search('退货|退款',answers))
        elif kind=='refund':
            goals[kind]=(any(t.ticket_type=='退款' for t in tickets) if '退款' in case['tickets'] else
                         not any(t.ticket_type=='退款' for t in tickets) and bool(re.search('待付款|无需退款|取消',answers)))
        elif kind=='complaint':goals[kind]=any(t.ticket_type=='投诉' for t in tickets)
        elif kind=='knowledge':goals[kind]=any(r['node']=='retrieve_knowledge' and '智能猫砂盆' in r['query'] for r in ACTIVE['retrieval'])
    checks={'selection_count':selections==case['selections'],'expected_tickets':sorted(t.ticket_type for t in tickets)==sorted(case['tickets']),
            'finished':out.get('interrupt') is None,'no_internal_state':not any(x in answers for x in ('task_id','operation_id','WAITING_USER','本次事项处理结果')),
            'reply_presented':bool(answers.strip()) and '本次已取得的结果暂时无法完整呈现' not in answers,
            'owned_expected_objects':all(a['args'].get('order_id') in case['orders'] for a in ACTIVE['audit'] if 'order_id' in (a['args'] or {}))}
    if args.version=='simple':
        checks['one_plan']=len(ACTIVE['planner'])==1
        checks['all_tasks_terminal']=all(t['status'] in simple_loop.TERMINAL for t in tasks.values())
        checks['no_repeat_order']=ACTIVE['duplicate_order_calls']==0
        checks['task_coverage']=all(any(t['kind']==k for t in tasks.values()) for k in case['kinds'])
        if case['id']=='S5':checks['independent_refund']=all(t.get('order_id') is None for t in ACTIVE['steps'][0]['state']['task_plan'].values() if t['kind']=='refund')
    ACTIVE.update(checks=checks,goals=goals,passed=all(checks.values()) and all(goals.values()),elapsed_seconds=time.monotonic()-started)
    return ACTIVE


async def main(args):
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((OUT/'cases.json').read_bytes()).hexdigest()==protocol['cases_sha256']
    cases=json.loads((OUT/'cases.json').read_text(encoding='utf-8'))
    if args.ids:cases=[c for c in cases if c['id'] in args.ids.split(',')]
    folder=OUT/args.run/args.version/f'r{args.rep}';folder.mkdir(parents=True,exist_ok=True)
    settings.multi_intent_enabled=args.version=='simple';settings.multi_intent_simple_enabled=args.version=='simple'
    settings.multi_intent_context_v2_enabled=True;settings.multi_intent_subflows_enabled=True
    runtime_settings.SETTINGS_PATH=Path('E:/Mewhelp/data/runtime_settings.json')
    settings.langfuse_public_key='';settings.model_pool_db_path=str(folder/'pool.sqlite')
    url=make_url(settings.database_url);db=f'caremate_simple_{args.run}_{args.version}_{args.rep}'
    if not db.replace('_','').isalnum():raise ValueError('Invalid isolated database name')
    admin=create_async_engine(url.set(database=None),isolation_level='AUTOCOMMIT')
    async with admin.begin() as conn:await conn.execute(text(f'CREATE DATABASE IF NOT EXISTS {db} CHARACTER SET utf8mb4'))
    await admin.dispose();old=base.engine
    base.engine=create_async_engine(url.set(database=db),pool_pre_ping=True);base.async_session=async_sessionmaker(base.engine,expire_on_commit=False)
    async with base.engine.begin() as conn:await conn.run_sync(base.Base.metadata.create_all)
    await workflow.ensure_tables()
    def existing():
        client=milvus_client.get_client()
        if not client.has_collection('knowledge'):raise RuntimeError('Existing KB required')
        client.load_collection('knowledge');milvus_client._ensured.add('knowledge');return milvus_client.count(client)
    kb_count=await milvus_client.acall(existing)
    save(folder/'manifest.json',{'database':db,'kb_count':kb_count,'context_policy':settings.context_policy,
         'legacy_recall':settings.context_legacy_recall_enabled,'confidence':runtime_settings.get_confidence_settings(),
         'models':[settings.chat_model,settings.embed_model,settings.rerank_model],
         'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'app').rglob('*.py')}})
    instrument()
    async with AsyncExitStack() as stack:
        for case in cases:
            path=folder/(case['id']+'.json')
            if path.exists():continue
            try:result=await run_case(case,args,folder,stack)
            except Exception as exc:
                import traceback
                result={**ACTIVE,'passed':False,'error':type(exc).__name__,'traceback':traceback.format_exc()}
            save(path,result)
            print(json.dumps({'case':case['id'],'version':args.version,'rep':args.rep,'passed':result['passed'],
                'checks':result.get('checks'),'goals':result.get('goals'),'error':result.get('error')},ensure_ascii=False),flush=True)
    await base.engine.dispose();await old.dispose()
    from app.core.model_transport import close_clients
    await close_clients()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--version',choices=['single','simple'],required=True)
    p.add_argument('--run',default='pilot1');p.add_argument('--rep',type=int,default=1);p.add_argument('--ids')
    asyncio.run(main(p.parse_args()))
