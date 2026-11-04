"""Real-model paired workflow evaluation in isolated databases.

python -m scripts.eval_multi_intent prepare
python -m scripts.eval_multi_intent run
python -m scripts.judge_multi_intent run
python -m scripts.report_multi_intent
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('MULTI_EVAL_OUTPUT',str(ROOT / 'data/multi-intent-eval/20261103')))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    temp.replace(path)


def cases():
    groups = {
        'explicit_two': [
            '订单1001商品有质量问题，我要退款，还要投诉商家。',
            '我要退款，订单是1001，还要投诉这家店。',
            '猫砂盆有质量问题，我要退款，也要投诉商家。',
            '订单1001我申请退货退款，并投诉商家售后不处理。',
            '这件商品质量不好，我想退款，还想投诉商家。',
            '我要对1001申请退款；同时请登记投诉工单，反映商家不处理质量问题。'],
        'explicit_three': [
            '查一下2002的物流，再对1001申请退款，还要投诉商家拖延处理质量问题。',
            '我要退款1001的质量问题商品，还要投诉商家，另外2002快递到哪里了？',
            '先告诉我1001订单现在什么状态，我要退款，还要投诉商家。',
            '订单1001多少钱、什么状态？商品有质量问题我要退款，也要投诉商家。',
            '2002物流走到哪了？1001商品有质量问题，我想退款，并登记投诉工单。',
            '我要退款，还要投诉商家。也请查询订单2002的物流位置。'],
        'implicit_two': [
            '1001这东西坏了，把钱退给我，店家不理人我要讨个说法。',
            '猫砂盆质量不行，钱我不想花了，得退回来；这家店也得有人管管。',
            '1001商品质量不好，给我办退钱，再找专员投诉这家店。',
            '这破东西不能用，想退掉把款拿回来，顺便登记商家的服务问题。',
            '订单1001有质量问题，不想留下了，麻烦办理退货退钱，商家推诿也请受理投诉。',
            '我不接受1001的质量，帮我走退款流程，店铺态度的问题也请交给人工跟进。'],
        'continuation': [
            '我要退款，还要投诉商家不处理商品质量问题。',
            '商品质量不好，我申请退款，同时要投诉商家。',
            '我要退款，还要登记投诉工单。',
            '猫砂盆坏了，我要退款，还要投诉商家。',
            '我要对质量问题申请退款，也想投诉商家。',
            '我要退款并投诉商家，另外查一下2002物流。'],
        'controls': [
            '订单1001商品有质量问题，我要退款，不要投诉，也别创建投诉工单。',
            '我要投诉商家服务差，不要退款。',
            '请查询订单1001的状态和实付金额。',
            '请查一下订单2002的物流在哪里。',
            '我要退款，订单1001有质量问题。',
            '我要退款，还要投诉商家；不要再替我查询2002的物流。'],
    }
    result = []
    for group, questions in groups.items():
        for i, query in enumerate(questions):
            required = ['refund_request', 'complaint_response'] if group != 'controls' else [
                ['refund_request'], ['complaint_response'], ['order_facts'], ['logistics_facts'],
                ['refund_request'], ['refund_request', 'complaint_response']][i]
            if group == 'explicit_three':
                required += ['order_facts' if i in (2, 3) else 'logistics_facts']
            if group == 'continuation' and i == 5:
                required += ['logistics_facts']
            result.append({'id': f'{group}-{i+1}', 'group': group, 'query': query,
                'required_tasks': required, 'restart_after_first_interrupt': group == 'continuation' and i in (0, 3),
                'cancel_order': group == 'continuation' and i == 1,
                'cancel_refund_submission': group == 'continuation' and i == 2,
                'cancel_complaint_submission': group == 'continuation' and i == 4,
                'forbidden_ticket_types': ['投诉'] if group == 'controls' and i in (0, 4) else
                                          ['退款'] if group == 'controls' and i == 1 else [],
                'forbid_logistics': group == 'controls' and i == 5})
    return result


async def prepare():
    from app.config import settings
    from app.kb import milvus_client
    if (OUT/'protocol.json').exists():
        raise RuntimeError('Protocol already frozen; do not overwrite experimental inputs.')
    OUT.mkdir(parents=True, exist_ok=True)
    before = ROOT/'data/issues/implementation-20261103/before/app'
    for arm, source in [('before', before), ('after', ROOT/'app')]:
        shutil.copytree(source, OUT/'source'/arm/'app', ignore=shutil.ignore_patterns('__pycache__'))
    rows = await milvus_client.acall(lambda: milvus_client.get_client().query('knowledge',filter='',
        output_fields=['id','question','answer','section_path','content_type','category'], limit=16384))
    rows.sort(key=lambda r:r['id'])
    save(OUT/'knowledge.json', rows)
    save(OUT/'cases.json', cases())
    fields = ['chat_model','chat_base_url','intent_model','embed_model','embed_base_url',
              'rerank_model','rerank_base_url','context_policy','max_agent_steps','recall_top_k','rerank_top_k']
    save(OUT/'protocol.json', {'sample_count':len(cases()),'arms':['before','after'],
        'model_config':{k:getattr(settings,k) for k in fields},
        'knowledge_rows':len(rows),'knowledge_sha256':hashlib.sha256((OUT/'knowledge.json').read_bytes()).hexdigest(),
        'source_hashes':{str(p.relative_to(OUT/'source')):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (OUT/'source').rglob('*.py')},
        'actor':'Same bounded cooperative actor; selects1001; independently confirms/cancels offered writes; no missing-card prompt rescue.',
        'metrics':'External response/SQL receipts; task_plan is diagnostic only; no urgent safety task counted.',
        'limits':['Constructed internal development cases, not blind test.',
                  'Graph/runtime and form APIs run; no browser/SSE-delivery score.',
                  'Business services are common deterministic fixtures; real chat/embedding/rerank API calls.',
                  'Current after uses deterministic review-request path, before may give eligibility advice; not an isolated scheduler ablation.',
                  'Cancel-order control requires cancellation support, not forced order selection.',
                  'Single repetition; paired calls alternate arm order; service failures recorded separately.']})
    print('Frozen',len(cases()),'cases, knowledge',len(rows),flush=True)


async def worker(arm, case_id):
    if arm == 'before' and read(OUT/'protocol.json').get('baseline_origin'):
        raise RuntimeError('This experiment reuses saved V1 trajectories; restore the original baseline artifact, do not resample it.')
    sys.path.insert(0, str(OUT/'source'/arm))
    from app.config import settings
    from app.db import base, models, repository
    from app.graph import runtime, nodes
    from app.core import model_pool
    from app.kb import milvus_client
    from sqlalchemy import select, text
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from langchain_core.tools import tool
    from app.tools import business, registry
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    from app.api.actions import router
    case = next(c for c in read(OUT/'cases.json') if c['id']==case_id)
    for k,v in read(OUT/'protocol.json')['model_config'].items():
        setattr(settings,k,v)
    settings.model_pool_db_path = str(ROOT/'data/model_pool.sqlite3')
    settings.chat_failover_enabled = False
    settings.langfuse_public_key = ''
    settings.context_legacy_recall_enabled = False
    if arm == 'after':
        settings.multi_intent_enabled = True
    directory=OUT/'runs'/arm/case_id
    directory.mkdir(parents=True,exist_ok=True)
    settings.checkpointer_db_path=str(directory/'checkpoints.sqlite')
    server=settings.test_database_url.rsplit('/',1)[0]
    dbname='caremate_multi_eval_'+hashlib.sha256(str(OUT).encode()).hexdigest()[:8]+'_'+arm
    admin=create_async_engine(server,isolation_level='AUTOCOMMIT')
    async with admin.connect() as conn:
        await conn.execute(text(f'CREATE DATABASE IF NOT EXISTS {dbname} CHARACTER SET utf8mb4'))
    await admin.dispose()
    base.engine=create_async_engine(server+'/'+dbname,pool_pre_ping=True)
    base.async_session=async_sessionmaker(base.engine,expire_on_commit=False)
    if arm=='after':
        from app.db import workflow
    async with base.engine.begin() as conn:
        await conn.run_sync(base.Base.metadata.create_all)
    if arm=='after':
        await workflow.ensure_tables()
    fixtures={
      '1001':{'order_id':'1001','status':'已签收','amount':399,'created_at':'2026-11-02 10:00',
              'delivered_at':'2026-11-02 15:00','product':'智能猫砂盆','tracking_no':'SF100100100001'},
      '2002':{'order_id':'2002','status':'已发货','amount':259,'created_at':'2026-11-02 11:00',
              'product':'自动饮水机','tracking_no':'SF200200200002'}}
    business.order_snapshot=lambda oid:dict(fixtures[oid])
    business.list_user_orders=lambda uid:[dict(v) for v in fixtures.values()] if uid=='benchmark' else []
    # Builtin orders imports order_snapshot by value; bind the identical fixture there too.
    registry.scan_builtin()
    import app.tools.builtin.orders as builtin_orders
    builtin_orders.order_snapshot=business.order_snapshot
    @tool
    async def query_logistics(tracking_no:str)->dict:
        """根据物流单号查询配送状态和位置，先query_order取得tracking_no。"""
        return {'tracking_no':tracking_no,'status':'运输中','current_city':'杭州',
                'trace':['杭州分拨中心 已发出']} if tracking_no in {'SF100100100001','SF200200200002'} else {'error':'单号不存在'}
    logistics_spec=registry.spec_from_langchain_tool(query_logistics,source='mcp')
    async def specs():
        return [*registry._BUILTIN.values(),logistics_spec]
    registry.get_all_specs=specs
    async def no_summary(cid):
        pass
    runtime.summarizer.maybe_schedule_summary=no_summary
    client=milvus_client.get_client()
    if not await milvus_client.acall(client.has_collection,'knowledge'):
        raise RuntimeError('Missing knowledge collection')
    milvus_client.ensure_collection=lambda *args,**kwargs:None
    await runtime.init_graph()
    api=FastAPI();api.include_router(router)
    started=time.perf_counter(); steps=[]; actor=[]; seen_forms=set(); cid=None
    result={'case_id':case_id,'arm':arm,'status':'running','case':case,'steps':steps,'actor':actor}
    async def capture(out):
        state=out['state']
        step={'answer':nodes.resolve_answer(state),'interrupt':out.get('interrupt'),
              'suggested_actions':runtime.dedup_actions(state.get('suggested_actions',[])),
              'intent':state.get('intent'),'trace':state.get('trace'),
              'task_plan':state.get('task_plan',{}),
              'messages':[m.model_dump(mode='json') for m in state.get('messages',[])]}
        steps.append(step); save(directory/'progress.json',result)
        return step
    try:
      with model_pool.call_context('multi-eval-'+arm+'-'+case_id,priority='background'):
        async with asyncio.timeout(300):
          out=await runtime.run_turn('benchmark',case['query'],None)
          cid=out['conversation_id'];result['conversation_id']=cid
          for index in range(9):
            step=await capture(out); intr=step['interrupt']
            if index==0 and intr and case['restart_after_first_interrupt']:
                await runtime.close_graph();await runtime.init_graph();actor.append({'restart':True})
            if intr:
                kind=intr['type']
                if kind=='select_order':
                    if case['cancel_order']:
                        if not intr.get('can_cancel'):
                            actor.append({'unsupported_cancel_order':True});break
                        value={'cancelled':True}
                    else:value='1001'
                elif kind=='confirm_ticket':
                    ticket_type=intr.get('preview',{}).get('ticket_type')
                    forbidden=ticket_type in case['forbidden_ticket_types']
                    if forbidden:actor.append({'forbidden_write_preview':ticket_type})
                    value={'confirmed':not (case['cancel_complaint_submission'] or forbidden)}
                elif kind=='task_actions':
                    action=intr['actions'][0]
                    ticket_type=action['preview']['ticket_type']
                    forbidden=ticket_type in case['forbidden_ticket_types']
                    if forbidden:actor.append({'forbidden_write_preview':ticket_type})
                    value={'operation_id':action['operation_id'],'confirmed':not (
                        (case['cancel_refund_submission'] if ticket_type=='退款' else case['cancel_complaint_submission']) or forbidden)}
                else:
                    actor.append({'unsupported_interrupt':kind});break
                actor.append({'kind':kind,'value':value,'interrupt_id':intr['interrupt_id']})
                out=await runtime.resume_turn(cid,value,user_id='benchmark',interrupt_id=intr['interrupt_id'])
                continue
            async with AsyncClient(transport=ASGITransport(app=api),base_url='http://local') as http:
                for action in step['suggested_actions']:
                    key=json.dumps(action,sort_keys=True,ensure_ascii=False)
                    if key in seen_forms:continue
                    seen_forms.add(key);draft=action.get('draft',{})
                    payload={'conversation_id':cid,'user_id':'benchmark','request_id':case_id+':'+str(len(seen_forms))}
                    if action['type']=='refund_form' and 'refund_request' in case['required_tasks'] and not case['cancel_order']:
                        if case['cancel_refund_submission']:
                            actor.append({'cancelled_refund_form':True});continue
                        payload.update(order_id=draft.get('order_id','1001'),reason='质量问题')
                        path='/api/actions/create-refund'
                    elif action['type']=='create_ticket' and 'complaint_response' in case['required_tasks']:
                        if case['cancel_complaint_submission']:
                            actor.append({'cancelled_complaint_form':True});continue
                        payload.update(description=case['query'],ticket_type='投诉');path='/api/actions/create-ticket'
                    else:continue
                    response=await http.post(path,json=payload)
                    actor.append({'path':path,'payload':payload,'status':response.status_code,'response':response.json()})
            break
          else:result['actor_limit']=True
      result['status']='ok'
    except Exception as exc:
      result.update(status='error',error_type=type(exc).__name__,error=str(exc))
    finally:
      if cid:
        async with base.async_session() as session:
          tickets=(await session.execute(select(models.Ticket).where(models.Ticket.conversation_id==cid))).scalars()
          result['tickets']=[{'ticket_no':t.ticket_no,'ticket_type':t.ticket_type,'description':t.description} for t in tickets]
          logs=(await session.execute(select(models.ToolAuditLog).where(models.ToolAuditLog.conversation_id==cid))).scalars()
          result['tool_audit']=[{'name':t.tool_name,'arguments':t.arguments,'status':t.status,
                                 'result':t.result_summary} for t in logs]
        result['dialogue']=[{'id':r.id,'role':r.role,'content':r.content} for r in await repository.list_dialog_messages(cid)]
      result['elapsed_seconds']=time.perf_counter()-started
      result['fixtures']=fixtures
      save(directory/'result.json',result)
      await runtime.close_graph()
      from app.core.model_transport import close_clients
      await close_clients();await base.engine.dispose()
    print(arm,case_id,result['status'],round(result['elapsed_seconds'],1),flush=True)


async def run(limit=None):
    selected=read(OUT/'cases.json')[:limit]
    if read(OUT/'protocol.json').get('baseline_origin'):
        missing=[c['id'] for c in selected if not (OUT/'runs/before'/c['id']/'result.json').exists()]
        if missing:raise RuntimeError('Restore saved V1 baseline trajectories first: '+','.join(missing))
    sem=asyncio.Semaphore(2)
    arm_locks={arm:asyncio.Lock() for arm in ('before','after')}
    async def pair(index,case):
      async with sem:
        for arm in (['before','after'] if index%2==0 else ['after','before']):
          directory=OUT/'runs'/arm/case['id'];directory.mkdir(parents=True,exist_ok=True)
          if (directory/'result.json').exists():continue
          async with arm_locks[arm]:
           with (directory/'worker.log').open('w',encoding='utf-8') as log:
            proc=await asyncio.create_subprocess_exec(sys.executable,str(Path(__file__).resolve()),'worker',
                '--arm',arm,'--case',case['id'],cwd=str(ROOT),stdout=log,stderr=log,
                env={**os.environ,'PYTHONPATH':str(OUT/'source'/arm),'PYTHONIOENCODING':'utf-8'})
            rc=await proc.wait()
          print(arm,case['id'],'worker_exit',rc,flush=True)
    await asyncio.gather(*(pair(i,c) for i,c in enumerate(selected)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','worker'])
    p.add_argument('--arm');p.add_argument('--case');p.add_argument('--limit',type=int)
    args=p.parse_args()
    asyncio.run(prepare() if args.stage=='prepare' else worker(args.arm,args.case) if args.stage=='worker' else run(args.limit))
