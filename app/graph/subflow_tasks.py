"""P2 orchestration: defer missing inputs, reuse business flows, confirm writes.

Selection and confirmation are pure interrupt nodes. Task preparation commits
before waiting; transactional workflow receipts handle commit/checkpoint gaps.
"""
from langgraph.graph import END
from langgraph.types import interrupt
from sqlalchemy.exc import SQLAlchemyError

from app.core import order_context
from app.core.task_reply import compose
import hashlib
import json
from app.db import workflow
from app.graph import nodes, tasks as legacy
from app.graph.read_subflows import execute_read
from app.tools import business

ORDER_KINDS = {'order', 'logistics', 'refund'}
PRIORITY = {'order':0, 'logistics':1, 'refund':2, 'complaint':3, 'knowledge':4, 'other':5}


def copied(state):
    return {k:dict(v) for k,v in state['task_plan'].items()}


async def schedule(state):
    items=copied(state)
    if state.get('task_plan_version')==2:
        from app.graph.task_dependencies import refresh
        refresh(items,state['user_id'])
    await workflow.save_plan(state['conversation_id'], state['user_id'], state['plan_id'],
                             state['current_message_id'], items)
    pending = [k for k,t in items.items() if t['status']=='PENDING']
    return {'active_task':min(pending,key=lambda k:PRIORITY.get(state['task_plan'][k]['kind'],9)) if pending else '',
            'task_plan':items,'task_events':[], 'subflow_presentation':True}


def route(state):
    if state.get('active_task'):
        return 'prepare'
    return 'present'


def wait_route(state):
    waits = [t for t in state['task_plan'].values() if t['status']=='WAITING_USER']
    if any(t.get('step')=='select_order' for t in waits):
        return 'select'
    return 'wait' if waits else 'finish'


async def prepare(state):
    items=copied(state); task=items[state['active_task']]
    if task['kind'] in ORDER_KINDS and not business.owns_order(state['user_id'],task.get('order_id') or ''):
        task.update(status='WAITING_USER',step='select_order',blocking=True)
    else:
        task.update(status='RUNNING',step='execute',blocking=False)
    if state.get('task_plan_version')==2:
        from app.core.task_context import packet
        task['task_context']=packet(state,task)
        if task['kind']=='knowledge' and task.get('context_order_id'):
            try:
                facts=business.order_snapshot(task['context_order_id'])
                if not facts.get('product'):raise ValueError('订单商品资料缺失')
                task['task_context']['order_facts']=facts
            except Exception as exc:
                task.update(status='FAILED',outcome='upstream_order:'+type(exc).__name__)
    return {'task_plan':items,'task_events':[]}


def dispatch(state):
    t=state['task_plan'][state['active_task']]
    if t['status']!='RUNNING':return 'schedule'
    return 'read' if t['kind'] in {'order','logistics'} else t['kind']


async def select_order(state):
    items=copied(state)
    key=next(k for k,t in items.items() if t['status']=='WAITING_USER' and t.get('step')=='select_order')
    task=items[key]; uid=state['user_id']
    orders=business.list_user_orders(uid)
    if not orders:
        if state.get('task_plan_version')==2:
            from app.graph.task_dependencies import cancel_group
            cancel_group(items,key)
        else:task.update(status='CANCELLED',step='no_orders')
        return {'task_plan':items,'task_events':[]}
    purposes={'refund':'申请退款审核','order':'查询订单商品','logistics':'查询物流','knowledge':'查询商品说明'}
    selection_prompt={'refund':'请选择要申请退款审核的订单：','order':'请选择要查询的订单：',
                       'logistics':'请选择要查询物流的订单：','knowledge':'请选择要查询商品说明的订单：'}[task['kind']]
    if state.get('task_plan_version')==2:
        group=task.get('subject_ref')
        labels=list(dict.fromkeys(purposes[t['kind']] for k,t in items.items()
            if (k==key or (group and t.get('subject_ref')==group)) and t['kind'] in purposes
            and t['status'] in {'WAITING_USER','WAITING_DEPENDENCY'}))
        selection_prompt='请确认用于'+ '、'.join(labels)+'的订单（同一笔订单只需选择一次）：'
    value=interrupt({'type':'select_order','orders':orders,'can_cancel':True,
        'plan_id':state['plan_id'],'task_id':task['task_id'],'task_kind':task['kind'],
        'selection_prompt':selection_prompt})
    if value=={'cancelled':True}:
        if state.get('task_plan_version')==2:
            from app.graph.task_dependencies import cancel_group
            cancel_group(items,key)
        else:task.update(status='CANCELLED',step='cancelled')
        return {'task_plan':items,'task_events':[]}
    if not isinstance(value,str) or not business.owns_order(uid,value):
        raise ValueError('无效订单选择，未执行业务')
    if task['kind']=='knowledge':task['context_order_id']=value
    task.update(order_id=value,binding={'status':'resolved','method':'user_selection','order_ids':[value]},
                status='PENDING',step='selected',blocking=False)
    context=order_context.selected(state,items,key,value)
    if state.get('task_plan_version')==2:
        from app.graph.task_dependencies import refresh
        refresh(items,uid)
    # A selection is shared only within the binder's explicit referent group.
    for other in items.values():
        if (other['status']=='WAITING_USER' and other.get('step')=='select_order'
                and business.owns_order(uid,other.get('order_id') or '')):
            other.update(status='PENDING',step='selected',blocking=False)
    return {'task_plan':items,'business_context':context,'task_events':[]}


async def read(state):
    key=state['active_task']; task=state['task_plan'][key]
    out=await execute_read({**state,'tasks':[task],'results':[],'active':0})
    items=copied(state); items[key]=out['tasks'][0]
    result=out['results'][0]
    return {'task_plan':items,'subflow_results':{**state.get('subflow_results',{}),key:result},'task_events':[]}


async def quiet(fn, state):
    with workflow.internal_events():
        out=await fn(state)
    key=state['active_task']
    texts=[e['text'] for e in out.get('task_events',[])]
    out['task_events']=[]
    if texts and state['task_plan'][key]['kind']=='knowledge':
        out['subflow_results']={**state.get('subflow_results',{}),key:{'answer':'\n'.join(texts)}}
    return out


async def refund(state):
    task=state['task_plan'][state['active_task']]
    # Already bound: reuse original deterministic ownership/order step, no reselect.
    verified=await nodes.fetch_order({**state,'order_id':task['order_id']})
    scoped={**state,**verified,'task_order':task['order_id'],'task_order_cancelled':False,
            'task_origin':task['source']}
    return await quiet(legacy.refund,scoped)


async def complaint(state):
    task=state['task_plan'][state['active_task']]
    return await quiet(legacy.complaint,{**state,'task_origin':task['source']})


async def knowledge(state):
    task=state['task_plan'][state['active_task']]
    return await quiet(legacy.knowledge,{**state,'subflow_presentation':True,'task_context':task.get('task_context',{})})


async def unsupported(state):
    return await quiet(legacy.unsupported,state)


async def wait_actions(state):
    actions=[{'task_id':t['task_id'],'operation_id':t['operation_id'],'preview':t['args']}
        for t in state['task_plan'].values() if t['status']=='WAITING_USER' and t.get('step')=='confirm_submission']
    if not actions:raise ValueError('没有可确认的任务')
    value=interrupt({'type':'task_actions','plan_id':state['plan_id'],'actions':actions})
    if (not isinstance(value,dict) or set(value)!={'operation_id','confirmed'}
        or type(value['confirmed']) is not bool or not any(a['operation_id']==value['operation_id'] for a in actions)):
        raise ValueError('确认参数无效或操作已处理')
    return {'task_decisions':value,'task_events':[]}


async def commit(state):
    try:
        with workflow.internal_events():
            return await _commit(state)
    except (SQLAlchemyError, ConnectionError, TimeoutError, OSError):
        items=copied(state)
        task=next(t for t in items.values() if t.get('operation_id')==state['task_decisions']['operation_id'])
        task['retry_submission']=True
        return {'task_plan':items,'task_decisions':{},'task_events':[]}


async def _commit(state):
    result=await legacy.commit_action(state)
    result['task_events']=[]
    return result


async def present(state):
    text,citations,seen=compose(state)
    if not text:return {'subflow_presented':seen,'subflow_reply_id':'','task_events':[]}
    digest=hashlib.sha256(json.dumps(seen,sort_keys=True).encode()).hexdigest()[:20]
    key=state['plan_id']+':reply:'+digest
    ev=await workflow.event(state['conversation_id'],state['user_id'],key,text,citations=citations)
    return {'subflow_presented':seen,'subflow_reply_id':key,'task_events':[ev]}


def install(b):
    for name,fn in {'sf_schedule':schedule,'sf_prepare':prepare,'sf_select':select_order,
        'sf_read':read,'sf_refund':refund,'sf_complaint':complaint,'sf_wait':wait_actions,
        'sf_commit':commit,'sf_present':present,'sf_knowledge':knowledge,'sf_unsupported':unsupported,
        'sf_finish':legacy.finish}.items():b.add_node(name,fn)
    b.add_conditional_edges('sf_schedule',route,{'prepare':'sf_prepare','present':'sf_present'})
    b.add_conditional_edges('sf_present',wait_route,{'select':'sf_select','wait':'sf_wait','finish':'sf_finish'})
    b.add_conditional_edges('sf_prepare',dispatch,{'schedule':'sf_schedule','read':'sf_read','refund':'sf_refund',
        'complaint':'sf_complaint','knowledge':'sf_knowledge','other':'sf_unsupported'})
    for name in ['sf_select','sf_read','sf_refund','sf_complaint','sf_commit','sf_knowledge','sf_unsupported']:b.add_edge(name,'sf_schedule')
    b.add_edge('sf_wait','sf_commit'); b.add_edge('sf_finish',END)
