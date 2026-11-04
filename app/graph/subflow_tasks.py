"""P2 orchestration: defer missing inputs, reuse business flows, confirm writes.

Selection and confirmation are pure interrupt nodes. Task preparation commits
before waiting; transactional workflow receipts handle commit/checkpoint gaps.
"""
from langgraph.graph import END
from langgraph.types import interrupt
from sqlalchemy.exc import SQLAlchemyError

from app.core import order_context
from app.db import workflow
from app.graph import nodes, tasks as legacy
from app.graph.read_subflows import execute_read
from app.tools import business

ORDER_KINDS = {'order', 'logistics', 'refund'}
PRIORITY = {'order':0, 'logistics':1, 'refund':2, 'complaint':3, 'knowledge':4, 'other':5}


def copied(state):
    return {k:dict(v) for k,v in state['task_plan'].items()}


async def schedule(state):
    await workflow.save_plan(state['conversation_id'], state['user_id'], state['plan_id'],
                             state['current_message_id'], state['task_plan'])
    pending = [k for k,t in state['task_plan'].items() if t['status']=='PENDING']
    return {'active_task':min(pending,key=lambda k:PRIORITY.get(state['task_plan'][k]['kind'],9)) if pending else '',
            'task_events':[]}


def route(state):
    if state.get('active_task'):
        return 'prepare'
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
    return {'task_plan':items,'task_events':[]}


def dispatch(state):
    t=state['task_plan'][state['active_task']]
    if t['status']=='WAITING_USER':return 'schedule'
    return 'read' if t['kind'] in {'order','logistics'} else t['kind']


async def select_order(state):
    items=copied(state)
    key=next(k for k,t in items.items() if t['status']=='WAITING_USER' and t.get('step')=='select_order')
    task=items[key]; uid=state['user_id']
    orders=business.list_user_orders(uid)
    if not orders:
        task.update(status='CANCELLED',step='no_orders')
        ev=await workflow.event(state['conversation_id'],uid,task['task_id']+':selection-result',
            '当前没有可选订单，本项未办理；其他事项会继续处理。',task_id=task['task_id'])
        return {'task_plan':items,'task_events':[ev]}
    value=interrupt({'type':'select_order','orders':orders,'can_cancel':True,
        'plan_id':state['plan_id'],'task_id':task['task_id'],'task_kind':task['kind'],
        'selection_prompt':f"请选择这项请求对应的订单：{task.get('query','')}"})
    if value=={'cancelled':True}:
        task.update(status='CANCELLED',step='cancelled')
        ev=await workflow.event(state['conversation_id'],uid,task['task_id']+':selection-result',
            '已取消本项选单，其他事项会继续处理。',task_id=task['task_id'])
        return {'task_plan':items,'task_events':[ev]}
    if not isinstance(value,str) or not business.owns_order(uid,value):
        raise ValueError('无效订单选择，未执行业务')
    task.update(order_id=value,binding={'status':'resolved','method':'user_selection','order_ids':[value]},
                status='PENDING',step='selected',blocking=False)
    context=order_context.selected(state,items,key,value)
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
    ev=await workflow.event(state['conversation_id'],state['user_id'],task['task_id']+':result',
        result.get('answer') or '本次查询暂时无法完成。',task_id=task['task_id'])
    return {'task_plan':items,'subflow_results':{**state.get('subflow_results',{}),key:result},'task_events':[ev]}


async def refund(state):
    task=state['task_plan'][state['active_task']]
    # Already bound: reuse original deterministic ownership/order step, no reselect.
    verified=await nodes.fetch_order({**state,'order_id':task['order_id']})
    scoped={**state,**verified,'task_order':task['order_id'],'task_order_cancelled':False,
            'task_origin':task['source']}
    return await legacy.refund(scoped)


async def complaint(state):
    task=state['task_plan'][state['active_task']]
    return await legacy.complaint({**state,'task_origin':task['source']})


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
        return await legacy.commit_action(state)
    except (SQLAlchemyError, ConnectionError, TimeoutError, OSError):
        # The SQL commit may have succeeded before a response/event failure.
        # Keep the SAME operation and immutable preview, return to a new card.
        # workflow.submit retrieves the original receipt on the next attempt.
        decision=state['task_decisions']
        task=next(t for t in state['task_plan'].values() if t.get('operation_id')==decision['operation_id'])
        events=[]
        try:
            events=[await workflow.event(state['conversation_id'],state['user_id'],task['operation_id']+':retry',
                '提交结果暂时无法确认，请重试原操作；如已提交成功，将返回原工单，不会重复创建。',task_id=task['task_id'])]
        except (SQLAlchemyError, ConnectionError, TimeoutError, OSError):
            pass  # SQLite still retains the waiting task even if MySQL is down.
        return {'task_decisions':{},'task_events':events}


def install(b):
    for name,fn in {'sf_schedule':schedule,'sf_prepare':prepare,'sf_select':select_order,
        'sf_read':read,'sf_refund':refund,'sf_complaint':complaint,'sf_wait':wait_actions,
        'sf_commit':commit,'sf_finish':legacy.finish}.items():b.add_node(name,fn)
    b.add_conditional_edges('sf_schedule',route,{'prepare':'sf_prepare','select':'sf_select','wait':'sf_wait','finish':'sf_finish'})
    b.add_conditional_edges('sf_prepare',dispatch,{'schedule':'sf_schedule','read':'sf_read','refund':'sf_refund',
        'complaint':'sf_complaint','knowledge':'task_knowledge','other':'task_unsupported'})
    for name in ['sf_select','sf_read','sf_refund','sf_complaint','sf_commit']:b.add_edge(name,'sf_schedule')
    b.add_edge('sf_wait','sf_commit'); b.add_edge('sf_finish',END)
