"""Task bookkeeping only. Business work stays in the existing graph nodes.

No model answer generation, tool invocation, policy retrieval or ticket creation
belongs here. The form wait only bridges the existing form back to the queue.
"""
import copy
import json

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import interrupt

from app.core import intent, task_context
from app.db import workflow
from app.graph import nodes
from app.graph.routing import INTENT_TO_ROUTE
from app.tools import business


def enabled(state):
    return state.get('task_plan_version') == 4 and bool(state.get('plan_id'))


def current(state):
    return state.get('task_plan', {}).get(state.get('active_task'), {}) if enabled(state) else {}


def priority(t):
    return {'订单':0, '物流':1, '商品咨询':2, '售后':2,
            '退款退货':3 if t['action']=='request' else 2,
            '人工':4, '投诉':4}.get(t['intent'],5)


async def classify(state):
    tasks=copy.deepcopy(state.get('task_plan',{}))
    try:
        if state.get('memory_error'):raise ValueError(state['memory_error'])
        context=task_context.build(state)
        proposals=await intent.classify_tasks(state['task_origin'],state['resolved_query'],
            nodes._history_text(state,stage='intent'),context['candidates'])
        candidates={x['order_id'] for x in context['candidates']}
        for i,item in enumerate(proposals):
            t=intent.TaskIntent.model_validate(item).model_dump()
            if t['source'] not in state['task_origin']:raise ValueError('任务不来自当前原话')
            oid=t.get('order_id')
            if oid and (oid not in candidates or not business.owns_order(state['user_id'],oid)):
                raise ValueError('模型给出了未经来源校验的订单')
            explicit={x for x in candidates if x in t['source']}
            if oid and len(explicit)==1 and oid not in explicit:
                raise ValueError('模型绑定与本轮明确订单冲突')
            key=str(i)
            tasks[key]={**t,'task_id':state['plan_id']+':'+key,'status':'pending','result':{},
                        'kind':t['intent'],'plan_version':4,'binding':{'method':'shared_agent','status':'resolved' if t.get('order_id') else 'missing'}}
        if not proposals:raise ValueError('没有识别出当前诉求')
        for ref in {t.get('order_ref') for t in tasks.values()}-{None,''}:
            if len({t['order_id'] for t in tasks.values() if t.get('order_ref')==ref and t.get('order_id')})>1:
                raise ValueError('不同订单不能共用一个引用')
        return {'task_plan':tasks,'planning_proposals':proposals,'planning_context':context,
                'queue_orders':{},'queue_answer':'','active_task':'','task_events':[]}
    except Exception as exc:
        return {'task_planning_error':True,'intent':'其他','route':'fallback_script',
                'trace':{'intent_status':'unavailable','intent_error':type(exc).__name__}}


async def take(state):
    tasks=copy.deepcopy(state['task_plan'])
    pending=[k for k,t in tasks.items() if t.get('status')=='pending']
    if not pending:return {'active_task':'','task_events':[]}
    key=min(pending,key=lambda k:priority(tasks[k]));t=tasks[key]
    shared=state.get('queue_orders',{}).get(t.get('order_ref'),{})
    oid=t.get('order_id') or shared.get('order_id')
    if oid and not business.owns_order(state['user_id'],oid):oid=None
    t.update(status='running',order_id=oid)
    order=shared.get('data',{}) if shared.get('order_id')==oid else {}
    query=t['query']
    if oid and oid not in query:query=f'订单{oid}：{query}'
    if t['intent']=='商品咨询' and order.get('product') and order['product'] not in query:
        query=order['product']+' '+query
    await workflow.save_plan(state['conversation_id'],state['user_id'],state['plan_id'],state['current_message_id'],tasks)
    return {'task_plan':tasks,'active_task':key,'intent':t['intent'],'route':INTENT_TO_ROUTE[t['intent']],
            'resolved_query':query,'order_id':oid or '', 'order_data':order,
            'queue_start':len(state.get('messages',[])), 'steps':0, 'answer':'',
            'suggested_actions':[], 'evidence':'','citations':[],'evidence_strong':False,
            'fallback_source':'','retrieved_snapshot':[],'fresh_business_evidence':{},'task_events':[],
            'trace':{'queue_current':t['task_id']}}


def instruction(state):
    t=current(state)
    if not t:return ''
    return ('本轮请求已拆为顺序任务。下面query是当前唯一待完成的任务；历史中最后一条用户原话是整轮背景，'
            '不能把原话中的其他诉求一起查询或回答。其他诉求由队列后续执行，不提前办理、不重复回答已完成项。'
            '自主选择当前任务需要的工具；当前轮已有成功工具回执可以复用。'
            '沿用正常Markdown客服回复，不输出内部任务状态，不为下一事项追问或每项重复客套。'
            'action=consult只咨询，不能提交或准备提交操作；选定订单不等于授权办理。'
            '如需用户确认，沿用现有工具和表单；已明确的工单描述无需重复口头确认，调用工具会展示原确认卡片。\n'
            'order_id非空表示当前对象已在服务端确认，不再向用户询问订单。submitted_forms是已提交回执，不再邀请重复提交。\n'
            +json.dumps({'query':state.get('resolved_query') or t['query'],'intent':t['intent'],'order_id':state.get('order_id'),
                        'action':t['action'],'submitted_forms':[x['result']['form_receipt']
                            for x in state.get('task_plan',{}).values() if x.get('result',{}).get('form_receipt')],
                        'cancelled_actions':[{'intent':x['intent'],'order_id':x.get('order_id')}
                            for x in state.get('task_plan',{}).values() if x.get('status')=='cancelled']},ensure_ascii=False))


def cached_read(state,call,spec):
    """Reuse successful receipts in this user turn, without choosing any tool."""
    if not current(state) or spec is None or spec.permission!='read':return None
    calls={}
    for msg in state.get('messages',[]):
        if msg.type=='human':calls={}
        if isinstance(msg,AIMessage):
            calls.update({c['id']:c for c in msg.tool_calls or []})
        if isinstance(msg,ToolMessage) and msg.status!='error':
            old=calls.get(msg.tool_call_id,{})
            if old.get('name')==call['name'] and old.get('args')==call.get('args'):
                return ToolMessage(content=msg.content,name=call['name'],tool_call_id=call['id'])
    return None


async def complete(state):
    tasks=copy.deepcopy(state['task_plan']);t=tasks[state['active_task']]
    answer=nodes.resolve_answer(state)
    recent=state.get('messages',[])[state['queue_start']:]
    receipts=[m for m in recent if isinstance(m,ToolMessage)]
    t.update(status='done',result={'answer':answer,'tool_receipts':[m.model_dump(mode='json') for m in receipts]})
    if state.get('fallback_source') or (receipts and all(m.status=='error' for m in receipts)):
        t['status']='failed'
    if t['intent']=='人工' and t['action']=='request' and not any(
            m.name=='create_ticket' and m.status!='error' for m in receipts):
        t['status']='failed'
        t['result']['completion_basis']='no_confirmed_write_receipt'
    shared=copy.deepcopy(state.get('queue_orders',{}));oid=state.get('order_id') or t.get('order_id')
    order=state.get('order_data',{})
    for m in receipts:
        if m.name=='query_order' and m.status!='error':
            try:data=json.loads(m.content)
            except (ValueError,TypeError):continue
            if isinstance(data,dict) and business.owns_order(state['user_id'],str(data.get('order_id',''))):
                if not oid or str(data['order_id'])==oid:oid=str(data['order_id']);order=data
    if oid and business.owns_order(state['user_id'],oid):
        t.update(order_id=oid,binding={'method':'shared_agent','status':'resolved'})
        if t.get('order_ref'):shared[t['order_ref']]={'order_id':oid,'data':order}
    forms=[a for a in state.get('suggested_actions',[]) if a['type'] in {'refund_form','create_ticket'}]
    if forms:
        t.update(status='blocked',form=forms[0],operation_id=t['task_id']+':form')
    await workflow.save_plan(state['conversation_id'],state['user_id'],state['plan_id'],state['current_message_id'],tasks)
    return {'task_plan':tasks,'queue_orders':shared,'queue_answer':'\n\n'.join(filter(None,[state.get('queue_answer'),answer]))}


def after_complete(state):
    return 'form' if current(state).get('status')=='blocked' else 'next'


async def wait_form(state):
    """The original form creates the ticket. Resume only verifies its receipt."""
    t=current(state)
    value=interrupt({'type':'task_form','form':t['form'],'request_id':t['operation_id']})
    receipt=await workflow.receipt(state['conversation_id'],state['user_id'],
        f"c{state['conversation_id']}:form:{t['operation_id']}")
    if value.get('confirmed') and not receipt:raise ValueError('表单还没有提交回执')
    tasks=copy.deepcopy(state['task_plan']);task=tasks[state['active_task']]
    task['status']='done' if receipt and receipt.get('ticket_no') else 'cancelled'
    task['result']['form_receipt']=receipt
    await workflow.save_plan(state['conversation_id'],state['user_id'],state['plan_id'],state['current_message_id'],tasks)
    return {'task_plan':tasks,'suggested_actions':[]}
