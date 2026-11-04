"""P1 pilot: reuse the existing business Agent and execution engine, no turn entry/log.

Only completed task boundaries are durable in this pilot. A crash inside a read
may repeat that read; write tools and user-interrupt flows are out of scope.
"""
import json
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

from app.core import order_context, task_planner
from app.graph import nodes
from app.graph.routing import should_continue
from app.graph.state import ConversationState
from app.tools import business


def receipts(state):
    return [m for m in state.get('messages', []) if isinstance(m, ToolMessage)
            and m.status != 'error']


def tracking_numbers(state):
    values = set()
    for m in receipts(state):
        if m.name != 'query_order':
            continue
        try:
            data = json.loads(m.content)
        except (ValueError, TypeError):
            continue
        if (isinstance(data, dict) and data.get('tracking_no') and data.get('status') != '待付款'
                and data.get('order_id') == state['task_scope']['order_id']):
            values.add(data['tracking_no'])
    return values


def denied_call(state, tc):
    scope = state['task_scope']
    name, args = tc['name'], tc.get('args') or {}
    if name not in scope['allowed_tools']:
        return '当前只读子任务不允许调用此工具。'
    if not business.owns_order(state['user_id'], scope['order_id']):
        return '订单归属校验失败。'
    if name == 'query_order' and str(args.get('order_id', '')) != scope['order_id']:
        return '订单必须与当前子任务已核验对象一致。'
    if name == 'query_logistics' and args.get('tracking_no') not in tracking_numbers(state):
        return '必须先查询当前订单，并使用本子任务订单回执中的物流单号。'
    return None


def build_business_subflow():
    b = StateGraph(ConversationState)
    b.add_node('agent', nodes.main_agent)
    b.add_node('tools', nodes.agent_tools)
    b.add_edge(START, 'agent')
    b.add_conditional_edges('agent', should_continue, {'continue': 'tools', 'stop': END})
    b.add_edge('tools', 'agent')
    return b.compile(checkpointer=False)


class PilotState(TypedDict, total=False):
    user_id: str
    conversation_id: int
    current_message_id: int
    query: str
    business_context: dict
    tasks: list
    results: list
    active: int


async def plan(state):
    items = await task_planner.extract(state['query'])
    context = state.get('business_context') or {}
    if context.get('user_id') != state['user_id'] or context.get('conversation_id') != state['conversation_id']:
        context = {}
    items = order_context.bind_tasks(state['query'], items, context, state['user_id'])
    tasks = []
    for i, item in enumerate(items):
        item = dict(item)
        item.update(task_id=f"{state['conversation_id']}:{state['current_message_id']}:{i}", status='PENDING')
        if item['kind'] not in {'order', 'logistics'}:
            item.update(status='FAILED', reason='P1尚未接入此类业务')
        elif not item.get('order_id') or not business.owns_order(state['user_id'], item['order_id']):
            item.update(status='WAITING_USER', reason='需要明确本人订单')
        tasks.append(item)
    return {'tasks': tasks, 'results': [], 'active': -1}


def schedule(state):
    candidates = [(i, t) for i, t in enumerate(state['tasks']) if t['status'] == 'PENDING']
    # Reads preserve task order within each priority; order facts precede logistics.
    chosen = min(candidates, key=lambda pair: (pair[1]['kind'] != 'order', pair[0]), default=None)
    return {'active': chosen[0] if chosen else -1}


async def execute_read(state, *, child=None):
    child = child or build_business_subflow()
    tasks = [dict(t) for t in state['tasks']]
    task = tasks[state['active']]
    oid = task['order_id']
    allowed = ['query_order'] + (['query_logistics'] if task['kind'] == 'logistics' else [])
    query = (f'仅查询订单{oid}的信息。' if task['kind'] == 'order' else f'仅查询订单{oid}的当前物流。')
    query += '必须实际调用当前任务工具后作答；不要办理或回答其他事项。'
    local = {'user_id':state['user_id'], 'conversation_id':state['conversation_id'],
             'current_message_id':state['current_message_id'],
             'messages':[HumanMessage(query)], 'resolved_query':query,
             'intent':'订单' if task['kind'] == 'order' else '物流', 'route':'business',
             'steps':0, 'tokens_used':0, 'trace':None, 'suggested_actions':[],
             'task_context':task.get('task_context',{}),
             'task_scope':{'order_id':oid,'allowed_tools':allowed,'task_id':task['task_id']}}
    try:
        out = await child.ainvoke(local)
        got = receipts(out)
        required = set(allowed)
        unpaid, verified_order = False, False
        for receipt in got:
            if receipt.name == 'query_order':
                try:
                    value = json.loads(receipt.content)
                    if isinstance(value,dict) and value.get('order_id') == oid and not value.get('error'):
                        verified_order = True
                        unpaid = value.get('status') == '待付款'
                except (ValueError, TypeError):
                    pass
        if unpaid:
            required = {'query_order'}
        success = verified_order and required <= {m.name for m in got} and not out['messages'][-1].tool_calls
        task['status'] = 'DONE' if success else 'FAILED'
        result = {'task_id':task['task_id'],'kind':task['kind'],'order_id':oid,
                  'status':task['status'], 'answer':nodes.resolve_answer(out) if success else '本次查询未能完成，请稍后重试。',
                  'receipts':[{'name':m.name,'content':m.content} for m in got],
                  'steps':out.get('steps'), 'tokens_used':out.get('tokens_used')}
        if success and unpaid and task['kind'] == 'logistics':
            result.update(outcome='UNPAID',answer=f'订单{oid}目前待付款，尚无可供查询的物流进度。')
    except Exception as exc:
        task['status'] = 'FAILED'
        result = {'task_id':task['task_id'],'status':'FAILED','error_type':type(exc).__name__}
    return {'tasks':tasks, 'results':[*state.get('results',[]),result]}


def build_pilot(checkpointer=None, *, interrupt_after=None):
    child = build_business_subflow()

    async def execute(state):
        return await execute_read(state, child=child)

    b = StateGraph(PilotState)
    b.add_node('plan', plan)
    b.add_node('schedule', schedule)
    b.add_node('execute', execute)
    b.add_edge(START, 'plan')
    b.add_edge('plan', 'schedule')
    b.add_conditional_edges('schedule', lambda s: 'execute' if s['active'] >= 0 else 'end',
                            {'execute':'execute','end':END})
    b.add_edge('execute', 'schedule')
    return b.compile(checkpointer=checkpointer, interrupt_after=interrupt_after)
