"""Bounded serial workflow for explicit safety/refund/complaint requests."""
import re
import json

from langgraph.types import interrupt

from app.db import workflow
from app.graph import nodes
from app.tools import business
from app.core import task_planner
from app.core import workflow_mode
from app.core import order_context

SAFETY_NOTICE = "请立即停止使用该设备，远离设备和潮湿区域，不要触碰疑似带电部位或自行拆修。请在确保安全的情况下联系专业人员处理；如有人员受伤或持续危险，请及时联系当地应急服务。"


def explicit_tasks(text):
    # Conservative MVP: quoted/hypothetical/negated clauses do not create actions.
    clauses = re.split(r"[，,。；;！？!?]", text)
    live = [c for c in clauses if not re.search(r"如果|假设|比如|例如|听说|他说|她说|[\"“”]|没有|没发生|不是|并未", c)]
    safety = any(re.search(r"漏电|触电|冒烟|起火", c) for c in live)
    refund = any(re.search(r"要退款|想退款|申请退款|要退货|申请退货", c)
                 and not re.search(r"不要|不想|不用|不需要", c) for c in live)
    complaint = any("投诉" in c and not re.search(r"不要|不想|不用|不需要|怎么|如何|能否", c) for c in live)
    return safety, refund, complaint


async def precheck(state):
    if not state.get('multi_intent_enabled', workflow_mode.enabled()):
        return {"task_plan": {},'plan_id':''}
    raw = nodes._user_text(state)
    await workflow.supersede(state["conversation_id"], state["user_id"], state["current_message_id"])
    safety, _, _ = explicit_tasks(raw)
    prefix = f"c{state['conversation_id']}:m{state['current_message_id']}"
    kinds = ["safety"] if safety else []
    tasks = {kind: {"task_id": prefix + ":" + kind, "kind": kind, "status": "PENDING"} for kind in kinds}
    return {"task_plan": tasks, "plan_id": prefix, "task_origin": raw,
            "task_events": [], "task_decisions": {}, "task_dispatches": 0}


async def plan(state):
    tasks = {k:dict(v) for k,v in state['task_plan'].items()}
    try:
        items = await task_planner.extract(state['task_origin'])
        items = order_context.bind_tasks(state['task_origin'], items, state.get('business_context') or {}, state['user_id'])
        counts={}
        for item in items:
            kind=item['kind'];counts[kind]=counts.get(kind,0)+1
            key=kind if counts[kind]==1 else f'{kind}:{counts[kind]}'
            tasks[key]={**item,'task_id':state['plan_id']+':'+key,'status':'PENDING'}
        if not tasks or (len(tasks)==1 and next(iter(tasks.values()))['kind'] in ('knowledge','other')
                         and not next(iter(tasks.values())).get('context_order_id')):
            return {'task_plan':{},'plan_id':'','task_events':[],'task_planning_error':False}
        return {'task_plan':tasks,'task_events':[], 'task_planning_error':False}
    except Exception:
        text='暂时无法可靠拆解这条消息中的全部事项，请分别说明需要办理的业务。本轮尚未发起其他业务操作。'
        ev=await workflow.event(state['conversation_id'],state['user_id'],state['plan_id']+':planning-error',text)
        return {'task_events':[ev],'answer':text,'task_planning_error':True}


def plan_route(state):
    if state.get('task_planning_error'):return 'failed'
    business_tasks=[t for t in state['task_plan'].values() if t['kind']!='safety']
    if not state['task_plan'] or (len(business_tasks)==1 and len(state['task_plan'])==1
                                  and business_tasks[0]['kind'] in ('knowledge','other')
                                  and not business_tasks[0].get('context_order_id')):
        return 'single'
    return 'multi'


def active_key(state, default='refund'):
    key=state.get('active_task')
    return key if key and key in state['task_plan'] else default


async def safety(state):
    tasks = {k: dict(v) for k, v in state["task_plan"].items()}
    emitted = []
    if "safety" in tasks:
        task = tasks["safety"]
        ev = await workflow.event(state["conversation_id"], state["user_id"],
                                  task["task_id"], SAFETY_NOTICE, task_id=task["task_id"])
        task.update(status="DONE", result_ref=ev["event_id"])
        emitted.append(ev)
    return {"task_plan": tasks, "task_events": emitted}


async def schedule(state):
    tasks = state["task_plan"]
    await workflow.save_plan(state["conversation_id"], state["user_id"], state["plan_id"],
                             state["current_message_id"], tasks)
    priority={'order':0,'logistics':0,'refund':1,'complaint':2,'knowledge':3,'other':4}
    pending = next((k for k in sorted(tasks,key=lambda k:priority.get(tasks[k]['kind'],9))
                    if tasks[k]['status']=='PENDING'), '')
    return {"active_task": pending, "task_dispatches": state.get("task_dispatches", 0) + 1,
            "task_events": []}


def next_task(state):
    if state.get("task_dispatches", 0) > 2*len(state['task_plan'])+4:
        raise RuntimeError("任务调度次数超限")
    if state['active_task']:
        kind=state['task_plan'][state['active_task']]['kind']
        return 'read' if kind in ('order','logistics') else kind
    return ("wait" if any(t["status"] == "WAITING_USER"
                                               for t in state["task_plan"].values()) else "finish")


async def prepare_refund(state):
    tasks = {k: dict(v) for k, v in state["task_plan"].items()}
    task=tasks[active_key(state)]
    known=business.owns_order(state['user_id'],task.get('order_id') or '')
    task.update(status='RUNNING' if known else 'WAITING_USER',
                step='verify_order' if known else 'select_order',blocking=not known)
    return {"task_plan": tasks, 'task_order':'', 'task_order_cancelled':False}


async def select_order(state):
    uid = state["user_id"]
    orders = business.list_user_orders(uid)
    task=state['task_plan'][active_key(state)]
    # Never borrow a number from another task or from a global sentence scan.
    oid = task.get('order_id')
    if not orders:
        return {"task_order": "", "task_order_cancelled": True}
    while not business.owns_order(uid, oid or ""):
        purpose={'refund':'退款/退货审核','order':'订单信息查询','logistics':'物流查询'}[task['kind']]
        prompt=(f'没有找到属于您的这笔订单，请为本项{purpose}选择订单。' if oid else
                f'请为本项{purpose}选择对应订单：{task.get("query","")}')
        value = interrupt({"type": "select_order", "orders": orders, "can_cancel": True,
                           "plan_id": state["plan_id"], "task_id": task['task_id'], 'task_kind':task['kind'],
                           'selection_prompt':prompt})
        if value == {"cancelled": True}:
            return {"task_order": "", "task_order_cancelled": True}
        oid = value
    tasks={k:dict(v) for k,v in state['task_plan'].items()}
    current=tasks[active_key(state)]
    if current.get('order_id')!=oid:
        current['binding']={'status':'resolved','method':'user_selection','order_ids':[oid]}
    current.update(order_id=oid,status='RUNNING',step='verify_order',blocking=False)
    context = order_context.selected(state, tasks, active_key(state), oid)
    return {'task_plan':tasks, 'business_context':context, "task_order": oid, "task_order_cancelled": False}


async def refund(state):
    tasks = {k: dict(v) for k, v in state["task_plan"].items()}
    task = tasks[active_key(state)]
    actions = []
    if state.get("task_order_cancelled"):
        task.update(status="CANCELLED", step="cancelled")
        text = "本次退款选单已取消或没有可选订单；仍会继续处理其他诉求。"
    else:
        oid = state["task_order"]
        if not business.owns_order(state["user_id"], oid):
            raise ValueError("订单归属已变化")
        order = business.order_snapshot(oid)
        if order["status"] == "待付款":
            task.update(status="DONE", outcome="unpaid_order")
            text = f"订单{oid}为待付款，无需退款；如需取消订单，请在订单页面操作。"
        else:
            try:
                policy = await nodes.retrieve_policy({**state, "resolved_query": "退款退货政策", "order_data": order})
                if policy.get("memory_error") or not policy.get("citations"):
                    raise ValueError("退款政策暂时无法核实")
            except Exception:
                task.update(status="FAILED", outcome="policy_unavailable")
                ev = await workflow.event(state["conversation_id"], state["user_id"], task["task_id"] + ":result",
                    "暂时无法核实退款政策，本次未提供提交操作；仍会继续处理投诉。", task_id=task["task_id"])
                return {"task_plan": tasks, "task_events": [ev]}
            task["policy_citations"] = policy["citations"]
            # Only offer a review request: never claim refund eligibility or funds success.
            task.update(status="WAITING_USER", step="confirm_submission", blocking=False,
                        operation_id=task["task_id"] + ":submit:v1",
                        args={"ticket_type": "退款", "description": f"退款审核申请 订单号={oid} 用户原述：{state['task_origin']}"},
                        order_id=oid)
            text = f"已核对订单{oid}属于当前用户。可确认提交退款审核工单，是否符合退款条件由售后核实；尚未提交，也未执行资金退款。"
            actions = [{"task_id": task["task_id"], "operation_id": task["operation_id"], "preview": task["args"]}]
    ev = await workflow.event(state["conversation_id"], state["user_id"], task["task_id"] + ":result",
                              text, task_id=task["task_id"], actions=actions)
    return {"task_plan": tasks, "task_events": [ev]}


async def complaint(state):
    tasks = {k: dict(v) for k, v in state["task_plan"].items()}
    task = tasks[active_key(state,'complaint')]
    task.update(status="WAITING_USER", step="confirm_submission", blocking=False,
                operation_id=task["task_id"] + ":submit:v1",
                args={"ticket_type": "投诉", "description": state["task_origin"]})
    ev = await workflow.event(state["conversation_id"], state["user_id"], task["task_id"] + ":result",
                              "已记录您的投诉诉求。可单独确认提交投诉工单，由人工进一步跟进；当前尚未建单。",
                              task_id=task["task_id"])
    return {"task_plan": tasks, "task_events": [ev]}


async def wait_actions(state):
    actions = [{"task_id": t["task_id"], "operation_id": t["operation_id"], "preview": t["args"]}
               for t in state["task_plan"].values() if t["status"] == "WAITING_USER"]
    decision = interrupt({"type": "task_actions", "plan_id": state["plan_id"], "actions": actions})
    valid = next((a for a in actions if a["operation_id"] == decision.get("operation_id")), None)
    if not valid or type(decision.get("confirmed")) is not bool:
        raise ValueError("操作卡已失效或确认参数不合法")
    return {"task_decisions": decision}


async def commit_action(state):
    tasks = {k: dict(v) for k, v in state["task_plan"].items()}
    decision = state["task_decisions"]
    _, task = next((k, t) for k, t in tasks.items() if t.get("operation_id") == decision["operation_id"])
    kind=task['kind']
    if task["status"] != "WAITING_USER":
        raise ValueError("该任务已处理")
    if kind == "refund" and not business.owns_order(state["user_id"], task["order_id"]):
        raise ValueError("订单归属已变化")
    denial = await nodes.context_runtime.write_denial(state, "submit_refund" if kind == "refund" else "create_ticket",
               {"order_id": task.get("order_id"), **task["args"]})
    if denial:
        receipt = {"status": "CANCELLED", "reason": denial}
    else:
        receipt = await workflow.submit(state["conversation_id"], state["user_id"], task["operation_id"],
                                        task["args"], confirmed=decision["confirmed"])
    task.update(status=receipt["status"], receipt=receipt)
    text = (f"{task['args']['ticket_type']}工单已创建：{receipt['ticket_no']}。" if receipt.get("ticket_no")
            else f"{task['args']['ticket_type']}工单未创建，已取消本次提交。")
    ev = await workflow.event(state["conversation_id"], state["user_id"], task["operation_id"] + ":receipt",
                              text, task_id=task["task_id"])
    return {"task_plan": tasks, "task_events": [ev], "task_decisions": {}}


async def finish(state):
    # Per-task user answers/receipts already exist. Statuses stay in the plan.
    return {'answer':'', 'task_events':[]}


def after_selection(state):
    return 'refund' if state['task_plan'][active_key(state)]['kind']=='refund' else 'read'


async def read_business(state):
    from app.tools import engine, registry
    tasks={k:dict(v) for k,v in state['task_plan'].items()};task=tasks[active_key(state)]
    oid=state.get('task_order')
    try:
        if state.get('task_order_cancelled'):
            task['status']='CANCELLED';text='已取消本项订单选择，继续处理其他事项。'
        else:
            if not business.owns_order(state['user_id'],oid):raise ValueError('订单归属不匹配')
            specs={s.name:s for s in await registry.get_all_specs()}
            async def invoke(name,args):
                if name not in specs or specs[name].permission!='read':raise ValueError('只读工具不可用')
                call={'id':task['task_id']+':'+name,'name':name,'args':args,'type':'tool_call'}
                result=await engine.execute_tool_call(call,state['conversation_id'],specs,
                    user_id=state['user_id'],message_watermark=state['current_message_id'])
                if not result.ok:raise ValueError('查询失败')
                data=json.loads(result.tool_message.content)
                if data.get('_context_receipt') and data.get('partial'):data=data.get('fields',{})
                if not isinstance(data,dict) or data.get('error'):raise ValueError('业务数据不可用')
                return data
            order=await invoke('query_order',{'order_id':oid})
            if task['kind']=='order':
                if 'status' not in order or 'amount' not in order:raise ValueError('缺少订单字段')
                text=f"订单{oid}：{order.get('product','')}，状态{order['status']}，实付金额{order['amount']}元。"
                task['result']=order
            else:
                tracking=order.get('tracking_no')
                if not tracking:raise ValueError('没有物流单号')
                result=await invoke('query_logistics',{'tracking_no':tracking})
                city=result.get('current_city');status=result.get('status') or result.get('status_code')
                if not city or not status:raise ValueError('物流字段不完整')
                statuses={'PICKED_UP':'已揽收','IN_TRANSIT':'运输中','DELIVERING':'派送中','DELIVERED':'已签收'}
                text=f"订单{oid}物流：{city}，{statuses.get(status,status)}。"
                task['result']=result
            task.update(status='DONE',order_id=oid)
    except Exception:
        task['status']='FAILED';text='本项业务查询暂时无法完成，本轮不会使用旧答复代替查询结果；继续处理其他事项。'
    ev=await workflow.event(state['conversation_id'],state['user_id'],task['task_id']+':result',text,task_id=task['task_id'])
    return {'task_plan':tasks,'task_events':[ev]}


async def unsupported(state):
    tasks={k:dict(v) for k,v in state['task_plan'].items()};task=tasks[active_key(state)]
    task.update(status='FAILED',outcome='unsupported_task')
    text=f"这项诉求暂未自动处理，请单独咨询或联系人工：{task.get('query','')}"
    ev=await workflow.event(state['conversation_id'],state['user_id'],task['task_id']+':result',text,task_id=task['task_id'])
    return {'task_plan':tasks,'task_events':[ev]}


async def knowledge(state):
    from langchain_core.messages import HumanMessage
    from app.core.llm import get_chat_model
    tasks={k:dict(v) for k,v in state['task_plan'].items()};task=tasks[active_key(state)]
    stage='retrieval'
    try:
        context = order_context.knowledge_context(state, task)
        query = order_context.policy_query(task['query'], context)
        task['retrieval_query'] = query
        scoped={**state,'messages':[HumanMessage(query)],'resolved_query':query,
                'answer':'','evidence':'','citations':[],'suggested_actions':[]}
        result=await nodes.retrieve_knowledge(scoped)
        task['retrieval_trace'] = result.get('trace', {})
        task['evidence_confidence'] = result.get('evidence_confidence')
        if not result.get('evidence_strong'):raise ValueError('缺少证据')
        context_note = ('\n当前订单资料：'+json.dumps(context,ensure_ascii=False)+
            '\norder_id、product、status来自本次核验的订单；user_reported_issue是用户此前的陈述，不代表故障已核实。'
            '直接承接已知订单和商品，不要再次询问其编号或品类；只对确实未知的签收、商品状况等必要条件澄清。'
            '历史是背景，当前问题是唯一任务。召回证据回答的是政策，不证明这笔订单已获准退货。'
            '用户反映故障时按质量问题申请审核的方向解释，不能套用非质量问题限制直接拒绝；是否属实、是否符合售后条件仍需核实。'
            '保修期、免费维修或换货权益不等于退货期限或退货批准，不能以保修政策推导必然可退货。'
            '退货咨询不代表申请退款，不恢复已取消的提交，不承诺审核通过。') if context else ''
        stage='generation'
        response=await get_chat_model().ainvoke([
            {'role':'system','content':'你是喵喵优选客服，直接对用户说话，用“您”称呼，简短回答当前问题。'
             '不要写“用户反映”“针对本项事项”“本次事项处理结果”等内部解说。'
             '只根据下列证据作答，保留适用条件，标注编号；没有依据的部分明确说明。不得执行或宣称业务操作。\n'+result['evidence']+context_note},
            {'role':'user','content':task['query']}])
        if not isinstance(response.content,str) or not response.content.strip():raise ValueError('空回答')
        text=response.content;task.update(status='DONE',citations=result['citations'])
    except Exception as exc:
        task['outcome'] = f'{stage}:{type(exc).__name__}'
        text='本项知识问题暂时缺少充分依据，请联系人工核实；其他事项会继续处理。';task['status']='FAILED'
    ev=await workflow.event(state['conversation_id'],state['user_id'],task['task_id']+':result',text,task_id=task['task_id'])
    return {'task_plan':tasks,'task_events':[ev]}
