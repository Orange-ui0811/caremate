"""Versioned task bookkeeping and receipt-based presentation for the v4 queue.

No business tools or answer models are executed here. All work and confirmations
remain in the original main_agent, agent_tools and fetch_order nodes.
"""
import copy
import json
import re
import unicodedata

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from app.config import settings
from app.core import intent, task_context
from app.db import workflow
from app.tools import business


def enabled(state):
    return state.get('task_plan_version') == 4 and state.get('queue_contract_version') == 1


def active(state):
    return state.get('task_plan', {}).get(state.get('active_task'), {})


def writing(state):
    t = active(state)
    return enabled(state) and t.get('action') == 'request' and t.get('intent') in {'退款退货','人工','投诉','售后'}


def literal_span(raw, value):
    if value in raw:
        return value
    # Formatting-only normalization, with an exact map back to original text.
    def normal(text):
        result, offsets = '', []
        for i, c in enumerate(text):
            for n in unicodedata.normalize('NFKC', c):
                if not n.isspace(): result += n; offsets.append(i)
        return result, offsets
    text, mapping = normal(raw); needle, _ = normal(value)
    pos = text.find(needle) if needle else -1
    return raw[mapping[pos]:mapping[pos + len(needle) - 1] + 1] if pos >= 0 else None


def validate_proposals(proposals, state, candidates):
    valid, errors = {}, []
    for i, proposal in enumerate(proposals):
        try:
            t = intent.TaskIntent.model_validate(proposal).model_dump()
            span = literal_span(state['task_origin'], t['source'])
            if not span: raise ValueError('source:必须逐字引用当前原话中的连续片段')
            t['source'] = span
            oid = t.get('order_id')
            if oid and (oid not in candidates or not business.owns_order(state['user_id'],oid)):
                raise ValueError('order_id:没有合法来源或不属于当前用户')
            explicit = set(re.findall(r'(?<!\d)\d{4,}(?!\d)', span)) & candidates
            if oid and len(explicit) == 1 and oid not in explicit:
                raise ValueError('order_id:与本项明确指定的订单冲突')
            if not oid and len(explicit)==1 and t['intent'] in {'订单','物流','退款退货','售后'}:
                raise ValueError('order_id:本项明确订单已在可信候选中，不应再次选单')
            # Detect contradictions only; never grant request authority by regex.
            source = span
            negated = re.search(r'不要|不用|不需要|别|如果|假设|他说|她说|[“”"]', source)
            request = not negated and re.search(r'提交.{0,8}退款申请|申请退款|办.{0,8}(?:退款|退钱)|登记.{0,15}(?:投诉|工单)', source)
            if request and t['action'] != 'request':
                raise ValueError('action:原话明确要求办理，不能改为咨询')
            if negated and t['action'] == 'request':
                raise ValueError('action:包含否定/假设/引用，需拆出实际授权片段')
            if t['action'] == 'request' and re.search(r'查询.{0,8}(?:审核|退款).{0,4}(?:状态|进度)|怎么申请|如何申请', t['query']):
                raise ValueError('query:不得把办理改写为方法或进度咨询')
            if t['intent'] == '人工' and t['action'] == 'consult' and re.search(r'在线|几点|时间|什么时候', t['query']):
                raise ValueError('intent:人工服务时间属于知识咨询')
            valid[str(i)] = t
        except Exception as exc:
            errors.append({'index':i,'type':type(exc).__name__,'reason':str(exc),'proposal':proposal})
    for ref in {t['order_ref'] for t in valid.values()}:
        group = [k for k,t in valid.items() if t['order_ref'] == ref]
        if len({valid[k]['order_id'] for k in group if valid[k]['order_id']}) > 1:
            for key in group:
                errors.append({'index':int(key),'type':'BindingConflict','reason':'不同订单不能共用引用','proposal':valid.pop(key)})
    if not proposals: errors.append({'type':'EmptyPlan','reason':'没有当前诉求'})
    return valid, errors


async def classify(state):
    from app.graph import nodes
    context = task_context.build(state)
    candidates = {x['order_id'] for x in context['candidates']}
    attempts, valid, unresolved = [], {}, []
    if state.get('memory_error'):
        unresolved = [{'type':'ContextError','reason':state['memory_error']}]
    else:
        for attempt in range(2):
            try:
                kwargs = {'feedback':{'attempts':attempts,'instruction':'修正错误，返回完整当前任务列表；保留原话、授权和对象。'}} if attempt else {}
                proposals = await intent.classify_tasks(state['task_origin'],state['resolved_query'],
                    nodes._history_text(state,stage='intent'),context['candidates'],**kwargs)
                checked, errors = validate_proposals(proposals,state,candidates)
                attempts.append({'proposals':proposals,'errors':errors})
                # Preserve validated independent proposals if correction loses them.
                for t in checked.values(): valid[(t['source'],t['intent'],t.get('order_id'))] = t
                unresolved = errors
                if not errors:
                    valid = {(t['source'],t['intent'],t.get('order_id')):t for t in checked.values()}
                    break
            except Exception as exc:
                unresolved = [{'type':type(exc).__name__,'reason':str(exc)}]
                attempts.append({'errors':unresolved})
                # Network failures are not repeated under the guise of schema repair.
                if not isinstance(exc, ValueError): break
    checked, merge_errors = validate_proposals(list(valid.values()),state,candidates) if valid else ({},[])
    valid = {(t['source'],t['intent'],t.get('order_id')):t for t in checked.values()}
    unresolved += merge_errors
    tasks = copy.deepcopy(state.get('task_plan',{}))
    for i,t in enumerate(valid.values()):
        tasks[str(i)] = {**t,'task_id':state['plan_id']+':'+str(i),'status':'pending','result':{},
            'kind':t['intent'],'plan_version':4,'queue_contract_version':1,'repair_attempts':0,
            'binding':{'method':'shared_agent','status':'resolved' if t.get('order_id') else 'missing'}}
    for i,e in enumerate(unresolved):
        tasks['invalid:'+str(i)] = {'task_id':state['plan_id']+':invalid:'+str(i),'status':'failed',
            'kind':'planning','queue_contract_version':1,'result':{'completion_basis':'planning_invalid','error':e}}
    await workflow.save_plan(state['conversation_id'],state['user_id'],state['plan_id'],state['current_message_id'],tasks)
    return {'task_plan':tasks,'planning_context':context,'planning_proposals':attempts,
        'planning_error':json.dumps(unresolved,ensure_ascii=False),'task_planning_error':not bool(valid),
        'intent':'其他','route':'fallback_script','queue_orders':{},'queue_answer':'','active_task':'',
        'task_events':[],'trace':{'planning_attempts':attempts,'intent_status':'invalid_plan' if not valid else 'ready'}}


def prepare_take(state, update):
    t = update['task_plan'][update['active_task']]
    need_order = t['intent'] in {'订单','物流','退款退货','售后'}
    if need_order and not update.get('order_id'):
        t.update(status='blocked',wait_reason='select_order')
    update['queue_review_retry'] = False
    return update


def dispatch(state):
    if not state.get('active_task'): return 'end'
    if enabled(state) and active(state).get('wait_reason') == 'select_order': return 'refund_flow'
    from app.graph.routing import route_by_intent
    return route_by_intent(state)


async def bind_order(state):
    tasks=copy.deepcopy(state['task_plan']);t=tasks[state['active_task']]
    oid=str(state['order_id'])
    if not business.owns_order(state['user_id'],oid): raise ValueError('订单归属校验失败')
    selected=t.get('wait_reason')=='select_order'
    shared=copy.deepcopy(state.get('queue_orders',{}))
    record={'order_id':oid,'data':state['order_data'],'source':'user_selection' if selected else 'verified_order'}
    shared[t['order_ref']]=record
    for item in tasks.values():
        if item.get('order_ref')==t['order_ref'] and item.get('status') in {'pending','running','blocked'}:
            if item.get('order_id') and item['order_id'] != oid: raise ValueError('共享对象冲突')
            item.update(order_id=oid,binding={'status':'resolved','method':record['source']})
    t.update(status='running',wait_reason='')
    await workflow.save_plan(state['conversation_id'],state['user_id'],state['plan_id'],state['current_message_id'],tasks)
    return {'task_plan':tasks,'queue_orders':shared,'resolved_query':f"订单{oid}：{t['query']}"}


def _data(msg):
    try:
        v=json.loads(msg.content)
        if isinstance(v,dict) and v.get('_context_receipt') and v.get('partial'):v=v.get('fields',{})
        return v if isinstance(v,dict) else {}
    except (ValueError,TypeError):return {}


def evidence(state):
    t=active(state);oid=str(state.get('order_id') or t.get('order_id') or '')
    messages=state.get('messages',[]);start=next((i+1 for i in range(len(messages)-1,-1,-1) if isinstance(messages[i],HumanMessage)),0)
    receipts=[m for m in messages[start:] if isinstance(m,ToolMessage) and m.status!='error']
    current_receipts=[m for m in messages[state.get('queue_start',0):] if isinstance(m,ToolMessage)]
    if state.get('fallback_source') or state.get('memory_error'):return 'failed','evidence_or_context_unavailable',{}
    if t['intent']=='投诉' and t['action']=='consult':
        form=next((a for a in state.get('suggested_actions',[]) if a['type']=='create_ticket'),None)
        if form:return 'blocked','awaiting_form_confirmation',{'form':form}
    if writing(state):
        if any(m.status=='error' and '用户在工单预览卡片上点了取消' in m.content for m in current_receipts):
            return 'cancelled','user_cancelled',{}
        for m in current_receipts:
            data=_data(m)
            if m.name=='create_ticket' and m.status!='error' and data.get('ticket_no'):
                return 'done','confirmed_ticket_receipt',data
        forms=[a for a in state.get('suggested_actions',[]) if a['type']=='refund_form' and str(a.get('draft',{}).get('order_id'))==oid]
        if forms:return 'blocked','awaiting_form_confirmation',{'form':forms[0]}
        return 'failed','no_confirmed_write_receipt',{}
    if t['intent'] in {'订单','物流'}:
        order=next((_data(m) for m in reversed(receipts) if m.name=='query_order' and str(_data(m).get('order_id'))==oid),None)
        if not order:return 'failed','missing_order_receipt',{}
        if t['intent']=='订单':return 'done','current_turn_order_receipt',order
        tracking=order.get('tracking_no')
        if not tracking and order.get('status') in {'待付款','已付款','待发货','已取消'}:
            return 'done','verified_no_tracking',order
        log=next((_data(m) for m in reversed(receipts) if m.name=='query_logistics' and tracking and _data(m).get('tracking_no')==tracking and not _data(m).get('error')),None)
        return ('done','current_turn_logistics_receipt',log) if log else ('failed','missing_logistics_receipt',{})
    return 'done','answer_or_evidence',{}


async def review(state):
    if not enabled(state):return {'queue_review_retry':False}
    status,basis,_=evidence(state);t=active(state)
    retry=(status=='failed' and basis in {'missing_order_receipt','missing_logistics_receipt','no_confirmed_write_receipt'}
        and t.get('repair_attempts',0)<1 and state.get('steps',0)<settings.max_agent_steps)
    if not retry:return {'queue_review_retry':False}
    tasks=copy.deepcopy(state['task_plan']);tasks[state['active_task']]['repair_attempts']=1
    return {'task_plan':tasks,'queue_review_retry':True,'messages':[SystemMessage(
        content='当前任务尚未取得对应结果（'+basis+'）。请继续使用本任务允许的原工具；已选订单无需再问。'
        '需要提交时必须通过原确认卡；不得复用其他任务工单。若政策或必填资料确实不足，请如实说明，不编造资料、不强行办理。',
        id=t['task_id']+':completion-check')]}


async def complete(state):
    from app.graph import nodes
    tasks=copy.deepcopy(state['task_plan']);t=tasks[state['active_task']]
    raw=nodes.resolve_answer(state);status,basis,proof=evidence(state)
    t.update(status=status,result={'raw_answer':raw,'completion_basis':basis,'evidence':proof,
        'tool_receipts':[m.model_dump(mode='json') for m in state.get('messages',[])[state.get('queue_start',0):] if isinstance(m,ToolMessage)]})
    answer=raw
    if status=='blocked':t.update(form=proof['form'],operation_id=t['task_id']+':form',wait_reason='confirm_form')
    if writing(state):
        label='退款申请' if t['intent']=='退款退货' else '工单'
        order=f"订单{state['order_id']}的" if state.get('order_id') else ''
        if status=='blocked':
            t.update(form=proof['form'],operation_id=t['task_id']+':form',wait_reason='confirm_form')
            answer=f'{order}{label}尚未提交，请在下方卡片确认；提交后由平台审核处理。'
        elif status=='done':answer=f"{order}{label}已提交，工单号：{proof['ticket_no']}。后续处理以平台审核为准。"
        elif status=='cancelled':answer=f'您已取消本次{label}，本次未创建该工单。'
        else:answer=f'{order}{label}尚未提交，当前信息或处理条件不足，未取得提交回执。可补充具体情况或联系人工客服核实。'
    t['result']['answer']=answer
    shared=copy.deepcopy(state.get('queue_orders',{}))
    oid=state.get('order_id') or t.get('order_id')
    if oid and business.owns_order(state['user_id'],oid):
        old=shared.get(t.get('order_ref'),{})
        data=state.get('order_data') or old.get('data',{})
        for m in reversed(state.get('messages',[])):
            if isinstance(m,HumanMessage):break
            if isinstance(m,ToolMessage) and m.status!='error' and m.name=='query_order' and str(_data(m).get('order_id'))==oid:
                data=_data(m);break
        t.update(order_id=oid,binding={'status':'resolved','method':old.get('source','current_turn_receipt')})
        shared[t['order_ref']]={**old,'order_id':oid,'data':data}
    replacements=[]
    if writing(state):
        # Keep tool call pairing but never persist unreleased model claims as customer history.
        for m in state.get('messages',[])[state.get('queue_start',0):]:
            if isinstance(m,AIMessage) and m.content: replacements.append(m.model_copy(update={'content':''}))
        replacements.append(AIMessage(answer,id=t['task_id']+':public'))
    await workflow.save_plan(state['conversation_id'],state['user_id'],state['plan_id'],state['current_message_id'],tasks)
    return {'task_plan':tasks,'queue_orders':shared,'answer':answer,'messages':replacements,
        'queue_answer':'\n\n'.join(filter(None,[state.get('queue_answer'),answer]))}


async def publish(state):
    t=active(state);answer=state.get('answer','')
    event=await workflow.event(state['conversation_id'],state['user_id'],t['task_id']+':public:'+t['status'],
        answer,task_id=t['task_id'],citations=state.get('citations'))
    return {'task_events':[event] if writing(state) else []}


async def form_resumed(state,update):
    task=update['task_plan'][state['active_task']];r=task['result'].get('form_receipt') or {}
    task['result']['completion_basis']='confirmed_form_receipt' if r.get('ticket_no') else 'user_cancelled'
    task['wait_reason']=''
    label='退款申请' if task.get('form',{}).get('type')=='refund_form' else '投诉工单'
    answer=(f"{label}已提交，工单号：{r['ticket_no']}。后续处理以平台审核为准。" if r.get('ticket_no')
        else f'您已取消本次{label}，本次未提交该申请。')
    event=await workflow.event(state['conversation_id'],state['user_id'],task['task_id']+':public:'+task['status'],answer,task_id=task['task_id'])
    await workflow.save_plan(state['conversation_id'],state['user_id'],state['plan_id'],state['current_message_id'],update['task_plan'])
    return {**update,'task_events':[event],'answer':answer,'messages':[AIMessage(answer,id=task['task_id']+':receipt')],
        'queue_answer':'\n\n'.join(filter(None,[state.get('queue_answer'),answer]))}
