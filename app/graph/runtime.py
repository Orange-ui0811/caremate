import logging
from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from app.config import settings
from app.core import summarizer, context_runtime, workflow_mode
from app.core.activities import RETRIEVAL_NODES, retrieval_activity
from app.core.observability import attach_observability
from app.db import repository
from app.graph.build import build_graph

logger = logging.getLogger(__name__)

# 会流式吐答复 token 的节点(其余节点的模型调用 token 不进 delta)
ANSWER_NODES = {"main_agent"}
# 确定性节点把答复写在 state['answer'],整块作为 delta 吐出
DETERMINISTIC_ANSWER_NODES = {"script_reply", "complaint_reply", "fallback_reply"}
# 产出编号引用的检索节点(知识路 retrieve_knowledge + 退款政策路 retrieve_policy)
CITATION_NODES = {"retrieve_knowledge", "retrieve_policy"}

_graph = None
_cm = None  # AsyncSqliteSaver context manager,持有以免被 GC


class ConversationNotFound(Exception):
    pass


class ResumeRejected(ValueError):
    """A stale or mismatched confirmation must not resume a different action."""


async def init_graph() -> None:
    """lifespan 起:开持久 checkpointer + setup + 编译图。"""
    global _graph, _cm
    _cm = AsyncSqliteSaver.from_conn_string(settings.checkpointer_db_path)
    checkpointer = await _cm.__aenter__()
    await checkpointer.setup()
    # ch09:编译时挂一次 Langfuse 回调,全图自动 trace(未配置时原样返回,零依赖)
    _graph = attach_observability(build_graph(checkpointer=checkpointer))
    logger.info("图已编译,checkpointer=%s", settings.checkpointer_db_path)


async def close_graph() -> None:
    global _graph, _cm
    if _cm is not None:
        await _cm.__aexit__(None, None, None)
        _cm = None
    _graph = None


def get_graph():
    if _graph is None:
        raise RuntimeError("图未初始化,应在 FastAPI lifespan 调 init_graph()")
    return _graph


async def ensure_no_pending_simple(conversation_id, user_id):
    """Check before appending a message; otherwise its watermark invalidates the card."""
    if conversation_id is None:return
    conv=await repository.get_conversation(conversation_id)
    # Leave existing not-found/ownership error behavior to the normal entry.
    if conv is None or conv.user_id!=user_id:return
    snapshot=await get_graph().aget_state({'configurable':{'thread_id':str(conversation_id)}})
    state=snapshot.values or {}
    if (state.get('task_plan_version')==3 and any(t.get('status')=='blocked' for t in state.get('task_plan',{}).values())) or (
            state.get('task_plan_version')==4 and any(t.interrupts for t in snapshot.tasks)):
        raise ResumeRejected('请先通过当前卡片选择订单、确认或取消，原来的待办事项会继续保留。')


async def conversation_mode(conversation_id: int) -> bool:
    """Read the durable conversation choice, including old multi-task checkpoints."""
    snapshot = await get_graph().aget_state({'configurable': {'thread_id': str(conversation_id)}})
    state = snapshot.values or {}
    value = state.get('multi_intent_enabled')
    if type(value) is bool:
        return value
    if state.get('task_plan') or state.get('plan_id'):
        return True
    return workflow_mode.enabled()


async def _turn_mode(conversation_id, requested, user_id):
    if requested is not None and type(requested) is not bool:
        raise ValueError('模式必须为布尔值')
    if conversation_id is None:
        return workflow_mode.enabled() if requested is None else requested
    conv = await repository.get_conversation(conversation_id)
    if conv is None or conv.user_id != user_id:
        raise ConversationNotFound(conversation_id)
    mode = await conversation_mode(conversation_id)
    if requested is not None and requested != mode:
        raise ResumeRejected('切换模式请开启新对话；已有对话和待确认任务保持原模式。')
    return mode


async def get_turn_snapshot(conversation_id: int) -> dict:
    """👎 快照尽力回捞(ch09):读该会话 checkpointer 终态,返回最近一轮的用户问题与召回快照。
    调用方拿去比对被踩的 question,对得上才用快照——踩历史消息时快照可能已是别轮的,不硬塞。"""
    config = {"configurable": {"thread_id": str(conversation_id)}}
    st = await get_graph().aget_state(config)
    values = getattr(st, "values", None) or {}
    question = ""
    for m in reversed(values.get("messages", [])):
        if getattr(m, "type", "") == "human":
            question = m.content if isinstance(m.content, str) else ""
            break
    return {"question": question, "snapshot": values.get("retrieved_snapshot") or []}


def dedup_actions(actions: list) -> list:
    """跨多个 ReAct 轮次累积的 suggested_actions 可能重复(如 agent 多次提议
    create_ticket);按 type 去重,保留每种 type 首次出现的那条。"""
    seen = set()
    out = []
    for action in actions:
        t = action.get("action_id") or action.get("operation_id") or action.get("type")
        if t in seen:
            continue
        seen.add(t)
        out.append(action)
    return out


async def _ensure_conversation(user_id: str, conversation_id: int | None) -> tuple[int, str, int]:
    """返回 (cid, summary, summary_upto_msg_id):入口一次库读把 ch07 摘要两字段一并带出。"""
    if conversation_id is None:
        return await repository.create_conversation(user_id), "", 0
    conv = await repository.get_conversation(conversation_id)
    if conv is None or conv.user_id != user_id:
        raise ConversationNotFound(conversation_id)
    return conversation_id, conv.summary or "", conv.summary_upto_msg_id or 0


def _graph_input(user_id: str, message: str, cid: int, msg_id: int,
                 summary: str, summary_upto: int) -> dict:
    # 显式重置每轮的输出通道:checkpointer 按 thread_id 持久化 State,这些无 reducer 的标量
    # 会跨轮残留(如上一轮闲聊/投诉的 answer、建工单可选项串到本轮),须在入口清零。
    # messages 使用 add_messages reducer;已有会话由 _new_turn_input 恢复完整历史。
    # steps/tokens_used 是本轮预算。
    # ch07:用户消息带 db-{msg_id} 锚点(滑窗与摘要边界对齐);summary 两字段每轮从库刷新。
    return {"messages": [HumanMessage(message, id=f"db-{msg_id}")], "user_id": user_id,
            'multi_intent_enabled': workflow_mode.enabled(),
            "conversation_id": cid, "steps": 0, "tokens_used": 0,
            "answer": "", "suggested_actions": [], "evidence": "",
            "evidence_strong": False, "citations": [],
            "evidence_confidence": 0.0, "fallback_source": "", "retrieved_snapshot": [],
            "resolved_query": "", "intent_confidence": 0.0,
            "order_id": "", "order_data": {}, "fresh_business_evidence": {},
            "summary": summary, "summary_upto_msg_id": summary_upto,
            "conversation_memory": {}, "memory_segments": [], "memory_error": "",
            "current_message_id": msg_id,
            "task_plan": {}, "task_events": [], "task_decisions": {}, "plan_id": "", 'task_planning_error':False,
            'task_plan_version':1,'planning_context':{},'planning_error':'','planning_proposals':[],'task_context':{},
            'simple_facts':{},'simple_presented':{},'simple_citations':[],'simple_decision':{},'simple_execution':False,
            'queue_orders':{},'queue_start':0,'queue_answer':'',
            'queue_contract_version':0,'queue_review_retry':False,'queue_public_events':[],
            'subflow_results': {},
            'subflow_presented': {}, 'subflow_reply_id':'', 'subflow_presentation':False,
            "trace": None}  # merge 通道:None=重置哨兵(见 merge_dict),清上一轮 trace 残留


def _interrupt_payload(state: dict):
    """从终态提取中断负载(整个 __interrupt__[0].value:含 type + orders/preview 等);
    无中断返回 None。ch08 起中断不止订单选择器,负载透传、消费方按 type 分发。"""
    intr = state.get("__interrupt__")
    if not intr:
        return None
    return {**intr[0].value, "interrupt_id": intr[0].id}


async def validate_resume(conversation_id: int, resume_value, *, user_id: str,
                          interrupt_id: str) -> None:
    """Bind user confirmation to one owned pending graph interrupt.

    user_id follows the demo's existing caller identity contract, not login auth.
    This checks live storage again even when the API already checked ownership.
    """
    conv = await repository.get_conversation(conversation_id)
    if conv is None or not user_id or conv.user_id != user_id:
        raise ConversationNotFound(conversation_id)
    snapshot = await get_graph().aget_state({"configurable": {"thread_id": str(conversation_id)}})
    state = snapshot.values or {}
    if state.get("user_id") != user_id or state.get("conversation_id") != conversation_id:
        raise ConversationNotFound(conversation_id)
    pending = [intr for task in snapshot.tasks for intr in task.interrupts]
    selected = next((intr for intr in pending if intr.id == interrupt_id), None)
    if selected is None or len(pending) != 1:
        raise ResumeRejected("该操作卡片已失效，请基于当前请求重新确认。")
    kind = selected.value.get("type") if isinstance(selected.value, dict) else None
    if kind in {"confirm_ticket", "task_form"}:
        if not isinstance(resume_value, dict) or set(resume_value) != {"confirmed"} or type(resume_value["confirmed"]) is not bool:
            raise ResumeRejected("当前卡片需要确认或取消工单，不能用于选择订单。")
        if kind=='task_form' and resume_value['confirmed']:
            from app.db import workflow
            receipt=await workflow.receipt(conversation_id,user_id,f"c{conversation_id}:form:{selected.value['request_id']}")
            if not receipt:raise ResumeRejected('请先通过原表单提交，未找到提交回执。')
    elif kind == "select_order":
        from app.tools import business
        if resume_value == {"cancelled": True} and selected.value.get("can_cancel"):
            pass
        elif not isinstance(resume_value, str) or not resume_value:
            raise ResumeRejected("当前卡片需要选择订单，不能用于确认工单。")
        elif not business.owns_order(user_id, resume_value):
            raise ResumeRejected("没有找到属于当前用户的这笔订单。")
    elif kind == "task_actions":
        if (not isinstance(resume_value, dict) or set(resume_value) != {"operation_id", "confirmed"}
                or type(resume_value["confirmed"]) is not bool
                or not any(a["operation_id"] == resume_value["operation_id"] for a in selected.value.get("actions", []))):
            raise ResumeRejected("该任务操作已失效或参数不匹配。")
    else:
        raise ResumeRejected("当前中断类型无法继续，请重新发起请求。")
    rows = await repository.list_dialog_messages(conversation_id)
    latest = max((row.id for row in rows if row.role == "user"), default=0)
    if latest > state.get("current_message_id", 0):
        raise ResumeRejected("已有新的用户消息，旧操作未执行，请基于最新请求重新确认。")


async def validate_form_request(req):
    """Bind the unchanged form endpoint to its pending queue task, if any."""
    snapshot=await get_graph().aget_state({'configurable':{'thread_id':str(req.conversation_id)}})
    if snapshot.values.get('task_plan_version')!=4:return
    pending=[i for t in snapshot.tasks for i in t.interrupts]
    if len(pending)!=1 or pending[0].value.get('type')!='task_form':
        raise ValueError('该表单已失效，请重新打开会话核实当前任务。')
    card=pending[0].value
    if card['request_id']!=req.request_id:raise ValueError('表单不属于当前待办事项。')
    form=card['form']
    if hasattr(req,'order_id'):
        if form['type']!='refund_form' or req.order_id!=form['draft']['order_id']:
            raise ValueError('退款表单订单与当前任务不一致。')
    elif form['type']!='create_ticket':raise ValueError('当前等待退款表单。')
    await validate_resume(req.conversation_id,{'confirmed':False},user_id=req.user_id,interrupt_id=pending[0].id)


async def _new_turn_input(user_id: str, message: str, cid: int, msg_id: int,
                          summary: str, upto: int, *, restore_history: bool) -> dict:
    """新轮以已展示的 MySQL 对话为准,修复旧 checkpoint 漏掉固定回复的历史。

    只在新用户轮执行。Command(resume) 不走这里,保留当前工具调用/结果和中断状态。
    旧轮工具中间消息不再作为新任务输入,审计和旧 checkpoint 仍保留。
    """
    inp = _graph_input(user_id, message, cid, msg_id, summary, upto)
    if workflow_mode.enabled():
        inp['business_context'] = {}
        if restore_history:
            from app.core import order_context
            from app.db import workflow
            previous = await get_graph().aget_state({'configurable':{'thread_id':str(cid)}})
            saved_plans = await workflow.plans(cid, user_id, msg_id)
            inp['business_context'] = order_context.restore(previous.values or {}, saved_plans, user_id, cid)
    if restore_history:
        rows = await repository.list_dialog_messages(cid)
        history = []
        for row in rows:
            if row.id >= msg_id or not row.content:
                continue
            cls = HumanMessage if row.role == "user" else AIMessage
            history.append(cls(content=row.content, id=f"db-{row.id}"))
        inp["messages"] = [RemoveMessage(id=REMOVE_ALL_MESSAGES),
                           *history, *inp["messages"]]
    if context_runtime.enabled():
        try:
            inp.update(await context_runtime.prepare_turn(user_id, cid, msg_id))
        except Exception:
            logger.exception("会话来源快照准备失败 conv=%s", cid)
            inp["memory_error"] = "会话来源暂时无法核实"
    return inp


async def run_turn(user_id, message, conversation_id, *, multi_intent_enabled=None) -> dict:
    with workflow_mode.use(await _turn_mode(conversation_id, multi_intent_enabled, user_id)):
        return await _run_turn(user_id, message, conversation_id)


def task_reply_text(final, saved, *, resumed=False):
    if final.get('subflow_presentation'):
        ids={final.get('subflow_reply_id')}
        if not resumed:
            ids.update(t['task_id'] for t in final.get('task_plan',{}).values() if t['kind']=='safety')
        return '\n\n'.join(e['text'] for e in saved if e['event_id'] in ids)
    return '\n\n'.join(e['text'] for e in saved if e['event_id'].startswith(final['plan_id']+':'))


async def _run_turn(user_id, message, conversation_id) -> dict:
    """非流式:落 user 消息 → ainvoke → 返回终态(供 /api/agent、eval)。遇 interrupt 带 orders。"""
    await ensure_no_pending_simple(conversation_id,user_id)
    cid, summary, upto = await _ensure_conversation(user_id, conversation_id)
    msg_id = await repository.append_message(cid, "user", content=message)
    # ch09:langfuse_session_id 让同一会话多轮在 Langfuse 界面串成一组(未挂回调时无害)
    config = {"recursion_limit":128,"configurable": {"thread_id": str(cid)},
              "metadata": {"langfuse_session_id": str(cid)}}
    inp = await _new_turn_input(user_id, message, cid, msg_id, summary, upto,
                                restore_history=conversation_id is not None)
    final = await get_graph().ainvoke(inp, config)
    if final.get('task_plan_version')==4:
        final['answer']=final.get('queue_answer') or final.get('answer','')
    elif final.get("task_plan"):
        from app.db.workflow import events
        saved = await events(cid, user_id)
        final["answer"] = task_reply_text(final,saved)
    await summarizer.maybe_schedule_summary(cid)   # ch07 轮后触发检查(后台跑,不阻塞返回)
    return {"conversation_id": cid, "state": final, "interrupt": _interrupt_payload(final)}


async def resume_turn(conversation_id: int, resume_value, *, user_id: str, interrupt_id: str) -> dict:
    await validate_resume(conversation_id, resume_value, user_id=user_id, interrupt_id=interrupt_id)
    with workflow_mode.use(await conversation_mode(conversation_id)):
        return await _resume_turn(conversation_id, resume_value, user_id=user_id, interrupt_id=interrupt_id)


async def _resume_turn(conversation_id: int, resume_value, *, user_id: str, interrupt_id: str) -> dict:
    """非流式续跑(供 /api/agent、eval):Command(resume) 回填后跑到下一个中断或结束。"""
    config = {"recursion_limit":128,"configurable": {"thread_id": str(conversation_id)},
              "metadata": {"langfuse_session_id": str(conversation_id)}}
    final = await get_graph().ainvoke(Command(resume=resume_value), config)
    if final.get('task_plan_version')==4:
        final['answer']=final.get('queue_answer') or final.get('answer','')
    elif final.get("task_plan"):
        from app.db.workflow import events
        saved = await events(conversation_id, user_id)
        final["answer"] = task_reply_text(final,saved,resumed=True)
    await summarizer.maybe_schedule_summary(conversation_id)
    return {"conversation_id": conversation_id, "state": final,
            "interrupt": _interrupt_payload(final)}


async def _stream_events(cid: int, stream_source) -> AsyncIterator[dict]:
    """astream 多模式 → 事件 dict。stream_source 为 _graph_input(...)(新一轮)或 Command(resume=...)(续跑)。
    遇 interrupt(updates 块含 __interrupt__,Task 1 冒烟钉死)→ 吐 interrupt 事件并结束本次流
    (前端点选后走 /api/actions/resume 续流)。"""
    config = {"recursion_limit":128,"configurable": {"thread_id": str(cid)},
              "metadata": {"langfuse_session_id": str(cid)}}
    actions: list = []
    recovery_sent = False
    pending_interrupt = None
    from app.graph import queue_contract
    initial = await get_graph().aget_state(config)
    stream_state = dict(initial.values or {})
    async for mode, chunk in get_graph().astream(
        stream_source, config, stream_mode=["messages", "updates"],
    ):
        if mode == "messages":
            msg, meta = chunk
            if meta.get("langgraph_node") in ANSWER_NODES:
                if queue_contract.writing(stream_state):continue
                text = msg.content if isinstance(msg.content, str) else ""
                if text:
                    if text == context_runtime.RECOVERY_REPLY:
                        if recovery_sent:
                            continue
                        recovery_sent = True
                    yield {"type": "delta", "text": text}
        elif mode == "updates":
            if "__interrupt__" in chunk:
                payload = chunk["__interrupt__"][0].value
                # 带上 conversation_id:中断时不发 done 帧,前端要靠这个才知道续跑用哪个会话。
                # ch08:负载按中断类型透传(select_order 带 orders,confirm_ticket 带 preview),
                # 前端按 kind 分发。
                ev = {"type": "interrupt", "kind": payload.get("type", ""), "conversation_id": cid}
                if getattr(chunk["__interrupt__"][0], "id", None):
                    ev["interrupt_id"] = chunk["__interrupt__"][0].id
                for k in ("orders", "preview", "actions", "plan_id", "task_id", "can_cancel", 'task_kind', 'selection_prompt','form','request_id'):
                    if payload.get(k) is not None:
                        ev[k] = payload[k]
                # Drain astream before exposing the card: its checkpoint writes
                # must finish before a disconnect/restart or immediate resume.
                pending_interrupt = ev
                continue
            for node, upd in chunk.items():
                if not isinstance(upd, dict):
                    continue
                stream_state.update(upd)
                if node=='queue_take' and upd.get('active_task'):
                    actions=[]
                    yield {'type':'task_start'}
                if node=='queue_complete' and any(t.get('status')=='blocked' for t in upd.get('task_plan',{}).values()):
                    actions=[]  # The original form is attached to the durable interrupt.
                for event in upd.get("task_events", []):
                    if not event.get('internal'):
                        yield {"type": "task_event", "conversation_id": cid, **event}
                if node in RETRIEVAL_NODES:
                    yield {"type": "activity", **retrieval_activity(node)}
                if node in DETERMINISTIC_ANSWER_NODES and upd.get("answer"):
                    yield {"type": "delta", "text": upd["answer"]}
                if node == "main_agent" and upd.get("memory_error") and not recovery_sent:
                    # No model call occurs on a budget/source block, hence there
                    # are no token callbacks to feed the usual streaming branch.
                    yield {"type": "delta", "text": context_runtime.RECOVERY_REPLY}
                    recovery_sent = True
                if node in CITATION_NODES and upd.get("citations"):
                    yield {"type": "citations", "items": upd["citations"]}
                if node in {"agent_tools", "prepare_business_evidence", "sl_tools"}:
                    for m in upd.get("messages", []):
                        name = getattr(m, "name", None)
                        # submit_refund 仍是「拦成前端表单」的合成消息,不发工具帧;
                        # create_ticket ch08 起确认后真执行,照常发帧
                        if name and name != "submit_refund":
                            yield {"type": "tool", "name": name}
                if upd.get("suggested_actions"):
                    actions.extend(upd["suggested_actions"])
    if actions:
        yield {"type": "actions", "items": dedup_actions(actions)}
    if pending_interrupt is not None:
        yield pending_interrupt
        return
    yield {"type": "done", "conversation_id": cid}


async def stream_turn(user_id, message, conversation_id, *, multi_intent_enabled=None) -> AsyncIterator[dict]:
    with workflow_mode.use(await _turn_mode(conversation_id, multi_intent_enabled, user_id)):
        async for event in _stream_turn(user_id, message, conversation_id):
            yield event


async def _stream_turn(user_id, message, conversation_id) -> AsyncIterator[dict]:
    """流式:落 user 消息 → astream 多模式 → 事件 dict(供 /api/chat 转 SSE)。"""
    await ensure_no_pending_simple(conversation_id,user_id)
    cid, summary, upto = await _ensure_conversation(user_id, conversation_id)
    msg_id = await repository.append_message(cid, "user", content=message)
    inp = await _new_turn_input(user_id, message, cid, msg_id, summary, upto,
                                restore_history=conversation_id is not None)
    async for ev in _stream_events(cid, inp):
        yield ev
    await summarizer.maybe_schedule_summary(cid)   # ch07 流吐完再触发(后台跑,不阻塞)


async def stream_resume(conversation_id: int, resume_value, *, user_id: str,
                        interrupt_id: str) -> AsyncIterator[dict]:
    await validate_resume(conversation_id, resume_value, user_id=user_id, interrupt_id=interrupt_id)
    with workflow_mode.use(await conversation_mode(conversation_id)):
        async for event in _stream_resume(conversation_id, resume_value, user_id=user_id, interrupt_id=interrupt_id):
            yield event


async def _stream_resume(conversation_id: int, resume_value, *, user_id: str,
                         interrupt_id: str) -> AsyncIterator[dict]:
    """前端点选订单后续跑同一会话的暂停流(Command(resume) 续 astream,事件与 stream_turn 同构)。"""
    async for ev in _stream_events(conversation_id, Command(resume=resume_value)):
        yield ev
    await summarizer.maybe_schedule_summary(conversation_id)
