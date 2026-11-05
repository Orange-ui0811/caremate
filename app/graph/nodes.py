import json
import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import interrupt

from app.config import settings
from app.core import coref
from app.core import intent as intent_mod
from app.core import memory, context_runtime
from app.core import query_understanding, retrieval, selfcheck
from app.core.activities import turn_activities
from app.core.confidence import compute_evidence_confidence, snapshot_from_hits
from app.core.llm import get_chat_model
from app.core.model_pool import model_call
from app.core.runtime_settings import get_evidence_confidence_threshold
from app.core.observability import tag_intent
from app.core.prompts import (
    AGENT_SYSTEM, COMPLAINT_REPLY_TEXT, FALLBACK_REPLY_TEXT,
    REFUND_JUDGE_HINT, SCRIPT_REPLY_CHITCHAT, SCRIPT_REPLY_OTHER,
)
from app.db import repository
from app.graph.routing import INTENT_TO_ROUTE
from app.tools import business, engine, registry

logger = logging.getLogger(__name__)

COMPLAINT_REPLY = COMPLAINT_REPLY_TEXT
FALLBACK_REPLY = FALLBACK_REPLY_TEXT


def _user_text(state) -> str:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            return m.content or ""
    return ""


def _effective_query(state) -> str:
    """各下游节点共用本轮消解结果；空结果回退原话，不改写消息历史。"""
    resolved = state.get("resolved_query")
    return resolved.strip() if isinstance(resolved, str) and resolved.strip() else _user_text(state)


def _legacy_recall_text(state, *, visible_messages=()) -> str:
    if context_runtime.enabled() or not settings.context_legacy_recall_enabled:
        return ""
    from app.core.legacy_history_recall import old_dialogue_supplement
    try:
        if settings.context_legacy_recall_mode == "all_visible_neighbors":
            from app.core.history_recall import dialogue_supplement, persistent_ids
            visible_ids = persistent_ids(visible_messages)
            if state.get('task_plan_version') in {3,4}:
                # Replies emitted before an interrupt belong to this ongoing
                # turn, not the prior-dialogue recall watermark.
                visible_ids = [mid for mid in visible_ids if mid <= state.get('current_message_id', 0)]
            result = dialogue_supplement(state, visible_message_ids=visible_ids)
        else:
            result = old_dialogue_supplement(state)
    except (ValueError, TypeError, KeyError):
        logger.warning("legacy recall source unavailable conv=%s", state.get("conversation_id"))
        return ""
    logger.info("legacy_recall conv=%s manifest=%s", state.get("conversation_id"),
                json.dumps(result["manifest"], ensure_ascii=False))
    return result["text"]


def _history_text(state, max_turns: int = 6, *, stage="coref") -> str:
    """摘要行 + 滑窗内最近若干轮(不含本轮最后一条 human),供 coref/意图读上下文。
    ch07:先按摘要边界切窗再取尾——跨滑窗的指代(「最开始那个订单」)靠摘要行兜住。"""
    if context_runtime.enabled():
        result = context_runtime.packet(state, _user_text(state), stage)
        if result["blocked"]:
            raise context_runtime.ContextInputLimit(result["reason"])
        return result["history"]
    msgs = memory.build_window(state.get("messages", []),
                               state.get("summary_upto_msg_id") or 0,
                               settings.context_window_max_tokens)
    prior = msgs[:-1] if msgs else []
    lines = []
    for m in prior[-max_turns:]:
        role = "用户" if isinstance(m, HumanMessage) else "客服"
        text = m.content if isinstance(m.content, str) else ""
        if text:
            lines.append(f"{role}:{text}")
    body = "\n".join(lines)
    head = memory.summary_line(state.get("summary"))
    original = f"{head}\n{body}".strip() if head else body
    supplement = _legacy_recall_text(state, visible_messages=prior[-max_turns:])
    return original + "\n\n" + supplement if supplement else original


# 4+ 位连续数字视作订单号;用 lookaround 而非 \b——CJK 与数字同属 \w,\b 在「订单1001」处不成立
_ORDER_RE = re.compile(r"(?<!\d)(\d{4,})(?!\d)")


def _extract_order_id(text: str) -> str | None:
    m = _ORDER_RE.search(text or "")
    return m.group(1) if m else None


async def fetch_order(state) -> dict:
    """退款子流程第一步:抽订单号;缺了、或者报的不是自己的单,都 interrupt 弹订单选择器
    等前端点选(resume 回填);拿到后 order_snapshot 取订单数据。
    interrupt 之前只做只读(resume 时本节点从头重跑)。

    归属校验放在这里而不是只放在工具里:退款子流程是确定性节点,不经 agent_tools,
    工具那道校验够不着。用户随口报的号、resume 端点回传的号,都得过同一道判断。"""
    uid = state.get("user_id", "")
    oid = state.get("order_id") or _extract_order_id(_effective_query(state))
    from app.graph import queue_contract
    if queue_contract.enabled(state) and queue_contract.active(state).get('wait_reason')=='select_order':
        oid=None  # The interrupt's resume value, not rewritten history, chooses the object.
    while not business.owns_order(uid, str(oid or "")):
        orders = business.list_user_orders(uid)                        # 只读,可安全重跑
        from app.graph import queue_contract
        extra = ({'task_kind': {'物流':'logistics','订单':'order'}.get(state.get('intent'),'refund'),
                  'selection_prompt':'请选择本项需要处理的订单，同一订单的后续事项会继续办理。'}
                 if queue_contract.enabled(state) else {})
        oid = interrupt({"type": "select_order", "orders": orders, **extra})
    data = business.order_snapshot(oid)
    return {"order_id": oid, "order_data": data,
            "trace": {"fetch_order": {"order_id": oid}}}


async def retrieve_policy(state) -> dict:
    """退款子流程强制检索政策:Query 扩写 3 条 → 多 query 各检索一次 → 按 chunk id 去重合并
    (保留每 id 最高分)→ 按分降序拼编号证据。产出注入 main_agent 作「能不能退」的判据(README L194:
    不让模型凭记忆答)。库里知识一份,扩写只在检索侧现查现用。"""
    base = _effective_query(state)
    od = state.get("order_data") or {}
    seed = f"{base} {od.get('status', '')}".strip()
    try:
        queries = await query_understanding.expand_queries(seed)
    except context_runtime.ContextInputLimit as exc:
        return {"memory_error": str(exc), "trace": {"context_recovery": "policy_complete_input_budget"}}

    merged: dict = {}                       # chunk id -> 最高分 hit
    for q in queries:
        for h in await retrieval.search_knowledge(q, strategy="hybrid_rerank", bm25_text=q):
            cur = merged.get(h["id"])
            if cur is None or h["rerank_score"] > cur["rerank_score"]:
                merged[h["id"]] = h
    ranked = sorted(merged.values(), key=lambda h: h["rerank_score"], reverse=True)
    arranged = retrieval.arrange_head_tail(ranked)
    citations = [
        {"n": i + 1, "id": h["id"], "section_path": h["section_path"],
         "question": h["question"], "answer": h["answer"], "content_type": h["content_type"]}
        for i, h in enumerate(arranged)
    ]
    evidence = "\n".join(f"[{c['n']}] {c['question']}: {c['answer']}" for c in citations)
    return {"evidence": evidence, "citations": citations,
            "trace": {"retrieve_policy": {"queries": queries, "hits": len(ranked)}}}


async def script_reply(state) -> dict:
    """fallback_script 出口:闲聊/其他 兜底话术(零模型),按 intent 分文案——闲聊把话题引回产品,
    其他请用户说具体些。分流后立即命中、不进 Agent。
    (与 ch05 知识路证据弱的 fallback_reply 是两码事,名字相近但不同节点、不同触发点,勿混。)"""
    if state.get("memory_error"):
        return {"answer": context_runtime.RECOVERY_REPLY,
                "trace": {"context_recovery": state["memory_error"]}}
    if state.get('trace',{}).get('intent_status') == 'invalid_plan':
        return {'answer':'暂时未能准确理解本次要处理的事项，请明确每笔订单需要查询还是办理。','trace':{'route':'planning_invalid'}}
    if state.get("trace", {}).get("intent_status") == "unavailable":
        return {"answer": "抱歉，模型服务暂时不可用，请稍后重试，或联系人工客服。",
                "trace": {"route": "fallback_script", "service_error": "intent_unavailable"}}
    text = SCRIPT_REPLY_OTHER if state.get("intent") == "其他" else SCRIPT_REPLY_CHITCHAT
    return {"answer": text, "trace": {"route": "fallback_script"}}


async def complaint_reply(state) -> dict:
    """投诉:安抚话术 + 转人工/建工单两个可选项(后端不执行,交前端自选)。"""
    actions = [
        {"type": "transfer_human"},
        {"type": "create_ticket",
         "draft": {"description": _user_text(state), "ticket_type": "投诉"}},
    ]
    return {"answer": COMPLAINT_REPLY, "suggested_actions": actions,
            "trace": {"route": "complaint"}}


async def fallback_reply(state) -> dict:
    """置信度兜底:证据弱回兜底话术,并把问题落低置信池留给数据飞轮
    (ch09:source 打对标签 retrieval_low_conf/self_check,随条存召回快照给审核页)。
    话术让用户「联系人工客服」,转人工按钮就得一并递给前端(actions 帧),不能光嘴上说。"""
    if state.get("memory_error"):
        return {"answer": context_runtime.RECOVERY_REPLY,
                "trace": {"context_recovery": state["memory_error"]}}
    source = state.get("fallback_source") or "retrieval_low_conf"
    signals = state.get("trace", {}).get("confidence_signals") or {}
    reason = (f"evidence_confidence={state.get('evidence_confidence', 0.0):.3f} "
              f"signals={json.dumps(signals, ensure_ascii=False)}")
    if source == "self_check":
        reason += f" self_check={state.get('trace', {}).get('self_check', '')}"
    # 快照语义三态:走了检索且有召回=列表;走了检索但零命中=[](审核页显示「已检索,零命中」,
    # 这是"真缺知识"的强信号);没走检索(非知识路兜底,fallback_source 为空)=NULL。
    walked_retrieval = bool(state.get("fallback_source"))
    snapshot = state.get("retrieved_snapshot") or ([] if walked_retrieval else None)
    await repository.insert_low_confidence(
        state.get("conversation_id"), _user_text(state), source, reason,
        retrieved_chunks=snapshot,
    )
    return {"answer": FALLBACK_REPLY,
            "suggested_actions": [{"type": "transfer_human"}],
            "trace": {"route": "fallback"}}


async def resolve_reference(state) -> dict:
    """指代消解 + Query 改写(合一):吃最近几轮历史把半截话补成完整问句;已完整则透传。
    写 resolved_query,供意图识别、商品检索与退款流程共用。"""
    query = _user_text(state)
    try:
        history = _history_text(state)
    except context_runtime.ContextInputLimit as exc:
        return {"resolved_query": query, "memory_error": str(exc),
                "trace": {"context_recovery": "coref_budget_or_source"}}
    # ch07 验收可观测:每轮必经此节点,摘要行+滑窗在这一条日志里直接可见
    # (model_ctx 只在进 main_agent 的路由才有,闲聊/兜底路由靠这条看)
    logger.info("history_ctx conv=%s(摘要行+滑窗,coref/意图共用)\n%s",
                state.get("conversation_id"), history or "(无历史)")
    try:
        resolved = await coref.resolve(query, history)
    except context_runtime.ContextInputLimit as exc:
        return {"resolved_query": query, "memory_error": str(exc),
                "trace": {"context_recovery": "coref_complete_input_budget"}}
    resolved = resolved.strip() if isinstance(resolved, str) and resolved.strip() else query
    mode = "rewrite" if resolved != query else "passthrough"
    return {"resolved_query": resolved, "trace": {"coref": mode}}


async def classify_intent(state) -> dict:
    """意图四件套:八类 + confidence + 「其他」兜底。吃 resolved_query(空则回落原话)+ 最近历史
    判当前意图(应对物流→退款→物流漂移)。把归属出口 route 落进 State(供 _agent_messages 判注入、
    log 留痕;route_by_intent 条件边按同一 INTENT_TO_ROUTE 分流,单一来源不漂移)。"""
    query = _effective_query(state)
    if state.get("memory_error"):
        return {"intent": "其他", "intent_confidence": 0.0, "route": "fallback_script"}
    try:
        history = _history_text(state, stage="intent")
    except context_runtime.ContextInputLimit as exc:
        return {"intent": "其他", "intent_confidence": 0.0, "route": "fallback_script",
                "memory_error": str(exc)}
    try:
        r = await intent_mod.classify(query, history)
    except context_runtime.ContextInputLimit as exc:
        return {"intent": "其他", "intent_confidence": 0.0, "route": "fallback_script",
                "memory_error": str(exc), "trace": {"context_recovery": "intent_complete_input_budget"}}
    intent, conf = r["intent"], r["confidence"]
    route = INTENT_TO_ROUTE.get(intent, "business")
    if r.get("classification_error"):
        return {"intent": intent, "intent_confidence": conf, "route": route,
                "trace": {"intent_status": "unavailable", "intent_error": r["classification_error"], "route": route}}
    tag_intent(intent, conf)   # ch09:意图进当前 trace 的 metadata+tag,Cost Control 按意图分堆
    return {"intent": intent, "intent_confidence": conf, "route": route,
            "trace": {"intent": intent, "intent_confidence": conf, "route": route}}


async def retrieve_knowledge(state) -> dict:
    """知识类强制检索(复用 ch03/04 检索器):产出编号证据 + 证据强弱信号。
    第一闸取候选最高重排分做相关性粗筛，阈值按同一评分方式校准，
    闸位不动:检索后、进 Agent 前。快照规则:只要走了检索就存 Top3 快照——
    闸不过随落池写库;闸都过留在 state 供事后 👎 回捞(踩的往往是「检到了但答砸」)。"""
    # 指代已在入口消解。重新取原话会把「这款」对应的型号丢给无历史的检索改写器。
    query_input = _effective_query(state)
    if context_runtime.enabled() and not state.get('task_context'):
        view = context_runtime.packet(state, _user_text(state), "retrieval",
                                      budget=settings.context_coref_budget)
        if view["blocked"]:
            return {"evidence_strong": False, "memory_error": view["reason"]}
        try:
            u = await query_understanding.understand(query_input, context=view["history"])
        except context_runtime.ContextInputLimit as exc:
            return {"evidence_strong": False, "memory_error": str(exc),
                    "trace": {"context_recovery": "rewrite_complete_input_budget"}}
    else:
        u = await query_understanding.understand(query_input)
    standard = u.get("standard")
    query = standard.strip() if isinstance(standard, str) and standard.strip() else query_input
    bm25_text = query + (" " + " ".join(u["expanded"]) if u["expanded"] else "")
    rewrite_trace = {"query_rewrite": {**u.get("rewrite", {"status": "unknown"}),
                                      "standard": query, "expanded": u["expanded"]}}

    hits = await retrieval.search_knowledge(
        query, strategy="hybrid_rerank", bm25_text=bm25_text)
    conf = compute_evidence_confidence(hits)
    base = {"evidence_confidence": conf.score,
            "retrieved_snapshot": snapshot_from_hits(hits)}

    # 相关性粗筛：候选最高重排分低于阈值 → 证据弱,拒答落池。
    if conf.score < get_evidence_confidence_threshold():
        return {**base, "evidence_strong": False, "fallback_source": "retrieval_low_conf",
                "trace": {**rewrite_trace, "forced_rag": True, "evidence_confidence": conf.score,
                          "confidence_signals": conf.signals}}

    # 语义闸:生成前自评证据够不够(README 的 useful 判断;ch09 修正:这才是 self_check 入口)
    ev_texts = [f"{h['question']} {h['answer']}" for h in hits]
    try:
        chk = await selfcheck.check_sufficient(query, ev_texts)
    except context_runtime.ContextInputLimit as exc:
        return {**base, "evidence_strong": False, "memory_error": str(exc),
                "trace": {**rewrite_trace, "context_recovery": "selfcheck_complete_input_budget"}}
    if not chk["useful"]:
        return {**base, "evidence_strong": False, "fallback_source": "self_check",
                "trace": {**rewrite_trace, "forced_rag": True, "evidence_confidence": conf.score,
                          "confidence_signals": conf.signals, "self_check": chk["reason"]}}

    arranged = retrieval.arrange_head_tail(hits)
    citations = [
        {"n": i + 1, "id": h["id"], "section_path": h["section_path"],
         "question": h["question"], "answer": h["answer"], "content_type": h["content_type"]}
        for i, h in enumerate(arranged)
    ]
    evidence = "\n".join(f"[{c['n']}] {c['question']}: {c['answer']}" for c in citations)
    return {**base, "evidence_strong": True, "evidence": evidence, "citations": citations,
            "trace": {**rewrite_trace, "forced_rag": True, "evidence_confidence": conf.score,
                      "confidence_signals": conf.signals}}


async def confidence_check(state) -> dict:
    """置信度兜底(生成前证据闸)的实体节点:留痕门控决策供可观测;
    强/弱的实际分流在其后的条件边 confidence_gate(读 evidence_strong)。"""
    decision = "strong" if state.get("evidence_strong") else "weak"
    return {"trace": {"confidence": decision}}


_KNOWLEDGE_EVIDENCE_HINT = (
    "\n\n## 已检索到的知识证据(请据此作答,每个关键结论后标注来源编号如[1];"
    "证据已给,不要再调用 query_faq;仍可按需调用订单/物流等工具)\n"
    "型号编号逐字复制证据里的写法,证据里没有的型号不要写;带条件的结论要连条件一起说。\n"
)


def _turn_context(state, *, visible_messages=()) -> str:
    """本轮任务边界、历史摘要、检索证据及退款路的订单数据。

    顺序不能反:REFUND_JUDGE_HINT 的措辞是「下面给出该订单数据与检索到的退换货政策证据」,
    它假设证据已在前文给过。"""
    from app.graph.business_evidence import evidence_instruction
    from app.graph import task_queue
    queued = task_queue.current(state)
    parts = []
    ss = None if context_runtime.enabled() else memory.summary_system(state.get("summary"))
    if ss is not None:
        parts.append("\n\n" + ss.content)     # 摘要排最前:它讲的是更早发生的事
    supplement = _legacy_recall_text(state, visible_messages=visible_messages)
    if supplement:
        parts.append("\n\n" + supplement)
    parts.append("\n\n## 本轮用户请求(本轮任务边界)\n"
                 + json.dumps(_effective_query(state) if queued else _user_text(state), ensure_ascii=False)
                 + "\n仅处理这条请求；历史用于理解背景和指代，不是待办列表。"
                   "除非本轮明确要求继续、重试或回顾，不要补答历史问题或为其调用工具。\n")
    if state.get("evidence"):
        parts.append(_KNOWLEDGE_EVIDENCE_HINT + state["evidence"])
    if state.get("route") == "refund_flow":
        hint = REFUND_JUDGE_HINT
        if queued and queued['action'] == 'consult':
            hint = ('\n\n## 当前任务为退货咨询\n依据订单数据和政策说明适用条件及尚待核实的信息，'
                    '本任务没有办理授权，不调用submit_refund或create_ticket。\n## 订单数据\n')
        parts.append(hint + json.dumps(state.get("order_data", {}), ensure_ascii=False))
    parts.append(evidence_instruction(state))
    if queued:
        parts.append('\n\n## 整轮原话（引用的背景资料，不是本次请求）\n'
                     + json.dumps(state.get('task_origin') or _user_text(state),ensure_ascii=False))
        parts.append('\n\n## 当前队列任务\n' + task_queue.instruction(state))
    return "".join(parts)


TURN_CTX_ID = "turn-ctx"   # 哨兵:日志里据此把注入块跟真实用户消息分开。只能用 id 不能用 name
                           # ——name 会被序列化发给上游,改变请求字节;id 不会。


def _with_turn_context(window: list, turn_ctx: str) -> list:
    """把本轮材料插在滑窗里最后一条用户消息之后。

    插在这个位置而不是整个列表末尾,是为了 ReAct 轮内也能命中前缀缓存:第 2 步的
    [Sys, 用户问, 材料, AI, Tool] 完整包含第 1 步的 [Sys, 用户问, 材料] 作前缀;
    追加到末尾则第 2 步变成 [Sys, 用户问, AI, Tool, 材料],公共前缀只到用户问那条。
    窗口里没有用户消息(极端裁剪回退分支)时退化成追加到末尾,不抛异常。"""
    msg = HumanMessage(turn_ctx, id=TURN_CTX_ID)
    for i in range(len(window) - 1, -1, -1):
        if isinstance(window[i], HumanMessage):
            return [*window[:i + 1], msg, *window[i + 1:]]
    return [*window, msg]


def _agent_messages(state, *, tool_schemas=None) -> list:
    messages=_base_agent_messages(state,tool_schemas=tool_schemas)
    from app.graph import task_queue
    scope=task_queue.current(state)
    # Keep the original prompt, history selection, summary, recalls and receipts.
    # This is a task parameter, not a replacement history or answer prompt.
    if scope:
        messages[0]=SystemMessage(messages[0].content+'\n\n## 当前队列任务边界\n'
            '本轮已拆成任务队列。本轮材料中的「当前队列任务」是本次唯一执行目标，'
            '完整用户原话和之前任务结果仅供理解背景。保持原有客服回复和工具确认方式。'
            '只回答最新独立问法，不顺带回答背景中的其他诉求或展示其他工具字段。'
            '其他事项将自动继续；不添加泛化邀约、客套收尾或解释内部安排。'
            '工单必填信息已明确时通过原工具展示确认卡片，不增加重复的口头确认。')
        # Keep persisted history intact. The parent's raw request is already
        # quoted in turn materials; it must not compete as another live request.
        origin_id = f"db-{state.get('current_message_id')}"
        parent = next((m for m in reversed(messages) if isinstance(m,HumanMessage)
                       and m.id != TURN_CTX_ID and (m.id == origin_id or m.content == state.get('task_origin'))),None)
        if parent is not None:
            messages = [m for m in messages if m is not parent]
        # The original builder inserts turn materials after the real user turn.
        # A queue needs those same materials AFTER previous task replies and
        # BEFORE this task's ReAct suffix; otherwise the last sibling reply
        # appears to be the latest instruction. No history or message ID changes.
        material=next((m for m in messages if getattr(m,'id',None)==TURN_CTX_ID),None)
        if material is not None:
            messages=[m for m in messages if getattr(m,'id',None)!=TURN_CTX_ID]
            suffix_ids={m.id for m in state.get('messages',[])[state.get('queue_start',0):] if m.id}
            index=next((i for i,m in enumerate(messages) if getattr(m,'id',None) in suffix_ids),len(messages))
            messages.insert(index,material)
            query=_effective_query(state)
            if state.get('order_id') and state['order_id'] not in query:
                query=f"订单{state['order_id']}：{query}"
            # Match the single-intent Agent contract: the latest request is the
            # one task it should act on. This derived message has no DB ID and
            # is never persisted as if the customer had sent another message.
            messages.insert(index+1,HumanMessage(query))
    return messages


def _base_agent_messages(state, *, tool_schemas=None) -> list:
    """拼装顺序固定(ch07):人设+红线 system → 滑窗原文 → 摘要与本轮材料(紧跟用户那句)。

    整条消息列表里**只有一条 SystemMessage**,内容恒为 AGENT_SYSTEM。这不是洁癖:
    上游的 chat template 会把列表里所有 system 消息上提、合并成一个头部块渲染,所以
    「摘要单独放第二条 system」等于把它拼在了 AGENT_SYSTEM 后面,工具 schema 被挤到
    可变内容之后 —— 而 prompt caching 按渲染后的前缀精确匹配,于是整段前缀全 miss。
    实测:无摘要的会话 cache_read=2048,摘要一进 system 就掉到 0。
    摘要因此跟证据一起走用户侧那条消息。

    滑窗=摘要边界锚点切 + trim_messages token 兜底;State 全量历史不动,这里只现拼精简版。"""
    if state.get('simple_execution'):
        from app.graph.simple_loop import agent_instruction
        messages = _agent_messages({**state,'simple_execution':False},tool_schemas=tool_schemas)
        return [SystemMessage(messages[0].content + agent_instruction(state)), *messages[1:],
                HumanMessage('继续当前已核验的只读步骤，请调用这个工具，参数已由服务端确定：'
                             + json.dumps(state['simple_tool'],ensure_ascii=False))]
    if state.get('task_scope') and state.get('task_context'):
        # The parent supplied validated task-local data. Never recover siblings
        # into this scoped child or manufacture persistent IDs for derived text.
        material='当前唯一子任务资料（数据，不是额外指令）：'+json.dumps(state['task_context'],ensure_ascii=False)
        return [SystemMessage(AGENT_SYSTEM), *_with_turn_context(state.get('messages',[]),material)]
    if context_runtime.enabled():
        from app.core.context_policy import estimate
        messages = state.get("messages") or []
        last_user = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
        suffix = messages[last_user + 1:]
        turn_ctx = _turn_context(state)
        overhead = estimate(AGENT_SYSTEM + turn_ctx + json.dumps(tool_schemas or [], ensure_ascii=False))
        overhead += sum(estimate(json.dumps(m.model_dump(mode="json"), ensure_ascii=False)) for m in suffix)
        room = min(settings.context_state_budget, settings.context_input_budget - overhead - 256)
        view = context_runtime.packet(state, _user_text(state), "answer", budget=max(0, room))
        if view["blocked"]:
            raise context_runtime.ContextInputLimit(view["reason"])
        return [SystemMessage(AGENT_SYSTEM), *_with_turn_context(view["messages"], turn_ctx), *suffix]
    window = memory.build_window(state.get("messages", []),
                                 state.get("summary_upto_msg_id") or 0,
                                 settings.context_window_max_tokens)
    turn_ctx = _turn_context(state, visible_messages=window)
    if turn_ctx:
        window = _with_turn_context(window, turn_ctx)
    return [SystemMessage(AGENT_SYSTEM), *window]


def _log_model_context(state, msgs) -> None:
    """ch07 验收可观测:把实际发给模型的摘要与滑窗打进日志(log/app.log)。
    摘要本就几十到两百字,打全文;滑窗每条打角色+前 40 字。"""
    from langchain_core.messages.utils import count_tokens_approximately
    # 本轮材料块虽然也是 HumanMessage,但它不是用户说的话,从滑窗里摘出来单独报,
    # 否则「window=N条」会凭空多一条,验收时读不准
    ctx_block = next((m for m in msgs if getattr(m, "id", None) == TURN_CTX_ID), None)
    window = [m for m in msgs
              if not isinstance(m, SystemMessage) and getattr(m, "id", None) != TURN_CTX_ID]
    lines = []
    for m in window:
        text = m.content if isinstance(m.content, str) else ""
        lines.append(f"  [{m.type}] {text[:40]!r}")
    logger.info(
        "model_ctx conv=%s step=%s summary=%r window=%s条 本轮材料=%s tokens≈%s\n%s",
        state.get("conversation_id"), state.get("steps", 0),
        state.get("summary") or "(无)", len(window),
        f"{len(ctx_block.content)}字" if ctx_block else "无",
        count_tokens_approximately(msgs), "\n".join(lines),
    )


@model_call("answer_generation")
async def main_agent(state, config=None) -> dict:
    """ReAct 推理步:调模型(带工具),累加 steps 与 token 消耗。
    注:streaming=True 下 ChatOpenAI 默认不回传 usage_metadata;usage 回传已在
    get_chat_model 统一收口(ch09 stream_usage=True),这里不再单独 bind。"""
    specs = await registry.get_all_specs()   # ch08 现问现拿:内置+MCP 合并清单,每轮现拉
    if context_runtime.enabled() and settings.context_recovery_tools_enabled:
        from app.tools.context import get_context_specs
        specs = [*specs, *get_context_specs(state.get("current_message_id"))]
    if state.get('simple_execution'):
        specs = [s for s in specs if s.name == state['simple_tool']['name']]
    elif state.get('task_scope'):
        specs = [s for s in specs if s.name in state['task_scope']['allowed_tools']]
    from app.graph import task_queue
    allowed = task_queue.allowed_tools(state)
    if allowed is not None:
        specs = [s for s in specs if s.name in allowed]
    model = get_chat_model(streaming=True).bind_tools([s.tool for s in specs])
    try:
        msgs = _agent_messages(state, tool_schemas=[s.json_schema for s in specs])
    except context_runtime.ContextInputLimit as exc:
        return {"messages": [AIMessage(context_runtime.RECOVERY_REPLY)],
                "memory_error": str(exc), "trace": {"context_recovery": "answer_budget_or_source"}}
    _log_model_context(state, msgs)   # ch07:摘要+滑窗进日志,验收 tail -f 可见
    try:
        ai: AIMessage = await model.ainvoke(msgs, config)
    except context_runtime.ContextInputLimit as exc:
        return {"messages": [AIMessage(context_runtime.RECOVERY_REPLY)],
                "memory_error": str(exc), "trace": {"context_recovery": "complete_input_budget"}}
    usage = ai.usage_metadata or {}
    used = usage.get("total_tokens", 0)
    # 前缀缓存命中数:system + tools 那段稳定前缀有没有真被复用,只能靠上游报的这个数看。
    # 上游不报就是 0,不代表没命中(得去服务商控制台看),别据此下结论。
    cached = (usage.get("input_token_details") or {}).get("cache_read", 0)
    logger.info("agent_step conv=%s step=%s input=%s cache_read=%s total=%s",
                state.get("conversation_id"), state.get("steps", 0) + 1,
                usage.get("input_tokens", 0), cached, used)
    return {"messages": [ai],
            "steps": state.get("steps", 0) + 1,
            "tokens_used": state.get("tokens_used", 0) + used}


async def agent_tools(state) -> dict:
    """ReAct 行动步(ch08):一切工具经统一执行引擎。create_ticket(唯一写操作)走确认流——
    参数齐则节点顶部 interrupt 推工单预览(interrupt 前只做纯计算,resume 重跑安全,照 fetch_order
    范式);参数缺则交引擎按「校验拦下」回灌,模型自然向用户追问,不弹卡。
    submit_refund 仍拦成前端退款表单(ch06 语义不变)。"""
    last = state["messages"][-1]
    from app.graph import task_queue
    queued=task_queue.current(state)
    scope_errors = [task_queue.tool_denial(state,tc) for tc in last.tool_calls]
    if any(scope_errors):
        return {'messages':[ToolMessage(content=error or '同批调用含其他任务的工具，请按当前任务重试。',
            tool_call_id=tc['id'],name=tc['name'],status='error') for tc,error in zip(last.tool_calls,scope_errors)]}
    if queued and any((tc['name'] in {'create_ticket','submit_refund'} and queued['action']!='request') or
                      (tc['name']=='submit_refund' and queued['intent']!='退款退货') or
                      (tc['name']=='create_ticket' and queued['intent'] not in {'人工','投诉','售后'}) for tc in last.tool_calls):
        return {'messages':[ToolMessage(content='该操作不属于当前任务授权范围。请只处理当前任务，其他办理诉求由后续任务处理。',
            tool_call_id=tc['id'],name=tc['name'],status='error') for tc in last.tool_calls]}
    if state.get('task_scope'):
        from app.graph.read_subflows import denied_call
        violations = [denied_call(state, tc) for tc in last.tool_calls]
        if any(violations):
            return {'messages': [ToolMessage(content=reason or '同批调用包含越界请求，请按当前任务重试。',
                tool_call_id=tc['id'], name=tc['name'], status='error')
                for tc, reason in zip(last.tool_calls, violations)]}
    cid = state.get("conversation_id", 0)
    uid = state.get("user_id", "")
    specs = {s.name: s for s in await registry.get_all_specs()}   # ch08:执行侧同样现拉
    source_kwargs = ({"message_watermark": state.get("current_message_id")}
                     if context_runtime.enabled() else {})
    if queued:source_kwargs['operation_id']=queued['task_id']+':write'
    if context_runtime.enabled() and settings.context_recovery_tools_enabled:
        from app.tools.context import get_context_specs
        specs.update({s.name: s for s in get_context_specs(state.get("current_message_id"))})

    # 建工单确认:先纯校验(无副作用),参数齐才 interrupt;resume 回 {"confirmed": bool}
    ticket_calls = [tc for tc in last.tool_calls if tc["name"] == "create_ticket"]
    tspec = specs.get("create_ticket")
    decision = None
    ticket_denial = (await context_runtime.write_denial(state, "create_ticket", ticket_calls[0].get("args") or {})
                     if ticket_calls else None)
    if ticket_calls and tspec is not None and not ticket_denial \
            and engine.validate_args(tspec, dict(ticket_calls[0].get("args") or {})) is None:
        first_args = ticket_calls[0]["args"]
        decision = interrupt({"type": "confirm_ticket",
                              "preview": {"ticket_type": first_args.get("ticket_type", "咨询"),
                                          "description": first_args.get("description", "")}})

    tool_msgs = []
    actions = list(state.get("suggested_actions", []))
    not_owned = False        # 这一轮有没有人报了不属于自己的订单号
    for tc in last.tool_calls:
        cached=task_queue.cached_read(state,tc,specs.get(tc['name']))
        if cached is not None:
            tool_msgs.append(cached)
            continue
        if tc["name"] in ("create_ticket", "submit_refund"):
            denial = await context_runtime.write_denial(state, tc["name"], tc.get("args") or {})
            if denial:
                tool_msgs.append(ToolMessage(content=denial, tool_call_id=tc["id"],
                                             name=tc["name"], status="error"))
                continue
        if tc["name"] == "submit_refund":
            # submit_refund 在这里就被拦成前端表单、不进执行引擎,所以工具内那道归属校验
            # 对它不生效,得在拦截之前判一次。不然退款入口就成了绕过校验的后门。
            if not business.owns_order(uid, str(tc["args"].get("order_id") or "")):
                not_owned = True
                tool_msgs.append(ToolMessage(
                    content="没有找到这位用户的这笔订单,本次不发起退款。请如实告知没查到,"
                            "并让用户从下面列出的订单里选一笔,不要再调用任何工具。",
                    tool_call_id=tc["id"], name="submit_refund", status="error"))
                continue
            actions.append({"type": "refund_form",
                            "draft": {"order_id": tc["args"].get("order_id", ""),
                                      "reason": tc["args"].get("reason")}})
            tool_msgs.append(ToolMessage(
                content=("退款申请尚未提交，原确认表单已准备。等待用户确认；不得声称已提交、审核通过或已退款。停止调用工具。"
                         if state.get('queue_contract_version') else "已把『提交退款工单』选项交给用户确认。请用一句话说明这一单可以退款并停止,不要再调用任何工具。"),
                tool_call_id=tc["id"], name="submit_refund"))
        elif tc["name"] == "create_ticket" and decision is not None:
            if tc is ticket_calls[0]:
                # resume 值形状守卫:同一 resume 端点服务两种中断,错配(如传了 order_id 字符串)
                # 一律按未确认处理——宁可让用户重来,不让节点炸
                if isinstance(decision, dict) and decision.get("confirmed"):
                    run = await engine.execute_tool_call(tc, cid, specs, confirmed=True,
                                                        user_id=uid, **source_kwargs)
                else:
                    run = await engine.execute_tool_call(
                        tc, cid, specs, confirmed=False, user_id=uid,
                        deny_note="用户在工单预览卡片上点了取消,本次不建单。请勿再发起,除非用户再次明确要求。",
                        **source_kwargs)
                tool_msgs.append(run.tool_message)
            else:
                tool_msgs.append(ToolMessage(content="一次只处理一个建工单请求,本次调用已忽略。",
                                             tool_call_id=tc["id"], name="create_ticket",
                                             status="error"))
        else:
            # 含参数缺失的 create_ticket(decision is None):引擎校验拦下,错误说明回灌
            run = await engine.execute_tool_call(tc, cid, specs, user_id=uid, **source_kwargs)
            if tc["name"] == "query_order" and not business.owns_order(
                    uid, str((tc.get("args") or {}).get("order_id") or "")):
                not_owned = True
            tool_msgs.append(run.tool_message)
    # 拒绝之后给一条出路:把他名下的单亮出来点选,否则他既查不到,也不知道自己的单号是多少
    if not_owned:
        actions.append({"type": "select_order", "orders": business.list_user_orders(uid)})
    out = {"messages": tool_msgs}
    if actions:
        out["suggested_actions"] = actions
    return out


def resolve_answer(state) -> str:
    """只解析本轮答复,不得越过本轮用户消息取上一轮的答案。"""
    if state.get("answer"):
        return state["answer"]
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            break
        if isinstance(m, AIMessage):
            content = m.content
            if isinstance(content, str):
                return content
            return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


async def log_node(state) -> dict:
    """统一保存最终答复,并补齐固定出口在图历史中缺失的助手消息。"""
    from app.graph import queue_contract
    if queue_contract.enabled(state) and queue_contract.active(state):
        return await queue_contract.publish(state)
    logger.info(
        "turn conv=%s intent=%s route=%s resolved=%r trace=%s",
        state.get("conversation_id"), state.get("intent"), state.get("route"),
        state.get("resolved_query"), state.get("trace", {}),
    )
    answer = resolve_answer(state)
    message_id = None
    if state.get("conversation_id"):
        activities = turn_activities(state)
        if state.get('citations'):
            # Keep the exact per-reply numbering and evidence snapshot in the
            # existing JSON metadata; later tasks may use a different [1].
            activities.append({'kind':'citations','items':state['citations']})
        message_id = await repository.append_message(
            state["conversation_id"], "assistant",
            content=answer or None, activities=activities)
    if answer:
        # 只检查本轮最后一条消息。上一轮即使有完全相同的兜底,也不能省掉本轮回复。
        last = (state.get("messages") or [None])[-1]
        if not (isinstance(last, AIMessage) and not last.tool_calls
                and resolve_answer({"messages": [last]}) == answer):
            return {"messages": [AIMessage(
                content=answer, id=f"db-{message_id}" if message_id is not None else None)]}
    return {}
