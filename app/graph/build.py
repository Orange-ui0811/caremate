from langgraph.graph import END, START, StateGraph

from app.graph import nodes
from app.graph import tasks
from app.graph import task_queue
from app.graph import queue_contract
from app.graph.business_evidence import prepare_business_evidence
from app.graph.routing import confidence_gate, route_by_intent, should_continue
from app.graph.state import ConversationState


def _builder(*, subflows=False) -> StateGraph:
    b = StateGraph(ConversationState)
    for name, fn in {"task_precheck": tasks.precheck, "task_safety": tasks.safety,
                     'task_plan':tasks.plan,'task_prepare_read':tasks.prepare_refund,
                     'task_read':tasks.read_business,'task_knowledge':tasks.knowledge,'task_unsupported':tasks.unsupported,
                     "task_schedule": tasks.schedule, "task_prepare_refund": tasks.prepare_refund,
                     "task_select_order": tasks.select_order, "task_refund": tasks.refund,
                     "task_complaint": tasks.complaint, "task_wait": tasks.wait_actions,
                     "task_commit": tasks.commit_action, "task_finish": tasks.finish}.items():
        b.add_node(name, fn)
    b.add_edge(START, "task_precheck")
    b.add_conditional_edges("task_precheck", lambda s: "multi" if s.get("plan_id") else "single",
                            {"multi": "task_safety", "single": "resolve_reference"})
    from app.graph import simple_loop
    simple_loop.install(b)
    b.add_conditional_edges('task_safety',lambda s:'shared' if task_queue.enabled(s) else ('simple' if s.get('task_plan_version')==3 else 'legacy'),
                            {'shared':'resolve_reference','simple':'sl_resolve','legacy':'task_plan'})
    b.add_node('classify_tasks',task_queue.classify)
    b.add_node('queue_take',task_queue.take)
    b.add_node('queue_complete',task_queue.complete)
    b.add_node('queue_form_wait',task_queue.wait_form)
    b.add_node('queue_bind_order',queue_contract.bind_order)
    b.add_node('queue_review',queue_contract.review)
    b.add_conditional_edges('queue_review',lambda s:'retry' if s.get('queue_review_retry') else ('complete' if queue_contract.enabled(s) else 'legacy'),
        {'retry':'main_agent','complete':'queue_complete','legacy':'log'})
    b.add_conditional_edges('classify_tasks',lambda s:'failed' if s.get('task_planning_error') else 'ready',
                            {'failed':'script_reply','ready':'queue_take'})
    b.add_conditional_edges('queue_take',queue_contract.dispatch,
        {'end':END,'escalate':'complaint_reply','fallback_script':'script_reply','knowledge':'retrieve_knowledge',
         'refund_flow':'fetch_order','business':'prepare_business_evidence'})
    b.add_conditional_edges('queue_complete',task_queue.after_complete,{'form':'queue_form_wait','next':'queue_take','log':'log'})
    b.add_edge('queue_form_wait','queue_take')
    if subflows:
        from app.graph import subflow_tasks
        subflow_tasks.install(b)
    b.add_conditional_edges('task_plan',tasks.plan_route,{'multi':'sf_schedule' if subflows else 'task_schedule','single':'resolve_reference','failed':END})
    b.add_conditional_edges("task_schedule", tasks.next_task,
                            {"refund": "task_prepare_refund", "complaint": "task_complaint",
                             'read':'task_prepare_read','knowledge':'task_knowledge','other':'task_unsupported',
                             "wait": "task_wait", "finish": "task_finish"})
    b.add_edge("task_prepare_refund", "task_select_order")
    b.add_edge('task_prepare_read','task_select_order')
    b.add_conditional_edges('task_select_order',tasks.after_selection,{'refund':'task_refund','read':'task_read'})
    b.add_edge('task_read','task_schedule')
    b.add_edge('task_knowledge','sf_schedule' if subflows else 'task_schedule')
    b.add_edge('task_unsupported','sf_schedule' if subflows else 'task_schedule')
    b.add_edge("task_refund", "task_schedule")
    b.add_edge("task_complaint", "task_schedule")
    b.add_edge("task_wait", "task_commit")
    b.add_edge("task_commit", "task_schedule")
    b.add_edge("task_finish", END)
    # 节点
    b.add_node("resolve_reference", nodes.resolve_reference)
    b.add_node("classify_intent", nodes.classify_intent)
    b.add_node("prepare_business_evidence", prepare_business_evidence)
    b.add_node("retrieve_knowledge", nodes.retrieve_knowledge)
    b.add_node("confidence_check", nodes.confidence_check)  # 实体节点(trace 门控);判断在其后条件边
    b.add_node("main_agent", nodes.main_agent)
    b.add_node("agent_tools", nodes.agent_tools)
    b.add_node("complaint_reply", nodes.complaint_reply)
    b.add_node("script_reply", nodes.script_reply)
    b.add_node("fallback_reply", nodes.fallback_reply)
    b.add_node("fetch_order", nodes.fetch_order)
    b.add_node("retrieve_policy", nodes.retrieve_policy)
    b.add_node("log", nodes.log_node)

    # 骨架:消解 → 意图 → 分流
    b.add_conditional_edges('resolve_reference',lambda s:'multi' if task_queue.enabled(s) else 'single',
                            {'multi':'classify_tasks','single':'classify_intent'})
    b.add_conditional_edges("classify_intent", route_by_intent, {
        "escalate": "complaint_reply",
        "fallback_script": "script_reply",
        "knowledge": "retrieve_knowledge",
        "refund_flow": "fetch_order",
        "business": "prepare_business_evidence",
    })
    b.add_edge("prepare_business_evidence", "main_agent")
    # 退款子流程确定性链:取单 → 检索政策 → 交主力 Agent 判能不能退
    b.add_conditional_edges('fetch_order',lambda s:'bind' if queue_contract.enabled(s) else 'policy',
        {'bind':'queue_bind_order','policy':'retrieve_policy'})
    b.add_conditional_edges('queue_bind_order',lambda s:'policy' if s.get('route')=='refund_flow' else 'business',
        {'policy':'retrieve_policy','business':'prepare_business_evidence'})
    b.add_edge("retrieve_policy", "main_agent")
    # 知识路:检索 → 生成前证据闸
    b.add_edge("retrieve_knowledge", "confidence_check")
    b.add_conditional_edges("confidence_check", confidence_gate, {
        "strong": "main_agent",
        "weak": "fallback_reply",
    })
    # 主力 Agent ReAct 环
    b.add_conditional_edges("main_agent", should_continue, {
        "continue": "agent_tools",
        "stop": "queue_review",
    })
    b.add_edge("agent_tools", "main_agent")
    # 确定性出口 → 日志 → END
    for name in ('complaint_reply','script_reply','fallback_reply'):
        b.add_conditional_edges(name,lambda s:'review' if queue_contract.enabled(s) and task_queue.current(s) else 'log',
            {'review':'queue_review','log':'log'})
    b.add_conditional_edges('log',lambda s:('form' if task_queue.current(s).get('status')=='blocked' else 'take')
                            if queue_contract.enabled(s) and task_queue.current(s) else ('next' if task_queue.current(s) else 'end'),
                            {'next':'queue_complete','end':END,'form':'queue_form_wait','take':'queue_take'})
    return b


def build_graph(checkpointer=None, *, subflows=None):
    from app.config import settings
    enabled = settings.multi_intent_subflows_enabled if subflows is None else subflows
    return _builder(subflows=enabled).compile(checkpointer=checkpointer)
