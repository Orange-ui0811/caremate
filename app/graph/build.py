from langgraph.graph import END, START, StateGraph

from app.graph import nodes
from app.graph import tasks
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
    b.add_edge('task_safety','task_plan')
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
    b.add_edge("resolve_reference", "classify_intent")
    b.add_conditional_edges("classify_intent", route_by_intent, {
        "escalate": "complaint_reply",
        "fallback_script": "script_reply",
        "knowledge": "retrieve_knowledge",
        "refund_flow": "fetch_order",
        "business": "prepare_business_evidence",
    })
    b.add_edge("prepare_business_evidence", "main_agent")
    # 退款子流程确定性链:取单 → 检索政策 → 交主力 Agent 判能不能退
    b.add_edge("fetch_order", "retrieve_policy")
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
        "stop": "log",
    })
    b.add_edge("agent_tools", "main_agent")
    # 确定性出口 → 日志 → END
    b.add_edge("complaint_reply", "log")
    b.add_edge("script_reply", "log")
    b.add_edge("fallback_reply", "log")
    b.add_edge("log", END)
    return b


def build_graph(checkpointer=None, *, subflows=None):
    from app.config import settings
    enabled = settings.multi_intent_subflows_enabled if subflows is None else subflows
    return _builder(subflows=enabled).compile(checkpointer=checkpointer)
