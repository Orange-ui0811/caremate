"""Production wiring for source-backed memory; no model calls or authorization grants."""
import json
import logging

from langchain_core.messages import HumanMessage, AIMessage

from app.config import settings
from app.core.context_policy import estimate

logger = logging.getLogger(__name__)

RECOVERY_REPLY = "当前会话资料需要进一步核实。请明确本次要处理的订单或商品，以及希望继续的操作。已经提交的操作请以系统记录为准。"


class ContextInputLimit(ValueError):
    """The complete serialized input cannot be sent within the configured budget."""


def enabled() -> bool:
    return settings.context_policy == "source"


def check_payload(payload: dict) -> int:
    """Include messages, tool definitions, response schema and tool-call arguments.

    This is an explicit heuristic, not the provider tokenizer. API usage remains
    the cost authority; no fabricated zero usage on blocked or failed requests.
    """
    parts = {k: payload[k] for k in ("messages", "tools", "functions", "response_format") if k in payload}
    used = estimate(json.dumps(parts, ensure_ascii=False, default=str))
    if used > settings.context_input_budget:
        logger.warning("context_budget_blocked estimated=%s limit=%s", used, settings.context_input_budget)
        raise ContextInputLimit("完整输入超过配置预算，未发送模型请求")
    return used


def snapshot_from_state(state):
    from app.core.context_state import build_snapshot
    snapshot = state.get("conversation_memory")
    if snapshot:
        return snapshot
    # Direct graph callers (including isolated tests) may not use the HTTP runtime.
    # Never invent persistent IDs; unanchored history cannot claim source recovery.
    rows = []
    for m in state.get("messages", []):
        if not isinstance(m, (HumanMessage, AIMessage)) or getattr(m, "tool_calls", None):
            continue
        mid = str(m.id or "")
        if not mid.startswith("db-") or not mid[3:].isdigit():
            raise ValueError("会话原文缺少稳定消息ID")
        rows.append({"id": int(mid[3:]), "role": "user" if isinstance(m, HumanMessage) else "assistant",
                     "content": m.content if isinstance(m.content, str) else ""})
    return build_snapshot(rows, state.get("conversation_id", 0))


def packet(state, query, stage="answer", budget=None):
    from app.core.context_state import assemble_context
    if state.get("memory_error"):
        return {"messages": [], "history": "", "blocked": True,
                "reason": state["memory_error"], "manifest": {"stage": stage}}
    limits = {"coref": settings.context_coref_budget, "intent": settings.context_intent_budget}
    budget = min(budget if budget is not None else limits.get(stage, settings.context_state_budget),
                 settings.context_state_budget)
    try:
        result = assemble_context(snapshot_from_state(state), query, stage=stage,
                                  budget=budget, recall=settings.context_recall,
                                  summaries=state.get("memory_segments") or [])
    except (ValueError, KeyError, TypeError) as exc:
        result = {"messages": [], "history": "", "blocked": True,
                  "reason": type(exc).__name__, "manifest": {"stage": stage}}
    logger.info("context_packet conv=%s stage=%s blocked=%s manifest=%s",
                state.get("conversation_id"), stage, result["blocked"],
                json.dumps(result.get("manifest", {}), ensure_ascii=False, default=str))
    return result


async def prepare_turn(user_id, conversation_id, message_id):
    """Snapshot exactly the visible prefix. Newer concurrent messages are excluded."""
    from app.db import repository, context_repository
    from app.core.context_state import build_snapshot
    rows = await repository.list_dialog_messages(conversation_id)
    visible = [{"id": m.id, "role": m.role, "content": m.content or ""}
               for m in rows if m.id <= message_id]
    current = await context_repository.load_snapshot(conversation_id, user_id)
    snapshot = build_snapshot(visible, conversation_id)
    saved = await context_repository.save_snapshot(
        conversation_id, user_id, snapshot, watermark=message_id,
        expected_version=current["version"] if current else 0)
    # A losing CAS does not allow this turn to overwrite a newer snapshot. Its
    # local input prefix remains replayable and write execution rechecks freshness.
    snapshot["store_version"] = saved.get("version")
    snapshot["store_saved"] = saved.get("saved", False)
    segments = await context_repository.list_summary_segments(
        conversation_id, user_id, upto_msg_id=message_id)
    return {"conversation_memory": snapshot, "memory_segments": segments,
            "current_message_id": message_id, "memory_error": ""}


async def write_denial(state, tool_name, args):
    """Memory can deny a write, never confirm it. Revalidate after interrupts."""
    if not enabled():
        return None
    if state.get("memory_error"):
        return "会话状态核验失败，本次操作未执行。"
    from app.core.context_state import write_constraint
    from app.db import repository
    try:
        cid, uid = state.get("conversation_id"), state.get("user_id", "")
        conv = await repository.get_conversation(cid)
        if conv is None or conv.user_id != uid:
            return "会话归属无法核实，本次操作未执行。"
        rows = await repository.list_dialog_messages(cid)
        latest = max((m.id for m in rows if m.role == "user"), default=0)
        if latest > state.get("current_message_id", 0):
            return "已有新的用户消息，请基于最新请求重新核实和确认，本次旧操作未执行。"
        return write_constraint(snapshot_from_state(state), tool_name, args)
    except Exception:
        logger.exception("context write validation unavailable conv=%s", state.get("conversation_id"))
        return "当前无法核实有效限制，本次操作未执行。"
