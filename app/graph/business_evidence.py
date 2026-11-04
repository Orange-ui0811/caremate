"""Opt-in deterministic freshness for read-only logistics requests.

Memory resolves the object; this node verifies ownership and obtains current
business evidence. It does not infer confirmation or execute writes.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage

from app.config import settings
from app.tools import business, engine, registry


async def prepare_business_evidence(state) -> dict:
    if not settings.context_fresh_business_evidence or state.get("intent") != "物流":
        return {"fresh_business_evidence": {}}
    original = next((m.content for m in reversed(state.get("messages", []))
                     if isinstance(m, HumanMessage)), "")
    if re.search(r"(?:不要|不用|不必|别)(?:再|继续)?(?:查|查询)", original):
        return {"fresh_business_evidence": {}}
    recalling = (re.search(r"刚才|刚刚|上次|之前|先前", original)
                 and re.search(r"结果|记录|轨迹|查到了什么|说一遍|回顾|复述", original))
    if recalling and not re.search(r"最新|现在|再查|重新查|刷新", original):
        return {"fresh_business_evidence": {}}
    query = state.get("resolved_query") or original
    ids = set(re.findall(r"(?<!\d)\d{4,}(?!\d)", query))
    record = {"message_id": state.get("current_message_id"),
              "observed_at": datetime.now(timezone.utc).isoformat(),
              "status": "needs_clarification", "call_ids": []}
    # Do not select the first of multiple objects. The unchanged Agent can ask
    # a question or perform an explicit multi-order request with normal tools.
    if len(ids) != 1:
        return {"fresh_business_evidence": record,
                "trace": {"fresh_business_evidence": record}}
    oid = next(iter(ids))
    uid, cid = state.get("user_id", ""), state.get("conversation_id", 0)
    if not business.owns_order(uid, oid):
        record["status"] = "unavailable"
        return {"fresh_business_evidence": record,
                "trace": {"fresh_business_evidence": record}}
    specs = {s.name: s for s in await registry.get_all_specs()}
    record["order_id"] = oid
    emitted = []

    async def invoke(name, args):
        spec = specs.get(name)
        if spec is None or spec.permission != "read":
            return None
        call = {"id": "fresh-" + uuid4().hex, "name": name, "args": args, "type": "tool_call"}
        result = await engine.execute_tool_call(call, cid, specs, user_id=uid,
            message_watermark=state.get("current_message_id"))
        emitted.extend([AIMessage(content="", tool_calls=[call]), result.tool_message])
        record["call_ids"].append(call["id"])
        if not result.ok:
            return None
        try:
            value = json.loads(result.tool_message.content)
        except (ValueError, TypeError):
            return None
        if isinstance(value, dict) and value.get("_context_receipt") and value.get("partial") is True:
            # Engine-generated projection after durable archiving. Do not
            # repeat a successful query because fields were externalized.
            value = value.get("fields")
        return value if isinstance(value, dict) and not value.get("error") else None

    order = await invoke("query_order", {"order_id": oid})
    tracking = order.get("tracking_no") if order else None
    if not isinstance(tracking, str) or not tracking.strip():
        record["status"] = "order_lookup_failed"
    else:
        record["tracking_no"] = tracking
        result = await invoke("query_logistics", {"tracking_no": tracking})
        record["status"] = "fresh" if result is not None else "logistics_lookup_failed"
    return {"messages": emitted, "fresh_business_evidence": record,
            "trace": {"fresh_business_evidence": record}}


def evidence_instruction(state) -> str:
    item = state.get("fresh_business_evidence") or {}
    if not item or item.get("message_id") != state.get("current_message_id"):
        return ""
    return ("\n\n## 本轮业务查询记录\n" + json.dumps(item, ensure_ascii=False)
            + "\n本轮工具回执在本条请求之后。状态fresh表示已完成本轮查询，直接依据回执作答，"
              "无需重复查询；历史客服说法不是最新业务状态。查询失败应明确说明，"
              "不得用旧答复冒充本轮查询成功。needs_clarification表示尚未确定唯一对象；"
              "若用户只说‘那笔’等指代且仍有多个可能对象，请先澄清。\n")
