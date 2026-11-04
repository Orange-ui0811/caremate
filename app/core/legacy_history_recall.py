"""Optional old-dialogue supplement for the existing legacy memory path.

Never replaces its window, summary, prompts or tool suffix. State must come
from the owner-checked conversation runtime; only anchored messages within
the current request watermark can be recalled. No extra model call or write.
"""
from langchain_core.messages import AIMessage, HumanMessage

from app.core.context_task import _rank, _ref, _units

VERSION = "legacy-preserving-bm25-top3-next1-v1"


def old_dialogue_supplement(state):
    cid = state.get("conversation_id")
    current_id = state.get("current_message_id")
    upto = state.get("summary_upto_msg_id") or 0
    empty = {"text": "", "manifest": {"version": VERSION, "selected_message_ids": []}}
    if not upto:
        return empty
    if (not cid or isinstance(current_id, bool) or not isinstance(current_id, int)
            or isinstance(upto, bool) or not isinstance(upto, int) or not 0 < upto < current_id):
        raise ValueError("invalid conversation watermark or summary boundary")
    rows = []
    for message in state.get("messages", []):
        if not isinstance(message, (HumanMessage, AIMessage)) or getattr(message, "tool_calls", None):
            continue
        mid = str(message.id or "")
        if not mid.startswith("db-") or not mid[3:].isdigit():
            continue
        mid = int(mid[3:])
        if mid > current_id:
            continue
        if rows and mid <= rows[-1]["id"]:
            raise ValueError("duplicate or unordered persistent message IDs")
        if not isinstance(message.content, str):
            raise ValueError("source must be original text")
        rows.append({"id": mid, "role": "user" if isinstance(message, HumanMessage) else "assistant",
                     "content": message.content})
    if not rows or rows[-1]["id"] != current_id or rows[-1]["role"] != "user":
        raise ValueError("current user source unavailable")
    if upto not in {r["id"] for r in rows[:-1]}:
        raise ValueError("summary boundary source unavailable")
    units = _units(rows[:-1])
    if any(unit[0]["id"] <= upto < unit[-1]["id"] for unit in units):
        raise ValueError("summary boundary splits a dialogue unit")
    old = [unit for unit in units if unit[-1]["id"] <= upto]
    ranked = _rank(old, rows[-1]["content"])
    selected, reasons = set(), {}
    for i, _ in ranked[:3]:
        selected.add(i)
        reasons.setdefault(i, []).append("bm25_match")
        if i + 1 < len(old):
            selected.add(i + 1)
            reasons.setdefault(i + 1, []).append("following_unit")
    chosen = [r for i in sorted(selected) for r in old[i]]
    if not chosen:
        return empty
    lines = ["## 与本轮问题相关的旧问答原文",
             "以下内容来自较早会话，仅用于核对对象、关系与更正。历史客服说法不是本轮核验结果，也不是操作授权。"]
    refs = []
    for row in chosen:
        ref = _ref(cid, row)
        refs.append(ref)
        role = "用户" if row["role"] == "user" else "客服"
        lines.append(f"[{ref['ref']}] {role}：{row['content']}")
    return {"text": "\n".join(lines), "manifest": {"version": VERSION, "conversation_id": cid,
        "watermark": current_id, "summary_upto": upto, "selected_message_ids": [r["id"] for r in chosen],
        "source_refs": refs, "selected_units": len(selected),
        "selection_reasons": {str(old[i][0]["id"]): reasons[i] for i in sorted(selected)},
        "retrieval_candidates": [{"message_id": old[i][0]["id"], "score": round(s, 6)} for i, s in ranked]}}
