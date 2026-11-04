"""All-visible BM25 history supplement with neighbors and per-node ID dedup.

Only owner-checked runtime state may be supplied. Original rows and model/tool
suffixes are never mutated; summaries are not treated as visible raw evidence.
"""
from langchain_core.messages import AIMessage, HumanMessage

from app.core.context_task import _rank, _ref, _units

VERSION = "all-visible-bm25-top3-prev-next-node-dedup-v1"


def persistent_ids(messages):
    result = []
    for message in messages:
        mid = str(getattr(message, "id", "") or "")
        if (isinstance(message, (HumanMessage, AIMessage))
                and not getattr(message, "tool_calls", None)
                and isinstance(message.content, str) and message.content
                and mid.startswith("db-") and mid[3:].isdigit()):
            result.append(int(mid[3:]))
    return sorted(set(result))


def dialogue_supplement(state, *, visible_message_ids=()):
    cid, watermark = state.get("conversation_id"), state.get("current_message_id")
    if not cid or isinstance(watermark, bool) or not isinstance(watermark, int) or watermark <= 0:
        raise ValueError("invalid conversation watermark")
    rows = []
    for message in state.get("messages", []):
        if not isinstance(message, (HumanMessage, AIMessage)) or getattr(message, "tool_calls", None):
            continue
        mid = str(message.id or "")
        if not mid.startswith("db-") or not mid[3:].isdigit():
            continue
        mid = int(mid[3:])
        if mid > watermark:
            continue
        if mid <= 0 or rows and mid <= rows[-1]["id"]:
            raise ValueError("duplicate or unordered persistent message IDs")
        if not isinstance(message.content, str):
            raise ValueError("source must be original text")
        rows.append({"id":mid, "role":"user" if isinstance(message,HumanMessage) else "assistant",
                     "content":message.content})
    if not rows or rows[-1]["id"] != watermark or rows[-1]["role"] != "user":
        raise ValueError("current user source unavailable")
    known = {row["id"] for row in rows}
    visible = set(visible_message_ids)
    if any(isinstance(mid,bool) or not isinstance(mid,int) or mid not in known for mid in visible):
        raise ValueError("node-visible source ID unavailable")
    units = _units(rows[:-1])
    ranked = _rank(units, rows[-1]["content"])
    selected, reasons = set(), {}
    for index, _ in ranked[:3]:
        for i, reason in ((index,"bm25_match"),(index-1,"previous_unit"),(index+1,"following_unit")):
            if 0 <= i < len(units):
                selected.add(i)
                reasons.setdefault(i,[]).append(reason)
    candidates = [row for i in sorted(selected) for row in units[i]]
    chosen = [row for row in candidates if row["id"] not in visible]
    refs = [_ref(cid,row) for row in chosen]
    manifest = {"version":VERSION, "conversation_id":cid, "watermark":watermark,
        "candidate_scope":"all_visible_prior_dialogue", "visible_message_ids":sorted(visible),
        "selected_message_ids":[row["id"] for row in chosen],
        "deduplicated_message_ids":[row["id"] for row in candidates if row["id"] in visible],
        "expanded_message_ids":[row["id"] for row in candidates],
        "source_refs":refs, "selected_units":sum(any(r["id"] not in visible for r in units[i]) for i in selected),
        "selection_reasons":{str(units[i][0]["id"]):reasons[i] for i in sorted(selected)},
        "retrieval_candidates":[{"message_id":units[i][0]["id"],"score":round(score,6)} for i,score in ranked]}
    if not chosen:
        return {"text":"", "manifest":manifest}
    lines = ["## 与本轮问题相关的旧问答原文",
        "以下内容来自较早会话，仅用于核对对象、关系与更正。历史客服说法不是本轮核验结果，也不是操作授权。"]
    for row,ref in zip(chosen,refs):
        role = "用户" if row["role"] == "user" else "客服"
        lines.append(f"[{ref['ref']}] {role}：{row['content']}")
    return {"text":"\n".join(lines),"manifest":manifest}
