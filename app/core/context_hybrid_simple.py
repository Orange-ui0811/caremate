"""Experimental rolling-summary view; not enabled in the service.

No token-based admission or truncation. The caller supplies an authorized,
watermarked source prefix and the rolling summary covering its older portion.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from app.core.context_task import _HEADER, _digest, _rank, _ref, _rows, _units
from app.core.context_policy import estimate

VERSION = "rolling30-keep6-bm25-top3-next1-v1"


def assemble_simple_context(snapshot, query, stage="answer", *, summary="",
                            summary_upto=0, retrieval=True):
    rows = _rows(snapshot)
    if not rows or rows[-1]["role"] != "user":
        raise ValueError("current user message must end the visible prefix")
    current = rows[-1]
    if not isinstance(query, str):
        raise ValueError("query must be text")
    if isinstance(summary_upto, bool) or not isinstance(summary_upto, int) or summary_upto < 0:
        raise ValueError("invalid summary boundary")
    prior = rows[:-1]
    if summary_upto and (not summary or summary_upto not in {r["id"] for r in prior}):
        raise ValueError("summary requires an existing historical boundary")
    if summary and not summary_upto:
        raise ValueError("summary lacks source boundary")
    units = _units(prior)
    if any(unit[0]["id"] <= summary_upto < unit[-1]["id"] for unit in units):
        raise ValueError("summary boundary splits a dialogue unit")
    old = [unit for unit in units if unit[-1]["id"] <= summary_upto]
    recent = [unit for unit in units if unit[0]["id"] > summary_upto]
    ranked = _rank(old, query) if retrieval else []
    selected, reasons = set(), {}
    for index, _ in ranked[:3]:
        selected.add(index)
        reasons.setdefault(index, []).append("bm25_top3")
        if index + 1 < len(old):
            selected.add(index + 1)
            reasons.setdefault(index + 1, []).append("following_unit")
    chosen_old = [r for i in sorted(selected) for r in old[i]]
    chosen_recent = [r for unit in recent for r in unit]
    chosen = chosen_old + chosen_recent
    cid = snapshot["conversation_id"]

    def render(row):
        ref = _ref(cid, row)
        label = "用户原话，未经业务核验" if row["role"] == "user" else "历史客服陈述，未经本轮核验"
        return f"[{ref['ref']}；{label}]\n{row['content']}"

    background = (f"[滚动摘要；覆盖至消息{summary_upto}；仅作背景，具体关系、更正须结合原文核对]\n{summary}"
                  if summary else "")
    messages = [HumanMessage(_HEADER, id="simple-history-header")]
    if background:
        messages.append(HumanMessage(background, id="simple-rolling-summary"))
    messages.extend((HumanMessage if r["role"] == "user" else AIMessage)(render(r), id=f"db-{r['id']}")
                    for r in chosen)
    messages.append(HumanMessage(query, id=f"db-{current['id']}" if current["content"] == query else "simple-rewritten-query"))
    history = "\n".join([_HEADER, *([background] if background else []),
                          *[("用户：" if r["role"] == "user" else "客服：") + render(r) for r in chosen]])
    refs = [_ref(cid, row) for row in chosen]
    manifest = {"version": VERSION, "policy": "hybrid6" if retrieval else "summary6", "stage": stage,
        "source_hash": _digest(rows), "watermark": snapshot.get("watermark", current["id"]),
        "current_message_id": current["id"], "summary_upto": summary_upto,
        "summary_hash": _digest([{"id": summary_upto, "role": "summary", "content": summary}]) if summary else None,
        "retrieval_enabled": retrieval, "token_admission": False, "budget": None,
        "retrieval_candidates": [{"message_id": old[i][0]["id"], "score": round(s, 6)} for i, s in ranked],
        "selection_reasons": {str(old[i][0]["id"]): reasons[i] for i in sorted(selected)},
        "old_message_ids": [r["id"] for r in chosen_old], "recent_message_ids": [r["id"] for r in chosen_recent],
        "selected_message_ids": [r["id"] for r in chosen], "source_refs": refs,
        "old_units": len(selected), "recent_units": len(recent),
        "approx_tokens": max(sum(estimate(m.content) for m in messages), estimate(history) + estimate(query))}
    return {"messages": messages, "history": history, "summary_text": background,
            "state_text": "", "quote_text": "\n".join(render(r) for r in chosen),
            "manifest": manifest, "blocked": False, "reason": ""}
