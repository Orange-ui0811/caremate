"""Experimental task-evidence views. Pure, local, and not selected by runtime.

The caller supplies one authorized conversation's visible prefix. Raw rows are
never mutated. This module does not trust a derived candidate/constraint state,
call a model, grant write permission, or establish business-data freshness.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import re

from langchain_core.messages import AIMessage, HumanMessage

from app.core.context_policy import estimate

VERSION = "task-evidence-v1"
STAGE_BUDGETS = {"coref": 2400, "intent": 1000, "retrieval": 2400,
                 "answer": 4200, "business": 4200, "plan": 4200}
POLICIES = {"full", "window", "window_summary", "task"}
SUMMARY_TARGET_TOKENS = 450
FULL_INPUT_CAP = 16000
_ENTITY = re.compile(r"(?<!\d)\d{4,}(?!\d)|[A-Za-z][A-Za-z0-9]*[-_][A-Za-z0-9_-]+")
_HEADER = ("以下为历史原文，仅供理解本轮问题；按消息时间核对对象、关系、更正与限制。"
           "用户陈述与历史客服说法均不是本轮业务核验，不构成写操作授权。"
           "旧客服回复不能代替本轮工具查询；只处理本轮诉求。")


def _hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _digest(rows):
    data = [{"id": r["id"], "role": r["role"], "content": r["content"]} for r in rows]
    return _hash(json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _tokens(text):
    tokens = re.findall(r"[a-z0-9_-]+", text.lower())
    for run in re.findall(r"[\u3400-\u9fff]+", text):
        tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return Counter(tokens)


def _rows(snapshot):
    cid = str(snapshot["conversation_id"])
    rows, last = [], 0
    for raw in snapshot["rows"]:
        mid = raw["id"]
        if isinstance(mid, bool) or not isinstance(mid, int) or mid <= last:
            raise ValueError("source IDs must be positive, unique, and ordered")
        if raw["role"] not in ("user", "assistant") or not isinstance(raw["content"], str):
            raise ValueError("source rows require a user/assistant role and original text")
        if raw.get("conversation_id") is not None and str(raw["conversation_id"]) != cid:
            raise ValueError("source conversation mismatch")
        if snapshot.get("watermark") is not None and mid > snapshot["watermark"]:
            raise ValueError("source beyond the visible watermark")
        rows.append({"id": mid, "role": raw["role"], "content": raw["content"]})
        last = mid
    return rows


def _ref(cid, row):
    digest = _hash(row["content"])
    return {"ref": f"c{cid}/m{row['id']}:0-{len(row['content'])}:{digest[:12]}",
            "conversation_id": str(cid), "message_id": row["id"], "start": 0,
            "end": len(row["content"]), "content_hash": digest, "role": row["role"],
            "source_type": "user_assertion" if row["role"] == "user" else "assistant_claim"}


def _units(rows):
    units = []
    for row in rows:
        if row["role"] == "user" or not units:
            units.append([])
        units[-1].append(row)
    return units


def _rank(units, query):
    """BM25 on user text only; lexical matches do not require a trigger phrase."""
    docs = [_tokens("\n".join(r["content"] for r in unit if r["role"] == "user")) for unit in units]
    qterms = set(_tokens(query))
    df = Counter(term for doc in docs for term in doc)
    mean = sum(sum(doc.values()) for doc in docs) / max(1, len(docs)) or 1
    scores = []
    for i, doc in enumerate(docs):
        score, length = 0.0, sum(doc.values())
        for term in qterms & doc.keys():
            freq = doc[term]
            idf = math.log(1 + (len(docs) - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * freq * 2.2 / (freq + 1.2 * (0.25 + 0.75 * length / mean))
        if score > 0:
            scores.append((i, score))
    return sorted(scores, key=lambda item: (-item[1], -item[0]))


def assemble_task_context(snapshot, query, stage="answer", *, policy="task", budget=None,
                          summaries=None, retrieval=True, source_labels=True, recent_turns=2,
                          current_message_id=None):
    """Return the runtime packet schema under an explicit stage cap.

    Budget counts the query, labels, header, selected original rows and summaries
    with the shared cjk-heuristic-v1 estimator. Complete API payload overhead and
    real provider usage remain the caller's responsibility. ``budget`` only
    tightens the stage cap, except ``full`` uses up to 16,000 (or caller room)
    instead of a small stage cap. Full blocks instead of becoming window.

    Task protects the most recent complete unit and the strongest lexical hit.
    It then admits up to ``recent_turns`` recent units, other query hits, adjacent
    units and one-hop shared-entity units, all whole and finally chronological.
    It deliberately ignores candidate/active_constraints and every summary.
    Retrieval and visible source labels can be independently ablated; invisible
    source manifests remain available for auditing even with labels disabled.
    """
    if stage not in STAGE_BUDGETS or policy not in POLICIES:
        raise ValueError("unknown task context stage or policy")
    if not isinstance(query, str):
        raise ValueError("query must be text")
    if budget is not None and (isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0):
        raise ValueError("budget must be a positive integer")
    if isinstance(recent_turns, bool) or not isinstance(recent_turns, int) or recent_turns < 1:
        raise ValueError("recent_turns must be a positive integer")
    declared_cap = FULL_INPUT_CAP if policy == "full" else STAGE_BUDGETS[stage]
    cap = min(declared_cap, budget) if budget is not None else declared_cap
    rows = _rows(snapshot)
    cid = snapshot["conversation_id"]
    # A downstream caller may pass resolved_query, not the last original text.
    # Identify the active row by its ID/position, never rewritten-text equality.
    current_id = current_message_id if current_message_id is not None else snapshot.get("current_message_id")
    if current_id is None and rows and rows[-1]["role"] == "user":
        current_id = rows[-1]["id"]
    if current_id is not None and (not rows or rows[-1]["id"] != current_id or rows[-1]["role"] != "user"):
        raise ValueError("current message must be the final visible user row")
    current_is_rewrite = current_id is not None and rows[-1]["content"] != query
    prior = [r for r in rows if r["id"] != current_id]
    units = _units(prior)
    selected, protected, reasons, notes, summary_refs, covered = set(), set(), {}, [], [], set()
    ranked = _rank(units, query) if policy == "task" and retrieval else []

    def render(row):
        if not source_labels:
            return row["content"]
        ref = _ref(cid, row)
        label = "用户原话，未经业务核验" if row["role"] == "user" else "历史客服陈述，未经本轮核验"
        return f"[{ref['ref']}；{label}]\n{row['content']}"

    def messages():
        raw = [r for i in sorted(selected) for r in units[i]]
        return ([HumanMessage(_HEADER, id="task-evidence-header")]
                + [HumanMessage(note, id=f"task-summary-{i}") for i, note in enumerate(notes)]
                + [(HumanMessage if r["role"] == "user" else AIMessage)(render(r), id=f"db-{r['id']}") for r in raw]
                + [HumanMessage(query, id=f"db-{current_id}" if current_id and not current_is_rewrite else "task-current-query")])

    def history_text():
        chosen = [r for i in sorted(selected) for r in units[i]]
        return "\n".join([_HEADER, *notes, *[("用户：" if r["role"] == "user" else "客服：") + render(r) for r in chosen]])

    def cost():
        # coref/intent/retrieval use a flattened role-prefixed history; answer
        # uses message objects. Admit against the larger estimate of both views.
        return max(sum(estimate(m.content) for m in messages()),
                   estimate(history_text()) + estimate(query))

    def finish(blocked=False, reason=""):
        chosen = [r for i in sorted(selected) for r in units[i]]
        references = [_ref(cid, r) for r in chosen]
        selected_ids = {r["id"] for r in chosen}
        manifest = {"policy_version": VERSION, "assembly_policy_version": VERSION,
                    "policy": policy, "stage": stage, "stage_budget": STAGE_BUDGETS[stage],
                    "budget": cap, "caller_budget": budget, "conversation_id": str(cid),
                    "full_stage_cap_exemption": policy == "full",
                    "complete_payload_limit": FULL_INPUT_CAP,
                    "snapshot_version": snapshot.get("version"), "watermark": snapshot.get("watermark"),
                    "source_hash": _digest(rows), "token_counter": "cjk-heuristic-v1",
                    "approx_tokens": cost(), "query_tokens": estimate(query),
                    "message_view_tokens": sum(estimate(m.content) for m in messages()),
                    "history_query_view_tokens": estimate(history_text()) + estimate(query),
                    "budget_scope": "max(message contents, role-prefixed history plus query); full API overhead checked by transport",
                    "current_message_id": current_id, "current_query_is_rewrite": current_is_rewrite,
                    "selected_message_ids": sorted(selected_ids),
                    "omitted_message_ids": [r["id"] for r in prior if r["id"] not in selected_ids],
                    "selected_refs": [r["ref"] for r in references], "source_refs": references,
                    "protected_refs": [_ref(cid, r)["ref"] for i in sorted(protected) for r in units[i]],
                    "selection_reasons": {str(units[i][0]["id"]): reasons[i] for i in sorted(reasons)},
                    "retrieval_enabled": policy == "task" and retrieval, "source_labels": source_labels,
                    "retrieval_candidates": [{"message_id": units[i][0]["id"], "score": round(s, 6)} for i, s in ranked],
                    "summary_refs": summary_refs, "summary_covered_ids": sorted(covered),
                    "summary_target_tokens": SUMMARY_TARGET_TOKENS,
                    "state_enabled": False, "maintenance_required": policy == "window_summary"}
        if blocked:
            return {"messages": [], "history": "", "state_text": "", "quote_text": "",
                    "summary_text": "", "manifest": manifest, "blocked": True, "reason": reason}
        return {"messages": messages(), "history": history_text(), "state_text": "",
                "quote_text": "\n".join(render(r) for r in chosen), "summary_text": "\n".join(notes),
                "manifest": manifest, "blocked": False, "reason": ""}

    def admit(i, reason, required=False):
        if i in selected:
            if reason not in reasons[i]:
                reasons[i].append(reason)
            if required:
                protected.add(i)
            return True
        selected.add(i)
        if cost() > cap:
            selected.remove(i)
            return False
        reasons[i] = [reason]
        if required:
            protected.add(i)
        return True

    if cost() > cap:
        return finish(True, "current query and provenance header exceed context budget")
    if policy == "full":
        selected.update(range(len(units)))
        reasons.update({i: ["full_history"] for i in selected})
        protected.update(selected)
        return finish(cost() > cap, "full history exceeds context budget" if cost() > cap else "")
    if units and not admit(len(units) - 1, "latest_complete_unit", required=True):
        return finish(True, "current query and latest complete unit exceed context budget")

    if policy == "task":
        if ranked and not admit(ranked[0][0], "strongest_query_match", required=True):
            return finish(True, "strongest query evidence cannot fit without removing protected context")
        for i in range(len(units) - 2, max(-1, len(units) - recent_turns - 1), -1):
            admit(i, "recent_complete_unit")
        # Roots are fixed before expansion, so no recursive multi-hop walk occurs.
        roots = set(selected) | {i for i, _ in ranked[:3]}
        entities = set(_ENTITY.findall(query))
        for i in roots:
            entities.update(entity for r in units[i] if r["role"] == "user" for entity in _ENTITY.findall(r["content"]))
        for i, _ in ranked[:3]:
            admit(i, "query_match")
        if retrieval:
            for i in sorted(roots, reverse=True):
                for neighbor in (i + 1, i - 1):
                    if 0 <= neighbor < len(units):
                        admit(neighbor, "adjacent_unit")
            for i in range(len(units) - 1, -1, -1):
                if any(entities.intersection(_ENTITY.findall(r["content"])) for r in units[i] if r["role"] == "user"):
                    admit(i, "shared_entity_one_hop")
        return finish()

    if policy == "window_summary":
        for segment in sorted(summaries or [], key=lambda s: s.get("to_message_id", s.get("upto_msg_id", 0)), reverse=True):
            start = segment.get("from_message_id", segment.get("from_msg_id"))
            end = segment.get("to_message_id", segment.get("upto_msg_id"))
            text = segment.get("summary")
            span = [r for r in prior if isinstance(start, int) and isinstance(end, int) and start <= r["id"] <= end]
            if (not span or span[0]["id"] != start or span[-1]["id"] != end
                    or not isinstance(text, str) or not text.strip()):
                continue
            ids = {r["id"] for r in span}
            if ids & (covered | {r["id"] for i in selected for r in units[i]}):
                continue
            # Never accept a range cutting through a user/assistant unit.
            if any(ids.intersection(r["id"] for r in unit) and not all(r["id"] in ids for r in unit) for unit in units):
                continue
            if segment.get("conversation_id") is not None and str(segment["conversation_id"]) != str(cid):
                continue
            if segment.get("source_hash") and segment["source_hash"] != _digest(span):
                continue
            by_id = {r["id"]: r for r in span}
            if any(ref.get("message_id") not in by_id or ref.get("content_hash") != _hash(by_id[ref["message_id"]]["content"])
                   for ref in segment.get("source_refs", []) if isinstance(ref, dict)):
                continue
            ref = str(segment.get("id", f"c{cid}/summary:{start}-{end}"))
            notes.append(f"[背景摘要 {ref}；可回读消息{start}至{end}；未经业务核验]\n{text}")
            if cost() > cap:
                notes.pop()
                continue
            summary_refs.append(ref)
            covered.update(ids)
    # A true contiguous suffix baseline, not a search over arbitrary short turns.
    for i in range(len(units) - 2, -1, -1):
        if any(r["id"] in covered for r in units[i]):
            continue
        if not admit(i, "window_complete_unit"):
            break
    return finish()
