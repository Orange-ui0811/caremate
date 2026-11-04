"""Experimental stage budgets, reusing the existing source-state assembler.

This module has no model, storage or settings side effects. It is deliberately
not selected by the live runtime. Callers must supply one authorized visible
conversation prefix, exactly as for ``context_state.assemble_context``.

The source variant changes only the stage/budget projection. The window variant
is an explicit ablation: remove derived state and historical source recall, but
keep the same raw turn selection, summary validation/admission and estimator.
Both inherit the original extractor and recall limitations. A small window is
not guaranteed to preserve facts or constraints that its ablation omits.
"""
from __future__ import annotations

from app.core.context_state import assemble_context

VERSION = "compact-stage-budget-v1"
STAGE_BUDGETS = {
    "coref": 1800,
    "intent": 1000,
    "plan": 4200,
    "answer": 4200,
    "retrieval": 4200,
    "business": 4200,
}
POLICIES = {"compact_window", "compact_source"}


def assemble_compact_context(snapshot, query, stage="answer", *,
                             policy="compact_source", budget=None, summaries=None):
    """Return the existing packet schema under an explicit per-stage cap.

    ``budget`` is a further caller cap, e.g. room left after the system prompt,
    tool schema and live tool receipts. It never expands the declared stage cap.
    The current query, source-state header, quoted evidence, admitted summaries
    and complete raw turns all count, using ``cjk-heuristic-v1``. Complete API
    payload limits remain the caller/transport's responsibility; actual API
    usage, not this estimator, is the Token cost authority.

    Source essential evidence that exceeds the cap returns ``blocked`` with no
    model messages. No original rows, derived state or source IDs are mutated.
    Window is a diagnostic ablation, not a substitute for backend write guards.
    """
    if stage not in STAGE_BUDGETS or policy not in POLICIES:
        raise ValueError("unknown compact context stage or policy")
    if budget is not None and (isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0):
        raise ValueError("caller budget must be a positive integer")
    cap = min(STAGE_BUDGETS[stage], budget) if budget is not None else STAGE_BUDGETS[stage]
    underlying_stage = "answer" if stage == "plan" else stage
    view = snapshot
    recall = "conditional"
    if policy == "compact_window":
        # Shallow copy is sufficient: assemble_context only reads the snapshot.
        # Preserve rows/source hash/index/identity for identical provenance and
        # summary checks; the absent state/recall is the declared ablation.
        view = {**snapshot, "candidate": None, "active_constraints": [], "events": []}
        recall = "off"
    packet = assemble_context(view, query, stage=underlying_stage, budget=cap,
                              recall=recall, summaries=summaries)
    packet["manifest"].update(
        assembly_policy_version=VERSION,
        compact_policy=policy,
        stage=stage,
        underlying_stage=underlying_stage,
        stage_budget=STAGE_BUDGETS[stage],
        caller_budget=budget,
        shared_assembly="context_state.assemble_context",
        state_enabled=policy == "compact_source",
        summary_admission="unchanged: per-segment min(1400, effective_budget//4), no truncation",
        change_scope=("stage projection and budget only" if policy == "compact_source" else
                      "same stage budget; derived state and historical recall removed"),
    )
    return packet
