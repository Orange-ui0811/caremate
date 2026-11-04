"""Offline matched-budget context ablation. Not wired into the live graph.

State extraction and history ranking deliberately reuse the frozen prototype.
The known narrow language rules and last-two-controls behavior are not repaired
here: this experiment measures organization, not a new extraction algorithm.
"""
from langchain_core.messages import AIMessage, HumanMessage

from app.core.context_policy import estimate
from app.core.service_context import ContextBudgetError, ServiceMemory, fingerprint

CONTEXT_BUDGET = 9800
EVIDENCE_BUDGET = 2800


def validate_rows(rows):
    if len(rows) % 2:
        raise ValueError('benchmark histories must contain complete turns')
    for index, row in enumerate(rows):
        if row.get('role') != ('user' if index % 2 == 0 else 'assistant') or not isinstance(row.get('content'), str):
            raise ValueError('benchmark requires alternating user/assistant turns')


def packet_cost(messages):
    return sum(estimate(message.content) for message in messages)


def raw_packet(rows, query, *, summary='', boundary=0, budget=CONTEXT_BUDGET):
    """One bounded context view for coreference, planning and response.

    Summary is charged inside the same budget. Only the unsummarized suffix is
    eligible in M; W has no summary or boundary. Every whole-turn candidate is
    considered newest first, including after a larger turn did not fit.
    """
    validate_rows(rows)
    if boundary < 0 or boundary > len(rows) or boundary % 2:
        raise ValueError('summary boundary must end on a complete turn')
    note = '历史对话摘要：' + summary if summary else ''
    fixed = ([HumanMessage(note)] if note else []) + [HumanMessage(query)]
    if packet_cost(fixed) > budget:
        raise ContextBudgetError('summary and current query exceed context budget')
    recent, omitted = [], []
    for end in range(len(rows), boundary, -2):
        ids = [end - 1, end]
        pair = [(HumanMessage if rows[i-1]['role'] == 'user' else AIMessage)(rows[i-1]['content'], id=f'db-{i}') for i in ids]
        if packet_cost(pair + recent + fixed) <= budget:
            recent = pair + recent
        else:
            omitted.append(ids)
    body = recent + fixed
    history = '\n'.join(('用户:' if isinstance(m, HumanMessage) else '客服:') + m.content for m in recent)
    if note:
        history += '\n' + note
    return {'body': body, 'coref_history': history.strip(), 'summary': summary,
            'boundary': boundary, 'omitted_turns': omitted, 'evidence_refs': [],
            'protected_refs': [], 'budget': budget, 'approx_context_tokens': packet_cost(body),
            'source_sha256': fingerprint(rows), 'source_count': len(rows)}


def state_packet(memory: ServiceMemory, query, *, recall, budget=CONTEXT_BUDGET):
    """S and R differ only in optional ranked source evidence admission."""
    validate_rows(memory.rows)
    return memory.assemble(query, budget=budget,
                           evidence_budget=EVIDENCE_BUDGET if recall else 0)
