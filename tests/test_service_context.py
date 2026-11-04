import pytest
from app.core.service_context import ServiceMemory,ContextBudgetError,project_tool_result
from app.core.context_policy import EvidenceState,assemble,estimate


def rows(*texts):
    return [r for text in texts for r in ({'role':'user','content':text},{'role':'assistant','content':'已收到。'})]


def test_oversized_latest_correction_keeps_literal_source_and_bound():
    data=rows('登记订单111111。','订单号111111报错了，应为222222。'+'普通包装背景。'*1800)
    old=EvidenceState();old.advance(data)
    assert '222222' not in old.render()
    m=ServiceMemory('a');m.advance(data,'a');p=m.assemble('查最后更正的订单。',budget=1200)
    text='\n'.join(x.content for x in p['body'])
    assert m.candidate['value']=='222222' and '222222' in text
    assert p['approx_context_tokens']<=1200 and p['omitted_turns']
    assert m.candidate['ref'] in p['protected_refs']
    for e in m.evidence:assert data[e.seq-1]['content'][e.start:e.end]==e.text


def test_scope_immutability_and_causal_incremental_replay():
    data=rows('这次只查订单111111。','目前仍没有选定订单，两单都有可能。')
    a=ServiceMemory('x');a.advance(data[:2],'x');assert a.candidate['value']=='111111'
    assert all(e.seq<=2 for e in a.evidence)
    a.advance(data,'x');assert a.candidate['value'] is None
    b=ServiceMemory('x');b.advance(data,'x');assert a.snapshot()==b.snapshot()
    with pytest.raises(ValueError):a.advance(data,'another-user')
    changed=[dict(r) for r in data];changed[0]['content']='changed'
    with pytest.raises(ValueError):a.advance(changed,'x')


def test_assistant_claim_and_hypothetical_never_update_user_candidate():
    data=rows('这次只查订单111111。','如果订单号报错，应为222222吗？')
    data[-1]['content']='这次只查订单333333。退款已经批准。'
    m=ServiceMemory('x');m.advance(data,'x');p=m.assemble('现在查哪笔？')
    assert m.candidate['value']=='111111'
    assert '未作业务验证' in p['coref_history']
    assert '不授予退款' in p['coref_history']


def test_no_silent_drop_of_protected_state_in_tiny_budget():
    m=ServiceMemory('x');m.advance(rows('这次只查订单111111。先别提交退款。'),'x')
    with pytest.raises(ContextBudgetError):m.assemble('查订单。',budget=20)
    with pytest.raises(ContextBudgetError):m.assemble('用户问题。'*1000,budget=300)


def test_long_unpunctuated_fragments_are_not_treated_as_complete_updates():
    data=rows('这次只查订单111111。','背景'*2000+'正确订单号是222222'+'背景'*2000)
    m=ServiceMemory('x');m.advance(data,'x')
    assert m.candidate['value'] is None
    assert any(not e.complete and '222222' in e.text for e in m.evidence)
    for e in m.evidence:assert data[e.seq-1]['content'][e.start:e.end]==e.text


def test_tool_projection_preserves_verified_payload_and_source_receipt():
    data={'order_id':'111111','tracking_no':'SF11111188','status':'运输中','debug':'后台日志。'*5000}
    r=project_tool_result(data,budget=400)
    assert r['value']['order_id']=='111111' and 'debug' not in r['value']
    assert r['source_sha256']==r['value']['_source_ref']
    assert estimate(str(r['value']))<400
    assert data['debug']  # source was not mutated
    with pytest.raises(ContextBudgetError):project_tool_result({'care':'重要条款。'*5000},budget=400)


def test_regression_and_pressure_packets_fit_and_sources_are_real():
    from scripts.service_context_cases import dataset
    cases=dataset()['cases'];assert len(cases)==24
    for c in cases:
        m=ServiceMemory(c['id'])
        for probe in c['probes']:
            prefix=c['history'][:2*probe['at_turn']];m.advance(prefix,c['id'])
            p=m.assemble(probe['query'])
            assert p['approx_context_tokens']<=9800
            assert all(e.seq<=len(prefix) for e in m.evidence)
