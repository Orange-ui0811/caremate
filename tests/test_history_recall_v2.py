from copy import deepcopy
from unittest.mock import patch
import pytest
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
from app.config import settings
from app.core.history_recall import dialogue_supplement,persistent_ids
from app.graph import nodes

def state(boundary=0):
    pairs=[('青禾点订单710003，白露点订单710008。','登记了'),
           ('青禾点改名书房店，白露点名称不变。','已记录改名')]
    pairs += [(f'一般包装说明{i}',f'保存包装资料{i}') for i in range(12)]
    messages=[]
    for q,a in pairs:
        messages.extend([HumanMessage(q,id=f'db-{len(messages)+1}'),AIMessage(a,id=f'db-{len(messages)+2}')])
    messages.append(HumanMessage('书房店那笔订单到哪了',id=f'db-{len(messages)+1}'))
    return {'conversation_id':123,'current_message_id':29,'summary_upto_msg_id':boundary,
            'summary':'曾登记两处订单' if boundary else '', 'messages':messages}

@pytest.fixture(autouse=True)
def new_mode(monkeypatch):
    monkeypatch.setattr(settings,'context_policy','legacy')
    monkeypatch.setattr(settings,'context_legacy_recall_mode','all_visible_neighbors')
    monkeypatch.setattr(settings,'context_legacy_recall_enabled',False)

def test_recall_before_first_summary_and_previous_neighbor():
    s=state();original=deepcopy(s);r=dialogue_supplement(s)
    ids=r['manifest']['selected_message_ids']
    assert {1,2,3,4}.issubset(ids) and 29 not in ids
    assert '青禾点订单710003' in r['text'] and '改名书房店' in r['text']
    assert s==original
    assert 'previous_unit' in r['manifest']['selection_reasons']['1']

def test_searches_unarchived_middle_and_dedup_uses_ids():
    s=state(4)
    s['messages'][12].content='北仓库对应订单710001。'
    s['messages'][-1].content='北仓库那笔到哪了'
    r=dialogue_supplement(s,visible_message_ids=[13,25,26,27,28])
    ids=r['manifest']['selected_message_ids']
    assert 14 in ids and 13 not in ids
    assert 13 in r['manifest']['deduplicated_message_ids']
    assert not set(ids)&{13,25,26,27,28}
    # Same text at a different source ID must not be treated as duplicate.
    s['messages'][10].content=s['messages'][12].content
    r=dialogue_supplement(s,visible_message_ids=[13])
    assert 11 in r['manifest']['selected_message_ids']

def test_actual_history_and_answer_use_different_visible_windows(monkeypatch):
    s=state()
    before_history=nodes._history_text(s);before_answer=nodes._agent_messages(s)
    monkeypatch.setattr(settings,'context_legacy_recall_enabled',True)
    assert '青禾点订单710003' in nodes._history_text(s)
    assert nodes._history_text(s)!=before_history
    # The answer window already contains every source; it must add none.
    assert nodes._agent_messages(s)==before_answer
    assert persistent_ids(before_answer)==list(range(1,30))

def test_answer_keeps_system_tool_suffix_and_deduplicates_raw(monkeypatch):
    s=state(8);s['messages'] += [AIMessage('',tool_calls=[{'id':'t','name':'query_order','args':{'order_id':'710003'}}]),ToolMessage('合成回执',tool_call_id='t')]
    original=nodes._agent_messages(s)
    supplement=dialogue_supplement(s,visible_message_ids=persistent_ids(original))['text']
    monkeypatch.setattr(settings,'context_legacy_recall_enabled',True)
    actual=nodes._agent_messages(s)
    assert len(actual)==len(original) and actual[0]==original[0] and actual[-2:]==original[-2:]
    for a,b in zip(actual,original):
        if a!=b:assert a.id==b.id=='turn-ctx' and a.content.replace('\n\n'+supplement,'',1)==b.content

def test_future_tool_and_invalid_sources():
    s=state();s['messages'].append(HumanMessage('书房店正确号码710007',id='db-30'))
    assert '710007' not in dialogue_supplement(s)['text']
    with pytest.raises(ValueError,match='visible'):dialogue_supplement(s,visible_message_ids=[999])
    s['messages']=[m for m in s['messages'] if m.id!='db-29']
    with pytest.raises(ValueError,match='current'):dialogue_supplement(s)

async def test_real_graph_mock_http_maintenance_and_independent_dedup_audit(tmp_path,monkeypatch):
    from scripts import eval_context_recall_v2 as exp
    from scripts import audit_context_keep3 as old_auditor
    from scripts.audit_context_recall_v2 import audit
    from tests.test_context_keep3 import test_actual_workflow_and_keep3_boundary_all_five_arms
    original_prepare=exp.previous.prepare
    with exp.binding(),patch.object(old_auditor,'audit',audit),patch.object(exp.previous,'prepare',
            lambda output,**kw:original_prepare(output,arms=exp.ARMS,**kw)):
        with monkeypatch.context() as local:
            await test_actual_workflow_and_keep3_boundary_all_five_arms(tmp_path,local)
    result=exp.finalize(tmp_path)
    assert result['complete'] and result['audit_passed'],result['issues']
    for arm in exp.ARMS:assert result['groups'][arm]['task']['pass']==3
