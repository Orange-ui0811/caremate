from copy import deepcopy

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.config import settings
from app.core import memory
from app.core.legacy_history_recall import old_dialogue_supplement
from app.graph import nodes


def state():
    messages = []
    pairs = [("工作室那单710001先记下", "登记了"), ("不对，应为710003", "收到更正")]
    pairs += [(f"包装问题{i}", f"包装说明{i}") for i in range(10)]
    for q,a in pairs:
        messages.append(HumanMessage(q,id=f"db-{len(messages)+1}"))
        messages.append(AIMessage(a,id=f"db-{len(messages)+1}"))
    messages.append(HumanMessage("工作室那单到哪了",id="db-25"))
    return {"conversation_id":123,"current_message_id":25,"summary_upto_msg_id":8,
            "summary":"此前讨论过订单710001，后更正为710003", "messages":messages}


@pytest.fixture(autouse=True)
def legacy(monkeypatch):
    monkeypatch.setattr(settings,"context_policy","legacy")
    monkeypatch.setattr(settings,"context_legacy_recall_enabled",False)
    # 本文件回归旧的 archived 方案，显式固定配置，避免受部署默认值影响。
    monkeypatch.setattr(settings,"context_legacy_recall_mode","archived")
    monkeypatch.setattr(settings,"context_window_turns",8)


def test_off_exactly_preserves_existing_history_and_default_parameters():
    s=state();window=memory.build_window(s['messages'],8,settings.context_window_max_tokens)
    expected=memory.summary_line(s['summary'])+'\n'+'\n'.join(
        ('用户' if isinstance(m,HumanMessage) else '客服')+':'+m.content for m in window[:-1][-6:])
    assert nodes._history_text(s)==expected
    assert settings.summary_trigger_messages==30 and settings.context_window_turns==8
    assert settings.context_window_max_tokens==3000


def test_only_old_match_and_next_unit_are_added(monkeypatch):
    s=state();original=deepcopy(s)
    r=old_dialogue_supplement(s)
    assert r['manifest']['selected_message_ids']==[1,2,3,4]
    assert '收到更正' in r['text']
    before=nodes._history_text(s)
    monkeypatch.setattr(settings,'context_legacy_recall_enabled',True)
    assert nodes._history_text(s)==before+'\n\n'+r['text']
    assert nodes._history_text(s,stage='intent')==before+'\n\n'+r['text']
    assert s==original


def test_answer_retains_original_messages_system_and_tool_pairs(monkeypatch):
    s=state()
    s['messages'] += [AIMessage('',tool_calls=[{'id':'t1','name':'query_order','args':{'order_id':'710003'}}]),
                      ToolMessage('合成回执',tool_call_id='t1')]
    before=nodes._agent_messages(s)
    supplement='\n\n'+old_dialogue_supplement(s)['text']
    monkeypatch.setattr(settings,'context_legacy_recall_enabled',True)
    after=nodes._agent_messages(s)
    assert len(after)==len(before)
    assert after[0]==before[0]
    changed=[]
    for i,(a,b) in enumerate(zip(after,before)):
        if a!=b:
            changed.append(i)
            assert a.id==b.id==nodes.TURN_CTX_ID
            assert a.content.replace(supplement,'',1)==b.content
    assert len(changed)==1
    assert [(m.type,getattr(m,'tool_call_id',None)) for m in after]==[(m.type,getattr(m,'tool_call_id',None)) for m in before]


def test_no_boundary_or_no_match_adds_nothing(monkeypatch):
    for boundary,query in [(0,'工作室'),(8,'XYZUNMATCHED')]:
        s=state();s['summary_upto_msg_id']=boundary;s['messages'][-1].content=query
        before=nodes._history_text(s);before_agent=nodes._agent_messages(s)
        monkeypatch.setattr(settings,'context_legacy_recall_enabled',True)
        assert not old_dialogue_supplement(s)['text']
        assert nodes._history_text(s)==before
        assert nodes._agent_messages(s)==before_agent
        monkeypatch.setattr(settings,'context_legacy_recall_enabled',False)


def test_future_and_current_tool_output_cannot_be_recalled():
    s=state();s['messages'].append(HumanMessage('工作室正确号码710008',id='db-26'))
    assert '710008' not in old_dialogue_supplement(s)['text']
    s['summary_upto_msg_id']=1
    with pytest.raises(ValueError,match='splits'):
        old_dialogue_supplement(s)


def test_source_policy_and_missing_anchors_do_not_change_default(monkeypatch):
    monkeypatch.setattr(settings,'context_legacy_recall_enabled',True)
    monkeypatch.setattr(settings,'context_policy','source')
    assert nodes._legacy_recall_text(state())==''
    monkeypatch.setattr(settings,'context_policy','legacy')
    s=state();s['messages'][-1].id='unanchored'
    assert nodes._legacy_recall_text(s)==''
