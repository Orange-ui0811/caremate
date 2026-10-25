import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from app.core.context_policy import EvidenceState, assemble, recent_pairs, estimate, recall_history
from scripts.eval_context_policy_v3 import Caller, sanitized_reply, score


def turns(*user_texts):
    return [r for text in user_texts for r in ({'role':'user','content':text},{'role':'assistant','content':'已收到。'})]


def test_correction_is_source_scoped_and_never_authorizes_action():
    rows=turns('订单12345报错了，应为98765；商品型号ABC-9。','先别提交退款。')
    state=EvidenceState();state.advance(rows)
    assert state.corrections[0]['new']=='98765'
    assert state.corrections[0]['source_id']==1
    assert '不代表订单全局失效' in state.corrections[0]['scope']
    assert '不授予提交/退款权限' in state.render()
    assert state.entities['ABC-9']['first_source']==1
    assert '先别提交退款' in state.render()


def test_explicit_target_clear_and_incremental_replay():
    rows=turns('现在先处理订单12345。','目前仍没有选定订单，两单都有可能。')
    a=EvidenceState();a.advance(rows[:2]);assert a.current_candidate=='12345'
    a.advance(rows);assert a.current_candidate is None
    b=EvidenceState();b.advance(rows)
    assert a.snapshot()==b.snapshot()
    with pytest.raises(ValueError):a.advance(rows[:2])


def test_recent_window_preserves_turns_and_budget():
    rows=turns(*['长消息'+('内容'*100)+str(i) for i in range(20)])
    ids=recent_pairs(rows,800)
    assert len(ids)%2==0 and ids[-1]==len(rows)
    assert ids[0]%2==1
    from app.core.context_policy import source_text
    assert sum(estimate(source_text(rows,ids[i:i+2])) for i in range(0,len(ids),2))<=800
    assert recent_pairs(rows,1)==[]


def test_history_recall_recovers_early_source_without_future_access():
    rows=turns('最早的订单号12345，型号ABC-1。',*['普通商品咨询。']*30)
    state=EvidenceState();state.advance(rows)
    context=assemble(rows,'最早那笔订单到哪了？',state,True,window_budget=100)
    assert any(1 in e['source_ids'] for e in context['retrieved'])
    assert all(max(e['source_ids'])<=len(rows) for e in context['retrieved'])
    assert recall_history(rows,'运费规则是什么',state,set())==[]


@pytest.mark.parametrize('text', ['作废号码12345未使用。','12345已作废，查询98765。','不使用订单12345。'])
def test_scoring_obsolete_exception(text):
    cleaned,excluded=sanitized_reply(text)
    assert '12345' not in cleaned and '12345' in excluded


@pytest.mark.parametrize('text',['查询订单12345。','订单12345没有收到货。','不要退款，但查询订单12345。'])
def test_scoring_does_not_excuse_unrelated_live_orders(text):
    assert sanitized_reply(text)==(text,[])


def test_wrong_action_cannot_be_rescued_by_good_answer():
    case={'all_orders':['12345','98765'],'history':[]}
    probe={'expected':{'action':'logistics','order_id':'98765','sku':'','date':''}}
    r={'coref':{'ok':True,'value':'查98765','calls':[{}]},
       'plan':{'ok':True,'value':{'action':'logistics','order_id':'12345'}},
       'reply':{'ok':True,'value':'订单98765运单SF9876588，作废号码12345未使用。'},'context_text':''}
    assert score(case,probe,r)['task_rule_v3'] is False


async def test_call_cache_keeps_empty_and_never_semantic_retries(tmp_path,monkeypatch):
    from scripts import eval_context_policy_v3 as run
    count=0
    class Fake:
        async def ainvoke(self,messages):
            nonlocal count
            count+=1
            return AIMessage('',response_metadata={'finish_reason':'stop'},usage_metadata={'input_tokens':12,'output_tokens':0,'total_tokens':12})
    monkeypatch.setattr(run.llm,'get_chat_model',lambda **kw:Fake())
    caller=Caller(tmp_path)
    r=await caller.call('one','reply',[AIMessage('test')])
    assert r['status']=='empty' and count==1
    assert await caller.call('one','reply',[AIMessage('test')])==r and count==1
    with pytest.raises(ValueError):await caller.call('one','reply',[AIMessage('changed')])


async def test_nontransport_exception_is_not_retried(tmp_path,monkeypatch):
    from scripts import eval_context_policy_v3 as run
    class Fake:
        async def ainvoke(self,messages): raise ValueError('bad parse')
    monkeypatch.setattr(run.llm,'get_chat_model',lambda **kw:Fake())
    r=await Caller(tmp_path).call('bad','reply',[AIMessage('test')])
    assert r['status']=='validation_error' and len(r['attempts'])==1


async def test_auto_tool_arguments_validated_and_immutable_attempt_recovers(tmp_path,monkeypatch):
    from scripts import eval_context_policy_v3 as run
    from langchain_core.messages import SystemMessage, HumanMessage
    count=0
    class Fake:
        def bind_tools(self,schemas,tool_choice):
            assert tool_choice=='auto'
            return self
        async def ainvoke(self,messages):
            nonlocal count
            count+=1
            assert isinstance(messages[0],SystemMessage)
            assert all(not isinstance(m,SystemMessage) for m in messages[1:])
            assert '请调用 Summary' in messages[0].content
            return AIMessage('',tool_calls=[{'name':'Summary','args':{'summary':'订单12345'},'id':'t','type':'tool_call'}],response_metadata={'finish_reason':'tool_calls'},usage_metadata={'input_tokens':10,'output_tokens':10,'total_tokens':20})
    monkeypatch.setattr(run.llm,'get_chat_model',lambda **kw:Fake())
    caller=Caller(tmp_path)
    messages=[SystemMessage('summarize'),HumanMessage('test')]
    r=await caller.call('summary','summary',messages,run.Summary)
    assert r['ok'] and r['value']=={'summary':'订单12345'}
    (tmp_path/'calls/summary.json').unlink() # simulate crash between attempt and final-cache writes
    recovered=await caller.call('summary','summary',messages,run.Summary)
    assert recovered['attempts']==r['attempts'] and count==1


def test_all_regression_cases_fit_declared_candidate_prompt_budget():
    from scripts.eval_context_policy_v3 import OLD_DATA,READER_SYSTEM,ServicePlan
    dataset=json.loads(OLD_DATA.read_text(encoding='utf-8'))
    for case in dataset['cases']:
        for probe in case['probes']:
            rows=case['history'][:probe['at_turn']*2]
            state=EvidenceState();state.advance(rows)
            for include_state,retrieval in [(False,False),(True,False),(True,True)]:
                c=assemble(rows,probe['query'],state if include_state else None,retrieval)
                approx=sum(estimate(m.content) for m in c['body'])+estimate(READER_SYSTEM)+estimate(json.dumps(ServicePlan.model_json_schema()))+512
                assert approx<=16000
