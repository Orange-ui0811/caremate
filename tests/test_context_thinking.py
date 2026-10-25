import json
from collections import Counter
import pytest
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from scripts import eval_context_thinking as run


def test_selection_and_only_mode_differs_in_wire_contract():
    d=run.dataset();assert len(d['cases'])==16
    assert set(Counter(c['kind'] for c in d['cases']).values())=={2}
    assert all(c['turns'] in (24,64) for c in d['cases'])
    messages=[SystemMessage('test'),HumanMessage('question')]
    a,off=run.wire_contract('off','plan',messages,run.ServicePlan)
    b,on=run.wire_contract('on','plan',messages,run.ServicePlan)
    assert [m.model_dump() for m in a]==[m.model_dump() for m in b]
    assert off['max_tokens']==on['max_tokens']==8192
    assert {k:v for k,v in off.items() if k not in ('mode','extra_body')}=={k:v for k,v in on.items() if k not in ('mode','extra_body')}
    assert len([m for m in a if isinstance(m,SystemMessage)])==1


async def test_empty_response_is_cached_without_semantic_retry(tmp_path,monkeypatch):
    count=0
    class Fake:
        extra_body={}
        async def ainvoke(self,msg):
            nonlocal count
            count+=1
            return AIMessage('',response_metadata={'finish_reason':'stop'},usage_metadata={'input_tokens':10,'output_tokens':0,'total_tokens':10})
    monkeypatch.setattr(run.llm,'get_chat_model',lambda **kw:Fake())
    caller=run.Caller(tmp_path)
    args=('empty','off','reply',[SystemMessage('test'),HumanMessage('question')])
    a=await caller.call(*args);b=await caller.call(*args)
    assert a==b and a['status']=='empty' and count==1


async def test_full_causal_replay_and_tamper_detection(tmp_path,monkeypatch):
    class Fake:
        extra_body={};schema=None
        def bind_tools(self,schema,tool_choice):
            self.schema=schema[0];assert tool_choice=='auto';return self
        async def ainvoke(self,msg):
            usage={'input_tokens':10,'output_tokens':30,'total_tokens':40,
                   'output_token_details':{'reasoning':20 if self.extra_body['enable_thinking'] else 0}}
            if self.schema:
                val={'summary':'用户查询订单610010。'} if self.schema==run.Summary else {'action':'logistics','order_id':'610010','sku':'','answer':''}
                return AIMessage('',tool_calls=[{'name':self.schema.__name__,'args':val,'id':'fake','type':'tool_call'}],response_metadata={'finish_reason':'tool_calls','model_name':'fake'},usage_metadata=usage)
            return AIMessage('订单610010，运单SF61001088，运输中。',response_metadata={'finish_reason':'stop','model_name':'fake'},usage_metadata=usage)
    monkeypatch.setattr(run.llm,'get_chat_model',lambda **kw:Fake())
    monkeypatch.setattr(run,'INTERVAL',0)
    monkeypatch.setattr(run,'check',lambda out:None)
    data=run.dataset();data['cases']=data['cases'][:1]
    run.write_json(tmp_path/'dataset.json',data)
    await run.run(tmp_path)
    report=run.read(tmp_path/'audited_report.json')
    assert report['complete'] and report['audit_passed']
    assert report['strategies']['C_on']['reasoning_tokens']>0
    assert report['strategies']['C_off']['reasoning_tokens']==0
    path=tmp_path/'results/svc-01-p1-S_on.json'
    value=run.read(path);value['state']['ingested']=100000
    run.write_json(path,value)
    assert not run.audit(tmp_path)['audit_passed']
