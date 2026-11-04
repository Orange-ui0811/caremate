import json
from langchain_core.messages import AIMessage
from scripts import eval_context_full_history_control as control
from scripts import eval_context_policy_v3 as base


def test_full_history_control_budget_and_causal_prefix():
    dataset=json.loads(base.OLD_DATA.read_text(encoding='utf-8'))
    for case in dataset['cases']:
        for probe in case['probes']:
            body,cr=control.prompts(case,probe)
            assert len(body)==2*probe['at_turn']+1
            assert [m.content for m in body[:-1]]==[r['content'] for r in case['history'][:2*probe['at_turn']]]
            assert sum(base.estimate(m.content) for m in cr)<16000
            assert sum(base.estimate(m.content) for m in body)+base.estimate(base.READER_SYSTEM)+base.estimate(json.dumps(base.ServicePlan.model_json_schema()))+512<16000


async def test_full_history_control_receipts_and_future_tamper(tmp_path,monkeypatch):
    case=json.loads(base.OLD_DATA.read_text(encoding='utf-8'))['cases'][0]
    class Fake:
        schema=None
        def bind_tools(self,schemas,tool_choice):
            self.schema=schemas[0];return self
        async def ainvoke(self,messages):
            usage={'input_tokens':10,'output_tokens':10,'total_tokens':20}
            if self.schema:
                return AIMessage('',tool_calls=[{'name':'ServicePlan','args':{'action':'logistics','order_id':'610010','sku':'','answer':''},'id':'fake','type':'tool_call'}],response_metadata={'finish_reason':'tool_calls'},usage_metadata=usage)
            return AIMessage('订单610010，运单SF61001088，运输中。',response_metadata={'finish_reason':'stop'},usage_metadata=usage)
    monkeypatch.setattr(base.llm,'get_chat_model',lambda **kw:Fake())
    data={'cases':[case]};base.write_json(tmp_path/'dataset.json',data)
    parent=tmp_path/'parent'
    base.write_json(tmp_path/'manifest.json',{'dataset_sha256':base.digest(data),'sources':{},'config':base.config_snapshot(),'parent':str(parent),'scope':'test'})
    caller=base.Caller(tmp_path)
    for probe in case['probes']:
        await control.probe_call(tmp_path,caller,case,probe)
        for arm in ('C_current','S_state8k'):
            base.write_json(parent/'results'/f'{case["id"]}-{probe["id"]}-{arm}.json',{'score':{'task_rule_v3':True}})
    assert control.audit(tmp_path)['audit_passed']
    path=tmp_path/'results'/'svc-01-p1-F_full.json'
    result=control.read(path);result['body'].append({'content':'future turn'})
    base.write_json(path,result)
    assert not control.audit(tmp_path)['audit_passed']
