"""Offline end-to-end audit test with a deterministic fake endpoint."""
import json

from langchain_core.messages import AIMessage

from scripts import eval_context_policy_v3 as run
from scripts.audit_context_policy_v3 import audit


async def test_independent_audit_replays_all_stages_and_detects_future_state(tmp_path,monkeypatch):
    class Fake:
        schema=None
        def bind_tools(self,schemas,tool_choice):
            self.schema=schemas[0]
            return self
        async def ainvoke(self,messages):
            usage={'input_tokens':100,'output_tokens':20,'total_tokens':120}
            if self.schema:
                value={'summary':'用户要求查询订单778812物流。'} if self.schema==run.Summary else {'action':'logistics','order_id':'778812','sku':'','answer':''}
                return AIMessage('',tool_calls=[{'name':self.schema.__name__,'args':value,'id':'fake','type':'tool_call'}],response_metadata={'finish_reason':'tool_calls','model_name':'fake'},usage_metadata=usage)
            return AIMessage('订单778812，运单SF77881288，运输中。',response_metadata={'finish_reason':'stop','model_name':'fake'},usage_metadata=usage)
    monkeypatch.setattr(run.llm,'get_chat_model',lambda **kw:Fake())
    history=[r for _ in range(24) for r in ({'role':'user','content':'订单778812请查物流。'},{'role':'assistant','content':'已收到查询请求。'})]
    expected={'action':'logistics','order_id':'778812','sku':'','date':''}
    case={'id':'unit-1','kind':'early_order','turns':24,'history':history,
          'probes':[{'id':'p1','at_turn':18,'query':'查询这笔订单物流。','expected':expected},
                    {'id':'p2','at_turn':24,'query':'前面那单呢？','expected':expected}],
          'tool_data':{'orders':{'778812':{'order_id':'778812','tracking_no':'SF77881288'}},'manuals':{}},
          'all_orders':['778812'],'all_skus':[]}
    data={'cases':[case]}
    run.write_json(tmp_path/'dataset.json',data)
    run.write_json(tmp_path/'manifest.json',{'config':run.config_snapshot(),'dataset_sha256':run.digest(data),'sources':{}})
    await run.evaluate_case(case,tmp_path,run.Caller(tmp_path))
    good=audit(tmp_path)
    assert good['passed'] and good['probes_checked']==8 and good['maintenance_checked']>0
    path=tmp_path/'states/unit-1-p1.json'
    state=json.loads(path.read_text(encoding='utf-8'));state['ingested']=999
    run.write_json(path,state)
    bad=audit(tmp_path)
    assert not bad['passed'] and any('state mismatch' in s for s in bad['issues'])
