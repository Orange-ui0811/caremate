from copy import deepcopy
import json

import httpx
import pytest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from scripts import eval_context_keep3 as exp


def state():
    pairs=[('工作室那单710001先登记','已登记'),('不对，应为710003','已更正')]
    pairs += [(f'包装资料{i}',f'保存说明{i}') for i in range(16)]
    messages=[]
    for q,a in pairs:
        messages.append(HumanMessage(q,id=f'db-{len(messages)+1}'))
        messages.append(AIMessage(a,id=f'db-{len(messages)+1}'))
    messages.append(HumanMessage('工作室那单物流呢',id='db-37'))
    return {'conversation_id':123,'current_message_id':37,'summary_upto_msg_id':24,
            'summary':'曾讨论710001，后更正为710003','messages':messages}


def test_four_views_only_add_supplement_and_full_has_all_raw():
    s=state()
    for arm in ('summary','hybrid','window','retrieval'):
        v=deepcopy(s)
        if arm in ('window','retrieval'):v['summary']=''
        r=exp.replay_view(v,arm,'history')
        assert '包装资料15' in r['packet']['history']
        assert '包装资料0' not in r['base_packet']['history']
        if arm in ('hybrid','retrieval'):
            assert r['supplement']['manifest']['selected_message_ids']==[1,2,3,4]
            assert r['packet']['history']==r['base_packet']['history']+'\n\n'+r['supplement']['text']
        else:assert r['packet']==r['base_packet']
    s['summary']='';s['summary_upto_msg_id']=0
    full=exp.replay_view(s,'full','history')
    for m in s['messages'][:-1]:assert m.content in full['packet']['history']


def test_agent_addition_preserves_system_raw_window_and_tool_pairs():
    s=state();s['messages'] += [AIMessage('',tool_calls=[{'id':'one','name':'query_order','args':{'order_id':'710003'}}]),ToolMessage('回执',tool_call_id='one')]
    r=exp.replay_view(s,'hybrid','answer')
    a,b=r['packet']['messages'],r['base_packet']['messages']
    assert len(a)==len(b) and a[0]==b[0]
    changed=[(x,y) for x,y in zip(a,b) if x!=y]
    assert len(changed)==1
    x,y=changed[0]
    assert x['data']['id']==y['data']['id']=='turn-ctx'
    assert x['data']['content'].replace('\n\n'+r['supplement']['text'],'',1)==y['data']['content']
    assert a[-2:]==b[-2:]


async def test_actual_workflow_and_keep3_boundary_all_five_arms(tmp_path,monkeypatch):
    impl=exp.base
    monkeypatch.setattr(impl.settings,'model_pool_enabled',False)
    monkeypatch.setattr(impl.settings,'chat_base_url','https://ai.gitee.com/v1')
    monkeypatch.setattr(impl.settings,'chat_model','fake-offline-only')
    case={'id':'keep3-offline','family':'offline','stratum':'long','cohort':'offline',
          'initial_history':[{'role':m.type if m.type=='user' else 'user' if isinstance(m,HumanMessage) else 'assistant','content':m.content} for m in state()['messages'][:-1]],
          'steps':[{'query':'请实际查订单710003最新物流。','expected':{'kind':'logistics','target':'710003'}} for _ in range(3)]}
    # Exactly27 rounds so initialization already crosses15 and27 triggers.
    while len(case['initial_history'])<54:
        case['initial_history'] += [{'role':'user','content':'一般包装资料'}, {'role':'assistant','content':'保留原始包装。'}]
    monkeypatch.setattr(exp,'dataset',lambda pilot=False:{'version':'offline','cases':[case],'pilot':True})
    requests=[]

    def reply(request):
        payload=json.loads(request.content);requests.append(payload)
        purpose,turn=impl.PURPOSE.get(),impl.COORDS.get()['turn']
        text=('请查询订单710003最新物流。' if purpose=='coreference' else
              f"订单710003，物流记录{turn}：{exp.unified.cases_module().logistics_event(max(1,turn))}。")
        args={'conversation_summary':{'summary':'工作室订单710001更正为710003，尚未查询。'},
              'intent_classification':{'intent':'物流','confidence':1.0}}.get(purpose)
        message={'role':'assistant','content':text}
        if args:
            message={'role':'assistant','content':'','tool_calls':[{'id':'offline','type':'function','function':{
                'name':payload['tools'][0]['function']['name'],'arguments':json.dumps(args,ensure_ascii=False)}}]}
        usage={'prompt_tokens':100,'completion_tokens':20,'total_tokens':120}
        common={'id':'offline','created':1,'model':'fake-offline-only','usage':usage}
        finish='tool_calls' if args else 'stop'
        if not payload.get('stream'):
            return httpx.Response(200,json={**common,'object':'chat.completion','choices':[{'index':0,'message':message,'finish_reason':finish}]})
        delta=deepcopy(message)
        if args:delta['tool_calls'][0]['index']=0
        chunk={**common,'object':'chat.completion.chunk','choices':[{'index':0,'delta':delta,'finish_reason':finish}]}
        return httpx.Response(200,content=('data: '+json.dumps(chunk,ensure_ascii=False)+'\n\ndata: [DONE]\n\n').encode(),headers={'content-type':'text/event-stream'})

    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        def factory(**kw):
            return impl.llm._CumulativeUsageChatOpenAI(model='fake-offline-only',base_url='https://ai.gitee.com/v1',
                api_key='offline-placeholder',http_async_client=client,max_retries=0,streaming=kw.get('streaming',False),stream_usage=True)
        monkeypatch.setattr(impl.llm,'get_chat_model',factory)
        exp.prepare(tmp_path,pilot=True)
        with exp.binding():report=await exp.unified.run(tmp_path)
    assert report['complete'] and report['audit_passed'],report['issues']
    for arm in exp.ARMS:
        g=report['groups'][arm]
        assert g['task_rule_pass']==3 and not g['interface_error_calls'] and not g['unknown_usage_calls']
        ms=exp.unified.read(tmp_path/'maintenance'/f'{arm}-{case["id"]}.json')
        triggered=[m for m in ms if m['triggered']]
        if arm=='full':assert not triggered
        else:
            assert [m['row_watermark'] for m in triggered]==[30,54]
            assert [m['legacy_summary_upto'] for m in triggered]==[24,48]
        if arm in ('window','retrieval','full'):assert g['maintenance_model_calls']==0
        else:assert g['maintenance_model_calls']==2
    assert sum(g['model_calls'] for g in report['groups'].values())==len(requests)
    from scripts.audit_context_keep3 import audit
    independent=audit(tmp_path)
    assert independent['passed'],independent['issues']
