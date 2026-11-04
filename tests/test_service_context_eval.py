from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from scripts import eval_service_context as run


async def test_budget_block_without_api_and_identical_cache(tmp_path, monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError('Oversize request must not reach the provider')
    monkeypatch.setattr(run.base.llm, 'get_chat_model', forbidden)
    caller = run.Caller(tmp_path)
    args = ('oversize', 'off', 'reply', [SystemMessage('test'), HumanMessage('长内容。'*6000)])
    first = await caller.call(*args)
    assert first == await caller.call(*args)
    assert first['status'] == 'budget_blocked' and first['attempts'] == []
    assert not first['ok'] and first['approx_input_tokens'] > 16000


async def test_three_arms_causal_replay_and_source_tamper(tmp_path, monkeypatch):
    class Fake:
        extra_body = {}
        schema = None
        def bind_tools(self, schema, tool_choice):
            self.schema = schema[0]
            assert tool_choice == 'auto'
            return self
        async def ainvoke(self, messages):
            assert self.extra_body == run.base.mode_body('off')
            usage = {'input_tokens': 10, 'output_tokens': 30, 'total_tokens': 40,
                     'output_token_details': {'reasoning': 0}}
            metadata = {'finish_reason': 'stop', 'model_name': 'fake'}
            if self.schema:
                value = {'summary': '用户查询订单610010。'} if self.schema == run.base.Summary else {
                    'action': 'logistics', 'order_id': '610010', 'sku': '', 'answer': ''}
                return AIMessage('', tool_calls=[{'name': self.schema.__name__, 'args': value,
                                 'id': 'fake', 'type': 'tool_call'}],
                                 response_metadata=metadata, usage_metadata=usage)
            return AIMessage('订单610010，运单SF61001088，运输中。',
                             response_metadata=metadata, usage_metadata=usage)
    monkeypatch.setattr(run.base.llm, 'get_chat_model', lambda **kwargs: Fake())
    monkeypatch.setattr(run.base, 'INTERVAL', 0)
    monkeypatch.setattr(run, 'check', lambda out: None)
    data = run.dataset()
    data['cases'] = data['cases'][:1]
    run.write_json(tmp_path/'dataset.json', data)
    await run.run(tmp_path)
    report = run.read(tmp_path/'audited_report.json')
    assert report['complete'] and report['audit_passed'] and report['nonthinking_verified']
    assert all(report['strategies'][a]['all']['n'] == 2 for a in run.ARMS)
    assert report['strategies']['C_off']['all']['summary']['calls'] > 0
    assert report['strategies']['N_off']['all']['summary']['calls'] == 0
    path = tmp_path/'results/svc-01-p1-N_off.json'
    result = run.read(path)
    result['state']['evidence'][0]['start'] = 999
    run.write_json(path, result)
    assert not run.audit(tmp_path)['audit_passed']
