"""No paid calls: business stages, task isolation, persistence and old history."""
import asyncio
import json
import time
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.runnables import RunnableLambda

from app.config import settings
from app.core import model_pool as pool
from app.core.model_transport import AsyncPoolTransport, SyncPoolTransport


@pytest.fixture(autouse=True)
def fake_providers(monkeypatch):
    monkeypatch.setattr(settings, 'chat_base_url', 'https://fake.test/v1')
    monkeypatch.setattr(settings, 'model_pool_chat_keys', 'fake-key-1,fake-key-2')
    monkeypatch.setattr(settings, 'model_pool_fallback_api_key', '')
    pool.get_store().set_policy(pool.Policy(global_limit=4, background_limit=2))


def test_nested_stage_preserves_turn_id_priority_and_restores_on_error():
    with pool.call_context('turn-1', 'background'):
        with pytest.raises(RuntimeError):
            with pool.call_context(purpose='coreference'):
                assert pool.request_id() == 'turn-1' and pool.priority() == 'background'
                assert pool.call_purpose() == 'coreference'
                raise RuntimeError('synthetic')
        assert pool.call_purpose() == 'chat_other'
    assert pool.PURPOSE.get() == '' and pool.REQUEST_ID.get() == ''


async def test_async_stages_remain_isolated_and_retries_keep_purpose():
    seen = {}
    async def upstream(request):
        tag = json.loads(request.content)['tag']
        seen[tag] = seen.get(tag, 0) + 1
        await asyncio.sleep(.005)
        return httpx.Response(503 if tag == 'rewrite' and seen[tag] == 1 else 200, json={'ok': True})
    async with httpx.AsyncClient(transport=AsyncPoolTransport(inner=httpx.MockTransport(upstream))) as client:
        async def send(tag, purpose):
            with pool.call_context(tag, 'background'):
                with pool.call_context(purpose=purpose):
                    return await client.post('https://fake.test/v1/chat/completions', json={'model':'m', 'tag':tag})
        a,b = await asyncio.gather(send('rewrite','coreference'), send('expand','query_expansion'))
    assert a.status_code == b.status_code == 200
    rows = pool.get_store().snapshot()['recent_calls']
    assert len(rows) == 3
    assert all(r['purpose'] == {'rewrite':'coreference','expand':'query_expansion'}[r['request_id']] for r in rows)
    assert all(r['priority'] == 'background' for r in rows)
    assert {r['outcome'] for r in rows} == {'RETRY','SUCCEEDED'}
    assert pool.PURPOSE.get() == ''


def test_sync_rejection_and_existing_database_history_are_compatible():
    store = pool.get_store()
    # Simulate an old process writing the original schema without purpose data.
    with store.connect() as c:
        c.execute('DROP TABLE attempt_purposes')
        c.execute('INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                  ('old','legacy',None,'chat','m','foreground',1,time.time(),time.time(),0,'SUCCEEDED',200))
    store = pool.Store(store.path)
    with pool.call_context('sync', purpose='evidence_check'):
        with httpx.Client(transport=SyncPoolTransport(inner=httpx.MockTransport(lambda r: httpx.Response(200, json={})))) as client:
            client.post('https://fake.test/v1/chat/completions', json={'model':'m'})
        store.reject('denied','sync','chat','m','foreground',1,time.time())
    records = {r['id']:r for r in pool.Store(store.path).snapshot()['recent_calls']}
    assert records['old']['purpose'] == ''
    assert records['denied']['purpose'] == 'evidence_check'
    assert all(r['purpose'] == 'evidence_check' for r in records.values() if r['request_id']=='sync')
    with store.transaction() as c:
        store.cleanup(c, time.time()+8*86400)
        assert c.execute('SELECT COUNT(*) FROM attempt_purposes').fetchone()[0] == 0


async def test_actual_business_entry_points_tag_invocations(monkeypatch):
    from app.core import coref,intent,query_understanding,selfcheck,summarizer,flywheel
    seen=[]
    async def invoke(value):
        seen.append((pool.call_purpose(),pool.request_id(),pool.priority()))
        return SimpleNamespace(content='完整问题',intent='物流',confidence=.9,standard='标准问题',
            expanded=['同义词'],queries=['查询1'],useful=True,reason='足够',summary='摘要',normalized_question='标准问题')
    fake = RunnableLambda(invoke)
    monkeypatch.setattr(coref,'get_chat_model',lambda **kw:fake)
    monkeypatch.setattr(intent.llm,'structured',lambda *a,**kw:fake)
    with pool.call_context('same-turn','background'):
        await coref.resolve('那单呢','订单111111')
        await intent.classify('订单查询')
        await query_understanding.understand('商品怎么清洗')
        await query_understanding.expand_queries('退货政策')
        await selfcheck.check_sufficient('问题',['证据'])
        await summarizer.summarize_dialog('旧摘要','新对话')
        await flywheel.normalize_and_match('原问题',[])
    assert [s[0] for s in seen] == ['coreference','intent_classification','query_rewrite','query_expansion',
        'evidence_check','conversation_summary','question_normalization']
    assert all(s[1:] == ('same-turn','background') for s in seen)


async def test_embedding_query_and_document_purposes_do_not_overwrite_each_other(monkeypatch):
    from app.core import embeddings
    seen=[]
    async def embed(texts):
        seen.append(pool.call_purpose())
        return [[1.] for _ in texts]
    monkeypatch.setattr(embeddings,'_embed',embed)
    assert await embeddings.embed_query('问题') == [1.]
    assert await embeddings.embed_texts(['知识']) == [[1.]]
    assert seen == ['retrieval_embedding','knowledge_embedding']


async def test_runtime_exposes_business_label_and_honest_legacy_label():
    from app.main import app
    store=pool.get_store()
    store.reject('tagged','turn','chat','m','foreground',1,time.time())
    with store.connect() as c:
        c.execute("DELETE FROM attempt_purposes WHERE attempt_id='tagged'")
    with pool.call_context(purpose='query_expansion'):
        store.reject('new','turn','chat','m','foreground',1,time.time())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
        response=await client.get('/api/model-pool/runtime')
    assert response.status_code == 200
    records={r['id']:r for r in response.json()['recent_calls']}
    assert records['new']['purpose_label'] == '多路查询扩写'
    assert records['tagged']['purpose_label'] == '历史调用，未记录环节'
    assert 'fake-key' not in response.text
