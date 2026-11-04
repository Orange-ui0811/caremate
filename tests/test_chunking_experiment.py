import json
import pytest
import httpx
from scripts import eval_chunking as exp


def test_all_arms_retain_every_gold_point_and_current_body_identity():
    docs=exp.corpus()
    samples=[json.loads(s) for s in (exp.ROOT/'tests/data/eval_ch04.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    gold,missing=exp.make_gold(docs,samples)
    assert not missing and len(samples)==300
    for arm in exp.ARMS:
        chunks=exp.make_chunks(docs,arm)
        for sample in samples:
            if sample['should_refuse']:continue
            # Give the scorer all chunks to verify source mapping completeness;
            # score() normally takes only the first10, so check each point by
            # filtering to chunks covering its source section.
            for point in gold[sample['id']]['points']:
                selected=[c for c in chunks if any(a['sid'] in c['section_ids'] for a in point['alternatives'])]
                metrics=exp.score(selected,{'groups':gold[sample['id']]['groups'],'points':[point]},chunks)
                assert metrics['source_point_coverage_at10']==1,(arm,sample['id'],point['point'])
        if arm=='structured':
            expected=[c.answer for name,typ in exp.SOURCE_TYPES.items() for c in exp.documents.build_chunks((exp.KB_DIR/name).read_text(encoding='utf-8'),typ)]
            assert [c['answer'] for c in chunks]==expected


def test_fixed_has_no_overlap_and_no_loss():
    for doc in exp.corpus():
        chunks=exp.make_chunks([doc],'fixed')
        assert ''.join(c['answer'] for c in chunks)==doc['text']
        assert all(len(c['answer'])<=400 for c in chunks)
        assert all(a['source_spans'][0][1]==b['source_spans'][0][0] for a,b in zip(chunks,chunks[1:]))


def test_section_title_match_does_not_replace_missing_body_evidence():
    chunks=[{'id':1,'document':'d','question':'退货','answer':'支持退货',
        'section_path':'退货','section_ids':['s'],'source_spans':[[0,4]]}]
    gold={'groups':[['退货']],'points':[{'point':'拆封不退','alternatives':[{'document':'d','sid':'s','span':[10,14]}]}]}
    score=exp.score(chunks,gold,chunks)
    assert score['section_recall_at5']==1 and score['source_point_coverage_at5']==0


async def test_wire_tap_records_response_usage_without_credentials(tmp_path):
    tap=exp.TapTransport(tmp_path)
    await tap.inner.aclose()
    tap.inner=httpx.MockTransport(lambda request:httpx.Response(200,json={'usage':{'prompt_tokens':12,'completion_tokens':3},'choices':[]}))
    async with httpx.AsyncClient(transport=tap) as client:
        response=await client.post('https://example.test/v1/chat/completions',json={'model':'test','messages':[]},headers={'Authorization':'Bearer must-not-save'})
        assert response.status_code==200
    record=exp.read(next((tmp_path/'calls').glob('*.json')))
    assert record['response']['usage']['prompt_tokens']==12 and record['complete']
    assert 'must-not-save' not in json.dumps(record)
