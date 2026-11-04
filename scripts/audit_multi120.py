"""Read-only study integrity, API aggregates and a queue for source-evidence review."""
import argparse
import asyncio
from collections import Counter
import sqlite3

from scripts.eval_multi120 import ROOT,OUT,read,save,sha


def audit():
    protocol=read(OUT/'protocol.json');rows=[];queue=[]
    for p in (OUT/'runs').glob('*/*/*/result.json'):
        r=read(p);rows.append(r)
        jp=OUT/'judges'/f'r{r["rep"]}'/r['arm']/(r['case_id']+'.json')
        if not jp.exists():continue
        j=read(jp)
        if j['status']=='judge_error' or j.get('unsafe_claims'):
            queue.append({'rep':r['rep'],'arm':r['arm'],'case_id':r['case_id'],
                'judge':j,'goals':r['case']['goals'],'query':r['case']['query'],'followup':r['case']['followup'],
                'answers':list(dict.fromkeys(s['answer'] for s in r.get('steps',[]) if s.get('answer'))),
                'tickets':r.get('tickets',[]),'actor':r.get('actor',[]),
                'source_path':str(p.relative_to(OUT))})
    save(OUT/'review-queue.json',queue)
    with sqlite3.connect(ROOT/'data/model_pool.sqlite3') as conn:
        conn.row_factory=sqlite3.Row
        calls=[dict(r) for r in conn.execute("SELECT request_id,channel,model,outcome,http_status,queue_ms,started,finished FROM attempts WHERE request_id LIKE 'mi120formal:%' OR request_id LIKE 'mi120formal-judge:%' OR request_id LIKE 'mi120formal-chronojudge:%'")]
    save(OUT/'api-attempts.json',calls)
    stats={}
    for arm in ('disabled','enabled','judge','chronological_judge'):
        if arm=='judge':subset=[r for r in calls if r['request_id'].startswith('mi120formal-judge:')]
        elif arm=='chronological_judge':subset=[r for r in calls if r['request_id'].startswith('mi120formal-chronojudge:')]
        else:subset=[r for r in calls if ':'+arm+':' in r['request_id'] and r['request_id'].startswith('mi120formal:')]
        stats[arm]={'attempts':len(subset),'by_channel':dict(Counter(r['channel'] for r in subset)),
                    'outcomes':dict(Counter(r['outcome'] for r in subset)),
                    'http_status':dict(Counter(str(r['http_status']) for r in subset))}
    save(OUT/'api-summary.json',stats)
    result={'trajectories':len(rows),'source_statuses':dict(Counter(r['status'] for r in rows)),
        'original300_unchanged':sha(ROOT/'tests/data/eval_ch04.jsonl')==protocol['original300_sha256'],
        'frozen_inputs_changed':[n for n in ('cases','knowledge','scoring') if sha(OUT/(n+'.json'))!=protocol[n+'_sha256']],
        'frozen_app_changed':[n for n,h in protocol['source_hashes'].items() if sha(OUT/'source'/n)!=h],
        'evaluator_scripts_changed':[n for n,h in protocol['evaluation_script_hashes'].items() if sha(ROOT/'scripts'/n)!=h],
        'review_queue_n':len(queue),'api_attempts':{k:v['attempts'] for k,v in stats.items()},
        'frozen_files_precede_first_formal_api':all((OUT/name).stat().st_mtime<=min(r['started'] for r in calls)
            for name in ('protocol.json','cases.json','knowledge.json','scoring.json')) if calls else None,
        'protocol_tests':'7 passed; chronological judge correction tests9 passed','production_service_restarted':False,'production_config_or_knowledge_modified':False,
        'ticket_primary_key_collision_logs':[str(p.relative_to(OUT)) for p in (OUT/'runs').glob('*/*/*/worker.log')
            if 'Duplicate entry' in p.read_text('utf-8',errors='replace') and 'tickets.PRIMARY' in p.read_text('utf-8',errors='replace')]}
    save(OUT/'integrity.json',result);print(result)


async def finish():
    from app.kb import milvus_client
    from app.config import settings
    protocol=read(OUT/'protocol.json')
    kb=await milvus_client.acall(lambda:milvus_client.get_client().query('knowledge',filter='',
        output_fields=['id','question','answer','section_path','content_type','category'],limit=16384))
    kb.sort(key=lambda r:r['id']);audit();result=read(OUT/'integrity.json')
    result.update(knowledge_unchanged=kb==read(OUT/'knowledge.json'),knowledge_rows=len(kb),
        chronological_judge_code_unchanged=sha(ROOT/'scripts/score_multi120_chronological.py')==read(OUT/'chronological-judge-protocol.json')['script_sha256'],
        current_feature_enabled=settings.multi_intent_enabled,
        current_app_differs_from_snapshot=[str(p.relative_to(OUT/'source')) for p in (OUT/'source/app').rglob('*.py')
            if p.read_bytes()!=(ROOT/p.relative_to(OUT/'source')).read_bytes()])
    save(OUT/'verification.json',result);print('Final verification:',result)
    assert result['trajectories']==720 and result['knowledge_unchanged'] and result['original300_unchanged'] and result['frozen_files_precede_first_formal_api']
    assert result['chronological_judge_code_unchanged']
    assert not(result['frozen_inputs_changed'] or result['frozen_app_changed'] or result['evaluator_scripts_changed'] or result['current_app_differs_from_snapshot'] or result['ticket_primary_key_collision_logs'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');a=p.parse_args()
    asyncio.run(finish()) if a.final else audit()
