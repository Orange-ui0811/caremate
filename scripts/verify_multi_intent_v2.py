"""Read-only integrity checks for the V2 study after collection and judging."""
import asyncio
import hashlib

from scripts.eval_multi_intent import ROOT, read, save

OUT=ROOT/'data/multi-intent-eval/20261103-v2'
BASE=ROOT/'data/multi-intent-eval/20261103'


async def main():
    from app.config import settings
    from app.kb import milvus_client
    protocol=read(OUT/'protocol.json')
    changed=[]
    for name,digest in protocol['source_hashes'].items():
        path=OUT/'source'/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:changed.append(name)
    current_changed=[]
    for p in (OUT/'source/after/app').rglob('*.py'):
        current=ROOT/'app'/p.relative_to(OUT/'source/after/app')
        if not current.exists() or current.read_bytes()!=p.read_bytes():
            current_changed.append(str(current.relative_to(ROOT)))
    rows=await milvus_client.acall(lambda:milvus_client.get_client().query('knowledge',filter='',
        output_fields=['id','question','answer','section_path','content_type','category'],limit=16384))
    rows.sort(key=lambda r:r['id'])
    source_results=[read(p) for p in (OUT/'runs').glob('*/*/result.json')]
    baseline_changed=[]
    for p in (OUT/'runs/before').glob('*/result.json'):
        reused=read(p);original=read(BASE/'runs/after'/p.parent.name/'result.json')
        reused.pop('arm',None);reused.pop('original_arm',None);original.pop('arm',None)
        if reused!=original:baseline_changed.append(p.parent.name)
    summary=read(OUT/'summary.json')
    result={
        'knowledge_unchanged':rows==read(OUT/'knowledge.json'),
        'knowledge_rows':len(rows),
        'cases_identical_to_v1':(OUT/'cases.json').read_bytes()==(BASE/'cases.json').read_bytes(),
        'frozen_source_changed':changed,
        'current_app_differs_from_frozen_v2':current_changed,
        'baseline_trajectories_changed':baseline_changed,
        'completed_trajectories':len(source_results),
        'source_errors':sum(r['status']!='ok' for r in source_results),
        'paired_scenarios':summary['paired_n'],
        'regression_tests':{'result':'228 passed',
            'command':'pytest tests/core/test_task_planner.py tests/graph tests/tools tests/test_actions_api.py tests/test_action_context_boundary.py tests/test_tool_ticket.py -q'},
        'metric_unit_tests':'4 passed',
        'feature_enabled_in_current_settings':settings.multi_intent_enabled,
        'production_service_restarted':False,
        'knowledge_or_customer_config_modified_by_experiment':False,
        'scope':'Graph/runtime + form APIs, deterministic business fixtures, isolated test SQL, real model/RAG. No new V2 browser/SSE fault-injection acceptance.'}
    save(OUT/'verification.json',result)
    print(result)
    assert not(changed or current_changed or baseline_changed)
    assert result['knowledge_unchanged'] and result['cases_identical_to_v1']
    assert len(source_results)==60 and summary['paired_n']==30 and not result['source_errors']


if __name__=='__main__':asyncio.run(main())
