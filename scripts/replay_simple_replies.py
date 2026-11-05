"""Replay only presentation from immutable receipts; never reexecute business tasks."""
import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config import settings
from app.graph import simple_loop
from scripts import eval_simple_loop as evaluation


async def main(rep):
    folder=evaluation.OUT/'render-replay'/f'r{rep}'
    folder.mkdir(parents=True,exist_ok=True)
    settings.model_pool_db_path=str(folder/'pool.sqlite')
    settings.langfuse_public_key=''
    evaluation.instrument()
    async def event(cid,uid,key,text,**kwargs):
        return {'event_id':key,'message_id':0,'text':text,**kwargs}
    simple_loop.workflow.event=event
    files=sorted((evaluation.OUT/'replyfix/simple'/f'r{rep}').glob('S*.json'))
    assert len(files)==8, 'All eight original trajectories must be frozen first'
    for path in files:
        target=folder/path.name
        if target.exists():continue
        original=json.loads(path.read_text(encoding='utf-8'))
        evaluation.ACTIVE={'http':[],'case':original['case'],'rep':rep,
            'source':str(path.relative_to(evaluation.OUT)),
            'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'renderer_sha256':hashlib.sha256(Path(simple_loop.__file__).read_bytes()).hexdigest(),'steps':[]}
        seen={}
        for step in original['steps']:
            state={**step['state'],'simple_presented':seen}
            # All eight scenarios are business queries. Serialized original
            # dialog is intentionally unnecessary in the fact renderer.
            assert not any(t['kind']=='other' for t in state['task_plan'].values())
            try:
                result=await simple_loop.present(state)
                evaluation.ACTIVE['steps'].append({'answer':'\n'.join(e['text'] for e in result.get('task_events',[])),
                    'trace':result.get('trace'),'interrupt':step.get('interrupt')})
                seen=result.get('simple_presented',seen)
            except Exception as exc:
                evaluation.ACTIVE['steps'].append({'error':type(exc).__name__+':'+str(exc)})
        evaluation.save(target,evaluation.ACTIVE)
        print(json.dumps({'case':path.stem,'rep':rep,'steps':len(evaluation.ACTIVE['steps']),
                          'model_calls':len(evaluation.ACTIVE['http'])}),flush=True)
    from app.core.model_transport import close_clients
    await close_clients()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--rep',type=int,required=True)
    asyncio.run(main(parser.parse_args().rep))
