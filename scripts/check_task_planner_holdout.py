"""Additional unseen-number binding probes; extraction only, no business writes."""
import asyncio
import hashlib
import json
from pathlib import Path
from app.config import settings
from app.core import task_planner, model_pool
from scripts.eval_multi_intent import save

OUT=Path('E:/Mewhelp/data/multi-intent-eval/20261103-v2/holdout-binding')
CASES=[
 ('先看3771快递到哪，再对4826申请退款，还要投诉商家。',[('logistics','3771'),('refund','4826'),('complaint',None)]),
 ('给4826办理退款，不要查询3771的物流，我要投诉商家。',[('refund','4826'),('complaint',None)]),
 ('物流查3771，退款处理4826，投诉商家服务差。',[('logistics','3771'),('refund','4826'),('complaint',None)]),
 ('查3771的物流，另外给我退钱，还想投诉店家。',[('logistics','3771'),('refund',None),('complaint',None)]),
 ('4826和5099都申请退款，3771只查快递。',[('refund','4826'),('refund','5099'),('logistics','3771')]),
 ('查3771的物流，4826申请退款，别投诉商家。',[('logistics','3771'),('refund','4826')])]


async def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'results.json').exists():raise RuntimeError('Do not resample holdout results')
    save(OUT/'protocol.json',{'cases':CASES,'source_sha256':hashlib.sha256(Path(task_planner.__file__).read_bytes()).hexdigest(),
                            'scope':'Extra binding probes after code freeze; not full workflow evaluation; no tuning after seeing outputs.'})
    settings.model_pool_db_path='E:/Mewhelp/data/model_pool.sqlite3'
    settings.chat_failover_enabled=False
    sem=asyncio.Semaphore(2)
    async def one(i,case):
      async with sem:
        raw,expected=case
        try:
          with model_pool.call_context('multi-v2-binding-'+str(i),priority='background'):
            tasks=await task_planner.extract(raw)
          actual=[(t['kind'],t['order_id']) for t in tasks]
          return {'id':i,'query':raw,'expected':expected,'actual':actual,'exact_match':sorted(actual,key=str)==sorted(expected,key=str),'tasks':tasks}
        except Exception as e:return {'id':i,'query':raw,'exact_match':False,'error':str(e)}
    rows=await asyncio.gather(*(one(i,c) for i,c in enumerate(CASES,1)))
    save(OUT/'results.json',{'exact_n':sum(r['exact_match'] for r in rows),'n':len(rows),'rows':rows})
    print([(r['id'],r['exact_match'],r.get('actual',r.get('error'))) for r in rows],flush=True)
    from app.core.model_transport import close_clients
    await close_clients()


if __name__=='__main__':asyncio.run(main())
