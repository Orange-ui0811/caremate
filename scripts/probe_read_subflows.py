"""Real model + local HTTP MCP; audit redirected to a local evidence file.

No production conversation, ticket, KB or customer configuration writes.
"""
import asyncio
import json
import sys
import time
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import settings
from app.graph.read_subflows import build_pilot
from app.tools import engine
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


async def main():
    folder = Path(sys.argv[1] if len(sys.argv)>1 else 'data/verification/live-p1')
    folder.mkdir(parents=True, exist_ok=True)
    code_files=['app/graph/read_subflows.py','app/graph/nodes.py','app/graph/state.py','app/core/task_planner.py']
    (folder/'code.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for p in code_files},indent=2),encoding='utf-8')
    settings.langfuse_public_key = ''
    settings.context_policy = 'legacy'
    settings.context_legacy_recall_enabled = False
    settings.chat_failover_enabled = False
    settings.model_pool_db_path = str(folder/'model-pool.sqlite')
    audits = []
    async def audit(*args, **kwargs):
        audits.append({'tool':args[2], 'args':args[4], 'status':args[6]})
    engine._audit = audit
    cases = [
        ('explicit', '查询订单1001的信息，它现在物流到哪了？', []),
        ('reference', '刚刚的订单是哪个，它现在物流到哪了？', ['1001']),
        ('different-orders', '查订单1001的信息，再查订单2002的物流。', []),
        ('ambiguous', '刚刚的订单是哪个，它现在物流到哪了？', ['1001','2002']),
    ]
    records = []
    for i,(name,query,ids) in enumerate(cases):
        started=time.monotonic(); before=len(audits)
        cid=91000+i
        inp={'user_id':'p1-synthetic','conversation_id':cid,'current_message_id':i+1,'query':query,
             'business_context':{'user_id':'p1-synthetic','conversation_id':cid,'order_ids':ids,'recent':True}}
        async with AsyncSqliteSaver.from_conn_string(str(folder/f'{name}.sqlite')) as saver:
            config={'configurable':{'thread_id':str(cid)}}
            first=await build_pilot(saver,interrupt_after=['execute']).ainvoke(inp,config)
        async with AsyncSqliteSaver.from_conn_string(str(folder/f'{name}.sqlite')) as saver:
            graph=build_pilot(saver)
            snapshot=await graph.aget_state(config)
            out=await graph.ainvoke(None,config) if snapshot.next else snapshot.values
        records.append({'name':name,'query':query,'context_orders':ids,'first':first,'final':out,
                        'audits':audits[before:], 'seconds':round(time.monotonic()-started,2)})
        (folder/'results.json').write_text(json.dumps(records,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
        print(json.dumps({'case':name,'statuses':[t['status'] for t in out['tasks']],
                          'orders':[t.get('order_id') for t in out['tasks']]},ensure_ascii=False),flush=True)
    # Evidence-based checks: successful receipts and correct object, not only DONE.
    checks={}
    for row in records:
        expected={'explicit':['1001','1001'],'reference':['1001','1001'],
                  'different-orders':['1001','2002']}.get(row['name'])
        out=row['final']
        checks[row['name']]=(all(t['status']=='WAITING_USER' for t in out['tasks']) and not row['audits']
            if expected is None else len(out['tasks'])==2 and [t.get('order_id') for t in out['tasks']]==expected
            and all(t['status']=='DONE' for t in out['tasks']) and len(out['results'])==2
            and all(r['receipts'] for r in out['results']))
        checks[row['name']+'-read-only']=all(a['tool'] in {'query_order','query_logistics'} for a in row['audits'])
        if expected is not None:
            checks[row['name']+'-fresh-order-receipts'] = all(
                any(receipt['name']=='query_order' and json.loads(receipt['content']).get('order_id')==result['order_id']
                    for receipt in result['receipts']) for result in out['results'])
            expected_calls = sum(1 if r.get('outcome')=='UNPAID' or r['kind']=='order' else 2 for r in out['results'])
            checks[row['name']+'-resume-no-repeat'] = len(row['first']['results'])==1 and len(row['audits'])==expected_calls
    (folder/'checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks),flush=True)
    if not all(checks.values()): raise SystemExit(1)


if __name__ == '__main__': asyncio.run(main())
