"""Diagnostic only: compare system-message placement on saved failed inputs."""
import asyncio
import json
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.core import llm
from app.core.model_pool import call_context
from app.core.model_transport import close_clients
from scripts.eval_context_policy_v3 import Summary, EXTRA_BODY, digest
from scripts.context_benchmark_cases import make_suite

ROOT=Path('data/ch07/context-policy/20261023-gitee-v3-nonthinking')
OUT=Path('data/verification/context-policy-v3-system-placement')


async def main():
    # Refuse external requests unless these are exactly the deterministic,
    # repository-generated fixtures, never production customer records.
    dataset=json.loads((ROOT/'dataset.json').read_text(encoding='utf-8'))
    if dataset!=make_suite():raise ValueError('Not the verified synthetic fixture dataset')
    OUT.mkdir(parents=True,exist_ok=True)
    cases=[]
    seen=set()
    for f in sorted((ROOT/'calls').glob('*-summary-*.json')):
        r=json.loads(f.read_text(encoding='utf-8'))
        cid=f.name[:6]
        if r['status']!='ok' and cid not in seen:
            cases.append((f.stem,r));seen.add(cid)
        if len(cases)==4:break
    gate=asyncio.Semaphore(4)
    async def one(key,r,mode):
        dest=OUT/(key+'-'+mode+'.json')
        if dest.exists():return
        rows=r['contract']['messages']
        cls={'system':SystemMessage,'human':HumanMessage,'ai':AIMessage}
        ms=[cls[m['type']](m['content']) for m in rows]
        if mode=='initial_system':
            ms=[SystemMessage(ms[0].content+'\n'+ms[-1].content),*ms[1:-1]]
        record={'source_call':key,'source_hash':r['input_sha256'],'variant':mode,
                'messages':[m.model_dump(mode='json') for m in ms],
                'max_tokens':1024,'extra_body':EXTRA_BODY}
        try:
            async with gate:
                with call_context(request_id='ctx-system-placement-'+key+'-'+mode,priority='background'):
                    model=llm.get_chat_model(streaming=True,temperature=.3)
                    model.max_tokens=1024
                    model.extra_body={**(model.extra_body or {}),**EXTRA_BODY}
                    raw=await asyncio.wait_for(model.bind_tools([Summary],tool_choice='auto').ainvoke(ms),100)
            record.update(raw_content=raw.content,tool_calls=raw.tool_calls,usage=raw.usage_metadata,
                          response_metadata=raw.response_metadata)
            found=[t for t in raw.tool_calls if t['name']=='Summary']
            record['schema_ok']=len(found)==1 and bool(Summary.model_validate(found[0]['args']).summary.strip())
        except Exception as e:
            record['error_type']=type(e).__name__;record['schema_ok']=False
        dest.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(key,mode,'schema',record['schema_ok'],'finish',record.get('response_metadata',{}).get('finish_reason'),flush=True)
    try:await asyncio.gather(*(one(k,r,mode) for k,r in cases for mode in ('trailing_system','initial_system')))
    finally:await close_clients()


if __name__=='__main__':asyncio.run(main())
