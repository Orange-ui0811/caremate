"""Small synthetic-only endpoint protocol diagnostic; never change live config."""
import asyncio
import json
from pathlib import Path
from langchain_core.messages import HumanMessage
from app.core import llm
from app.core.model_pool import call_context
from app.core.model_transport import close_clients

OUT=Path('data/verification/thinking-controls-20261025')
OPTIONS=[('default',{},None),
    ('native',{'thinking':{'type':'enabled'}},None),
    ('boolean',{'enable_thinking':True},None),
    ('template',{'chat_template_kwargs':{'thinking':True,'enable_thinking':True}},None),
    ('native_high',{'thinking':{'type':'enabled'},'enable_thinking':True},'high'),
    ('template_high',{'chat_template_kwargs':{'thinking':True,'enable_thinking':True,'reasoning_effort':'high'}},None)]


async def main():
    OUT.mkdir(parents=True,exist_ok=True)
    try:
        for name,extra,effort in OPTIONS:
            path=OUT/(name+'.json')
            if path.exists():continue
            record={'name':name,'extra_body':extra,'reasoning_effort':effort,'max_tokens':2048}
            try:
                with call_context(request_id='thinking-controls-20261025-'+name,priority='background'):
                    m=llm.get_chat_model(streaming=True,temperature=0)
                    m.extra_body=extra;m.reasoning_effort=effort;m.max_tokens=2048
                    r=await asyncio.wait_for(m.ainvoke([HumanMessage('有三个整数，甲是乙的两倍，丙比甲少3，三者之和为42。只输出乙的值。')]),100)
                record.update(usage=r.usage_metadata,metadata=r.response_metadata,content=r.content,
                    reasoning_field_present=bool(r.additional_kwargs.get('reasoning_content') or r.additional_kwargs.get('reasoning')))
            except Exception as e:record.update(error_type=type(e).__name__,http_status=getattr(e,'status_code',None))
            path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            print(name,record.get('usage'),record.get('error_type'),flush=True)
            await asyncio.sleep(1)
    finally:await close_clients()


if __name__=='__main__':asyncio.run(main())
