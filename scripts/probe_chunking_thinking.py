"""Synthetic transport-control probe, separate from benchmark scores."""
import asyncio
from pathlib import Path
import time
from scripts import eval_chunking as exp
from scripts.resume_chunking import DecodedTapTransport,durable_write

async def main():
    out=Path('data/chunking-comparison/20261028-v2/thinking-probe')
    exp.write=durable_write;exp.TapTransport=DecodedTapTransport
    exp.write(out/'manifest.json',{'diagnostic_only':True,'included_in_scores':False})
    runner=exp.Runner(out);results=[]
    variants={
        'thinking_disabled':{'thinking':{'type':'disabled'}},
        'template_thinking_false':{'thinking':{'type':'disabled'},'chat_template_kwargs':{'thinking':False}},
        'template_enable_false':{'thinking':{'type':'disabled'},'chat_template_kwargs':{'enable_thinking':False}},
        'enable_false':{'thinking':{'type':'disabled'},'enable_thinking':False},
    }
    try:
        for name,extra in variants.items():
            payload={'model':exp.settings.chat_model,'messages':[{'role':'user','content':'请只输出OK两个英文字母，不要解释。'}],
                'max_tokens':256,'temperature':0,'stream':False,**extra}
            start=time.monotonic()
            try:
                r=await runner.api('chat',payload,{'stage':'thinking_probe','variant':name})
                message=r['choices'][0]['message'];usage=r.get('usage') or {}
                result={'variant':name,'finish_reason':r['choices'][0]['finish_reason'],
                    'content':message.get('content'),'reasoning_chars':len(message.get('reasoning_content') or ''),
                    'usage':usage,'seconds':time.monotonic()-start}
            except Exception as exc:result={'variant':name,'error_type':type(exc).__name__}
            results.append(result);exp.write(out/'results.json',results);print(result,flush=True)
    finally:await runner.close()

if __name__=='__main__':asyncio.run(main())
