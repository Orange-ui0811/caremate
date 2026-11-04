"""V3 experiment driver: normalize explicit DSML Check/Grade transport results.

This never executes model tools. Only known evaluation schemas are accepted;
free text and malformed/ambiguous output still fail. Earlier raw wires stay intact.
"""
import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import re
import time
from scripts import eval_chunking as exp
from scripts.resume_chunking import durable_write,DecodedTapTransport

DSML=re.compile(r'\s*(?:<｜DSML｜tool_calls>\s*)?<｜DSML｜invoke name="(Check|Grade)">\s*(\{[\s\S]*\})\s*</｜DSML｜invoke>\s*(?:</｜DSML｜tool_calls>\s*)?')

def normalized(response,payload):
    tools=payload.get('tools') or []
    if len(tools)!=1:return None
    name=tools[0]['function']['name']
    if name not in ('Check','Grade'):return None
    choices=response.get('choices') or []
    if len(choices)!=1 or choices[0].get('finish_reason')!='tool_calls':return None
    message=choices[0].get('message') or {}
    if message.get('tool_calls'):return None
    match=DSML.fullmatch(message.get('content') or '')
    if not match or match[1]!=name:return None
    try:args=json.loads(match[2])
    except ValueError:return None
    if not isinstance(args,dict) or not isinstance(args.get('reason'),str):return None
    if name=='Check':
        if type(args.get('useful')) is not bool:return None
    else:
        if not isinstance(args.get('covered'),list) or not all(type(v) is bool for v in args['covered']):return None
        if type(args.get('faithful')) is not bool or type(args.get('refused')) is not bool:return None
    result=deepcopy(response)
    result['choices'][0]['message']['tool_calls']=[{'id':'dsml-normalized','type':'function',
        'function':{'name':name,'arguments':json.dumps(args,ensure_ascii=False)}}]
    return result

def install():
    exp.write=durable_write;exp.TapTransport=DecodedTapTransport;original=exp.Runner.api
    async def api(self,channel,payload,coords):
        if channel!='chat':return await original(self,channel,payload,coords)
        payload={**payload,'chat_template_kwargs':{'thinking':False}}
        def log(raw,norm,wire_id,from_cache):
            durable_write(self.out/'format-normalizations'/f'{exp.digest([wire_id,coords])}.json',{
                'source_wire_id':wire_id,'source_response_id':raw.get('id'),'coords':coords,
                'source_response_sha256':exp.digest(raw),'normalized_response':norm,
                'reason':'Exact DSML tool envelope in content with finish_reason=tool_calls; schema-checked, no semantic edit',
                'recovered_without_new_call':from_cache})
        # Recover only an existing failed stage from an identical saved request;
        # never choose among successful semantic answers or change a bool/grade.
        stage='generation' if coords['stage']=='absent_check' else coords['stage']
        case=self.out/stage/f"{coords['arm']}-{coords['id']}.json"
        if case.exists() and exp.read(case).get('status')=='error':
            candidates=[]
            for path in (self.out/'calls').glob('*.json'):
                wire=exp.read(path)
                if wire.get('coords')==coords and wire.get('payload')==payload and wire.get('complete') and wire.get('status')==200:
                    value=normalized(wire.get('response') or {},payload)
                    if value is not None:candidates.append((wire.get('started_at',0),wire,value))
            if candidates:
                _,wire,value=min(candidates,key=lambda x:x[0]);log(wire['response'],value,wire['id'],True);return value
        result=await original(self,channel,payload,coords);value=normalized(result,payload)
        if value is not None:
            wire_id=next((p.stem for p in (self.out/'calls').glob('*.json') if exp.read(p).get('response',{}).get('id')==result.get('id')),None)
            assert wire_id is not None
            log(result,value,wire_id,False);return value
        return result
    exp.Runner.api=api

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--stage',choices=('all','retrieval','generation','judge'),default='all');p.add_argument('--limit',type=int);args=p.parse_args()
    out=Path(args.output);assert not (out/'writer.lock').exists(),'writer active'
    durable_write(out/'recovery-phases'/f'{time.time_ns()}.json',{'adapter':'scripts/resume_chunking_v3.py',
        'sha256':exp.sha(Path(__file__)),'change':'Normalize only exact known DSML Check/Grade envelopes; recover prior failed format from identical first valid wire; raw result unmodified',
        'gitee_actual_nonthinking':True,'semantic_results_prompts_models_data_scores_changed':False})
    install();asyncio.run(exp.run(args))
