"""Resume experiment with audited file/wire adapters and optional Gitee mode control.

--nonthinking is for a separately frozen generation phase: Gitee requires the
template thinking=False switch. It must never mix with earlier thinking output.
"""
import argparse
import asyncio
import base64
import json
import os
from pathlib import Path
import time
import uuid
import httpx

from scripts import eval_chunking as exp

def durable_write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    try:
        for attempt in range(12):
            try:temp.replace(path);return
            except PermissionError:
                if attempt==11:raise
                time.sleep(.02*(attempt+1))
    finally:
        if temp.exists():temp.unlink()


class DecodedStream(httpx.AsyncByteStream):
    def __init__(self,stream,record,path):
        self.stream,self.record,self.path=stream,record,path;self.body=bytearray()
    async def __aiter__(self):
        try:
            async for part in self.stream:
                self.body.extend(part);yield part
            self.record['complete']=True
        except BaseException as exc:
            self.record['error_type']=type(exc).__name__;raise
        finally:
            self.record['raw_body_base64']=base64.b64encode(self.body).decode('ascii')
            try:
                decoded=httpx.Response(self.record['status'],headers=self.record['safe_response_headers'],content=bytes(self.body))
                self.record['response']=decoded.json()
            except (ValueError,UnicodeError,httpx.DecodingError):
                self.record['response_parse_error']=True
            self.record['seconds']=time.monotonic()-self.record.pop('_start')
            durable_write(self.path,self.record)
    async def aclose(self):await self.stream.aclose()


class DecodedTapTransport(httpx.AsyncBaseTransport):
    def __init__(self,out):self.out=out;self.inner=httpx.AsyncHTTPTransport(retries=0)
    async def handle_async_request(self,request):
        rid=uuid.uuid4().hex;path=self.out/'calls'/f'{rid}.json'
        record={'id':rid,'coords':exp.COORD.get(),'payload':json.loads(await request.aread()),
            'host':request.url.host,'endpoint':request.url.path,'complete':False,
            'started_at':time.time(),'recorder':'decoded-wire-v2','_start':time.monotonic()}
        durable_write(path,{k:v for k,v in record.items() if k!='_start'})
        try:response=await self.inner.handle_async_request(request)
        except BaseException as exc:
            record.update(error_type=type(exc).__name__,seconds=time.monotonic()-record.pop('_start'));durable_write(path,record);raise
        record.update(status=response.status_code,safe_response_headers={k:v for k,v in response.headers.items()
            if k.lower() in ('content-encoding','content-type')})
        durable_write(path,{k:v for k,v in record.items() if k!='_start'})
        return httpx.Response(response.status_code,headers=response.headers,
            stream=DecodedStream(response.stream,record,path),extensions=response.extensions)
    async def aclose(self):await self.inner.aclose()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    parser.add_argument('--stage',default='all',choices=('all','retrieval','generation','judge'));parser.add_argument('--limit',type=int)
    parser.add_argument('--nonthinking',action='store_true')
    args=parser.parse_args();out=Path(args.output)
    assert not (out/'writer.lock').exists(),'writer active'
    durable_write(out/'recovery-phases'/f'{time.time_ns()}.json',{
        'reason':'Resume failed/unwritten work; handle Windows replacement contention and decode compressed HTTP wire responses for accounting',
        'adapter':'scripts/resume_chunking.py','adapter_sha256':exp.sha(Path(__file__)),
        'change':'unique temporary filenames, transient replacement retry, preserve raw response bytes and decode HTTP compression; unchanged request payloads and experiment logic',
        'models_prompts_data_scores_changed':False,'prior_failures_preserved':True,
        'gitee_actual_nonthinking':args.nonthinking,
        'pid':os.getpid(),'time':time.time()})
    exp.write=durable_write
    exp.TapTransport=DecodedTapTransport
    if args.nonthinking:
        original_api=exp.Runner.api
        async def nonthinking_api(self,channel,payload,coords):
            if channel=='chat':payload={**payload,'chat_template_kwargs':{'thinking':False}}
            return await original_api(self,channel,payload,coords)
        exp.Runner.api=nonthinking_api
    asyncio.run(exp.run(args))
