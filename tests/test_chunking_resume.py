from pathlib import Path
from scripts.resume_chunking import durable_write
from scripts.eval_chunking import read
import gzip
import json
import httpx
import pytest
from scripts.resume_chunking import DecodedTapTransport

def test_retry_file_contention_without_changing_record(tmp_path,monkeypatch):
    actual=Path.replace;attempts=[]
    def blocked_twice(self,target):
        attempts.append(target)
        if len(attempts)<=2:raise PermissionError('transient reader lock')
        return actual(self,target)
    monkeypatch.setattr(Path,'replace',blocked_twice)
    path=tmp_path/'progress.json';value={'stage':'retrieval','status':'ok','n':160}
    durable_write(path,value)
    assert read(path)==value and len(attempts)==3
    assert list(tmp_path.glob('*.tmp'))==[]


async def test_capture_gzip_usage_and_raw_replay(tmp_path):
    tap=DecodedTapTransport(tmp_path);await tap.inner.aclose()
    original={'choices':[],'usage':{'prompt_tokens':15,'completion_tokens':4,'total_tokens':19}}
    body=gzip.compress(json.dumps(original).encode())
    async def response(request):
        return httpx.Response(200,headers={'content-encoding':'gzip'},stream=httpx.ByteStream(body))
    tap.inner=httpx.MockTransport(response)
    async with httpx.AsyncClient(transport=tap) as client:
        actual=await client.post('https://example.test/v1/chat/completions',json={'model':'fake'})
        assert actual.json()==original
    r=read(next((tmp_path/'calls').glob('*.json')))
    assert r['response']==original and r['complete']
    import base64
    assert gzip.decompress(base64.b64decode(r['raw_body_base64']))==json.dumps(original).encode()
