"""Recover transport failures, reusing every unaffected result including wrong ones."""
import argparse
import asyncio
import hashlib
from pathlib import Path
from langchain_core.messages import SystemMessage
from scripts import eval_context_full_history_control as control
from scripts.eval_context_policy_v3 import Caller, LIMITS, EXTRA_BODY, digest, write_json, writer_lock, ROOT, stamp


class RecoveryCaller(Caller):
    def __init__(self,out):
        super().__init__(out)
        self.sem=asyncio.Semaphore(1)
        self.original=Path(control.read(out/'manifest.json')['original_control'])

    async def call(self,key,stage,messages,schema=None,model=None):
        ms=messages
        if schema:ms=[SystemMessage(ms[0].content+'\n请调用 '+schema.__name__+' 工具返回结构化结果，不要用正文代替工具参数。'),*ms[1:]]
        manifest=control.read(self.out/'manifest.json')
        expected={'messages':[m.model_dump(mode='json') for m in ms],'stage':stage,
                  'model':model or manifest['config']['chat_model'],'max_tokens':LIMITS[stage],
                  'schema':schema.model_json_schema() if schema else None,
                  'tool_choice':'auto' if schema else None,'extra_body':EXTRA_BODY}
        source=control.read(self.original/'calls'/(key+'.json'))
        if source['status']!='transport_error' and source['input_sha256']==digest(expected):
            write_json(self.out/'calls'/(key+'.json'),source)
            for i,a in enumerate(source['attempts'],1):write_json(self.out/'attempts'/(key+f'-{i}.json'),a)
            decision='reused_unchanged_even_if_semantically_wrong'
            result=source
        else:
            decision='transport_recovery' if source['status']=='transport_error' else 'dependency_input_changed_after_upstream_recovery'
            result=await super().call(key,stage,messages,schema,model)
        write_json(self.out/'recovery_decisions'/(key+'.json'),{'decision':decision,'original_input':source['input_sha256'],'new_input':digest(expected),'original_status':source['status']})
        return result


def prepare(original,out):
    if out.exists():raise ValueError('new recovery output required')
    old=control.read(original/'manifest.json')
    if not control.read(original/'audited_report.json')['audit_passed']:raise ValueError('original audit required')
    out.mkdir(parents=True)
    write_json(out/'dataset.json',control.read(original/'dataset.json'))
    sources={**old['sources']};name='scripts/recover_context_full_history_control.py'
    sources[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    for name,h in sources.items():
        data=(ROOT/name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=h:raise ValueError('source changed '+name)
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    write_json(out/'manifest.json',{**old,'created_at':stamp(),'sources':sources,
        'concurrency':1,'original_control':str(original.resolve()),
        'scope':old['scope']+' Transport-only recovery at concurrency 1; unchanged successes and wrong answers reused, dependent stages regenerated only if their input changes.'})


def recovery_audit(out):
    manifest=control.read(out/'manifest.json');original=Path(manifest['original_control'])
    counts={};issues=[];new_attempts=[];all_decisions={}
    for path in (out/'recovery_decisions').glob('*.json'):
        d=control.read(path);counts[d['decision']]=counts.get(d['decision'],0)+1
        old=control.read(original/'calls'/path.name);new=control.read(out/'calls'/path.name)
        all_decisions[path.stem]=d
        if d['decision']=='reused_unchanged_even_if_semantically_wrong':
            if old!=new or old['status']=='transport_error':issues.append('invalid reuse '+path.name)
        else:
            new_attempts+=new['attempts']
            if old['status']!='transport_error' and old['input_sha256']==new['input_sha256']:issues.append('forbidden semantic retry '+path.name)
    for key,d in all_decisions.items():
        if d['decision']=='dependency_input_changed_after_upstream_recovery':
            root,stage=key.rsplit('-',1)
            upstream=['coref'] if stage=='plan' else ['coref','plan']
            if not any(all_decisions[root+'-'+s]['decision']!='reused_unchanged_even_if_semantically_wrong' for s in upstream):issues.append('no repaired dependency '+key)
    original_attempts=[a for p in (original/'calls').glob('*.json') for a in control.read(p)['attempts']]
    result={'passed':not issues and len(all_decisions)==288,'issues':issues,'decisions':counts,
        'additional_reported_usage':{k:sum(a.get('usage',{}).get(k,0) or 0 for a in new_attempts) for k in ('input_tokens','output_tokens','total_tokens')},
        'original_plus_recovery_reported_usage':{k:sum(a.get('usage',{}).get(k,0) or 0 for a in original_attempts+new_attempts) for k in ('input_tokens','output_tokens','total_tokens')},
        'missing_usage_attempts_including_original_failures':sum(not a.get('usage') for a in original_attempts+new_attempts)}
    write_json(out/'recovery_audit.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--original');p.add_argument('--output',required=True);p.add_argument('--prepare',action='store_true')
    args=p.parse_args();out=Path(args.output).resolve()
    if args.prepare:prepare(Path(args.original),out)
    else:
        control.Caller=RecoveryCaller
        with writer_lock(out):asyncio.run(control.run(out))
        print(recovery_audit(out),flush=True)
