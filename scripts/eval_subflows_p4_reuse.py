"""User scope correction: original complete baseline + fresh candidate360 only.

Already collected fresh baseline records remain untouched as supplementary data.
"""
import argparse,asyncio,json,os,sys,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import eval_subflows_p4 as core
OUT=core.OUT;BASE=core.BASE


def raw(arm,rep,cid):
    return (BASE if arm=='disabled' else OUT)/'runs'/f'r{rep}'/arm/cid/'result.json'


def prepare():
    core.verify_inputs()
    if (OUT/'comparison-protocol.json').exists():raise RuntimeError('Already frozen')
    p=core.read(OUT/'protocol.json');paths=[]
    for f in (BASE/'runs').glob('*/disabled/*/result.json'):paths.append(f)
    for f in (BASE/'judges-chronological').glob('*/disabled/*.json'):paths.append(f)
    assert len(paths)==720
    core.save(OUT/'comparison-protocol.json',{
        'reason':'User requested reuse of already-tested baseline rather than redundant720 business rerun',
        'baseline':'Original360 disabled trajectories + original chronological judgments, no source-reviewed overrides',
        'candidate':'360 fresh enabled trajectories from P3, continue existing results without resampling',
        'original_protocol_sha256':core.sha(OUT/'protocol.json'),
        'baseline_hashes':{str(f.relative_to(BASE)):core.sha(f) for f in paths},
        'original_inputs_unchanged':True,'model_config':p['model_config'],
        'quality':p['quality'],'limits':['Baseline and candidate called at different times; not contemporary randomized pairs.',
            'Extra fresh disabled results from superseded plan retained, excluded from primary comparison.'],
        'script_sha256':core.sha(Path(__file__))})
    shutil.copy2(__file__,OUT/'evaluator-source/eval_subflows_p4_reuse.py')


async def run():
    core.verify_inputs();sem=asyncio.Semaphore(6)
    async def one(c,rep):
        async with sem:
            directory=OUT/'runs'/f'r{rep}'/'enabled'/c['id'];directory.mkdir(parents=True,exist_ok=True)
            if (directory/'result.json').exists():return
            with (directory/'worker.log').open('w',encoding='utf-8') as log:
                proc=await asyncio.create_subprocess_exec(sys.executable,str(ROOT/'scripts/eval_subflows_p4.py'),'worker',
                    '--arm','enabled','--case',c['id'],'--rep',str(rep),cwd=str(ROOT),stdout=log,stderr=log,
                    env={**os.environ,'PYTHONIOENCODING':'utf-8'})
                try:rc=await asyncio.wait_for(proc.wait(),620)
                except TimeoutError:proc.kill();await proc.wait();rc=-999
            if not (directory/'result.json').exists():core.save(directory/'result.json',{
                'case':c,'case_id':c['id'],'rep':rep,'arm':'enabled','status':'worker_error','exit_code':rc,
                'steps':[],'tickets':[],'actor':[],'authorized':[],'tool_audit':[]})
            print(len(list((OUT/'runs').glob('*/enabled/*/result.json'))),'/360',rep,c['id'],rc,flush=True)
    cs=sorted(core.read(OUT/'cases.json'),key=lambda c:(int(c['id'].split('-')[1]),c['group']))
    for rep in (1,2,3):await asyncio.gather(*(one(c,rep) for c in cs))


async def judge():
    core.wire()
    from scripts import score_multi120_chronological as scorer
    scorer.OUT=OUT;sem=asyncio.Semaphore(2)
    async def one(f):
        async with sem:await scorer.judge(f)
    while True:
        fs=list((OUT/'runs').glob('*/enabled/*/result.json'))
        missing=[f for f in fs if not (OUT/'judges-chronological'/f.parents[2].name/'enabled'/(f.parent.name+'.json')).exists()]
        await asyncio.gather(*(one(f) for f in missing))
        n=len(list((OUT/'judges-chronological').glob('*/enabled/*.json')))
        core.save(OUT/'candidate-progress.json',{'runs':len(fs),'judges':n,'baseline_reused':360})
        if len(fs)==n==360:break
        await asyncio.sleep(8)
    from app.core.model_transport import close_clients
    await close_clients()


async def quality():
    core.wire()
    from pydantic import BaseModel,Field
    from app.core import llm,model_pool
    from scripts.score_multi120_chronological import timeline
    class Issue(BaseModel):
        kind:str;event_id:str;quote:str;reason:str
    class Quality(BaseModel):
        readability:int=Field(ge=1,le=5)
        issues:list[Issue];comment:str
    sem=asyncio.Semaphore(2)
    async def one(arm,cid):
        async with sem:
            dest=OUT/'quality'/arm/(cid+'.json')
            if dest.exists():return
            r=core.read(raw(arm,1,cid));events=timeline(r)
            texts={e['event_id']:e['content'] for e in events if e['role']=='assistant'}
            if not texts:core.save(dest,{'status':'no_answer'});return
            errors=[]
            for attempt in range(2):
                try:
                    with model_pool.call_context(f"{core.read(OUT/'protocol.json')['request_namespace']}-quality-reused-{arm}-{cid}-{attempt}",priority='background'):
                        result=await llm.structured(Quality,temperature=0).ainvoke([
                            {'role':'system','content':core.QUALITY_PROMPT},{'role':'user','content':json.dumps({
                                'query':r['case']['query'],'goals':r['case']['goals'],'timeline':events,
                                'orders':r['case']['orders'],'logistics':r['case']['logistics']},ensure_ascii=False)}])
                    data=result.model_dump();core.save(dest.parent/'attempts'/f'{cid}-{attempt}.json',data)
                    for i in data['issues']:
                        i['evidence_valid']=bool(i['quote'].strip()) and i['event_id'] in texts and ''.join(i['quote'].split()) in ''.join(texts[i['event_id']].split())
                    core.save(dest,{'status':'ok',**data,'errors':errors});return
                except Exception as exc:errors.append(str(exc))
            core.save(dest,{'status':'judge_error','errors':errors})
    await asyncio.gather(*(one(a,cid) for a in core.ARMS for cid in core.read(OUT/'protocol.json')['quality']['sample_case_ids']))
    from app.core.model_transport import close_clients
    await close_clients()


def report():
    core.wire()
    from scripts.score_multi120_chronological import evaluate,timeline
    from scripts.score_multi120 import summarize
    from collections import Counter
    rows=[];flags=[]
    for arm in core.ARMS:
        origin=BASE if arm=='disabled' else OUT
        for rep in (1,2,3):
            for case in core.read(OUT/'cases.json'):
                r=core.read(raw(arm,rep,case['id']))
                j=core.read(origin/'judges-chronological'/f'r{rep}'/arm/(case['id']+'.json'))
                rows.append(evaluate(r,j))
                texts=[e['content'] for e in timeline(r) if e['role']=='assistant'];text='\n'.join(texts)
                import re
                flags.append({'arm':arm,'rep':rep,'case_id':case['id'],
                    'internal_state_candidate':bool(re.search(r'本次事项处理结果|\b(?:WAITING_USER|PENDING|task_id|operation_id)\b',text)),
                    'invitation_candidate':bool(re.search(r'需要我|还有其他|随时告诉我|如需.*(?:告诉我|继续.*查询)',text)),
                    'exact_repeated_reply_candidate':len(texts)!=len(set(texts)),'visible_reply_n':len(texts)})
    summary={a:summarize([r for r in rows if r['arm']==a]) for a in core.ARMS}
    old=core.read(BASE/'summary-chronological.json')['summary']['disabled']
    assert all(summary['disabled'][k]==old[k] for k in ['n','success_n','terminal_n','task_n','unknown_n'])
    look={a:{(r['rep'],r['case_id']):r for r in rows if r['arm']==a} for a in core.ARMS}
    common=look['disabled'].keys()&look['enabled'].keys()
    data={'summary':summary,'rows':rows,'paired_n':len(common),
        'paired_improved':sum(look['enabled'][k]['success'] is True and look['disabled'][k]['success'] is False for k in common),
        'paired_regressed':sum(look['enabled'][k]['success'] is False and look['disabled'][k]['success'] is True for k in common),
        'repetitions':{str(i):{a:summarize([r for r in rows if r['arm']==a and r['rep']==i]) for a in core.ARMS} for i in (1,2,3)}}
    core.save(OUT/'summary.json',data);core.save(OUT/'quality-diagnostics.json',flags)
    qs={}
    for arm in core.ARMS:
        labels=[core.read(p) for p in (OUT/'quality'/arm).glob('*.json')];good=[x for x in labels if x['status']=='ok']
        qs[arm]={'sample_n':len(labels),'valid_n':len(good),'readability_mean':sum(x['readability'] for x in good)/len(good) if good else None,
            'issues':dict(Counter(i['kind'] for x in good for i in x['issues'] if i['evidence_valid'])),
            'unverified_quotes':sum(not i['evidence_valid'] for x in good for i in x['issues'])}
    core.save(OUT/'quality-summary.json',qs)
    a,b=summary['disabled'],summary['enabled'];rate=lambda n,d:f'{n}/{d}（{n/d:.1%}）'
    lines=['# P4：复用原单意图基线，与P3最新版对比','',
        '主比较：原已完成的单意图360条轨迹及其原时序自动评分，和P3最新版360条真实接口轨迹。不是重新运行720条业务；中途多跑的基线原样保留，但不进入主表。','',
        '| 指标 | 原单意图基线 | P3最新版 |','|---|---:|---:|',
        f"| 场景通过 | {rate(a['success_n'],a['n'])} | {rate(b['success_n'],b['n'])} |",
        f"| 场景待定 | {a['unknown_n']} | {b['unknown_n']} |",
        f"| 必需事项目标完成 | {rate(a['terminal_n'],a['task_n'])} | {rate(b['terminal_n'],b['task_n'])} |",
        f"| 目标待定 | {a['terminal_unknown_n']} | {b['terminal_unknown_n']} |",
        f"| 有效回应 | {rate(a['response_n'],a['task_n'])} | {rate(b['response_n'],b['task_n'])} |"]
    for key,label in [('extra_selection_n','多余选单'),('duplicate_ticket_n','重复工单'),('wrong_refund_order_n','错订单退款工单'),
                      ('unauthorized_ticket_n','未确认写入'),('forbidden_read_n','禁止查询'),('forbidden_ticket_n','禁止类型工单')]:
        lines.append(f"| {label} | {a['hard'].get(key,0)} | {b['hard'].get(key,0)} |")
    lines+=['','| 分组 | 基线通过 | P3通过 |','|---|---:|---:|']
    for group,label in core.read(OUT/'protocol.json')['groups'].items():
        x=a['by_group'][group];y=b['by_group'][group];lines.append(f"| {label} | {x['success_n']}/{x['n']} | {y['success_n']}/{y['n']} |")
    lines+=['','## 回复质量','',
        '全量沿用原时序语义与事实评分；预先选定的40个语义族代表场景、每版40条公开轨迹，另外做同口径盲评。基线质量复核直接读取旧回复，不重新运行业务。',
        f"- 基线：有效样本{qs['disabled']['valid_n']}/40，可读性均值{qs['disabled']['readability_mean']}，原文有效的问题计数{qs['disabled']['issues']}。",
        f"- P3：有效样本{qs['enabled']['valid_n']}/40，可读性均值{qs['enabled']['readability_mean']}，原文有效的问题计数{qs['enabled']['issues']}。",'',
        '具体原文、问题理由与引用有效性见quality/；全量规则候选见quality-diagnostics.json。模型评分不是独立人工审计，可读性不能替代业务完成。','',
        '## 口径与保留','',
        '- 原120题、目标、知识、动作策略、主评分公式逐字节保持；原基线原始结果和原自动标签不改，不套用助手来源复核标签。',
        '- 每版均按120场景×3轮统计，774项必需目标；不挑最好一轮。未知结果保留，不静默删除失败。',
        '- 退款/投诉完成指正确且经确认的隔离SQL审核工单；不是实际资金退款。额外选单要与完成任务数一起看。',
        '- 基线与新版调用时间不同，不是同期随机配对实验；按相同场景/轮次对齐只用于诊断。',
        '- 最新版使用真实Gitee模型、BGE-M3、BGE-Reranker及固定业务夹具；未做真实浏览器/网络投递评分。原知识54条，写入独立SQL和检查点。',
        '- 作者可见的内部回归集，语义族/变体/重复相关；已知4个资料/标注问题保持，辅助敏感性分析不替代主表。',
        '- 原720重测计划、预检错误、已跑的新基线和中止日志均保留；用户指出重复工作后，仅继续新版缺失轨迹。',
        '- 原客服配置、知识内容及既有结果未修改，业务源码冻结在P3。','',
        '资料：comparison-protocol.json、scope-adjustment.json、CASES.md、summary.json、runs/、judges-chronological/、quality/、verification.json。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    core.save(OUT/'failure-index.json',[r for r in rows if r['success'] is not True])


async def verify():
    core.verify_inputs();p=core.read(OUT/'comparison-protocol.json')
    assert core.sha(Path(__file__))==p['script_sha256']
    assert core.sha(OUT/'protocol.json')==p['original_protocol_sha256']
    changed=[name for name,h in p['baseline_hashes'].items() if core.sha(BASE/name)!=h]
    from app.kb import milvus_client
    kb=await milvus_client.acall(lambda:milvus_client.get_client().query('knowledge',filter='',
        output_fields=['id','question','answer','section_path','content_type','category'],limit=16384))
    kb.sort(key=lambda x:x['id'])
    result={'candidate_runs':len(list((OUT/'runs').glob('*/enabled/*/result.json'))),
        'candidate_judges':len(list((OUT/'judges-chronological').glob('*/enabled/*.json'))),
        'baseline_reused':360,'extra_fresh_baseline_retained':len(list((OUT/'runs').glob('*/disabled/*/result.json'))),
        'quality_samples':len(list((OUT/'quality').glob('*/*.json'))),'baseline_changed':changed,
        'knowledge_unchanged':kb==core.read(OUT/'knowledge.json'),
        'original300_unchanged':core.sha(core.HOME/'tests/data/eval_ch04.jsonl')==core.read(OUT/'protocol.json')['original300_sha256'],
        'cases_identical':core.sha(OUT/'cases.json')==core.sha(BASE/'cases.json'),
        'scoring_identical':core.sha(OUT/'scoring.json')==core.sha(BASE/'scoring.json'),
        'customer_service_restarted':False,'primary_comparison':'original baseline360 + candidate360, not fresh720'}
    core.save(OUT/'verification.json',result)
    assert result['candidate_runs']==result['candidate_judges']==360 and result['quality_samples']==80
    assert not changed and result['knowledge_unchanged'] and result['original300_unchanged'] and result['cases_identical'] and result['scoring_identical']


async def full():
    with (OUT/'candidate-execution.log').open('a',encoding='utf-8') as log,(OUT/'candidate-judging.log').open('a',encoding='utf-8') as jl:
        proc=await asyncio.create_subprocess_exec(sys.executable,__file__,'run',cwd=str(ROOT),stdout=log,stderr=log)
        jp=await asyncio.create_subprocess_exec(sys.executable,__file__,'judge',cwd=str(ROOT),stdout=jl,stderr=jl)
        rc=await proc.wait()
        if rc:jp.terminate();await jp.wait();raise RuntimeError('Candidate execution orchestration failed')
        if await jp.wait():raise RuntimeError('Candidate judge failed')
    await quality();report();await verify()
    core.save(OUT/'completion.json',{'complete':True,'baseline_reused':True,'time':time.strftime('%Y-%m-%d %H:%M:%S')})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run','judge','quality','report','verify','full']);a=parser.parse_args()
    if a.stage in ('prepare','report'):globals()[a.stage]()
    else:asyncio.run(globals()[a.stage]())
