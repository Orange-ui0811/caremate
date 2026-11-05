"""User-requested post-hoc scope change: remove primary bucket C, no new model calls."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import shutil

from scripts.score_multi120 import summarize

HOME=Path('E:/Mewhelp')
SOURCE=HOME/'data/multi-intent-eval/20261105-orchestration150-once'
DEST=HOME/'data/multi-intent-eval/20261105-orchestration120-no-c'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')


def exclude_c(cases,mapping,rows):
    kept=[copy.deepcopy(c) for c in cases if mapping[c['id']]['bucket']!='C_clarification']
    excluded=[copy.deepcopy(c) for c in cases if mapping[c['id']]['bucket']=='C_clarification']
    ids={c['id'] for c in kept}
    return kept,excluded,[copy.deepcopy(r) for r in rows if r['case_id'] in ids]


def build(source=SOURCE,dest=DEST):
    if dest.exists():raise RuntimeError('Never overwrite a derived scope or original results')
    inputs=['cases.json','bucket-map.json','buckets.json','summary.json','scoring.json','protocol.json','knowledge.json','verification.json','REPORT.md']
    before={name:sha(source/name) for name in inputs}
    original=read(source/'cases.json');mapping=read(source/'bucket-map.json');scores=read(source/'summary.json')
    kept,excluded,rows=exclude_c(original,mapping,scores['rows'])
    assert len(kept)==120 and len(excluded)==30 and len(rows)==240
    ids={c['id'] for c in kept};newmap={k:v for k,v in mapping.items() if k in ids}
    buckets=[b for b in read(source/'buckets.json') if b['id']!='C_clarification']
    assert Counter(v['bucket'] for v in newmap.values())==Counter({b['id']:30 for b in buckets})
    dest.mkdir(parents=True)
    save(dest/'cases.json',kept);save(dest/'excluded-C.json',excluded);save(dest/'bucket-map.json',newmap);save(dest/'buckets.json',buckets)
    (dest/'eval_orchestration120_no_c.jsonl').write_text('\n'.join(json.dumps({**c,**newmap[c['id']]},ensure_ascii=False) for c in kept)+'\n',encoding='utf-8')
    contract=read(source/'scoring.json')
    contract.update(version='orchestration120-no-c-derived',main_case_ids=[k for k in contract['main_case_ids'] if k in ids],control_case_ids=[k for k in contract['control_case_ids'] if k in ids],
                    main_goals=sum(len(c['goals']) for c in kept if len(c['goals'])>1),control_goals=sum(len(c['goals']) for c in kept if len(c['goals'])==1),
                    derivation='Same completed single-round trajectories, labels and formulas; remove all30 primary-C cases from both arms by explicit user request. No rerun or relabeling.')
    save(dest/'scoring.json',contract);shutil.copy2(source/'knowledge.json',dest/'knowledge.json')
    origins=[]
    for arm in ['disabled','enabled']:
        for c in kept:
            raw=source/'runs/r1'/arm/c['id']/'result.json';label=source/'judges/r1'/arm/c['id']/'label.json'
            assert raw.exists() and label.exists()
            origins.append({'arm':arm,'case_id':c['id'],'rep':1,'result_path':str(raw),'result_sha256':sha(raw),'label_path':str(label),'label_sha256':sha(label)})
    save(dest/'origins.json',origins)
    summaries={a:{s:summarize([r for r in rows if r['arm']==a and (s=='all' or r['split']==s)]) for s in ['main','control','all']} for a in ['disabled','enabled']}
    bybucket={a:{b['id']:summarize([r for r in rows if r['arm']==a and r['bucket']==b['id']]) for b in buckets} for a in summaries}
    save(dest/'summary.json',{'summary':summaries,'by_bucket':bybucket,'rows':rows,'new_model_calls':0,'new_business_trajectories':0})
    save(dest/'failure-index.json',[r for r in rows if r['success'] is not True])
    p=read(source/'protocol.json')
    protocol={'version':'orchestration120-no-c-derived','user_scope_change':'去除C桶','source':str(source),'source_artifact_hashes':before,
              'scope_change_after_results_seen':True,'removed_primary_bucket':'C_clarification','removed_cases':[c['id'] for c in excluded],
              'retained_cases':[c['id'] for c in kept],'buckets':{b['id']:30 for b in buckets},
              'main_cases':len(contract['main_case_ids']),'control_cases':len(contract['control_case_ids']),
              'source_repetitions':1,'new_model_calls':0,'new_business_trajectories':0,'reused_trajectories':240,
              'model_config':p['model_config'],'source_app_snapshots':{'path':str(source/'source'),'hashes':p['source_hashes']},
              'limits':['User-requested post-hoc scope subset, not an improvement to model or architecture.',
                        'Only primary C removed. D restart-with-selection cases remain unchanged; this does not remove every missing-order tag.',
                        '102 multi-task main cases and18 single-task controls reported separately; unknown stays in denominators.',
                        'This is not the older20261103 original120 dataset. Original150 and all raw results remain preserved.']}
    save(dest/'protocol.json',protocol)
    fmt=lambda n,d:f'{n}/{d}（{n/d:.1%}）'
    a,b=summaries['disabled']['main'],summaries['enabled']['main']
    lines=['# 去除C桶后的四桶对比','','按用户要求，从已完成的150题单轮结果中移出C桶30题，保留A/B/D/E各30题，共120题。两版使用完全相同范围，复用原240条轨迹；未重新运行模型、业务或评分。原150题和原报告不改。','','这是看过结果后的范围调整，分数变化不代表模型或架构改进；也不代表缺参澄清能力已解决。仅移出主桶C，D桶的重连选单题仍保留。此120题不是旧20261103版120题。','','## 多任务主集（102题，18题单任务控制另报）','','| 指标 | 基线 | 当前多意图 |','|---|---:|---:|']
    for label,key,den in [('场景通过','success_n','n'),('必需任务完成','terminal_n','task_n')]:lines.append(f'| {label} | {fmt(a[key],a[den])} | {fmt(b[key],b[den])} |')
    lines.append(f'| 场景待定 | {a["unknown_n"]} | {b["unknown_n"]} |')
    for label,key in [('全部实际选单','observed_selector_n'),('显式订单额外选单','extra_selection_n'),('重复工单','duplicate_ticket_n'),('未经确认写入','unauthorized_ticket_n')]:lines.append(f'| {label} | {a["hard"][key]} | {b["hard"][key]} |')
    lines+=['','## 四桶（每桶30题，E含单任务控制）','','| 桶 | 基线通过 | 多意图通过 |','|---|---:|---:|']
    for bucket in buckets:
        x,y=bybucket['disabled'][bucket['id']],bybucket['enabled'][bucket['id']];lines.append(f'| {bucket["name"]} | {fmt(x["success_n"],x["n"])} | {fmt(y["success_n"],y["n"])} |')
    lines+=['','## 全部120题总览（含控制）','','| 指标 | 基线 | 当前多意图 |','|---|---:|---:|']
    x,y=summaries['disabled']['all'],summaries['enabled']['all']
    for label,key,den in [('全部场景通过','success_n','n'),('全部目标完成','terminal_n','task_n')]:lines.append(f'| {label} | {fmt(x[key],x[den])} | {fmt(y[key],y[den])} |')
    x,y=summaries['disabled']['control'],summaries['enabled']['control'];lines.append(f'| 单任务控制通过 | {fmt(x["success_n"],x["n"])} | {fmt(y["success_n"],y["n"])} |')
    lines+=['','未知不算通过，不删失败。原评分为自动评审与SQL/确认核验，不是独立人工真值；接口及评分故障仍保留。一轮记录不能证明稳定性。','','数据：cases.json与eval_orchestration120_no_c.jsonl；移出题：excluded-C.json；同题原始轨迹和标签来源：origins.json；统计：summary.json；来源指纹与限制：protocol.json。','']
    (dest/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    lines=['# 四桶保留题（120题）','','移出C桶后，题目、目标、标注、业务夹具均保持原样。','']
    for bucket in buckets:
        lines += [f'## {bucket["id"]} {bucket["name"]}','']
        for c in kept:
            if newmap[c['id']]['bucket']==bucket['id']:lines += [f'**{c["id"]}** {c["query"]}','']
    (dest/'CASES.md').write_text('\n'.join(lines),encoding='utf-8')
    assert all(sha(source/n)==h for n,h in before.items())
    assert all(sha(Path(r['result_path']))==r['result_sha256'] and sha(Path(r['label_path']))==r['label_sha256'] for r in origins)
    save(dest/'verification.json',{'original150_files_unchanged':True,'original_raw_results_and_labels_unchanged':True,'same120_cases_both_arms':True,'only_primary_C_excluded':True,'new_model_calls':0,'new_business_trajectories':0})
    print(json.dumps({'summary':summaries,'buckets':{k:{b:[v['success_n'],v['n']] for b,v in bs.items()} for k,bs in bybucket.items()}},ensure_ascii=False))


if __name__=='__main__':build()
