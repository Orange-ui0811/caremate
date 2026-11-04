"""Offline replay and accounting for the isolated three-arm chunking study."""
import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
import random
import statistics

from scripts import eval_chunking as exp

def avg(values):
    values=[v for v in values if v is not None]
    return statistics.mean(values) if values else None

def percentile(values,p):
    values=sorted(values)
    return values[min(len(values)-1,math.ceil(len(values)*p)-1)] if values else None

def cluster_difference(samples,left,right,key):
    clusters=defaultdict(list)
    for sample in samples:
        sid=sample['id']
        if sid not in left or sid not in right:continue
        a,b=left[sid].get(key),right[sid].get(key)
        if a is None or b is None:continue
        groups=sample.get('expect_sections_all') or [sample['expect_section']]
        signature=exp.digest(sorted(sorted(g if isinstance(g,list) else [g]) for g in groups))
        clusters[signature].append(a-b)
    if not clusters:return None
    groups=list(clusters.values());rng=random.Random(20261028);draws=[]
    for _ in range(2000):
        values=[v for group in rng.choices(groups,k=len(groups)) for v in group]
        draws.append(statistics.mean(values))
    return {'difference':statistics.mean(v for g in groups for v in g),
        'cluster_bootstrap_95ci':[percentile(draws,.025),percentile(draws,.975)],
        'clusters':len(groups),'questions':sum(map(len,groups)),
        'scope':'resample expected-evidence-signature clusters; internal reused benchmark, not independent blind examples'}

def audit(out,partial=False):
    out=Path(out).resolve()
    if (out/'writer.lock').exists() and not partial:raise ValueError('Writer active')
    manifest=exp.read(out/'manifest.json');samples=exp.read(out/'samples.json');byid={s['id']:s for s in samples}
    issues=[];checked=Counter()
    def check(ok,code,where):
        checked[code]+=1
        if not ok:issues.append({'code':code,'where':str(where)})
    check(len(samples)==300 and len(byid)==300,'sample_count',out)
    check(Counter(s['bucket'] for s in samples)==dict.fromkeys(('A_policy','B_model','C_colloquial','D_absent','E_multi'),60),'bucket_counts',out)
    check(exp.sha(exp.ROOT/'tests/data/eval_ch04.jsonl')==manifest['dataset_sha256'],'dataset_unchanged',out)
    for name,value in manifest['sources'].items():
        check(exp.sha(exp.ROOT/name)==value,'current_source_hash',name)
        check(exp.sha(out/'source_snapshot'/name)==value,'frozen_source_hash',name)
    check(exp.protected()==manifest['protected'],'protected_files_unchanged',out)
    before,after=out/'production-before.json',out/'production-after.json'
    check(before.exists() and after.exists() and exp.read(before)==exp.read(after),'production_knowledge_unchanged',out)
    docs=exp.corpus();check(docs==exp.read(out/'corpus.json'),'corpus_replay',out)
    gold,missing=exp.make_gold(docs,samples);check(not missing and gold==exp.read(out/'gold.json'),'gold_mapping_replay',out)
    chunks={a:exp.read(out/'chunks'/f'{a}.json') for a in exp.ARMS}
    for arm in exp.ARMS:check(exp.make_chunks(docs,arm)==chunks[arm],'chunk_replay',arm)
    records={};groups={}
    for arm in exp.ARMS:
        phase={stage:{} for stage in ('retrieval','generation','judge')}
        for stage in phase:
            expected=[s for s in samples if stage!='judge' or not s['should_refuse']]
            for s in expected:
                path=out/stage/f'{arm}-{s["id"]}.json'
                if not path.exists():continue
                r=exp.read(path)
                if r.get('status')!='ok':continue
                check((r['arm'],r['id'],r['bucket'],r['stage'])==(arm,s['id'],s['bucket'],stage),'record_coordinates',path.name)
                phase[stage][s['id']]=r
            check(len(phase[stage])==len(expected),'complete_'+stage,arm)
        records[arm]=phase;bybucket={}
        for sid,r in phase['retrieval'].items():
            g=gold[sid];score=exp.score(r['hits'],g,chunks[arm]);dense=exp.score(r['dense_hits'],g,chunks[arm])
            check(score==r['metrics'] and dense==r['dense_metrics'],'retrieval_score_replay',f'{arm}/{sid}')
            cindex={c['id']:c for c in chunks[arm]}
            for hit in [*r['hits'],*r['dense_hits'],*(h for leaf in r['leaves'] for h in leaf['candidates'])]:
                original=cindex.get(int(hit['id']))
                check(original is not None and all(hit[k]==original[k] for k in ('question','answer','section_path','category','content_type')),'retrieval_source_identity',f'{arm}/{sid}')
        for sid,r in phase['generation'].items():
            if byid[sid]['should_refuse']:
                source=phase['retrieval'].get(sid)
                if source:
                    confidence=source['metrics']['max_rerank_score']
                    check(confidence==r['confidence'],'absent_confidence_lineage',f'{arm}/{sid}')
                    expected_refusal=(confidence<manifest['threshold'] or not r.get('check',{}).get('useful',False))
                    check(r['refused']==expected_refusal,'absent_gate_replay',f'{arm}/{sid}')
                    check(r['model_called']==(confidence>=manifest['threshold']),'absent_check_call_policy',f'{arm}/{sid}')
                continue
            source=out/'retrieval'/f'{arm}-{sid}.json'
            check(exp.sha(source)==r['retrieval_sha256'],'generation_retrieval_lineage',f'{arm}/{sid}')
            hits=exp.read(source)['hits'];evidence='\n'.join(f"[{i+1}] {h['question']}:{h['answer']}" for i,h in enumerate(hits))
            check(evidence==r['evidence'],'generation_evidence_identity',f'{arm}/{sid}')
        for sid,r in phase['judge'].items():
            check(exp.sha(out/'generation'/f'{arm}-{sid}.json')==r['generation_sha256'],'judge_generation_lineage',f'{arm}/{sid}')
            g=r['grade'];check(len(g['covered'])==len(byid[sid]['expect_points']) and all(type(v) is bool for v in g['covered']),'judge_shape',f'{arm}/{sid}')
        for bucket in ('A_policy','B_model','C_colloquial','E_multi','D_absent','answerable_overall'):
            ids=[s['id'] for s in samples if (not s['should_refuse'] if bucket=='answerable_overall' else s['bucket']==bucket)]
            rr=[phase['retrieval'][sid] for sid in ids if sid in phase['retrieval']]
            gg=[phase['generation'][sid] for sid in ids if sid in phase['generation']]
            jj=[phase['judge'][sid]['grade'] for sid in ids if sid in phase['judge']]
            if bucket=='D_absent':
                bybucket[bucket]={'retrieval_completed':len(rr),'checks_completed':len(gg),
                    'correct_refusals':sum(r['refused'] for r in gg),'false_accepts':sum(not r['refused'] for r in gg),
                    'refusal_rate':avg([r['refused'] for r in gg]),'fixed_threshold':manifest['threshold']}
            else:
                keys=('section_recall_at5','section_mrr_at10','source_point_coverage_at5','source_point_coverage_at10','all_points_at5','all_points_at10','mechanical_point_coverage','evidence_chars')
                bybucket[bucket]={'retrieval_completed':len(rr),'generated':len(gg),'judged':len(jj),
                    'hybrid_rerank':{k:avg([r['metrics'][k] for r in rr]) for k in keys},
                    'dense':{k:avg([r['dense_metrics'][k] for r in rr]) for k in keys},
                    'answer_point_coverage_judge':avg([avg(g['covered']) for g in jj]),
                    'all_answer_points_judge':avg([all(g['covered']) for g in jj]),
                    'faithful_judge':avg([g['faithful'] for g in jj]),'answer_refusal_rate_judge':avg([g['refused'] for g in jj])}
        groups[arm]={'label':exp.LABELS[arm],'chunks':len(chunks[arm]),
            'body_chars':sum(len(c['answer']) for c in chunks[arm]),'buckets':bybucket,
            'retrieval_latency_seconds':{'p50':percentile([r['seconds'] for r in phase['retrieval'].values()],.5),'p95':percentile([r['seconds'] for r in phase['retrieval'].values()],.95)},
            'generation_latency_seconds':{'p50':percentile([r['seconds'] for r in phase['generation'].values() if 'answer' in r],.5),'p95':percentile([r['seconds'] for r in phase['generation'].values() if 'answer' in r],.95)}}
        curve=[]
        for k in (1,3,5,10):
            values=[exp.score(r['hits'][:k],gold[sid],chunks[arm]) for sid,r in phase['retrieval'].items() if not byid[sid]['should_refuse']]
            curve.append({'k':k,'questions':len(values),'point_coverage':avg([v['source_point_coverage_at10'] for v in values]),
                'all_points':avg([v['all_points_at10'] for v in values]),'evidence_chars':avg([v['evidence_chars'] for v in values])})
        groups[arm]['coverage_vs_evidence_chars']=curve
    accounting={};calls=[]
    for path in (out/'calls').glob('*.json'):
        r=exp.read(path);calls.append(r);check(r.get('host')=='ai.gitee.com','wire_provider',path.name)
        payload=r['payload'];endpoint=r['endpoint'];cfg=manifest['config']
        model=cfg['embed_model'] if endpoint.endswith('embeddings') else cfg['rerank_model'] if endpoint.endswith('rerank') else cfg['chat_model']
        check(payload['model']==model,'wire_model',path.name)
        if endpoint.endswith('chat/completions'):
            check(payload.get('thinking')=={'type':'disabled'} and payload['temperature']==0,'wire_chat_config',path.name)
            if r.get('coords',{}).get('stage')=='generation':
                coord=r['coords'];s=byid[coord['id']]
                source=records[coord['arm']]['retrieval'].get(s['id'])
                if source:
                    evidence='\n'.join(f"[{i+1}] {h['question']}:{h['answer']}" for i,h in enumerate(source['hits']))
                    expected=[{'role':'system','content':exp.RAG_ANSWER_SYSTEM},
                        {'role':'user','content':f"用户问题:{s['query']}\n\n知识证据:\n{evidence}"}]
                    check(payload['messages']==expected,'generation_no_gold_and_prompt',path.name)
        coord=r.get('coords',{});stage=coord.get('stage','unknown');arm=coord.get('arm','shared')
        key=arm+'/'+stage
        if key not in accounting:accounting[key]={'attempts':0,'http_or_transport_failed':0,'known_usage_attempts':0,'unknown_usage_attempts':0,'known_input_tokens':0,'known_output_tokens':0,'known_total_tokens':0,'input_usage_attempts':0,'output_usage_attempts':0,'submitted_text_chars':0}
        a=accounting[key];a['attempts']+=1
        text_parts=payload.get('input') or payload.get('documents') or [m.get('content','') for m in payload.get('messages',[])]
        if isinstance(text_parts,str):text_parts=[text_parts]
        a['submitted_text_chars']+=sum(len(t) for t in text_parts if isinstance(t,str))+len(payload.get('query',''))
        if not r.get('complete') or r.get('status',0)>=400:a['http_or_transport_failed']+=1
        usage=(r.get('response') or {}).get('usage')
        if not usage:a['unknown_usage_attempts']+=1;continue
        a['known_usage_attempts']+=1
        a['input_usage_attempts']+=int('prompt_tokens' in usage or 'input_tokens' in usage)
        a['output_usage_attempts']+=int('completion_tokens' in usage or 'output_tokens' in usage)
        a['known_input_tokens']+=usage.get('prompt_tokens',usage.get('input_tokens',0)) or 0
        a['known_output_tokens']+=usage.get('completion_tokens',usage.get('output_tokens',0)) or 0
        a['known_total_tokens']+=usage.get('total_tokens',0) or 0
    differences={}
    for baseline in ('fixed','recursive'):
        left={sid:r['metrics'] for sid,r in records['structured']['retrieval'].items()}
        right={sid:r['metrics'] for sid,r in records[baseline]['retrieval'].items()}
        differences['structured_minus_'+baseline]={k:cluster_difference([s for s in samples if not s['should_refuse']],left,right,k)
            for k in ('source_point_coverage_at5','all_points_at5','section_recall_at5')}
    failures={a:{'missing_points_at5':[sid for sid,r in records[a]['retrieval'].items() if byid[sid]['expect_points'] and not r['metrics']['all_points_at5']],
        'judge_not_faithful':[sid for sid,r in records[a]['judge'].items() if not r['grade']['faithful']],
        'absent_false_accepts':[sid for sid,r in records[a]['generation'].items() if byid[sid]['should_refuse'] and not r['refused']]} for a in exp.ARMS}
    result={'complete':not any(x['code'].startswith('complete_') for x in issues),'audit_passed':not issues,
        'issues':issues,'checked':dict(checked),'groups':groups,'accounting':accounting,'wire_attempts':len(calls),
        'paired_differences':differences,'failure_ids':failures,'scope_limits':manifest['limits'],
        'audit_script_sha256':exp.sha(Path(__file__))}
    exp.write(out/('PARTIAL_REPORT.json' if partial else 'audited_report.json'),result)
    if not partial:
        lines=['# 三种分块方法对照实验','',f"完成：{result['complete']}；审计通过：{result['audit_passed']}。",'',
            '现有300题五桶，各桶60题；可答240题，库外60题。原始六份文档，未新增挑战样本。',
            '固定Top10证据，主检索为混合召回＋重排；纯向量结果作为补充。回答是统一提示词的单轮受控生成，不是完整客服Workflow。','',
            '| 方法 | 块数 | 证据要点覆盖@5 | 全要点命中@5 | 标题Recall@5（次要） | 答案要点覆盖（模型判分） | D桶误放 |',
            '|---|---:|---:|---:|---:|---:|---:|']
        fmt=lambda v:'未完成' if v is None else f'{v:.2%}'
        for a,g in groups.items():
            b=g['buckets']['answerable_overall'];h=b['hybrid_rerank'];d=g['buckets']['D_absent']
            lines.append(f"| {g['label']} | {g['chunks']} | {fmt(h['source_point_coverage_at5'])} | {fmt(h['all_points_at5'])} | {fmt(h['section_recall_at5'])} | {fmt(b['answer_point_coverage_judge'])} | {d['false_accepts']}/{d['checks_completed']} |")
        lines+=['','D桶误放指固定阈值＋证据充分性闸门放行，不是已生成错误答案的比例。答案忠实性为模型裁判代理指标，不能写成人工认证幻觉率。','',
            '## 用量（包括所有已记录重试）','', '| 分组/阶段 | 调用尝试 | 已知输入Token | 已知输出Token | 已知总Token | 用量未知尝试 |','|---|---:|---:|---:|---:|---:|']
        for k,a in sorted(accounting.items()):
            shown_input=a['known_input_tokens'] if a['input_usage_attempts'] else '未返回'
            shown_output=a['known_output_tokens'] if a['output_usage_attempts'] else '未返回'
            shown_total=a['known_total_tokens'] if a['known_usage_attempts'] else '未返回'
            lines.append(f"| {k} | {a['attempts']} | {shown_input} | {shown_output} | {shown_total} | {a['unknown_usage_attempts']} |")
        lines+=['','嵌入结果按完整文本缓存并在组间共享，shared/embedding不重复摊入每组。重排仅返回total_tokens时，输入/输出列不据此推算；未知用量不是0。','',
            '## 证据覆盖与文本量','', '| 方法 | TopK | 要点覆盖 | 全要点命中 | 平均证据字符数 |','|---|---:|---:|---:|---:|']
        for a,g in groups.items():
            for v in g['coverage_vs_evidence_chars']:
                shown_chars='未完成' if v['evidence_chars'] is None else f"{v['evidence_chars']:.1f}"
                lines.append(f"| {g['label']} | {v['k']} | {fmt(v['point_coverage'])} | {fmt(v['all_points'])} | {shown_chars} |")
        lines+=['','## 解释范围','',*['- '+s for s in manifest['limits']], '',
            '章节标题命中可能高估大块的检索效果，应优先看原文要点覆盖及全要点命中。400字符不等于400Token；当前方法的完整句重叠、表格拆分不是硬长度上限。',
            '三组元数据按覆盖的原文小节统一生成；固定/递归跨小节时携带多个标题。因此结论包含完整切分与元数据组织策略，不能归因于单个组件。',
            '延迟仅为本次受控实验的观测值；查询向量共享缓存、并发排队和阶段顺序都会影响它，不是线上P95延迟。',
            '运行期错误、原始模型输出、证据、判分和逐次用量均保留；重算入口：python -m scripts.audit_chunking --output '+str(out), '',
            '具体分桶、纯向量结果、配对置信区间及失败题号见 audited_report.json。']
        (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--partial',action='store_true');args=p.parse_args()
    r=audit(args.output,args.partial);print({'complete':r['complete'],'audit_passed':r['audit_passed'],'issues':len(r['issues']),'wire_attempts':r['wire_attempts']})
