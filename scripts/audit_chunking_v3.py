"""Additional wire/control/lineage audit for corrected nonthinking generation."""
import argparse
import base64
from collections import Counter, defaultdict
from pathlib import Path
import json
import statistics
import httpx
from scripts import eval_chunking as exp
from scripts import audit_chunking as base
from scripts.resume_chunking_v3 import normalized

def audit(out,partial=False):
    out=Path(out).resolve();result=base.audit(out,partial)
    manifest=exp.read(out/'manifest.json');issues=[];checked=Counter()
    def check(ok,code,where):
        checked[code]+=1
        if not ok:issues.append({'code':code,'where':str(where)})
    normalizations={}
    for path in (out/'format-normalizations').glob('*.json'):
        item=exp.read(path);source=exp.read(out/'calls'/f"{item['source_wire_id']}.json")
        check(exp.digest(source['response'])==item['source_response_sha256'],'normalization_source_hash',path.name)
        check(normalized(source['response'],source['payload'])==item['normalized_response'],'strict_dsml_normalization_replay',path.name)
        normalizations[item['source_wire_id']]=item['normalized_response']
    parent=manifest['retrieval_parent'];parentdir=Path(parent['output'])
    check(exp.sha(parentdir/'manifest.json')==parent['manifest_sha256'],'parent_manifest_unchanged',parentdir)
    for name,value in parent['records'].items():
        check(exp.sha(parentdir/'retrieval'/name)==value and exp.sha(out/'retrieval'/name)==value,'retrieval_reused_exactly',name)
    wires=defaultdict(list);costs={a:{'successful_answer_requests':0,'input_tokens':0,'output_tokens':0,'total_tokens':0,'cached_input_tokens':0,'reasoning_tokens':0} for a in exp.ARMS}
    for path in (out/'calls').glob('*.json'):
        r=exp.read(path);coord=r['coords'];wires[(coord['stage'],coord.get('arm'),coord.get('id'))].append(r)
        payload=r['payload'];check(payload.get('chat_template_kwargs')=={'thinking':False},'actual_off_request_control',path.name)
        if 'raw_body_base64' not in r:continue
        raw=base64.b64decode(r['raw_body_base64'])
        if not raw:continue
        decoded=httpx.Response(r['status'],headers=r['safe_response_headers'],content=raw).json()
        check(decoded==r.get('response'),'compressed_wire_replay',path.name)
        if not r.get('complete') or r.get('status')!=200:continue
        check(decoded.get('model','').lower()==manifest['config']['chat_model'].lower(),'response_model_identity',path.name)
        usage=decoded.get('usage') or {};reasoning=(usage.get('completion_tokens_details') or {}).get('reasoning_tokens',0)
        check(reasoning==0 and all(not c.get('message',{}).get('reasoning_content') for c in decoded.get('choices',[])),'actual_off_response_verified',path.name)
        check(all(k in usage for k in ('prompt_tokens','completion_tokens','total_tokens')),'wire_usage_present',path.name)
    for arm in exp.ARMS:
        for path in (out/'generation').glob(f'{arm}-*.json'):
            r=exp.read(path)
            if r.get('status')!='ok':continue
            if 'answer' not in r:continue
            valid=[w for w in wires[('generation',arm,r['id'])] if w.get('complete') and w.get('status')==200
                and any(c.get('finish_reason')=='stop' and c.get('message',{}).get('content')==r['answer'] for c in w.get('response',{}).get('choices',[]))]
            check(len(valid)==1,'answer_has_one_successful_wire',path.name)
            if valid:
                usage=valid[0]['response']['usage'];c=costs[arm];c['successful_answer_requests']+=1
                c['input_tokens']+=usage['prompt_tokens'];c['output_tokens']+=usage['completion_tokens'];c['total_tokens']+=usage['total_tokens']
                c['cached_input_tokens']+=(usage.get('prompt_tokens_details') or {}).get('cached_tokens',0)
                c['reasoning_tokens']+=(usage.get('completion_tokens_details') or {}).get('reasoning_tokens',0)
        for stage in ('generation','judge'):
            for path in (out/stage).glob(f'{arm}-*.json'):
                r=exp.read(path)
                if r.get('status')!='ok':continue
                target=r.get('grade') if stage=='judge' else r.get('check')
                if target is None:continue
                wirestage='judge' if stage=='judge' else 'absent_check';matches=[]
                for w in wires[(wirestage,arm,r['id'])]:
                    if not w.get('complete') or w.get('status')!=200:continue
                    effective=normalizations.get(w['id'],w.get('response',{}))
                    for choice in effective.get('choices',[]):
                        for call in choice.get('message',{}).get('tool_calls',[]):
                            try:args=json.loads(call['function']['arguments'])
                            except (ValueError,KeyError):continue
                            if args==target and choice.get('finish_reason')!='length':matches.append(w['id'])
                check(len(set(matches))==1,'structured_result_has_one_successful_wire',path.name)
    # These are real expenses too, but never mix the initial uncontrolled
    # thinking answers into formal quality comparisons or pretend unknown=0.
    preliminary={}
    for source,label in ((parentdir/'calls','initial_and_retrieval'),(parentdir/'thinking-probe/calls','synthetic_control_probes')):
        for path in source.glob('*.json'):
            r=exp.read(path);coord=r.get('coords',{});key=label+'/'+coord.get('stage','unknown')
            if key not in preliminary:preliminary[key]={'attempts':0,'usage_unknown':0,'known_input_tokens':0,'known_output_tokens':0,'known_total_tokens':0,'compressed_response_not_archived':0}
            c=preliminary[key];c['attempts']+=1;usage=(r.get('response') or {}).get('usage')
            if r.get('response_parse_error') and 'raw_body_base64' not in r:c['compressed_response_not_archived']+=1
            if not usage:c['usage_unknown']+=1;continue
            c['known_input_tokens']+=usage.get('prompt_tokens',0);c['known_output_tokens']+=usage.get('completion_tokens',0);c['known_total_tokens']+=usage.get('total_tokens',0)
    result['issues']+=issues;result['audit_passed']=not result['issues'];result['v3_checked']=dict(checked)
    result['successful_generation_usage']=costs;result['preliminary_and_retrieval_accounting']=preliminary
    result['quality_control_note']=manifest['protocol_correction']
    result['retrieval_wire_limit']='Initial recorder did not decode/archive compressed wire response bodies. Retrieval candidate/result JSON and embedding vectors remain; scores/source can be replayed, raw rerank response equality cannot be audited. Unknown usage is never zero.'
    result['known_label_ambiguity']={'id':'D3','main_300_unchanged':True,
        'reason':'Question asks mailing an invoice to a different address; bundled billing-shipping source permits specifying paper-invoice mailing address at application. The absent label is ambiguous; do not automatically interpret D3 acceptance as a genuine unsupported-service error.',
        'sensitivity_scope':'Exploratory subtraction only; primary D60 counts remain unchanged.'}
    sensitivity={}
    for arm in exp.ARMS:
        rows=[exp.read(p) for p in (out/'generation').glob(f'{arm}-D*.json')]
        usable=[r for r in rows if r.get('status')=='ok' and r['id']!='D3']
        sensitivity[arm]={'n':len(usable),'false_accepts':sum(not r['refused'] for r in usable),
            'D3':next(({'refused':r['refused'],'reason':r.get('check',{}).get('reason',r.get('reason'))} for r in rows if r.get('status')=='ok' and r['id']=='D3'),None)}
    result['D3_exploratory_sensitivity']=sensitivity
    statistics_by_arm={}
    for arm in exp.ARMS:
        chunks=exp.read(out/'chunks'/f'{arm}.json');bydoc=defaultdict(list)
        for c in chunks:bydoc[c['document']].extend(c['source_spans'])
        mapped=sum(b-a for c in chunks for a,b in c['source_spans'])
        unique=sum(b-a for spans in bydoc.values() for a,b in exp.merged(spans))
        statistics_by_arm[arm]={'chunks':len(chunks),'body_characters':sum(len(c['answer']) for c in chunks),
            'mapped_source_characters_with_repetitions':mapped,'unique_source_characters':unique,
            'repeated_mapped_source_characters':mapped-unique,
            'max_body_characters':max(len(c['answer']) for c in chunks)}
    result['chunk_statistics']=statistics_by_arm
    answer_values={a:{} for a in exp.ARMS}
    for arm in exp.ARMS:
        for path in (out/'judge').glob(f'{arm}-*.json'):
            row=exp.read(path)
            if row.get('status')!='ok':continue
            g=row['grade'];answer_values[arm][row['id']]={'point_coverage':statistics.mean(g['covered']),
                'all_points':all(g['covered']),'faithful':g['faithful']}
    samples=[s for s in exp.read(out/'samples.json') if not s['should_refuse']]
    result['answer_paired_differences']={baseline:{key:base.cluster_difference(samples,answer_values['structured'],answer_values[baseline],key)
        for key in ('point_coverage','all_points','faithful')} for baseline in ('fixed','recursive')}
    result['format_normalizations']={'count':len(normalizations),'source_wire_ids':sorted(normalizations),
        'scope':'Exact DSML Check/Grade envelope normalized into equivalent tool_calls; no free text fallback, no unknown tool, no bool coercion or judgment change.'}
    review_path=out/'ANNOTATION_AND_JUDGE_REVIEW.json'
    if review_path.exists():
        result['manual_annotation_and_judge_review']=exp.read(review_path)
        result['manual_review_sha256']=exp.sha(review_path)
    result['supplemental_auditor_sha256']=exp.sha(Path(__file__))
    exp.write(out/('PARTIAL_REPORT.json' if partial else 'audited_report.json'),result)
    if not partial:
        with (out/'REPORT.md').open('a',encoding='utf-8') as f:
            f.write('\n## 正式生成输入与输出用量\n\n| 方法 | 成功回答数 | 输入Token | 输出Token | 总Token | 平均输入Token |\n|---|---:|---:|---:|---:|---:|\n')
            for arm,c in costs.items():
                n=c['successful_answer_requests'];mean=c['input_tokens']/n if n else 0
                f.write(f"| {exp.LABELS[arm]} | {n} | {c['input_tokens']} | {c['output_tokens']} | {c['total_tokens']} | {mean:.1f} |\n")
            f.write('\n此表仅比较240道可答题的成功回答请求；不含拒答检查、裁判、嵌入、重排与失败重试。完整阶段用量见前表；初始预跑与接口探测的额外开销单列在JSON，部分用量未知，不能把此表称为实验全部成本。\n')
            f.write('\n## 执行修正与保留记录\n\n初始请求thinking.type=disabled仍返回思考内容，造成一题输出额度耗尽。初始生成完整保留于v2但不进入正式比较；三组正式生成全部重新开始并增加经合成接口探测验证的chat_template_kwargs.thinking=false。模型、原文、问题、提示词、评分和检索结果不变。正式响应逐条检查思考内容与思考Token均为零。\n')
            f.write('\n初始HTTP记录器遗漏压缩解码，早期响应正文和用量无法从其日志恢复。已修复并保留后续原始压缩字节；早期检索候选与结果、嵌入向量和调用请求仍保留，可以复算评分，但不能声称检索原始响应已全部通过逐次wire审计。\n')
            f.write('\n既有D3题询问发票寄往另一地址，而原文允许申请纸质发票时备注邮寄地址，其库外标签存在歧义。主结果仍保留D60；JSON另外提供排除D3的探索性敏感性结果，不替换主分数。\n')
            f.write('\nD桶使用同一个现行阈值，未分别重新校准；它反映相同运行条件下的闸门表现，不代表各方法可达到的最佳拒答效果。配对置信区间按既有证据族重采样，不覆盖模型跨次运行的随机波动。\n')
            f.write(f"\n严格DSML格式归一化共{len(normalizations)}条：只接受finish_reason=tool_calls、已知Check/Grade工具名与类型正确的JSON，恢复等价tool_calls，不改变判断值。原始响应与转换记录均保留并独立复算。\n")
            f.write('\n人工检查还发现既有E27的保修期限要点可能超出布艺商品适用范围，且一条裁判布尔值与理由不一致；E40的初始化要点是否属于断电问题必需证据也存在疑问。统一提示词禁止承诺具体时效，而部分旧要点要求时效。主分数不作事后修正，详见ANNOTATION_AND_JUDGE_REVIEW.json；这些数字是相对既有标注的内部代理指标，不能直接等同业务正确率。\n')
            f.write(f"\n补充控制审计：{not issues}；问题数：{len(issues)}。重算：python -m scripts.audit_chunking_v3 --output {out}\n")
        current=costs['structured'];fixed=costs['fixed'];recursive=costs['recursive']
        if all(c['successful_answer_requests']==240 for c in costs.values()):
            saved=1-current['input_tokens']/fixed['input_tokens']
            saved_recursive=1-current['input_tokens']/recursive['input_tokens']
            summary=(f"\n相同300题在三组分别测试：900项检索、720份回答、180项库外拒答检查、720份回答评分。未新增题目。\n\n"
                f"主要结论：当前结构感知方案的成功回答平均输入为{current['input_tokens']/240:.1f} Token，"
                f"固定长度为{fixed['input_tokens']/240:.1f}，递归分块为{recursive['input_tokens']/240:.1f}；分别减少{saved:.2%}、{saved_recursive:.2%}。"
                "这一比较仅含240道可答题的成功回答请求，不含建库、重排、拒答检查、裁判、失败重试或初始预跑。\n\n"
                "当前方案的既有要点检索覆盖与模型判分略低于固定长度，不能宣称召回更高或效果已证明等同。"
                "本轮体现的是文本量与覆盖的取舍；旧标注适用性及裁判一致性问题另行保留，不能将覆盖率写成真实业务正确率。\n")
            path=out/'REPORT.md';text=path.read_text(encoding='utf-8');first,rest=text.split('\n',1)
            path.write_text(first+'\n'+summary+rest,encoding='utf-8')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--partial',action='store_true');args=p.parse_args()
    result=audit(args.output,args.partial);print({'complete':result['complete'],'audit_passed':result['audit_passed'],'issues':len(result['issues'])})
