"""Read-only analysis of a completed and independently audited context experiment.

No model calls and no rescore. Adds cost decomposition, paired clean-subset
diagnostics and readable failures without overwriting the frozen run artifacts.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


ARMS = ('C_current', 'W_raw8k', 'S_state8k', 'R_recall8k')
FAMILIES = {
    'early_order': '早期订单', 'early_product': '早期商品',
    'correction': '编号更正', 'topic_switch': '话题切换',
    'multi_hop': '多跳关系', 'temporal_update': '时间更新',
    'negation': '否定/撤回', 'ambiguous': '对象歧义',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def analyze(out):
    audited = read(out/'audited_report.json')
    independent = read(out/'independent_audit.json')
    if not audited['complete'] or not audited['audit_passed'] or not independent['passed']:
        raise ValueError('Require complete results and both passing audits')
    data = read(out/'dataset.json')['cases']
    stats = audited['strategies']
    results = {}
    for c in data:
        for p in c['probes']:
            for a in ARMS:
                results[c['id'], p['id'], a] = read(out/'results'/f'{c["id"]}-{p["id"]}-{a}.json')
    calls = [read(p) for p in (out/'calls').glob('*.json')]
    clean = [(c['id'],p['id']) for c in data for p in c['probes']
             if all(all(results[c['id'],p['id'],a][s]['status']=='ok'
                        for s in ('coref','plan','reply')) for a in ARMS)]
    clean_cases = [c['id'] for c in data if all((c['id'],p['id']) in clean for p in c['probes'])]
    maintenance_by_case = {
        c['id']: dict(Counter(e['status'] for e in read(out/'maintenance'/f'{c["id"]}.json')))
        for c in data
    }
    details = {}
    for a in ARMS:
        rr = [r for (cid,pid,arm),r in results.items() if arm==a]
        records = [r[s] for r in rr for s in ('coref','plan','reply')]
        maintenance = [c for c in calls if c['contract']['stage']=='summary'] if a=='C_current' else []
        by_stage = {}
        for stage in ('coref','plan','reply','summary'):
            selected = [c for c in records+maintenance if c['contract']['stage']==stage]
            attempts = [t for c in selected for t in c['attempts']]
            by_stage[stage] = {
                'calls': len(selected), 'status': dict(Counter(c['status'] for c in selected)),
                'input': sum(t.get('usage',{}).get('input_tokens',0) or 0 for t in attempts),
                'output': sum(t.get('usage',{}).get('output_tokens',0) or 0 for t in attempts),
                'max_reported_input': max((t.get('usage',{}).get('input_tokens',0) or 0 for t in attempts), default=0),
                'repeated_format_instruction_attempts':sum(
                    str(t.get('raw_content','')).count('不要用正文代替工具参数')>=3 for t in attempts),
            }
        examples = []
        for c in data:
            for p in c['probes']:
                r = results[c['id'],p['id'],a]
                if r['score']['task_rule_v3']:
                    continue
                if len(examples)<5:
                    examples.append({'case':c['id'],'family':c['kind'],'probe':p['id'],
                                     'query':p['query'],'expected':p['expected'],
                                     'coref':r['coref']['value'],'plan':r['plan']['value'],
                                     'reply':r['reply']['value'],'score':r['score']})
        details[a] = {
            'stages':by_stage,
            'clean_probe_passed':sum(results[c,p,a]['score']['task_rule_v3'] for c,p in clean),
            'clean_probe_n':len(clean),
            'clean_conversation_passed':sum(results[c,p,a]['score']['task_rule_v3'] for c,p in clean if c in clean_cases),
            'clean_conversation_n':2*len(clean_cases),
            'target_correct_final_failed':sum(r['score']['target_pass'] and not r['score']['task_rule_v3'] for r in rr),
            'required_literal_visible':sum(r['score']['required_fact_literal_present'] for r in rr),
            'assembly_cpu_mean_ms':statistics.mean(r['assembly_cpu_seconds'] for r in rr)*1000,
            'obsolete_exception_changed_result':sum(r['score']['task_rule_v3'] and not r['score']['task_pass'] for r in rr),
            'failure_examples':examples,
            'by_history_turns':{
                n:{'passed':sum(results[c['id'],p['id'],a]['score']['task_rule_v3']
                                for c in data for p in c['probes'] if p['at_turn']==n),
                   'n':sum(p['at_turn']==n for c in data for p in c['probes'])}
                for n in sorted({p['at_turn'] for c in data for p in c['probes']})
            },
            'paired_improved':sum(r['score']['task_rule_v3'] and not results[c,p,'C_current']['score']['task_rule_v3']
                                  for (c,p,arm),r in results.items() if arm==a),
            'paired_regressed':sum(not r['score']['task_rule_v3'] and results[c,p,'C_current']['score']['task_rule_v3']
                                   for (c,p,arm),r in results.items() if arm==a),
        }
    analysis = {'by_arm':details,'clean_probe_ids':clean,'clean_case_ids':clean_cases,
                'maintenance_status_by_case':maintenance_by_case,
                'clean_subset_warning':'Selected after observing call validity; potentially biased. Full-set results remain primary.',
                'cost_warning':'Reported tokens only; transport exceptions may omit billed usage. Historical dialogue generation is excluded.'}
    (out/'analysis.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines = ['# 客服上下文工程：结果分解与落地边界','',
             '以下只读取冻结结果，不调用模型、不重试答错、不改评分。两项审计均通过。', '',
             '## 主结果（每组96次追问）','',
             '|方案|任务规则通过|对象/动作正确|相对当前差值及配对95%区间|维护+追问输入Token|维护+追问输出Token|三阶段P50/P95秒|',
             '|---|---:|---:|---|---:|---:|---:|']
    for a,s in stats.items():
        lo,hi=s['paired_conversation_bootstrap_95ci']
        delta=s['difference_vs_current']*100
        lines.append(f'|{s["label"]}|{s["passed"]}/{s["n"]} ({s["passed"]/s["n"]:.1%})|{s["target_passed"]}/{s["n"]}|{delta:+.1f}百分点 [{lo*100:+.1f}, {hi*100:+.1f}]|{s["usage"]["input_tokens"]:,}|{s["usage"]["output_tokens"]:,}|{s["p50_seconds"]:.2f}/{s["p95_seconds"]:.2f}|')
    lines += ['', '区间按48条完整会话进行配对bootstrap（4000次），不是把96次追问当作独立样本。四组单次模型采样；区间没有估计重新生成的随机波动。',
              '', '## 开销与故障拆分','',
              '|方案|阶段|调用数|结果状态|重复格式指令≥3次的尝试|输入Token|输出Token|最大单请求输入Token|',
              '|---|---|---:|---|---:|---:|---:|---:|']
    for a,d in details.items():
        for stage,s in d['stages'].items():
            if s['calls']:
                lines.append(f'|{stats[a]["label"]}|{stage}|{s["calls"]}|{json.dumps(s["status"],ensure_ascii=False)}|{s["repeated_format_instruction_attempts"]}|{s["input"]:,}|{s["output"]:,}|{s["max_reported_input"]:,}|')
    lines += ['', '输入/输出按最终正式运行的接口usage累计，含已记录重试；前两个诊断阶段及格式预检已消耗的Token另外归档，不混入策略主表，也不能把主表当本次工作总账单。历史轮次由数据集给定，未计入生成这些轮次的费用。模型池内部传输重试可能没有完整计费回执。延迟为三阶段调用等待累计，排除实验排队和后台摘要，不是线上首字延迟。',
              '', '## 共同有效调用子集（仅诊断）','',
              f'四组的改写、规划、答复均返回有效格式的共同追问：{len(clean)}/96；两个追问都符合该条件的共同会话：{len(clean_cases)}/48。未排除此前摘要维护失败，因此也不是纯记忆能力对照；这个子集由结果筛选，存在选择偏差。', '',
              '|方案|共同追问子集通过|共同完整会话子集通过|目标正确但最终未通过|旧规则→新规则额外通过|',
              '|---|---:|---:|---:|---:|']
    for a,d in details.items():
        lines.append(f'|{stats[a]["label"]}|{d["clean_probe_passed"]}/{d["clean_probe_n"]}|{d["clean_conversation_passed"]}/{d["clean_conversation_n"]}|{d["target_correct_final_failed"]}|{d["obsolete_exception_changed_result"]}|')
    lines += ['', '## 场景分布','', '|场景族|当前摘要窗口|原文窗口|来源状态|按需回查|','|---|---:|---:|---:|---:|']
    for kind in stats['C_current']['by_kind']:
        cells=[f'{stats[a]["by_kind"][kind]["passed"]}/{stats[a]["by_kind"][kind]["n"]}' for a in ARMS]
        lines.append('|'+FAMILIES.get(kind,kind)+'|'+'|'.join(cells)+'|')
    lines += ['', '## 追问时的历史长度','', '|历史轮数|当前摘要窗口|原文窗口|来源状态|按需回查|', '|---|---:|---:|---:|---:|']
    for n in details['C_current']['by_history_turns']:
        lines.append('|'+str(n)+'|'+'|'.join(f'{details[a]["by_history_turns"][n]["passed"]}/{details[a]["by_history_turns"][n]["n"]}' for a in ARMS)+'|')
    lines += ['', '## 对当前方案的逐条变化','', '|方案|当前未过而该方案通过|当前通过而该方案未过|', '|---|---:|---:|']
    for a,d in details.items():
        lines.append(f'|{stats[a]["label"]}|{d["paired_improved"]}|{d["paired_regressed"]}|')
    lines += ['', '## 可追溯失败样例','', '每组按固定数据顺序列出前5例失败。完整输出见results与analysis.json；这些样例不能代替总体分布。']
    for a,d in details.items():
        lines += ['', '### '+stats[a]['label'], '']
        for e in d['failure_examples']:
            lines += [f'- **{e["case"]}/{e["probe"]}，{e["family"]}**：{e["query"]}',
                      '  - 期望：'+json.dumps(e['expected'],ensure_ascii=False),
                      '  - 规划：'+json.dumps(e['plan'],ensure_ascii=False),
                      '  - 回答：'+str(e['reply']).replace('\n',' '),
                      '  - 阶段状态：'+json.dumps(e['score']['stage_status'],ensure_ascii=False)]
    lines += ['', '## 使用结论的边界','',
              '- 48条会话来自8个已知合成场景族，24～64轮，不是独立新场景或真实客服流量。',
              '- 当前组复用当前摘要提示词、触发规则和上下文拼装，四组调用器、输出上限、工具格式及关闭思考参数统一；不是未修改线上端到端测试。',
              '- 候选状态是带来源的原文索引和少量确定性候选槽位，不是完整业务状态机；尚未接入生产图。',
              '- W/S/R和C使用不同近似Token计数法，并同步扩大指代可见范围；差值不能只归因于窗口数值。',
              '- 规则通过不等于人工语义准确率。旧严格规则保留；新规则只豁免明确标为作废/未使用的引用。',
              '- 不用结果推导真实写操作安全、线上并发能力、完整会话成本或1M长上下文能力。',
              '- 最终上线前仍需独立客服场景、授权/取消状态及实时业务工具端到端验证。']
    (out/'ANALYSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return {a:{'passed':stats[a]['passed'],'clean':details[a]['clean_probe_passed']} for a in ARMS}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(json.dumps(analyze(Path(args.output)),ensure_ascii=False))
