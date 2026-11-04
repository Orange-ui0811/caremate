"""Read-only supplementary interpretation; never changes primary labels/scores."""
from collections import Counter,defaultdict
from pathlib import Path
import json

OUT=Path('E:/Mewhelp/data/multi-intent-eval/20261104-p4-subflows')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    assert read(OUT/'completion.json')['complete']
    data=read(OUT/'summary.json');cases={c['id']:c for c in read(OUT/'cases.json')}
    rows=data['rows'];kinds={};risks={};extra={};sample={}
    arm_names={'disabled':'单意图基线','enabled':'P3最新版'}
    for arm in arm_names:
        by_kind=defaultdict(Counter);unsafe=[];selections=[]
        for row in rows:
            if row['arm']!=arm:continue
            for goal in cases[row['case_id']]['goals']:
                value=row['terminal'][goal['id']];cell=by_kind[goal['kind']]
                cell['total']+=1;cell['passed']+=value is True;cell['unknown']+=value is None
            if row['hard']['extra_selection_n']:
                selections.append({'rep':row['rep'],'case_id':row['case_id'],'count':row['hard']['extra_selection_n']})
            for risk in row['unsafe_claims']:
                unsafe.append({'rep':row['rep'],'case_id':row['case_id'],**risk})
        kinds[arm]={k:dict(v) for k,v in by_kind.items()};risks[arm]=unsafe;extra[arm]=selections
        ids=set(read(OUT/'protocol.json')['quality']['sample_case_ids'])
        sample[arm]={'n':len(ids),'scene_passed':sum(r['success'] is True for r in rows if r['arm']==arm and r['rep']==1 and r['case_id'] in ids)}
    audit=read(OUT/'knowledge-conflict-audit.json')
    excluded={c['case_id'] for c in audit.get('conservatively_flagged_cases',[])}|{c['case_id'] for c in audit.get('label_entity_mismatch_cases',[])}
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from scripts.score_multi120 import summarize
    sensitivity={a:summarize([r for r in rows if r['arm']==a and r['case_id'] not in excluded]) for a in arm_names}
    look={a:{(r['rep'],r['case_id']):r for r in rows if r['arm']==a} for a in arm_names}
    both=[key for key in look['disabled'].keys()&look['enabled'].keys() if all(look[a][key]['success'] is True for a in arm_names)]
    both_selection={a:sum(look[a][key]['hard']['extra_selection_n'] for key in both) for a in arm_names}
    failed=[{'arm':r['arm'],'rep':r['rep'],'case_id':r['case_id'],'source_status':r['source_status'],'judge_status':r['judge_status'],
             'false_goals':[g for g,v in r['terminal'].items() if v is False],
             'unknown_goals':[g for g,v in r['terminal'].items() if v is None],
             'hard':r['hard'],'risks':r['unsafe_claims']} for r in rows if r['success'] is not True]
    supplementary={'by_kind':kinds,'unsafe_claims':risks,'extra_selection_cases':extra,
        'known_conflict_sensitivity':{'excluded_cases':sorted(excluded),'summary':sensitivity},
        'both_success_pair_count':len(both),'both_success_extra_selection':both_selection,'quality_sample_completion':sample}
    write(OUT/'supplementary-analysis.json',supplementary);write(OUT/'failed-goal-index.json',failed)
    lines=['# P4补充分析：任务分项、澄清与回复质量','',
        '主成绩见REPORT.md。本文件只重组已保存结果；不覆盖模型标签、不修改任务评分、不重跑轨迹。','',
        '| 目标类型 | 单意图完成/总数（待定） | P3完成/总数（待定） |','|---|---:|---:|']
    for kind in sorted(set(kinds['disabled'])|set(kinds['enabled'])):
        x=kinds['disabled'][kind];y=kinds['enabled'][kind]
        lines.append(f"| {kind} | {x['passed']}/{x['total']}（{x['unknown']}） | {y['passed']}/{y['total']}（{y['unknown']}） |")
    lines+=['','## 多余选单','',
        f'两版都完整通过的配对有{len(both)}条；此条件子集中的多余选单分别为{both_selection["disabled"]}与{both_selection["enabled"]}。',
        '这是条件诊断，不是因果估计。主表仍报告全部轨迹的次数，同时保留遗漏事项和失败分母。',
        '逐次多余选单位置见supplementary-analysis.json，可回到runs/r*/版本/场景/result.json检查公开选单、原文与实际绑定。','',
        '## 回复质量与不确定性','']
    q=read(OUT/'quality-summary.json')
    for a,label in arm_names.items():
        valid=sum(x.get('evidence_valid') is True for x in risks[a])
        invalid=len(risks[a])-valid
        lines.append(f'- {label}：全量时序裁判记录{valid}条有有效原文的不安全声明候选，另{invalid}条引用无效。代表样本40条中，业务完整通过{sample[a]["scene_passed"]}条；可读性均值{q[a]["readability_mean"]}。')
    lines+=['','不安全声明和可读性均为模型裁判结果，不等于独立人工审计。引用有效只证明文本存在，不保证裁判推理正确；对具体争议应查看quote/reason与原事实。主评分保留原判断，未用助手事后复核替换。',
        '可读性高不能弥补遗漏业务；通用拒答也可能简洁。全量邀约/重复文本规则只产生候选信号，不把正常确认步骤、取消说明或必要限制当作自动质量扣分。','',
        '## 既有资料冲突敏感性','',
        '以下仅排除评测前已记录的4个存在资料/标注问题的场景作敏感性展示，主成绩和所有原始轨迹保持不变。',
        '涉及场景：'+', '.join(sorted(excluded))+'。','',
        '| 排除已知问题后的辅助口径 | 单意图 | P3 |','|---|---:|---:|']
    a=sensitivity['disabled'];b=sensitivity['enabled']
    lines+=[f"| 场景通过 | {a['success_n']}/{a['n']}，待定{a['unknown_n']} | {b['success_n']}/{b['n']}，待定{b['unknown_n']} |",
            f"| 目标完成 | {a['terminal_n']}/{a['task_n']}，待定{a['terminal_unknown_n']} | {b['terminal_n']}/{b['task_n']}，待定{b['terminal_unknown_n']} |",'',
            '失败目标、未知评分和硬约束证据索引见failed-goal-index.json；其余未发现的数据问题不因此视为不存在。']
    (OUT/'ANALYSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
