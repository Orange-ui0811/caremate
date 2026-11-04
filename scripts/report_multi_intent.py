"""Produce a transparent report of paired multi-task outcomes, not reminder timing."""
import json
from pathlib import Path
from scripts.eval_multi_intent import OUT, ROOT, read, save
from scripts.judge_multi_intent import report


def pct(n,d):
    return f'{n}/{d}（{n/d:.1%}）' if d else '无数据'


def main():
    report()
    data=read(OUT/'summary.json');a=data['summary']['before'];b=data['summary']['after']
    raw=[read(p) for p in (OUT/'runs').glob('*/*/result.json')]
    judges=[read(p) for p in (OUT/'judges').glob('*/*.json')]
    errors=[{k:j.get(k) for k in ('case_id','arm','status','error')} for j in judges if j['status']!='ok']
    save(OUT/'scoring-errors.json',errors)
    v2=bool(read(OUT/'protocol.json').get('baseline_origin'))
    before,after=('V1固定规则','V2完整任务计划') if v2 else ('修改前','修改后')
    lines=['# 多意图任务能力：'+before+'与'+after+'真实模型实验','',
      '本实验评估多个业务诉求的回应、终态与续办；不以安全提醒率作为主指标。',
      f'30个固定场景、修改前/后两组，当前已采集{len(raw)}/60条轨迹，可共同评分{data["paired_n"]}/30对。',
      '使用真实DeepSeek-V4-Flash、BGE-M3、BGE-Reranker接口；业务订单/物流是两组共用的固定夹具，工单只写隔离MySQL库。',
      '执行图、真实RAG及表单API；未评估浏览器/SSE投递时序。任务执行采用确定性流程，旧路径保留原Agent；V2额外使用真实模型提取全部任务及各自参数。',
      ('V1复用上一轮已保存的30条轨迹及评分，V2按相同30题、知识快照和交互策略重新调用。两版本调用时间不同，非同步随机试验。' if v2 else '两组逐题执行，原始记录全部保留。'),'',
      f'| 主指标 | {before} | {after} |','| --- | ---: | ---: |',
      f'| 必需事项得到有效回应 | {pct(a["addressed_tasks"],a["required_tasks"])} | {pct(b["addressed_tasks"],b["required_tasks"])} |',
      f'| 必需事项达到本实验规定终态 | {pct(a["terminal_tasks"],a["required_tasks"])} | {pct(b["terminal_tasks"],b["required_tasks"])} |',
      f'| 整场景成功 | {pct(a["whole_scenario_success_n"],a["scored_n"])} | {pct(b["whole_scenario_success_n"],b["scored_n"])} |',
      f'| 错订单退款工单数 | {a["wrong_refund_orders"]} | {b["wrong_refund_orders"]} |',
      f'| 重复工单数 | {a["duplicate_tickets"]} | {b["duplicate_tickets"]} |',
      f'| 明确禁止类型的写入数 | {a["forbidden_writes"]} | {b["forbidden_writes"]} |','',
      f'| 分组整场景成功 | {before} | {after} |','| --- | ---: | ---: |']
    labels={'explicit_two':'明确退款+投诉','explicit_three':'订单/物流+退款+投诉','implicit_two':'口语/隐含双诉求',
            'continuation':'选单、取消与重启续办','controls':'单意图/否定控制'}
    for g in a['by_group']:
        x,y=a['by_group'][g],b['by_group'][g]
        lines.append(f'| {labels[g]} | {pct(x["success_n"],x["n"])} | {pct(y["success_n"],y["n"])} |')
    paired={arm:{r['case_id']:r for r in data['rows'] if r['arm']==arm} for arm in ('before','after')}
    gains=[i for i in data['paired_ids'] if not paired['before'][i]['all_success'] and paired['after'][i]['all_success']]
    regressions=[i for i in data['paired_ids'] if paired['before'][i]['all_success'] and not paired['after'][i]['all_success']]
    lines+=['',f'逐题变化：{len(gains)}题由失败变成功，{len(regressions)}题由成功变失败。',
            '新增退步场景：'+('、'.join(regressions) or '无')+'。']
    lines+=['','## 评分口径','',
      '- 有效回应：真实模型对可见客服答复逐项判定，物流和订单事实再做输出/业务调用校验。新task_plan只用于诊断，不作为成功证据。',
      '- 终态：退款要求正确订单1001的审核工单回执，或观察到用户明确取消；不能把“已展示入口”或内部DONE当提交成功。退款资格咨询与最终资金退款不在该提交目标内。',
      '- 对未明确退款订单的题，模拟用户计划在选单时选择1001；2002只出现在物流诉求中。未追问而直接借用2002办理退款也计为目标绑定失败。错对象计数包括这种未澄清情形，不能全部解释为违背了问题中明说的退款订单号。',
      '- 投诉通常评价是否有实质性承接/跟进入口；明确要求登记投诉工单时额外要求SQL回执。',
      '- 整场景成功：每项终态都满足，且没有错订单写入、明确禁止写入、重复工单或禁止的物流查询。',
      '- 两组使用相同预设操作策略：选1001、按场景分别确认或取消、遇计划重启则重建SQLite连接；旧版缺少对应卡片时记录缺失，不凭空构造resume参数。不额外追问提醒系统补做遗漏事项。',
      '- 因此终态成功率是固定卡片交互轨迹下的结果，不代表允许人工自由对话补救后的极限成功率。',
      '- 评分错误与接口失败独立保留；主表只比较两组均有有效评分的相同场景。','',
      'evidence-reviews.json记录模型裁判格式错误后的助手明文复核（如有），不属于独立人工标注；错误文件和原始答复保留。','',
      '## 限制','',
      '这30题是围绕当前能力构造的内部开发集，不能作为盲测。每组一次调用，模型有随机性；不作显著性或稳定泛化提升声明。',
      '新路径提供人工审核工单，旧路径可能进行资格判断；比较的是本次整体实现，不能将全部差异单独归因于调度器。',
      '没有HTTP网络掉线/浏览器恢复故障注入；续办组包含程序化恢复与重建SQLite连接，不等同于完整生产恢复验收。',
      '模拟用户按预设策略确认所提供的操作卡，没有人工纠正系统拿错订单；错误对象的工单是隔离库测试结果，不是真实用户资金损失。',
      '单模型裁判不是人工真值。发现过裁判引用并不存在的物流答复，已增加源答复和正确物流调用校验；原判断与修正原因均保留在逐题数据。',
      '本轮没有修改客服配置、知识库或生产数据库；共享模型池会正常记录调用与用量。','',
      '## 复现与原始记录','',
      '- cases.json：30题及动作策略；protocol.json：冻结源码哈希、模型与知识快照。',
      '- runs/：每组每题答复、中断、操作、业务回执、工具审计和错误。',
      '- judges/：原始逐事项判断；summary.json：事实校验后的逐题结果；scoring-errors.json：未成功评分记录。',
      '- 脚本：scripts/eval_multi_intent.py、scripts/judge_multi_intent.py、scripts/report_multi_intent.py。']
    if v2:
        failures=[r for r in data['rows'] if r['arm']=='after' and not r['all_success']]
        lines+=['','## V2失败诊断','']
        for row in failures:
            r=read(OUT/'runs/after'/row['case_id']/'result.json')
            failed=[k for k,v in row['completed'].items() if not v]
            lines.append(f'- {row["case_id"]}：{r["case"]["query"]}；未达终态：{", ".join(failed) or "约束违规"}。详见逐题轨迹。')
            first=r['steps'][0] if r.get('steps') else {}
            intr=first.get('interrupt') or {}
            if intr.get('type')=='select_order' and intr.get('task_kind')=='logistics':
                task=next((t for t in first.get('task_plan',{}).values() if t['kind']=='logistics'),{})
                lines.append('  诊断：物流绑定原文为“'+task.get('order_reference','')+
                    '”，未通过任务语义绑定校验，触发了额外选单；固定测试用户选择1001，故没有查询要求的2002。此结果不按“有物流输出”算成功。')
        holdout=OUT/'holdout-binding/results.json'
        if holdout.exists():
            h=read(holdout)
            lines+=['','## 冻结实现后的新订单绑定探针','',
                f'6条新构造问题，仅评任务和订单提取，不评业务全流程；精确匹配{h["exact_n"]}/{h["n"]}。',
                '探针使用不同订单号，在实现冻结后生成；未依据结果回调本轮实现。失败：']
            lines += ['- '+r['query']+'；实际：'+json.dumps(r['actual'],ensure_ascii=False) for r in h['rows'] if not r['exact_match']]
            lines+=['这6条仅为小规模补充检查，不能代表独立的大规模泛化评估。原始提取保存在holdout-binding/。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Report:',OUT/'REPORT.md')


if __name__=='__main__':main()
