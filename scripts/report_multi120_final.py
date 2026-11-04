"""Explain the unchanged frozen scores, uncertainty, data conflicts and resource tradeoffs."""
from collections import Counter
import json
import statistics

from scripts.eval_multi120 import OUT,read,save
from scripts.score_multi120 import summarize


def main():
    corrected=(OUT/'summary-chronological.json').exists()
    reviewed=(OUT/'summary-source-reviewed.json').exists()
    data=read(OUT/('summary-source-reviewed.json' if reviewed else 'summary-chronological.json' if corrected else 'summary.json'))
    rows=data['rows'];p=read(OUT/'protocol.json')
    conflict=read(OUT/'knowledge-conflict-audit.json') if (OUT/'knowledge-conflict-audit.json').exists() else {}
    flagged={x['case_id'] for x in conflict.get('conservatively_flagged_cases',[])}
    flagged.update(x['case_id'] for x in conflict.get('label_entity_mismatch_cases',[]))
    api=read(OUT/'api-summary.json') if (OUT/'api-summary.json').exists() else {}
    analyses={}
    for arm in ('disabled','enabled'):
        rs=[r for r in rows if r['arm']==arm]
        stable=[]
        for cid in sorted({r['case_id'] for r in rs}):
            values=[r['success'] for r in rs if r['case_id']==cid]
            stable.append({'case_id':cid,'pass_n':sum(v is True for v in values),'fail_n':sum(v is False for v in values),
                           'unknown_n':sum(v is None for v in values),'runs':len(values)})
        analyses[arm]={
            'normal':summarize([r for r in rs if r['scenario_class']=='normal']),
            'control':summarize([r for r in rs if r['scenario_class']=='control']),
            'fault':summarize([r for r in rs if r['scenario_class']=='fault']),
            'without_flagged_sections_sensitivity':summarize([r for r in rs if r['case_id'] not in flagged]),
            'stable_all3_success_n':sum(x['runs']==3 and x['pass_n']==3 for x in stable),
            'case_stability':stable,
            'unsafe_claim_label_n':sum(len(r['unsafe_claims']) for r in rs),
            'judge_missing_or_error_n':sum(r['judge_status'] not in ('ok','no_answer') for r in rs),
            'judge_no_answer_n':sum(r['judge_status']=='no_answer' for r in rs)}
    save(OUT/'analysis.json',analyses)
    def bounds(s):
        if not s['n']:return '尚无数据'
        value=f"{s['success_n']}/{s['n']}（{s['success_n']/s['n']:.1%}）"
        if s['unknown_n']:value+=f"；待定{s['unknown_n']}，范围{s['success_n']/s['n']:.1%}～{(s['success_n']+s['unknown_n'])/s['n']:.1%}"
        return value
    a,b=[data['summary'][k] for k in ('disabled','enabled')]
    lines=['# 原300题派生120业务场景：正式对比报告','',
        f"本次构造120个完整业务场景，来源于原300道RAG题中的{p['source_questions_used']}道，原题文件未改；8类各15题。每场景每模式运行3次，计划720条真实模型轨迹，目前{len(rows)}/720条。",
        '比较对象为同一份冻结代码的新功能关闭（原单主意图路径）与开启（多任务规划与调度路径）；不是上一轮V1固定规则工作流与V2的比较。双方全部重新调用，没有复用旧分数。','',
        '| 指标 | 原单主意图路径 | 多任务路径 |','|---|---:|---:|',
        f'| 场景验收（时序评分及已注明来源复核） | {bounds(a)} | {bounds(b)} |',
        f"| 必需事项目标终态 | {a['terminal_n']}/{a['task_n']}；待定{a['terminal_unknown_n']} | {b['terminal_n']}/{b['task_n']}；待定{b['terminal_unknown_n']} |",
        f"| 3次全部通过的场景 | {analyses['disabled']['stable_all3_success_n']}/120 | {analyses['enabled']['stable_all3_success_n']}/120 |"]
    for cls,label in [('normal','正常业务场景'),('control','单意图/否定/库外控制'),('fault','注入物流不可用后的正确处理')]:
        lines.append(f"| {label} | {bounds(analyses['disabled'][cls])} | {bounds(analyses['enabled'][cls])} |")
    for key,label in [('wrong_refund_order_n','错订单退款工单'),('wrong_preview_n','错误或未要求的操作预览'),
        ('duplicate_ticket_n','重复工单'),('unauthorized_ticket_n','未经确认的工单写入'),('forbidden_read_n','禁止的查询'),
        ('forbidden_ticket_n','禁止类型工单'),('extra_selection_n','多余选单')]:
        lines.append(f"| {label} | {a['hard'].get(key,0)} | {b['hard'].get(key,0)} |")
    lines+=['','## 八类结果','', '| 类别 | 原单主意图路径 | 多任务路径 |','|---|---:|---:|']
    for group,label in p['groups'].items():
        def one(arm):
            s=data['summary'][arm]['by_group'].get(group,{'n':0,'success_n':0,'unknown_n':0})
            return bounds(s)
        lines.append(f'| {label} | {one("disabled")} | {one("enabled")} |')
    lines+=['','## 三次重复','', '| 重复 | 原单主意图路径 | 多任务路径 |','|---|---:|---:|']
    for i,arms in data['repetitions'].items():lines.append(f'| {i} | {bounds(arms["disabled"])} | {bounds(arms["enabled"])} |')
    lines+=['',f"逐次配对改善{data['paired_wins']}条、退步{data['paired_losses']}条。重复数据及同语义族变体相关，不将360条当成360个独立场景，也不取最好一次。",'',
       '## 原始标注、证据与评分','',
       '- 来源题、首问、目标订单、目标事实、知识出处、确认/取消策略、后续改口和故障条件均在首次调用前确定；来源B14未被询问的摄像头像素不作为新题必答要点。',
       '- SQL回执核对退款审核/投诉是否真正提交，工具调用与可见答复联合核对订单物流；任务计划的DONE不作为成功证据。退款审核工单不是实际资金退款。',
       '- 模拟用户按事项选单并拒绝错对象预览，不会自动批准错误订单，也不会主动提醒系统补办漏项；已有缺参追问允许一次定向补充。多余澄清单独计数。',
       '- 知识回答由真实模型裁判对照冻结事实判定，并校验所引答复确实存在。裁判解析/引文失败及服务商内容过滤保留为评分问题，待定结果用上下界表示；不据此换题或重跑业务挑优。',
       '- 场景分包含模型裁判的风险标签，这不是独立人工审核的成功率或幻觉率；应结合SQL/工具硬指标与原始答复查看。','',
       '## 评分输入修正与原判保留','',
       '初版裁判把累计答复作为无时间标注的列表，曾把确认前“尚未建单”和确认后“工单已创建”误判为同时矛盾。原始judges/、REPORT.md和summary.json全部保留。',
       '本表采用按时间顺序的重新评分：拆出每次新展示内容，插入实际用户确认/取消/补充；每个目标可引用多个真实片段。坏引文只令对应项目待定，不牵连已有合法证据的其他项目。',
       '这是发现评测输入缺陷后的公开修正，修正发生在首轮业务执行后，不能宣称全部评分实现从未修改。120题、目标事实、SQL判据和业务轨迹均未改变，也未重跑业务挑选结果。修正提示与哈希在chronological-judge-protocol.json。',
       '修正后的模型裁判仍可能出错，未进行独立人工全量复核。无有效证据和服务商评分过滤继续保留待定；不能把待定自动算成功。','',
       f"另外对{data.get('review_n',0)}条明确的风险标签错误进行了助手来源复核，证据包括原始操作卡、确认前SQL状态、确认后回执及知识原文；不是独立人工标注。只移除已经证实不成立的风险标签，不修改任务目标或模型原判文件。详见source-risk-reviews.json。",
       '未作来源复核的时序自动分数另存summary-chronological.json；含来源复核的分数存summary-source-reviewed.json。下表保留自动评分，便于比较复核的影响。','',
       '| 时序自动评分（来源复核前） | 原单主意图路径 | 多任务路径 |','|---|---:|---:|',
       f"| 场景验收 | {bounds(read(OUT/'summary-chronological.json')['summary']['disabled'])} | {bounds(read(OUT/'summary-chronological.json')['summary']['enabled'])} |",'',
       '## 知识冲突敏感性','',
       '冻结库中FAQ运费条款写不满99元收10元，配送条款和积分示例却写6元。原标注取自指定FAQ出处，但不能把引用另一处6元材料的回答一概称为模型凭空编造。',
       '另外发现源题A30的“仅企业账户”在当前原文中修饰电子专票，未明确修饰纸质发票；派生M2-10把该限制用于纸质专票存在实体错配。逐字命中校验未能发现这项语义问题，不能把该标签当可靠事实真值。',
       '主表完整保留120场景原标注分数。下面仅做附加敏感性分析：去掉3个冲突小节相关场景（M1-02、M1-06、M1-08）及1个实体标注错配场景（M2-10）。不代替主表，也不证明其余材料已通过独立人工审核。','',
       '| 附加分析 | 原单主意图路径 | 多任务路径 |','|---|---:|---:|',
       f"| 排除4个资料/标注问题场景 | {bounds(analyses['disabled']['without_flagged_sections_sensitivity'])} | {bounds(analyses['enabled']['without_flagged_sections_sensitivity'])} |",'',
       '证据原文与受影响目标见knowledge-conflict-audit.json。库中冲突尚未修复，本轮未更新客服知识库。','',
       '## 耗时与调用代价','', '| 项目 | 原单主意图路径 | 多任务路径 |','|---|---:|---:|',
       f"| 服务端轨迹耗时中位数（秒） | {a['median_seconds']} | {b['median_seconds']} |"]
    for key in ('attempts',):lines.append(f"| API尝试次数（含重试） | {api.get('disabled',{}).get(key,'未采集')} | {api.get('enabled',{}).get(key,'未采集')} |")
    for channel in ('chat','embed','rerank'):
        lines.append(f"| {channel}尝试次数 | {api.get('disabled',{}).get('by_channel',{}).get(channel,0)} | {api.get('enabled',{}).get('by_channel',{}).get(channel,0)} |")
    lines+=['',
       '延迟包含排队、网络、检索及程序化交互，不含真人思考时间；旧路径可能少做事项，因此不能只按耗时判断效率。未独立采集完整Token及账单费用，不能据调用次数直接计算金额。',
       '多余选单只统计已提供有效订单后仍出现的选单卡，不覆盖全部文字澄清。旧路径漏办事项较多，也减少了触发选单的机会，不能只凭绝对次数比较绑定准确率；多任务路径当前额外选单的交互负担仍需优化。',
       '两组沿用相同配置值，但旧Agent的6步/2000 Token累计预算与新任务各自执行的实际预算机制不同。这是整体实现的对比，不能把差异全部归因于任务规划器，或声称严格等Token预算。','',
       '## 预跑、隔离与验收边界','',
       '初次预跑因多进程共享测试库而碰撞旧版时间戳工单号，已完整归档至../20261103-business120-preflight，并标记不用于能力结论。正式评测为每条轨迹分配独立序号范围，使用新测试库和新请求标识；原120题、知识、评分文件与预跑逐字节一致。见preflight-correction.json。',
       '这项修正只用于隔离测试扇出，不是生产工单编号修复；原工单号生成器的多进程碰撞风险仍应另行处理。正式运行的碰撞日志与文件完整性见verification.json。',
       '恢复测试重建SQLite连接及执行图；未执行浏览器/SSE投递、实际进程断电或全套网络故障验收。订单物流为确定性夹具，工单写隔离MySQL；模型、向量编码及重排为真实接口。',
       '取消/改口消息在首次答复后、任何确认之前发送，并明确重述继续投诉或替换订单。这些限定轨迹通过，不等于已支持任意省略指代、已完成写操作后的改口或原计划任务ID保持不变。',
       '本次原300题只用作知识与主题来源，没有重新跑300题的RAG召回/生成指标；本报告不能替代原RAG回归报告。',
       '原300题未改、客服配置和生产知识库未改、服务未切换新功能。数据为内部构造验收集，不能把结果直接表述为线上成功率。','',
       '文件：[120场景](CASES.md) · [完整字段](cases.json) · [冻结规则](scoring.json) · [冻结协议](protocol.json) · [含来源复核结果](summary-source-reviewed.json) · [时序自动分](summary-chronological.json) · [失败与退步样例](FAILURES.md) · [原始严格报告](REPORT.md) · [核验](verification.json)']
    (OUT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n','utf-8')
    print('Final-readable report:',OUT/'FINAL_REPORT.md')


if __name__=='__main__':main()
