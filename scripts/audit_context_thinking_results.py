"""Read-only supplementary accounting, paired outcomes and pool evidence.

Does not change the frozen scorer or any endpoint output. No network calls.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def main(out):
    report=read(out/'audited_report.json')
    if not report['complete'] or not report['audit_passed']:
        raise ValueError('Complete passing causal audit required')
    data=read(out/'dataset.json')['cases']
    arms=read(out/'manifest.json')['arms']
    calls={p.stem:read(p) for p in (out/'calls').glob('*.json')}
    results={p.stem:read(p) for p in (out/'results').glob('*.json')}
    all_attempts=[a for c in calls.values() for a in c['attempts']]
    checks={
        'result_count':len(results)==len(data)*len(arms)*2,
        'maintenance_count':len(list((out/'maintenance').glob('*.json')))==len(data)*len(arms),
        'receipt_count':len(list((out/'attempts').glob('*.json')))==len(all_attempts),
    }
    for arm in arms:
        rr=[r for k,r in results.items() if k.endswith('-'+arm)]
        aa=[a for k,c in calls.items() if '-'+arm+'-' in k for a in c['attempts']]
        checks['score_'+arm]=sum(r['score']['task_rule_v3'] for r in rr)==report['strategies'][arm]['passed']
        for field in ('input_tokens','output_tokens','total_tokens'):
            checks[field+'_'+arm]=sum(a.get('usage',{}).get(field,0) or 0 for a in aa)==report['strategies'][arm]['usage'][field]
    paired={}
    for policy in ('C','S'):
        pairs=[(f'{c["id"]}-{p["id"]}',results[f'{c["id"]}-{p["id"]}-{policy}_off'],results[f'{c["id"]}-{p["id"]}-{policy}_on']) for c in data for p in c['probes']]
        # Conditioning is post hoc and potentially biased; keep full sample first.
        valid=[t for t in pairs if all(r[s]['ok'] for r in t[1:] for s in ('coref','plan','reply'))]
        paired[policy]={
            'off_fail_on_pass':[k for k,a,b in pairs if not a['score']['task_rule_v3'] and b['score']['task_rule_v3']],
            'off_pass_on_fail':[k for k,a,b in pairs if a['score']['task_rule_v3'] and not b['score']['task_rule_v3']],
            'common_valid':{'n':len(valid),'off':sum(a['score']['task_rule_v3'] for _,a,b in valid),'on':sum(b['score']['task_rule_v3'] for _,a,b in valid)},
        }
    failures=[{'key':k,'score':r['score'],'coref':r['coref']['value'],'plan':r['plan']['value'],'reply':r['reply']['value']} for k,r in sorted(results.items()) if not r['score']['task_rule_v3']]
    # Limit read to this experiment's attempt metadata; never read credentials.
    with sqlite3.connect('file:data/model_pool.sqlite3?mode=ro',uri=True) as con:
        con.row_factory=sqlite3.Row
        pool=[dict(r) for r in con.execute('SELECT request_id,attempt,started,finished,outcome,http_status FROM attempts WHERE request_id LIKE ? ORDER BY started',('think-'+out.name+'-%',))]
    expected_ids={f'think-{out.name}-{key}-{i}' for key,c in calls.items() for i in range(len(c['attempts']))}
    checks['pool_requests_match']=expected_ids=={r['request_id'] for r in pool}
    checks['pool_finished']=all(r['finished'] is not None and r['outcome']!='RUNNING' for r in pool)
    summary={'checks':checks,'passed':all(checks.values()),'counts':{'calls':len(calls),'attempts':len(all_attempts),'results':len(results)},
        'attempt_status':dict(Counter(a['status'] for a in all_attempts)),
        'final_call_status':dict(Counter(c['status'] for c in calls.values())),
        'missing_usage_attempts':sum(not a.get('usage') for a in all_attempts),
        'pool_status':dict(Counter(r['outcome'] for r in pool)),
        'pool_extra_attempts':len(pool)-len(all_attempts),
        'raw_tool_markup_replies':[k for k,r in results.items() if 'DSML' in str(r['reply']['value'])],
        'pairs':paired,'failure_review_material':failures,
        'caveats':['Common-valid subset is post hoc and selection-biased.',
                   'Missing usage on timed-out attempts means reported token totals are lower bounds.',
                   'This checks accounting independently of the workflow, not independent semantic ground truth.']}
    (out/'POOL_ATTEMPTS.json').write_text(json.dumps(pool,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'SUPPLEMENTARY_AUDIT.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# 思考开关实验：补充核对与解读','',
        '本报告不修改冻结评分。16条既有合成会话，8个场景族，各选24轮和64轮一条，每条两次追问；四组共128条结果。',
        '同一Gitee DeepSeek-V4-Flash，开关影响摘要、改写、规划、回答全部模型阶段。来源状态由本地规则生成。',
        '这是当前上下文组件加共享测试执行器的对照，未覆盖真实客服全流程，也未部署到线上。','',
        '|策略|任务规则通过|规划对象/动作正确|API等待P50/P95秒|输入Token含维护|输出Token含思考维护|',
        '|---|---:|---:|---:|---:|---:|']
    labels={'C_off':'当前摘要窗口·关闭','C_on':'当前摘要窗口·开启','S_off':'原文窗口+来源状态·关闭','S_on':'原文窗口+来源状态·开启'}
    for arm,s in report['strategies'].items():
        lines.append(f'|{labels[arm]}|{s["passed"]}/{s["n"]}|{s["target_passed"]}/{s["n"]}|{s["p50_seconds"]:.2f}/{s["p95_seconds"]:.2f}|{s["usage"]["input_tokens"]:,}|{s["usage"]["output_tokens"]:,}|')
    lines += ['',f'544个逻辑请求，550次实际尝试：541次成功、9次超时；重试后3个请求仍失败。调用池额外隐藏尝试数：{summary["pool_extra_attempts"]}。',
        '9次超时均在开启组，均缺少Token用量，因此开启组列出的Token是已返回用量的下限。输出已包含思考Token，不可重复相加，也不能不考虑输入/输出单价、缓存就换算为账单成本。',
        '等待时间为改写、规划、回答三阶段API耗时之和，包含失败尝试；不含实验节流排队、重试退避和后台摘要维护，不是线上端到端或首字延迟。',
        '所有80×2次摘要维护已计入Token；不包含生成给定历史文本的费用。关闭组无超时、无缺失用量；全部组无截断，成功响应均返回思考用量字段。','',
        '## 差异与不确定性','']
    for policy,label in [('C','当前摘要窗口'),('S','来源状态')]:
        p=report['thinking_on_minus_off'][policy];lo,hi=p['conversation_bootstrap_95ci'];v=paired[policy]['common_valid']
        lines.append(f'- {label}：开启减关闭的任务规则通过率差为{p["difference"]*100:.3f}个百分点，会话级配对bootstrap 95%区间[{lo*100:.3f}, {hi*100:.3f}]；区间跨零。本组两边阶段均成功的事后子集为{v["n"]}条，关闭{v["off"]}条通过、开启{v["on"]}条通过，存在筛选偏差。')
    lines += ['', '## 失败样例定性复核（不改分）','',
        '- svc-13/p2：当前关闭组查旧号610132，开启组规划改为正确的610133。两组指代改写均选了旧号；最新更正在规划原文里，却不在改写器最近6条消息里。开启规划纠正了上游错误。',
        '- svc-37/p2：当前关闭组查询旧对象610371；开启组规划选对610370，但最终正文泄漏DSML工具标记且仍引用610371，是实际回答失败。规划32/32不等于最终回答32/32。',
        '- svc-33/p2的当前关闭组：正确新日期后说明旧日期作废，冻结规则仍判失败。',
        '- svc-37及svc-39的一部分答案正确查询当前订单并说明另一个订单“未查询”，冻结规则将否定引用算作错误。详见SUPPLEMENTARY_AUDIT.json逐条输出。',
        '- svc-37/p1当前开启组、svc-43/p2当前开启组、svc-43/p1来源状态开启组：指代阶段重试后仍超时，后续规划/回答能回退处理；主任务规则因调用失败判未通过。不能归因于记忆丢失。','',
        '本次没有证据支持全链统一开启思考。更稳妥的下一步是统一改写与规划的证据可见范围，保留带来源的更正/撤回关系，再单独测试仅复杂规划开启思考。该混合策略尚未测试。','',
        '## 复算','',
        '先运行 scripts.eval_context_thinking 的 --audit，再运行 scripts.audit_context_thinking_results；二者 --output 均指向本目录。原始响应、尝试、冻结源码/数据及初次未生效预检均保留。',
        '补充账目检查独立于工作流生成器，但不是独立人工语义标注。对照集已知且规模小，不宜宣称真实客服准确率或通用显著提升。']
    (out/'INTERPRETATION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='failure_review_material'},ensure_ascii=False,indent=2))
    if not summary['passed']:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    main(Path(p.parse_args().output).resolve())
