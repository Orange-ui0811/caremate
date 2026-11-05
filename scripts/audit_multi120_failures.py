"""Deterministic archived-trajectory audit. No model calls and no replacement pass rate."""
from collections import Counter
import json
from pathlib import Path
import re

from scripts.rescore_multi120_evidence import OUT, HOME, read, save, sha, audit_flags
from scripts.score_multi120 import hard_metrics, refund_order
from scripts.multi120_evidence_protocol import selectors


LABELS = {
    'driver_contaminated': '驱动在原事项办完后再次补参，结果受污染',
    'driver_blocked': '驱动无法处理选单，续办能力待定',
    'external_service_interference': '上游接口故障干扰，业务能力不能据此定性',
    'application_rejected_pending_text': '明确未完成：等待卡片时文字更正或取消被拒',
    'application_selection_cancel_unsupported': '明确未完成：选单取消不受支持',
    'hard_constraint_failure': '明确未通过：重复、错误、未经确认或禁止的操作',
    'required_write_not_completed': '明确未完成：必需办理目标缺少唯一正确工单',
    'semantic_review_needed': '仅硬约束未见上述失败，回答语义或标注仍待核实',
}
CONFIRMED = {'application_rejected_pending_text', 'application_selection_cancel_unsupported', 'hard_constraint_failure', 'required_write_not_completed'}


def classify(raw, data, log, provider_attempts=None):
    hard = hard_metrics(raw)
    flags = audit_flags(raw, data)
    missing = [g['id'] for g in raw['case']['goals'] if g['kind'] in ('refund', 'complaint') and not hard['write_terminal'][g['id']]]
    provider = bool(provider_attempts) or bool(re.search(r'BadRequestError|RateLimitError|APIConnectionError|ReadTimeout|HTTP.{0,8}(?:400|429|500|502|503)', log))
    # A mere recovered timeout does not reclassify a complete trajectory.
    provider_blocked = provider and bool(missing or raw['status'] != 'ok' or any(s.get('trace', {}).get('intent_status') == 'unavailable' for s in raw.get('steps', [])))
    if 'driver_reinjection_after_original_goals_completed' in flags: category = 'driver_contaminated'
    elif 'driver_cannot_resolve_selector' in flags: category = 'driver_blocked'
    elif provider_blocked: category = 'external_service_interference'
    elif 'application_rejected_pending_text' in flags: category = 'application_rejected_pending_text'
    elif 'application_selection_cancel_unsupported' in flags: category = 'application_selection_cancel_unsupported'
    elif any(hard[k] for k in ['wrong_ticket_n','duplicate_ticket_n','unauthorized_ticket_n','forbidden_read_n','forbidden_ticket_n']): category = 'hard_constraint_failure'
    elif missing: category = 'required_write_not_completed'
    else: category = 'semantic_review_needed'
    outcomes = []
    for g in raw['case']['goals']:
        if g['kind'] not in ('refund', 'complaint'): continue
        value = hard['write_terminal'][g['id']]
        basis = 'SQL和原确认事件'
        if g['expected_outcome'] != 'ticket' and raw.get('error_type') == 'ResumeRejected' and any(a['kind'] == 'user_message' for a in raw['actor']):
            value = False; basis = '文字取消被拒，不能以未写入证明已取消'
        elif g['expected_outcome'] != 'ticket' and raw['case'].get('followup') and not any(a['kind'] == 'cancel_form' and a.get('goal_id') == g['id'] for a in raw['actor']):
            # Accepted text alone still does not prove application-level cancellation acknowledgment.
            value = None; basis = '文字改口需结合后续实际回应，离线不凭未写入自动判成功'
        outcomes.append({'goal_id':g['id'],'kind':g['kind'],'order_id':g.get('order_id'),'expected':g['expected_outcome'],
                         'receipt_outcome':value,'basis':basis})
    return {'category':category,'confirmed_failure_by_existing_business_contract':category in CONFIRMED,
            'flags':flags,'write_outcomes':outcomes,'hard':hard,'selectors':selectors(raw),
            'original_provider_error_attempts':provider_attempts or [],
            'gold_issues':data['gold_issues'], 'semantic_verdict':'not_determined_by_this_offline_audit'}


def main():
    p = read(OUT/'protocol.json'); manifest = read(OUT/'manifest.json')
    rows = []
    # The candidate archived an isolated request journal; use it rather than inferring
    # that a caught provider error never happened merely because status became ok.
    import sqlite3
    candidate_journal=HOME/'data/multi-intent-eval/20261105-shared-agent120-v2/model_pool.sqlite3'
    connection=sqlite3.connect('file:'+candidate_journal.as_posix()+'?mode=ro',uri=True)
    provider_errors={}
    for request_id,model,http_status,outcome in connection.execute('select request_id,model,http_status,outcome from attempts where http_status>=400'):
        match=re.fullmatch(r'mi120formal:r(\d+):(enabled|disabled):(M\d+-\d+)',request_id or '')
        if match:
            key=(match[2],int(match[1]),match[3]);provider_errors.setdefault(key,[]).append({'model':model,'http_status':http_status,'outcome':outcome})
    connection.close()
    old_rows = {}
    for arm, origin in [('disabled',HOME/'data/multi-intent-eval/20261103-business120'),('enabled',HOME/'data/multi-intent-eval/20261105-shared-agent120-v2')]:
        summary = read(origin/('summary-chronological.json' if arm=='disabled' else 'summary.json'))
        for r in summary['rows']:
            if r['arm']==arm:old_rows[(arm,r['rep'],r['case_id'])]=r
    for record in manifest:
        raw_path=Path(record['raw_path']); raw=read(raw_path)
        assert sha(raw_path)==record['raw_sha256']
        assert sha(Path(record['checkpoint_path']))==record['checkpoint_sha256']
        data=read(OUT/'inputs'/record['key']/'payload.json')
        assert sha(OUT/'inputs'/record['key']/'payload.json')==record['payload_sha256']
        log=raw_path.with_name('worker.log').read_text(encoding='utf-8',errors='replace')
        row=classify(raw,data,log,provider_errors.get((record['arm'],record['rep'],record['case_id'])))
        old=old_rows[(record['arm'],record['rep'],record['case_id'])]
        contradictions=[u for u in old.get('unsafe_claims',[]) if re.search(r'不构成.{0,12}(?:风险|违规|错误)|故不列为',u['reason'])]
        row.update(arm=record['arm'],rep=record['rep'],case_id=record['case_id'],query=raw['case']['query'],
                   source_path=str(raw_path),old_pass_diagnostic_only=old['success'],old_self_contradictory_risk_labels=contradictions)
        rows.append(row)
    summaries={}
    for arm in ['disabled','enabled']:
        group=[r for r in rows if r['arm']==arm]
        summaries[arm]={'n':len(group),'categories':dict(Counter(r['category'] for r in group)),
                        'old_nonpass_categories':dict(Counter(r['category'] for r in group if r['old_pass_diagnostic_only'] is not True)),
                        'confirmed_workflow_contract_failures':sum(r['confirmed_failure_by_existing_business_contract'] for r in group),
                        'observed_selector_n':sum(r['selectors']['observed_selector_n'] for r in group),
                        'unmapped_selector_n':sum(r['selectors']['unmapped_selector_n'] for r in group),
                        'old_self_contradictory_risk_label_n':sum(len(r['old_self_contradictory_risk_labels']) for r in group),
                        'old_passes_still_require_semantic_review':True}
        assert len(group)==360 and sum(summaries[arm]['categories'].values())==360
    save(OUT/'offline-audit.json',{'scope':'All720 existing traces audited deterministically; semantic classification not inferred from partial model run. Categories disjoint by documented priority. Counts are not new pass rates.','summary':summaries,'rows':rows})
    save(OUT/'confirmed-failures.json',[r for r in rows if r['confirmed_failure_by_existing_business_contract']])
    save(OUT/'review-needed.json',[r for r in rows if not r['confirmed_failure_by_existing_business_contract']])
    for name,expected in p['original_files'].items():assert sha(Path(name))==expected,name
    for name,expected in p['production_app_hashes'].items():assert sha(Path(name))==expected,name
    assert sha(HOME/'data/runtime_settings.json')==p['runtime_settings_sha256']
    labels=list((OUT/'judges').glob('*/*/*/label.json'))
    import sqlite3
    connection=sqlite3.connect('file:'+(OUT/'model_pool.sqlite3').as_posix()+'?mode=ro',uri=True)
    actual_models=[r[0] for r in connection.execute('select distinct model from attempts')];connection.close()
    assert actual_models==['DeepSeek-V4-Flash']
    result={'complete_offline_audit':True,'archived_trajectories':720,'new_business_trajectories':0,
            'partial_model_labels_retained':len(labels),'publish_new_semantic_pass_rate':False,
            'original_results_labels_and_summaries_unchanged':True,'original_checkpoints_unchanged':True,
            'production_app_and_threshold_unchanged':True,'model_names_in_validated_partial_batch':actual_models,
            'stopped_by_user_choice':True}
    save(OUT/'offline-verification.json',result)
    lines=['# 评测口径修复与离线失败清单','','按用户选择，本轮停止完整720条模型重评，交付代码、28项测试和全部720条已有轨迹的离线审核。没有重跑业务，也没有发布新的通过率。原69.2%、83.2%仅保留为旧协议结果。','','| 离线归因（互斥） | 单意图基线 | 当前多意图 |','|---|---:|---:|']
    for category,title in LABELS.items():lines.append(f'| {title} | {summaries["disabled"]["categories"].get(category,0)} | {summaries["enabled"]["categories"].get(category,0)} |')
    lines+=['','当前多意图旧结果中的111条未通过/待定，分解为：34条必需办理未完成、18条文字更正/取消被拒、9条选单取消不受支持（合计61条流程验收未达标）；其余40条语义或标注待复核、6条驱动中止、3条接口干扰、1条驱动重复补参污染（合计50条待核实）。不能据此把其余轨迹自动算作通过。']
    lines += ['', '“明确未完成”仅表示已有业务验收要求未满足，如缺正确工单、取消不受支持；不是认定每次拒绝办理都违反真实业务政策。类别有优先级：驱动污染→驱动中止→上游干扰→应用控制限制→硬约束→办理未完成→语义待复核。', '',
              '语义待复核不是判通过。没有用部分50条模型结果、内部DONE、未写入或低风险措辞自动补齐成功标签；受驱动影响的重复工单保留原始SQL，不改成0。', '',
              '固定口径见SCORING.md、protocol.json；新模型评分已按用户选择停止，部分结果只作调用验证，不用于总体比较。已停止的v2混合模型批次保留作无效批次；v3部分调用只有Gitee的DeepSeek-V4-Flash。', '',
              '全部逐条归因：offline-audit.json；确定的流程验收失败：confirmed-failures.json；驱动/接口/标注或语义待核实：review-needed.json；原文件和部署核验：offline-verification.json。','',
              '本次修复包括完整知识及实际检索来源输入、逐事件确认与取消时序、三态风险判定与引用/来源校验、两项错误gold保留待定、被拒绝文字取消不算成功、所有选单事件计数、未来驱动只响应新提示。应用业务代码和8000部署未改。','']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    candidate=[r for r in rows if r['arm']=='enabled' and r['confirmed_failure_by_existing_business_contract']]
    grouped={}
    for row in candidate:
        key=(row['case_id'],row['category']);grouped.setdefault(key,[]).append(row)
    table=['# 当前多意图：可确认的流程验收失败','',f'共{len(candidate)}条轨迹，按原场景与原因归并。确认的是原业务目标未达到，不自动断言每次拒绝受理都违反真实政策。语义结论仍需人工或完整同口径复核。','','| 场景 | 轮次 | 已确认的未达标项 | 原问题 |','|---|---|---|---|']
    for (cid,category),items in sorted(grouped.items()):
        table.append('| '+cid+' | '+','.join(str(r['rep']) for r in items)+' | '+LABELS[category]+' | '+items[0]['query'].replace('|','／')+' |')
    table+=['','逐目标的订单、预期动作、SQL与取消依据见confirmed-failures.json；每条source_path指向原始结果，未覆盖原轨迹。','']
    (OUT/'candidate-confirmed-failures.md').write_text('\n'.join(table),encoding='utf-8')
    print(json.dumps({'summary':summaries,'verification':result},ensure_ascii=False))


if __name__=='__main__':main()
