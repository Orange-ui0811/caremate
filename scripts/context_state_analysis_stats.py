"""Read-only aggregates from frozen receipts; never reconstruct or change grades."""
from collections import Counter, defaultdict
import random

STAGES = ('coref', 'plan', 'reply')


def success(result):
    return all(result.get(stage, {}).get('status') == 'ok' for stage in STAGES)


def metrics(rows):
    rows = list(rows); n = len(rows)
    return {'n': n, 'conversations': len({r['case_id'] for r in rows}),
            'task_passed': sum(r['score']['task_pass'] for r in rows),
            'task_rate': sum(r['score']['task_pass'] for r in rows) / n if n else None,
            'target_passed': sum(r['score']['target_pass'] for r in rows),
            'answer_rule_passed': sum(r['score']['answer_rule_pass'] for r in rows),
            'legacy_passed': sum(r['score']['legacy']['task_pass'] for r in rows),
            'all_stages_ok': sum(success(r) for r in rows),
            'transport_or_validation_failed': sum(r['score']['transport_failure'] for r in rows),
            'protocol_failed': sum(r['score']['protocol_failure'] for r in rows),
            'budget_blocked': sum(r['score']['budget_blocked'] for r in rows),
            'rule_failure_with_all_stages_ok': sum(success(r) and not r['score']['task_pass'] for r in rows),
            'fabricated_identifier_flags': sum(bool(r['score'].get('fabricated_identifiers')) for r in rows),
            'stage_status': {s: dict(Counter(r.get(s, {}).get('status', 'missing') for r in rows)) for s in STAGES}}


def usage(records):
    attempts = [a for record in records for a in record.get('attempts', [])]
    return {key: sum((a.get('usage') or {}).get(key, 0) or 0 for a in attempts)
            for key in ('input_tokens', 'output_tokens', 'total_tokens')} | {
        'attempt_count': len(attempts),
        'successful_attempts': sum(a.get('transport_ok', False) for a in attempts),
        'missing_usage_attempts': sum(not a.get('usage') for a in attempts),
        'cache_read_tokens': sum((a.get('usage') or {}).get('input_token_details', {}).get('cache_read', 0) or 0 for a in attempts),
        'successful_attempts_without_cache_read_detail': sum(a.get('transport_ok', False) and 'cache_read' not in (a.get('usage') or {}).get('input_token_details', {}) for a in attempts),
        'tokens_are_known_receipt_sum': True}


def reconcile_attempt_usage(calls, attempts, grouped_cost):
    """Unique persisted attempt files, not unique inputs, are the billing unit."""
    references, duplicate_refs, mismatches = {}, [], []
    for key, call in calls.items():
        for index, attempt in enumerate(call.get('attempts', []), 1):
            ref = key + '-' + str(index)
            if ref in references:
                duplicate_refs.append(ref)
            references[ref] = attempt
            if ref in attempts and attempts[ref] != attempt:
                mismatches.append(ref)
    unique = usage([{'attempts': list(attempts.values())}])
    cited = usage([{'attempts': list(references.values())}])
    keys = ('input_tokens', 'output_tokens', 'total_tokens', 'cache_read_tokens')
    grouped = {k: sum(v['all'][k] for v in grouped_cost.values()) for k in keys}
    equal = all(grouped[k] == cited[k] == unique[k] for k in keys)
    pending = sorted(set(attempts) - set(references)); missing = sorted(set(references) - set(attempts))
    return {'all_receipts_accounted': not duplicate_refs and not mismatches and not pending and not missing and equal,
            'unique_attempt_files': len(attempts), 'referenced_attempts': len(references),
            'unique_attempt_file_usage': unique, 'call_reference_usage': cited, 'grouped_usage': grouped,
            'totals_equal': equal, 'duplicate_reference_names': duplicate_refs,
            'content_mismatch_names': mismatches, 'pending_or_unreferenced_attempt_files': pending,
            'missing_attempt_files': missing,
            'identity_policy': 'Count each unique persisted attempt filename once. Distinct submitted attempts with identical input/content hashes remain distinct. Reading cached records adds no usage.',
            'cache_policy': 'cache_read_tokens is a reported subset of input_tokens, not additional tokens to add to input or total. Provider prices/cache discounts are not inferred.',
            'completion_scope': 'During a live run an attempt may precede its final call receipt; report pending files without claiming loss. This is a read-only receipt reconciliation, not a new API or semantic audit.'}


def group_calls(cases, arms, calls):
    assigned = defaultdict(list); unassigned = []
    for key, call in calls.items():
        owners = []
        for cid, case in cases.items():
            for arm in arms:
                if key.startswith(f'{cid}-{arm}-') or any(key.startswith(f'{cid}-{s["id"]}-{arm}-') for s in case['steps']):
                    owners.append((cid, arm))
        if len(owners) == 1:
            assigned[owners[0]].append(call)
        else:
            unassigned.append(key)
    return assigned, unassigned


def compare_cost(first, second):
    output = {}
    for key in ('input_tokens', 'output_tokens', 'total_tokens'):
        a, b = first[key], second[key]
        output[key] = {'C': a, 'R': b, 'R_minus_C': b - a,
                       'R_relative_C_change_pct': (b / a - 1) * 100 if a else None}
    return output


def family_cluster_difference(cases, results, first='C', second='R', *, seed=20261027, samples=5000):
    """Descriptive paired family bootstrap; preserve all variants and probes."""
    families = defaultdict(list)
    for case in cases.values():
        families[case['kind']].append(case)
    rows, incomplete = [], []
    for family, members in sorted(families.items()):
        case_rows = []
        for case in members:
            a = [results.get(f'{case["id"]}-{step["id"]}-{first}') for step in case['steps']]
            b = [results.get(f'{case["id"]}-{step["id"]}-{second}') for step in case['steps']]
            if not a or any(row is None for row in a + b):
                break
            case_rows.append({'case_id': case['id'], 'steps': len(a),
                              'first_task_rate': sum(row['score']['task_pass'] for row in a) / len(a),
                              'second_task_rate': sum(row['score']['task_pass'] for row in b) / len(b)})
        if len(case_rows) != len(members):
            incomplete.append(family)
            continue
        deltas = [row['second_task_rate'] - row['first_task_rate'] for row in case_rows]
        rows.append({'family': family, 'conversations': len(members), 'case_details': case_rows,
                     'mean_difference_pp': sum(deltas) / len(deltas) * 100})
    differences = [row['mean_difference_pp'] for row in rows]
    interval = None
    if len(differences) >= 2:
        rng = random.Random(seed)
        draws = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences) for _ in range(samples))
        def percentile(q):
            position = (len(draws) - 1) * q; lower = int(position); upper = min(lower + 1, len(draws) - 1)
            return draws[lower] + (draws[upper] - draws[lower]) * (position - lower)
        interval = [percentile(.025), percentile(.975)]
    return {'name': 'descriptive_family_cluster_bootstrap', 'comparison': first + '->' + second,
            'family_clusters': len(rows), 'declared_families': len(families),
            'conversations_included': sum(row['conversations'] for row in rows),
            'mean_difference_pp': sum(differences) / len(differences) if differences else None,
            'descriptive_95_percentile_interval_pp': interval, 'bootstrap_samples': samples, 'seed': seed,
            'family_details': rows, 'incomplete_families_excluded': incomplete,
            'weighting': 'Equal weight to each family, equal weight to its conversations; all correlated steps remain together. Balanced formal data makes this equal to the paired overall mean.',
            'failure_policy': 'Use frozen task_pass for every complete pair, including transport, protocol, budget and semantic failures; no success-only selection.',
            'limits': 'Descriptive uncertainty across internally constructed families, not a guarantee of independent family sampling or external-population confidence. Shared templates/provider and trajectory divergence limit causal claims. Fewer than two complete families yields no interval.'}


def family_cluster_cost(cases, paired, *, seed=20261027, samples=5000):
    """Family bootstrap of known total-token costs, never currency estimates."""
    by_case = {r['case_id']: r for r in paired}; families = defaultdict(list)
    for cid, case in cases.items():
        families[case['kind']].append(cid)
    groups, incomplete = [], []
    for family, ids in sorted(families.items()):
        if any(cid not in by_case for cid in ids):
            incomplete.append(family); continue
        groups.append({'family': family, 'conversations': len(ids),
                       'C_mean_total_tokens': sum(by_case[cid]['comparison']['total_tokens']['C'] for cid in ids) / len(ids),
                       'R_mean_total_tokens': sum(by_case[cid]['comparison']['total_tokens']['R'] for cid in ids) / len(ids),
                       'missing_usage_attempts': sum(by_case[cid]['missing_usage_attempts'] for cid in ids)})
    def summary(values):
        first = sum(r['C_mean_total_tokens'] for r in values) / len(values)
        second = sum(r['R_mean_total_tokens'] for r in values) / len(values)
        return {'C_mean_tokens_per_conversation': first, 'R_mean_tokens_per_conversation': second,
                'R_minus_C_tokens_per_conversation': second - first,
                'R_relative_C_change_pct': (second / first - 1) * 100 if first else None}
    point = summary(groups) if groups else None
    intervals = {}
    if len(groups) >= 2:
        rng = random.Random(seed)
        draws = [summary(rng.choices(groups, k=len(groups))) for _ in range(samples)]
        for metric in ('R_minus_C_tokens_per_conversation', 'R_relative_C_change_pct'):
            values = sorted(r[metric] for r in draws if r[metric] is not None)
            if len(values) != samples:
                intervals[metric] = None; continue
            def quantile(q):
                index = (len(values) - 1) * q; low = int(index); high = min(low + 1, len(values) - 1)
                return values[low] + (values[high] - values[low]) * (index - low)
            intervals[metric] = [quantile(.025), quantile(.975)]
    return {'name': 'descriptive_family_cluster_token_cost', 'family_clusters': len(groups),
            'families': groups, 'incomplete_families_excluded': incomplete, 'point_estimate': point,
            'descriptive_95_percentile_intervals': intervals, 'bootstrap_samples': samples, 'seed': seed,
            'scope': 'All recorded attempt tokens including maintenance; equal family weight, variants retained together. Missing usage implies a known-receipt sum, not complete billed usage. These are tokens, not currency or wall-clock latency.',
            'limits': 'Descriptive internal-template uncertainty, not independent sampling or external-population confidence.'}


def build_stats(manifest, cases, results, calls, maintenance):
    arms = manifest['arms']
    grouped, unassigned = group_calls(cases, arms, calls)
    by_arm = {a: [r for r in results.values() if r['arm'] == a] for a in arms}
    declared = {'conversations': len(cases), 'families': dict(Counter(c['kind'] for c in cases.values())),
                'length_strata': dict(Counter(c['length_stratum'] for c in cases.values())),
                'scripted_checkpoints': sum(len(c['steps']) for c in cases.values()),
                'independent_human_review': False}
    strata = {}
    for field in ('length_stratum', 'kind'):
        strata[field] = {value: {a: metrics(r for r in by_arm[a] if cases[r['case_id']][field] == value) for a in arms}
                         for value in sorted({c[field] for c in cases.values()})}
    common_points, excluded_points = [], []
    for cid, case in cases.items():
        for step in case['steps']:
            pair = (cid, step['id'])
            records = [results.get(f'{cid}-{step["id"]}-{a}') for a in arms]
            if all(record and success(record) for record in records):
                common_points.append(pair)
            else:
                excluded_points.append({'case_id': cid, 'probe_id': step['id'],
                                        'reasons': {a: 'missing' if r is None else {s: r[s]['status'] for s in STAGES if r[s]['status'] != 'ok'}
                                                    for a, r in zip(arms, records) if r is None or not success(r)}})
    common_set = set(common_points)
    common_conversations = []
    for cid, case in cases.items():
        if not all((cid, s['id']) in common_set for s in case['steps']):
            continue
        if any(f'{cid}-{a}' not in maintenance for a in arms):
            continue
        if any(len(grouped[cid, arm]) < 3 * len(case['steps']) for arm in arms):
            continue
        if any(any(call['status'] != 'ok' for call in grouped[cid, arm]) or
               any(event.get('accepted') is False for event in maintenance[f'{cid}-{arm}']) for arm in arms):
            continue
        common_conversations.append(cid)
    point_results = {a: metrics(r for r in by_arm[a] if (r['case_id'], r['probe_id']) in common_set) for a in arms}
    strict_results = {a: metrics(r for r in by_arm[a] if r['case_id'] in common_conversations) for a in arms}
    common = {'definition': 'Same checkpoints across all requested arms; coref, plan and reply must all be ok. No semantic outcome is used for selection.',
              'selection_bias': 'Post-treatment selection; not representative of all tasks and not a replacement for the full denominator. Checkpoint subset does not exclude maintenance failures.',
              'checkpoints': [{'case_id': c, 'probe_id': s} for c, s in common_points],
              'excluded': excluded_points, 'by_arm': point_results,
              'strict_conversation_definition': 'All checkpoints and all model/maintenance receipts across every arm ok; all maintenance files present; rejected summaries also excluded.',
              'strict_conversations': common_conversations, 'strict_by_arm': strict_results,
              'family_distribution': dict(Counter(cases[c]['kind'] for c in common_conversations)),
              'length_distribution': dict(Counter(cases[c]['length_stratum'] for c in common_conversations))}
    cost = {'scope': 'Known provider tokens for every saved attempt, including summary maintenance; not money, not all supplied-history generation. Missing usage may undercount.',
            'unassigned_call_keys': unassigned, 'per_arm': {}, 'C_R_paired_conversations': []}
    for arm in arms:
        records = [call for cid in cases for call in grouped[cid, arm]]
        cost['per_arm'][arm] = {'all': usage(records),
                               'maintenance': usage(c for c in records if c['contract']['stage'] == 'summary'),
                               'response_pipeline': usage(c for c in records if c['contract']['stage'] != 'summary')}
    if 'C' in arms and 'R' in arms:
        for cid, case in cases.items():
            if any(f'{cid}-{a}' not in maintenance or any(f'{cid}-{s["id"]}-{a}' not in results for s in case['steps']) for a in ('C', 'R')):
                continue
            first, second = usage(grouped[cid, 'C']), usage(grouped[cid, 'R'])
            cost['C_R_paired_conversations'].append({'case_id': cid, 'family': case['kind'], 'length_stratum': case['length_stratum'],
                                                   'comparison': compare_cost(first, second),
                                                   'missing_usage_attempts': first['missing_usage_attempts'] + second['missing_usage_attempts']})
        cost['C_R_all_available'] = compare_cost(cost['per_arm']['C']['all'], cost['per_arm']['R']['all'])
        cost['C_R_descriptive_family_cluster'] = family_cluster_cost(cases, cost['C_R_paired_conversations'])
    summary = {'scope': 'M only: distinguish called, accepted and actually included; a generated summary is not necessarily read by a model.',
               'maintenance_files': 0, 'events': 0, 'accepted_events': 0, 'rejected_events': 0,
               'event_status': Counter(), 'rejection_reasons': Counter(), 'checkpoints_with_summary': 0,
               'distinct_used_segment_ids': set(), 'per_conversation': {}}
    if 'M' in arms:
        for cid, case in cases.items():
            events = maintenance.get(f'{cid}-M', [])
            rr = [r for r in by_arm['M'] if r['case_id'] == cid]
            refs = {ref for r in rr for ref in r['context_manifest'].get('summary_refs', [])}
            calls_summary = [c for c in grouped[cid, 'M'] if c['contract']['stage'] == 'summary']
            row = {'events': len(events), 'calls': len(calls_summary), 'call_status': dict(Counter(c['status'] for c in calls_summary)),
                   'accepted': sum(e.get('accepted', False) for e in events),
                   'rejected': sum(e.get('accepted') is False for e in events),
                   'checkpoints_with_summary': sum(bool(r['context_manifest'].get('summary_refs')) for r in rr),
                   'used_segment_ids': sorted(refs), 'usage': usage(calls_summary),
                   'length_stratum': case['length_stratum']}
            summary['per_conversation'][cid] = row
            summary['maintenance_files'] += f'{cid}-M' in maintenance
            summary['events'] += len(events); summary['accepted_events'] += row['accepted']; summary['rejected_events'] += row['rejected']
            summary['event_status'].update(e.get('status') for e in events)
            summary['rejection_reasons'].update(e.get('rejection_reason') for e in events if e.get('accepted') is False)
            summary['checkpoints_with_summary'] += row['checkpoints_with_summary']; summary['distinct_used_segment_ids'].update(refs)
    summary['event_status'] = dict(summary['event_status']); summary['rejection_reasons'] = dict(summary['rejection_reasons'])
    summary['distinct_used_segment_ids'] = sorted(summary['distinct_used_segment_ids'])
    fixed = {}
    for a, b in (('W', 'P'), ('P', 'Q'), ('Q', 'S'), ('A', 'B'), ('A', 'R'), ('B', 'R'), ('R', 'M'), ('C', 'R')):
        if a not in arms or b not in arms:
            continue
        first_rows, second_rows, excluded = [], [], []
        for cid, case in cases.items():
            for step in case['steps']:
                first, second = results.get(f'{cid}-{step["id"]}-{a}'), results.get(f'{cid}-{step["id"]}-{b}')
                if not first or not second:
                    continue
                if first['history_sha256'] == second['history_sha256']:
                    first_rows.append(first); second_rows.append(second)
                else:
                    excluded.append({'case_id': cid, 'probe_id': step['id']})
        fixed[a + '->' + b] = {'first': metrics(first_rows), 'second': metrics(second_rows),
                              'included_checkpoints': [{'case_id': r['case_id'], 'probe_id': r['probe_id']} for r in first_rows],
                              'divergent_history_excluded': excluded,
                              'definition': 'Observed requests with identical full prior-history hashes. No counterfactual model output or new call. In trajectory mode later agreement is selected, not randomized.',
                              'raw_window_equivalence_not_implied': a not in ('W', 'P', 'Q') or b not in ('P', 'Q', 'S')}
    return {'experiment_version': manifest.get('version'), 'mode': manifest['mode'], 'declared_data': declared,
            'experimental_scope': {
                'reader': 'Shared offline coref -> ServicePlan -> read-only fixture -> reply pipeline, not the production graph.',
                'source_context_projection': 'V2 shared reader uses assemble_context(stage=answer) material to build coref_history; it does not reproduce production coref 4000 / intent 1800 or all node-specific projections.',
                'legacy_reference': 'C calls real legacy assembly helpers, but its downstream pipeline is still the shared offline reader.',
                'production_validation': 'Separate production Workflow checks cover short continuous trajectories; long synthetic comparison numbers are not full deployed SSE/RAG accuracy.',
                'human_review': 'Programmatically constructed synthetic cases; rule scoring and assistant text spot-checks are not independent human review.'},
            'full_by_arm': {a: metrics(rr) for a, rr in by_arm.items()}, 'stratified': strata,
            'common_zero_failure_subset': common, 'costs': cost, 'M_summary_accounting': summary,
            'C_R_descriptive_family_bootstrap': family_cluster_difference(cases, results) if 'C' in arms and 'R' in arms else None,
            'fixed_history_observed_comparisons': fixed,
            'semantic_review_scope': 'Frozen rule scores only. Any textual review must be labelled model-assisted, not independent human review, and must not replace these scores.'}


def markdown(stats):
    lines = ['# 只读分层与成本分析', '',
             f'实验版本：{stats["experiment_version"]}；模式：{stats["mode"]}。保留冻结规则，未重打分或调用模型。', '',
             '本报告是共用离线 reader：指代改写→计划→只读工具夹具→回答。V2 source 组以 answer 装配材料构造 coref_history，'
             '并未复现线上 coref 4000 / intent 1800 等每个节点的完整装配。C 虽调用真实 legacy helper，下游同样是共用 reader。'
             '真实 Workflow 的短连续轨迹验证应单独报告；24 条长短合成会话的数字不能称完整上线 SSE/RAG 准确率。', '',
             '|组|结果数|规则通过|全阶段成功|协议失败|接口/校验失败|预算阻断|', '|---|---:|---:|---:|---:|---:|---:|']
    for arm, r in stats['full_by_arm'].items():
        lines.append(f'|{arm}|{r["n"]}|{r["task_passed"]}|{r["all_stages_ok"]}|{r["protocol_failed"]}|{r["transport_or_validation_failed"]}|{r["budget_blocked"]}|')
    lines += ['', '|长度组|方案|结果数|规则通过|协议失败|接口/校验失败|预算阻断|', '|---|---|---:|---:|---:|---:|---:|']
    for stratum, groups in stats['stratified']['length_stratum'].items():
        for arm, r in groups.items():
            lines.append(f'|{stratum}|{arm}|{r["n"]}|{r["task_passed"]}|{r["protocol_failed"]}|{r["transport_or_validation_failed"]}|{r["budget_blocked"]}|')
    arms = list(stats['full_by_arm'])
    lines += ['', '|家族|会话数|' + '|'.join(arms) + '|', '|---|---:|' + '|'.join('---:' for _ in arms) + '|']
    for family, groups in stats['stratified']['kind'].items():
        values = [f'{groups[a]["task_passed"]}/{groups[a]["n"]}' for a in arms]
        lines.append('|' + family + '|' + str(stats['declared_data']['families'][family]) + '|' + '|'.join(values) + '|')
    common = stats['common_zero_failure_subset']
    lines += ['', f'所有组共同无调用错误检查点：{len(common["checkpoints"])}；连同维护全部成功的共同会话：{len(common["strict_conversations"])}。',
              '共同子集是在观察调用结果后筛选，存在选择偏差，不能替代全样本口径；语义答错仍完整保留。', '',
              '|方案|输入含维护|输出含维护|总Token含维护|缺失usage的attempt|', '|---|---:|---:|---:|---:|']
    for arm, values in stats['costs']['per_arm'].items():
        u = values['all']; lines.append(f'|{arm}|{u["input_tokens"]}|{u["output_tokens"]}|{u["total_tokens"]}|{u["missing_usage_attempts"]}|')
    if 'C_R_all_available' in stats['costs']:
        delta = stats['costs']['C_R_all_available']['total_tokens']
        pct = delta['R_relative_C_change_pct']
        lines += ['', f'现有完整记录 R 相对 C 的总Token差：{delta["R_minus_C"]}；相对变化：{pct:.2f}%。' if pct is not None else '',
                  '分会话、分家族、分长度配对成本保存在 JSON；运行未完成时总体差不能作为最终结论。']
        clustered_cost = stats['costs'].get('C_R_descriptive_family_cluster', {})
        cost_interval = clustered_cost.get('descriptive_95_percentile_intervals', {}).get('R_relative_C_change_pct')
        if cost_interval is not None:
            lines.append(f'按{clustered_cost["family_clusters"]}个完整家族重采样，R/C含维护总Token相对变化的95%描述性区间：'
                         f'[{cost_interval[0]:.2f}%, {cost_interval[1]:.2f}%]。这是回执Token量，不是费用或完整会话生成成本。')
    m = stats['M_summary_accounting']
    lines += ['', f'M 摘要维护事件 {m["events"]}，接受 {m["accepted_events"]}，拒收 {m["rejected_events"]}；'
                  f'实际装入 {m["checkpoints_with_summary"]} 个检查点，使用 {len(m["distinct_used_segment_ids"])} 个不同摘要段。',
              '摘要未触发或未入模的 R/M 差异不能归因于摘要。成本含历史维护和后续追问，不含给定历史生成。', '',
              '固定历史比较仅使用已有请求中历史哈希相同的配对；trajectory 后续历史可能分叉。没有追加模型调用，也没有制造 checkpoints 反事实结果。',
              '规则评分不是独立人工语义准确率；文本抽查应另标“模型辅助复核”，并保留规则假阴性与假阳性。']
    receipts = stats.get('attempt_usage_reconciliation')
    if receipts:
        lines += ['', f'费用口径只读核对：{receipts["unique_attempt_files"]} 个唯一 attempt 文件；'
                      f'分组汇总、calls引用与唯一attempt回执一致：{receipts["all_receipts_accounted"]}。',
                  f'回执 cache_read_tokens 合计 {receipts["unique_attempt_file_usage"]["cache_read_tokens"]}，'
                  '已包含在输入Token中，不重复相加；不能仅据此换算为供应商实际费用或缓存折扣。'
                  '相同输入的不同真实请求分别计数，重读断点缓存不产生新计数。']
    clustered = stats.get('C_R_descriptive_family_bootstrap')
    if clustered and clustered['mean_difference_pp'] is not None:
        interval = clustered['descriptive_95_percentile_interval_pp']
        formatted = f'[{interval[0]:.2f}, {interval[1]:.2f}]' if interval is not None else '家族数不足，不给区间'
        lines += ['', f'补充的家族聚类描述性分析：C→R 的家族宏平均任务规则通过率差 '
                      f'{clustered["mean_difference_pp"]:.2f} 个百分点；{clustered["family_clusters"]} 个完整场景家族、'
                      f'{clustered["conversations_included"]} 条会话；5000次家族重采样的95%百分位描述区间：{formatted}。',
                  '每次抽中家族时保留其全部变体和追问；24个模板变体及72次追问不当作完全独立样本。'
                  '各家族等权，失败照原分保留。该区间描述本组合成场景的不确定性，不保证家族独立采样，也不是外部人群泛化置信区间；未改变主冻结报告的算法。']
    return '\n'.join(lines) + '\n'
