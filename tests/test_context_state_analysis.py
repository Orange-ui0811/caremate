from scripts.context_state_analysis_stats import build_stats, family_cluster_cost, family_cluster_difference, markdown, reconcile_attempt_usage, usage


def receipt(status='ok', stage='plan', tokens=10):
    return {'status': status, 'contract': {'stage': stage},
            'attempts': [{'transport_ok': True, 'usage': {'input_tokens': tokens, 'output_tokens': 2, 'total_tokens': tokens + 2}}]}


def record(cid, arm, probe, passed=True, status='ok', summary_refs=()):
    return {'case_id': cid, 'arm': arm, 'probe_id': probe,
            'coref': {'status': 'ok'}, 'plan': {'status': status}, 'reply': {'status': 'ok'},
            'history_sha256': probe,
            'context_manifest': {'summary_refs': list(summary_refs)},
            'score': {'task_pass': passed, 'target_pass': passed, 'answer_rule_pass': passed,
                      'legacy': {'task_pass': passed}, 'transport_failure': False,
                      'protocol_failure': status == 'parse_error', 'budget_blocked': False,
                      'fabricated_identifiers': []}}


def test_common_subset_uses_call_success_and_keeps_semantic_failures():
    cases = {'case': {'id': 'case', 'kind': 'ambiguous', 'length_stratum': 'short',
                      'steps': [{'id': p} for p in ('p1', 'p2', 'p3')]}}
    results, calls = {}, {}
    for arm in ('C', 'R'):
        for probe in ('p1', 'p2', 'p3'):
            status = 'parse_error' if arm == 'C' and probe == 'p2' else 'ok'
            passed = not (arm == 'R' and probe == 'p1') and status == 'ok'
            results[f'case-{probe}-{arm}'] = record('case', arm, probe, passed, status)
            for stage in ('coref', 'plan', 'reply'):
                calls[f'case-{probe}-{arm}-{stage}'] = receipt(status if stage == 'plan' else 'ok', stage)
    manifest = {'version': 'v2', 'mode': 'trajectory', 'arms': ['C', 'R']}
    output = build_stats(manifest, cases, results, calls, {'case-C': [], 'case-R': []})
    shared = output['common_zero_failure_subset']
    assert len(shared['checkpoints']) == 2
    assert shared['by_arm']['R']['n'] == 2 and shared['by_arm']['R']['task_passed'] == 1
    assert not shared['strict_conversations']
    assert output['costs']['C_R_all_available']['total_tokens']['R_minus_C'] == 0
    assert len(output['costs']['C_R_paired_conversations']) == 1
    assert output['declared_data']['families'] == {'ambiguous': 1}
    assert '家族' in markdown(output)


def test_summary_acceptance_is_distinct_from_actual_loading_and_cost():
    cases = {'case': {'id': 'case', 'kind': 'early_order', 'length_stratum': 'long', 'steps': [{'id': 'p1'}]}}
    results = {'case-p1-M': record('case', 'M', 'p1', summary_refs=['segment1'])}
    calls = {f'case-p1-M-{s}': receipt(stage=s) for s in ('coref', 'plan', 'reply')}
    calls.update({'case-M-summary-1': receipt(stage='summary', tokens=100),
                  'case-M-summary-2': receipt(stage='summary', tokens=200)})
    events = [{'key': 'case-M-summary-1', 'accepted': True, 'status': 'ok'},
              {'key': 'case-M-summary-2', 'accepted': True, 'status': 'ok'}]
    out = build_stats({'version': 'v2', 'mode': 'checkpoints', 'arms': ['M']}, cases, results, calls, {'case-M': events})
    m = out['M_summary_accounting']
    assert m['accepted_events'] == 2 and m['checkpoints_with_summary'] == 1
    assert m['distinct_used_segment_ids'] == ['segment1']
    assert m['per_conversation']['case']['usage']['input_tokens'] == 300
    assert out['costs']['per_arm']['M']['all']['input_tokens'] == 330
    assert out['common_zero_failure_subset']['strict_conversations'] == ['case']


def test_missing_usage_and_divergent_trajectory_are_explicit():
    assert usage([{'attempts': [{'transport_ok': False, 'usage': {}}]}])['missing_usage_attempts'] == 1
    cases = {'case': {'id': 'case', 'kind': 'missing_fact', 'length_stratum': 'short', 'steps': [{'id': 'p1'}]}}
    left, right = record('case', 'Q', 'p1'), record('case', 'S', 'p1')
    right['history_sha256'] = 'divergent'
    out = build_stats({'version': 'v2', 'mode': 'trajectory', 'arms': ['Q', 'S']}, cases,
                      {'case-p1-Q': left, 'case-p1-S': right}, {}, {})
    pair = out['fixed_history_observed_comparisons']['Q->S']
    assert pair['first']['n'] == 0 and len(pair['divergent_history_excluded']) == 1


def test_family_bootstrap_keeps_variants_and_steps_together():
    cases, results = {}, {}
    for cid, family in (('a1', 'family_a'), ('a2', 'family_a'), ('b1', 'family_b')):
        cases[cid] = {'id': cid, 'kind': family, 'steps': [{'id': p} for p in ('p1', 'p2', 'p3')]}
        for step in cases[cid]['steps']:
            for arm in ('C', 'R'):
                passed = (family == 'family_a') == (arm == 'R')
                results[f'{cid}-{step["id"]}-{arm}'] = record(cid, arm, step['id'], passed=passed)
    output = family_cluster_difference(cases, results)
    assert output['family_clusters'] == 2 and output['conversations_included'] == 3
    # Equal family weight yields zero; treating all 9 probes as independent
    # or weighting the duplicated A template twice would give a different mean.
    assert output['mean_difference_pp'] == 0
    assert output['descriptive_95_percentile_interval_pp'] == [-100, 100]
    assert output == family_cluster_difference(cases, results)
    del results['a2-p3-R']
    partial = family_cluster_difference(cases, results)
    assert partial['family_clusters'] == 1
    assert partial['incomplete_families_excluded'] == ['family_a']
    assert partial['descriptive_95_percentile_interval_pp'] is None


def test_cost_family_bootstrap_includes_maintenance_receipt_sum():
    cases = {cid: {'kind': family} for cid, family in (('a1', 'a'), ('a2', 'a'), ('b1', 'b'))}
    paired = [{'case_id': cid, 'missing_usage_attempts': 0,
               'comparison': {'total_tokens': {'C': 100, 'R': 50}}} for cid in cases]
    result = family_cluster_cost(cases, paired)
    assert result['family_clusters'] == 2
    assert result['point_estimate']['R_relative_C_change_pct'] == -50
    assert result['descriptive_95_percentile_intervals']['R_minus_C_tokens_per_conversation'] == [-50, -50]
    assert result == family_cluster_cost(cases, paired)


def test_attempt_file_reconciliation_does_not_deduplicate_identical_requests():
    call = receipt(); call['attempts'][0]['usage']['input_token_details'] = {'cache_read': 4}
    calls = {'a': call, 'b': call}
    files = {'a-1': call['attempts'][0], 'b-1': call['attempts'][0]}
    groups = {'C': {'all': usage([call])}, 'R': {'all': usage([call])}}
    checked = reconcile_attempt_usage(calls, files, groups)
    assert checked['all_receipts_accounted'] and checked['unique_attempt_files'] == 2
    assert checked['unique_attempt_file_usage']['input_tokens'] == 20
    assert checked['unique_attempt_file_usage']['cache_read_tokens'] == 8
    assert checked['unique_attempt_file_usage']['total_tokens'] == 24
    files['pending-1'] = call['attempts'][0]
    pending = reconcile_attempt_usage(calls, files, groups)
    assert not pending['all_receipts_accounted']
    assert pending['pending_or_unreferenced_attempt_files'] == ['pending-1']
