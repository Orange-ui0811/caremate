from copy import deepcopy
import json
from pathlib import Path

from scripts.build_orchestration_dataset import exclusion_reasons, partition, evaluation_contract, assign_bucket, bucket_tags, BUCKETS


CASES=Path('E:/Mewhelp/data/multi-intent-eval/20261105-shared-agent120-v2/cases.json')


def originals():return json.loads(CASES.read_text(encoding='utf-8'))


def case(cid):return next(c for c in originals() if c['id']==cid)


def test_selection_cancel_excluded_but_form_cancel_and_continuation_retained():
    assert 'R2' in exclusion_reasons(case('M6-03'))
    assert not exclusion_reasons(case('M6-01'))
    assert not exclusion_reasons(case('M6-02'))
    assert 'R1' in exclusion_reasons(case('M6-04'))


def test_clear_complaint_does_not_require_unrequested_order_or_extra_details():
    assert not exclusion_reasons(case('M1-04'))
    assert 'R6' in exclusion_reasons(case('M2-02'))


def test_policy_filter_does_not_remove_unrelated_membership_or_return_shipping():
    assert 'R3' in exclusion_reasons(case('M1-02'))
    assert not exclusion_reasons(case('M1-06'))
    assert not exclusion_reasons(case('M1-08'))
    assert not exclusion_reasons(case('M1-09'))


def test_readable_but_difficult_cases_are_not_removed_by_previous_failures():
    assert not exclusion_reasons(case('M3-08'))  # Service-hours question is explicit.
    assert not exclusion_reasons(case('M3-15'))  # Troubleshooting evidence exists.
    assert not exclusion_reasons(case('M4-08'))  # Provider failure is not a question defect.


def test_partition_is_disjoint_complete_and_does_not_rewrite_cases():
    source=originals();snapshot=deepcopy(source);kept,excluded=partition(source)
    assert source==snapshot
    assert {c['id'] for c in kept}|{e['id'] for e in excluded}=={c['id'] for c in source}
    assert {c['id'] for c in kept}.isdisjoint(e['id'] for e in excluded)
    assert kept==[c for c in source if not exclusion_reasons(c)]
    assert all(e['original_case']==next(c for c in source if c['id']==e['id']) for e in excluded)


def test_outcome_metadata_cannot_change_selection():
    source=originals();modified=deepcopy(source)
    for i,c in enumerate(modified):c['historical_pass']=bool(i%2);c['success']=False;c['judge_status']='failed'
    assert [c['id'] for c in partition(source)[0]]==[c['id'] for c in partition(modified)[0]]


def test_main_and_controls_are_separate_and_business_evidence_is_not_relaxed():
    kept,_=partition(originals());contract=evaluation_contract(kept)
    assert all(len(c['goals'])>=2 for c in kept if c['id'] in contract['main_case_ids'])
    assert all(len(c['goals'])==1 for c in kept if c['id'] in contract['single_task_control_ids'])
    assert set(contract['main_case_ids']).isdisjoint(contract['single_task_control_ids'])
    assert len(contract['main_case_ids'])+len(contract['single_task_control_ids'])==len(kept)
    assert contract['main_goal_n']+contract['control_goal_n']==sum(len(c['goals']) for c in kept)
    assert any(c['fault']=='restart_connection' for c in kept)
    assert any(len([g for g in c['goals'] if g['kind']=='refund'])==2 for c in kept)
    assert any(any(g['expected_outcome']=='cancelled' for g in c['goals']) for c in kept)
    assert {c['group'] for c in kept}=={c['group'] for c in originals()}
    assert contract['run_now'] is False


def test_five_buckets_correspond_to_different_business_requirements():
    assert assign_bucket(case('M1-05'))=='A_explicit'
    assert assign_bucket(case('M4-02'))=='B_binding'
    assert assign_bucket(case('M4-04'))=='B_binding'
    assert assign_bucket(case('M5-02'))=='C_clarification'
    assert assign_bucket(case('M6-01'))=='D_recovery'
    assert assign_bucket(case('M7-03'))=='D_recovery'
    assert assign_bucket(case('M8-04'))=='E_boundary'


def test_special_condition_decides_bucket_without_losing_other_tags():
    c=case('M7-04')
    assert assign_bucket(c)=='D_recovery'
    assert 'missing_order' in bucket_tags(c) and 'restart_connection' in bucket_tags(c)
    assert assign_bucket(case('M5-03'))=='C_clarification'
    assert 'multiple_orders' in bucket_tags(case('M5-03'))


def test_new_bucket_does_not_just_copy_the_legacy_group():
    c=case('M4-04');c['group']='different_legacy_template'
    assert assign_bucket(c)=='B_binding'
    c=case('M3-01')
    assert assign_bucket(c)=='A_explicit' and 'colloquial' in bucket_tags(c)


def test_bucket_partition_has_no_duplicates_and_separates_boundary_controls():
    kept,_=partition(originals())
    mapping={c['id']:assign_bucket(c) for c in kept}
    assert len(mapping)==len(kept)
    assert set(mapping.values())=={b['id'] for b in BUCKETS}
    boundary=[c for c in kept if mapping[c['id']]=='E_boundary']
    assert any(len(c['goals'])>=2 for c in boundary) and any(len(c['goals'])==1 for c in boundary)
    assert all(c['scenario_class']=='control' for c in boundary)
