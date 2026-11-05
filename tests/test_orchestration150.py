from collections import Counter
import json

from scripts.orchestration150_cases import build_cases,originals,read,SUBSET
from scripts.build_orchestration_dataset import assign_bucket,exclusion_reasons
from scripts.eval_multi120 import ticket_sequence_namespace


def test_equal_five_buckets_and_unique_complete_scenarios():
    cases=build_cases()
    assert len(cases)==150 and len({c['id'] for c in cases})==150
    assert len({c['query'] for c in cases})==150
    assert Counter(assign_bucket(c)[0] for c in cases)==Counter({b:30 for b in 'ABCDE'})
    assert sum(len(c['goals'])>1 for c in cases)==132
    assert sum(len(c['goals'])==1 for c in cases)==18


def test_original83_are_unchanged_and_new67_have_provenance():
    cases=build_cases();old=read(SUBSET/'cases.json');look={c['id']:c for c in cases};src=originals()
    assert all(look[c['id']]==c for c in old)
    new=[c for c in cases if c['id'].startswith('N')]
    assert len(new)==67
    assert all(c['source_question_ids'] and all(s in src for s in c['source_question_ids']) for c in new)
    assert all(not exclusion_reasons(c) and c['followup'] is None and not c['actor']['cancel_selection'] for c in cases)


def test_new_knowledge_points_have_frozen_evidence_and_unknowns_stay_unknown():
    for c in build_cases():
        if not c['id'].startswith('N'):continue
        for g in c['goals']:
            if g['kind']=='knowledge':
                corpus=''.join(''.join((k['question']+k['answer']).split()) for k in g['evidence'])
                assert all(''.join(p.split()) in corpus for p in g['required_points'])
            if g['kind']=='refuse':assert not g['evidence'] and g['source']['should_refuse']


def test_recovery_has_five_explicit_fault_or_cancel_types_each_six():
    counts=Counter()
    for c in build_cases():
        if not assign_bucket(c).startswith('D_'):continue
        if c['fault']:counts[c['fault']]+=1
        else:
            cancelled=next(g for g in c['goals'] if g['expected_outcome']=='cancelled')
            counts[cancelled['kind']+'_cancel']+=1
    assert counts==Counter({k:6 for k in ['refund_cancel','complaint_cancel','restart_connection','logistics_unavailable','duplicate_submission']})


def test_150_case_ticket_sequences_do_not_collide_between_arms():
    namespaces=[ticket_sequence_namespace(arm,i,1,150) for arm in ['disabled','enabled'] for i in range(150)]
    assert len(set(namespaces))==300
    assert all(abs(a-b)>=1000 for a,b in zip(sorted(namespaces),sorted(namespaces)[1:]))


def test_paired_wire_disables_fallback_independently_of_optional_settings(monkeypatch,tmp_path):
    from app.config import settings
    from app.core import runtime_settings
    from scripts import eval_orchestration150 as runner,eval_multi120 as old
    names=['chat_model','model_pool_db_path','langfuse_public_key','chat_failover_enabled',
           'model_pool_fallback_base_url','model_pool_fallback_api_key','model_pool_fallback_model']
    for name in names:monkeypatch.setattr(settings,name,getattr(settings,name))
    monkeypatch.setattr(runtime_settings,'SETTINGS_PATH',runtime_settings.SETTINGS_PATH)
    monkeypatch.setattr(old,'OUT',old.OUT);monkeypatch.setattr(old,'ROOT',old.ROOT)
    monkeypatch.setattr(runner,'OUT',tmp_path)
    (tmp_path/'protocol.json').write_text(json.dumps({'model_config':{'chat_model':'DeepSeek-V4-Flash'},'evaluation_settings':{},'model_pool_db_path':str(tmp_path/'pool.sqlite3')}),encoding='utf-8')
    settings.model_pool_fallback_base_url='https://unused.invalid';settings.model_pool_fallback_model='different-model'
    runner.wire('enabled')
    assert settings.chat_failover_enabled is False and not settings.model_pool_fallback_base_url and not settings.model_pool_fallback_model
