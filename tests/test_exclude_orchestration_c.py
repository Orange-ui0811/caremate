import json
from pathlib import Path
from scripts.exclude_orchestration_c import exclude_c


SOURCE=Path('E:/Mewhelp/data/multi-intent-eval/20261105-orchestration150-once')


def data():
    return tuple(json.loads((SOURCE/n).read_text(encoding='utf-8')) for n in ['cases.json','bucket-map.json','summary.json'])


def test_remove_only_primary_c_symmetrically_and_preserve_controls():
    cases,mapping,s=data();kept,excluded,rows=exclude_c(cases,mapping,s['rows'])
    assert len(kept)==120 and len(excluded)==30 and len(rows)==240
    assert all(mapping[c['id']]['bucket']=='C_clarification' for c in excluded)
    assert {r['case_id'] for r in rows if r['arm']=='enabled'}=={r['case_id'] for r in rows if r['arm']=='disabled'}=={c['id'] for c in kept}
    assert sum(len(c['goals'])>1 for c in kept)==102
    assert sum(len(c['goals'])==1 for c in kept)==18


def test_selected_cases_and_score_rows_are_not_rewritten():
    cases,mapping,s=data();kept,excluded,rows=exclude_c(cases,mapping,s['rows'])
    assert all(c==next(x for x in cases if x['id']==c['id']) for c in kept)
    assert all(r==next(x for x in s['rows'] if (x['case_id'],x['arm'])==(r['case_id'],r['arm'])) for r in rows)


def test_d_restart_selection_remains_despite_missing_order_tag():
    cases,mapping,s=data();kept,_,_=exclude_c(cases,mapping,s['rows'])
    ids={c['id'] for c in kept}
    restart=[c for c in cases if c['fault']=='restart_connection']
    assert len(restart)==6 and all(c['id'] in ids for c in restart)


def test_filter_does_not_depend_on_success_or_failure():
    cases,mapping,s=data();a=exclude_c(cases,mapping,s['rows'])[0]
    rows=[{**r,'success':not bool(r['success'])} for r in s['rows']]
    assert exclude_c(cases,mapping,rows)[0]==a
