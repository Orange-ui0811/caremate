"""Independent source-level acceptance audit; never overwrites raw run labels."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import statistics

OUT=Path('E:/Mewhelp/data/verification/multi-intent-context-v2/20261104-v1')


def audit(d):
    c=d['case'];checks=dict(d.get('checks',{}));notes=[]
    if d.get('error'):checks['no_execution_error']=False
    steps=d.get('steps') or [{}]
    initial=steps[0].get('state',{}).get('task_plan',{})
    final=steps[-1].get('state',{}).get('task_plan',{})
    # The frozen contract is correct product context. Existing policy_query
    # intentionally searches general governing rules for return eligibility.
    for key in list(checks):
        if c['id']=='A1' and key.startswith('product_query_'):
            tasks=[t for t in final.values() if t['kind']=='knowledge']
            valid=len(tasks)==1 and tasks[0].get('context_order_id')=='1001' and (
                tasks[0].get('task_context',{}).get('order_facts',{}).get('product')=='智能猫砂盆') and (
                '退货退款申请条件' in tasks[0].get('retrieval_query',''))
            checks[key]=valid
            notes.append('A1原自动检查要求政策检索词包含商品名；既有policy_query按设计检索通用退货规则。依据冻结的对象正确性口径核对订单商品上下文，不改原标签。')
    actual=Counter(t['kind'] for t in initial.values() if t['kind']!='safety')
    required=Counter(c['required'].keys())
    if c['id']=='B3':checks['no_extra_business_tasks']=not any(k in actual for k in ['order','logistics','refund','complaint'])
    else:checks['no_extra_or_missing_tasks']=actual==required
    manual={'A2','A4','D1','D2','E1','F1','F2','F4'}
    if c['id'] in manual:
        ts=[t for t in final.values() if t['kind']=='knowledge']
        if c['action']=='cancel':checks['manual_dependency_cancelled']=len(ts)==1 and ts[0]['status']=='CANCELLED'
        else:
            checks['manual_bound_product']=len(ts)==1 and ts[0].get('context_order_id')=='1001'
            if c['action']!='order_failure':
                checks['manual_received_facts']=len(ts)==1 and ts[0].get('task_context',{}).get('order_facts',{}).get('product')=='智能猫砂盆'
                checks['manual_retrieved_with_product']=len(ts)==1 and '智能猫砂盆' in ts[0].get('retrieval_query','')
    if c['id'] in {'C3','C4'}:
        checks['general_not_bound']=all(not t.get('context_order_id') for t in final.values() if t['kind']=='knowledge')
    allowed_writes={'C2':Counter({'退款':1}),'D2':Counter({'投诉':1}),'E3':Counter({'退款':1,'投诉':1})}.get(c['id'],Counter())
    checks['expected_tickets_only']=Counter(t['type'] for t in d.get('tickets',[]))==allowed_writes
    return {'id':c['id'],'rep':d['rep'],'passed':all(checks.values()),'raw_passed':d.get('passed'),
            'checks':checks,'notes':notes,'failed':[k for k,v in checks.items() if not v]}


def main():
    rows=[];source={};records=[];setup_failures=[]
    for p in sorted((OUT/'formal1').glob('r*/*.json')):
        if p.stem=='manifest':continue
        d=json.loads(p.read_text(encoding='utf-8'))
        source[str(p.relative_to(OUT))]=hashlib.sha256(p.read_bytes()).hexdigest()
        if d['case']['id']=='B1' and d.get('error')=='TypeError' and not d.get('http') and not d.get('steps'):
            setup_failures.append(str(p.relative_to(OUT)))
            fixed=OUT/'setupfix'/p.parent.name/p.name
            if fixed.exists():
                p=fixed;d=json.loads(p.read_text(encoding='utf-8'))
        records.append(d)
        rows.append(audit(d));source[str(p.relative_to(OUT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    usage=Counter();unknown=Counter();calls=Counter();gates=Counter();latencies=[];initial_bindings=Counter();selections=0
    for d in records:
        for h in d.get('http',[]):
            channel=h['endpoint'];calls[channel]+=1;u=h.get('usage')
            if u is None:unknown[channel]+=1
            else:
                for k in ['prompt_tokens','completion_tokens','total_tokens','promptTokens','totalTokens']:
                    if isinstance(u.get(k),(int,float)):usage[channel+':'+k]+=u[k]
        if d.get('elapsed_seconds'):latencies.append(d['elapsed_seconds'])
        for s in d.get('steps',[]):
            if (s.get('interrupt') or {}).get('type')=='select_order':selections+=1
        for t in (d.get('steps') or [{}])[-1].get('state',{}).get('task_plan',{}).values():
            if t['kind']=='knowledge':gates[t.get('evidence_gate',t.get('outcome',t['status']))]+=1
        for k,v in d.get('checks',{}).items():
            if k.startswith('initial_'):initial_bindings['correct' if v else 'incorrect']+=1
    report={'complete':len(rows)==72 and all(bool(d.get('steps')) for d in records),'passed':sum(r['passed'] for r in rows),'total':len(rows),
        'setup_failures_preserved':setup_failures,'attempt_files':len(source),
        'raw_passed':sum(bool(r.get('raw_passed')) for r in rows),'rows':rows,'raw_sha256':source,
        'calls':dict(calls),'reported_usage':dict(usage),'unknown_usage_calls':dict(unknown),
        'knowledge_gates':dict(gates),'initial_binding_checks':dict(initial_bindings),
        'selection_events':selections,'expected_selection_events':sum(d['case']['selections'] for d in records),
        'elapsed_median':statistics.median(latencies) if latencies else None,
        'elapsed_max':max(latencies) if latencies else None,
        'limits':['Candidate-only internal acceptance set; no baseline comparison or general accuracy claim.',
                  'A1 policy-query checker correction is separate from immutable original checks and source data.',
                  'Provider usage may be absent or zero; no estimates are substituted.',
                  'Knowledge refusals are not counted as answered or factual coverage success.']}
    (OUT/'AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in {'rows','raw_sha256'}},ensure_ascii=False))
    for r in rows:
        if not r['passed']:print(r['rep'],r['id'],r['failed'])


if __name__=='__main__':main()
