"""Input readiness for versioned plans; no model calls and no business writes."""
from app.tools import business

TERMINAL = {'DONE','FAILED','CANCELLED','SUPERSEDED'}
ORDER_KINDS = {'order','logistics','refund'}
PRIORITY = {'order':0,'logistics':1,'refund':2,'knowledge':4}


def requires_order(t):
    return t['kind'] in ORDER_KINDS or (t['kind']=='knowledge' and t.get('knowledge_scope')=='product'
                                       and bool(t.get('subject_ref')))


def refresh(tasks, uid):
    groups={}
    for key,t in tasks.items():
        if requires_order(t):groups.setdefault(t.get('subject_ref') or key,[]).append((key,t))
    for group,members in groups.items():
        ids={t.get('order_id') or t.get('context_order_id') for _,t in members}-{None}
        known=next(iter(ids)) if len(ids)==1 else None
        known=known if known and business.owns_order(uid,known) else None
        unresolved=[]
        for key,t in members:
            if t['status'] in TERMINAL or t.get('step')=='confirm_submission':continue
            t['required_inputs']=['owned_order']+(['product'] if t['kind']=='knowledge' else [])
            if known:
                if t['kind']=='knowledge':t['context_order_id']=known
                elif not t.get('order_id'):
                    t.update(order_id=known,binding={'status':'resolved','method':'context_v2','order_ids':[known],
                                                     'source':{'kind':'shared_reference_group','group':group}})
                if t['status'] in {'WAITING_DEPENDENCY','WAITING_USER'}:
                    t.update(status='PENDING',step='inputs_ready',blocking=False)
            else:unresolved.append((key,t))
        if unresolved:
            leader=min(unresolved,key=lambda p:PRIORITY.get(p[1]['kind'],9))[0]
            for key,t in unresolved:
                t.update(status='WAITING_USER' if key==leader else 'WAITING_DEPENDENCY',
                         step='select_order' if key==leader else 'await_order',blocking=True,
                         depends_on=[] if key==leader else [leader])
    return tasks


def cancel_group(tasks, key):
    group=tasks[key].get('subject_ref')
    for other,t in tasks.items():
        if other==key or (group and t.get('subject_ref')==group and requires_order(t)):
            if t['status'] not in TERMINAL:t.update(status='CANCELLED',step='cancelled',blocking=False)
