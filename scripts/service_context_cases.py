"""Fixed regression selection plus declared, purpose-built overflow stress cases."""
import copy
from scripts.context_benchmark_cases import make_suite


def dataset():
    suite=make_suite()
    normal=[copy.deepcopy(c) for i,c in enumerate(suite['cases']) if i%6 in (0,2)]
    for c in normal:c['cohort']='regression'
    stress=[]
    noise='附加背景只介绍普通包装颜色和外观细节，不构成新的操作要求。'*320
    for idx,c in enumerate(suite['cases'][::6]):
        c=copy.deepcopy(c);c['id']='pressure-'+c['id'];c['cohort']='pressure'
        kind=c['kind']
        if kind=='early_order':
            for turn in range(4,24):
                oid=str(820000+turn)
                c['history'][2*turn-2]['content']=f'另登记订单{oid}，这里只登记，不改变最早订单。'+noise[:900]
                c['all_orders'].append(oid)
                c['tool_data']['orders'][oid]={'order_id':oid,'tracking_no':'SF'+oid+'88','status':'运输中','last_event':'包裹已到杭州分拨中心'}
            c['stress_description']='许多后续登记与长背景挤出早期对象'
        else:
            turns={'early_product':[16,22], 'correction':[9,20], 'topic_switch':[17,22],
                   'multi_hop':[7,11], 'temporal_update':[8,20], 'negation':[10,20],
                   'ambiguous':[10,20]}[kind]
            for turn in turns:c['history'][2*turn-2]['content']+='\n'+noise
            c['stress_description']='关键事实与超长背景出现在同一条消息，事实与原标准答案不变'
        stress.append(c)
    return {'version':'service-context-v1','cases':normal+stress,
            'selection':'16 pre-existing regression histories (variants 0/2 per family), 8 purpose-built pressure variants. Not a blind holdout.',
            'scope':'Given synthetic histories and two causal probes; no production customer data.'}
