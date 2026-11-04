"""Predeclared business scenarios derived from the existing 300 RAG questions."""
import copy
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
GROUPS={
    'explicit_two':'明确双诉求', 'three_plus':'三项及以上诉求', 'colloquial':'口语与隐含诉求',
    'binding':'多订单及同类任务', 'clarification':'缺参与澄清', 'continuation':'取消与改口',
    'recovery':'恢复、重复请求及依赖故障', 'controls':'单意图、否定与库外控制'}
ATOMS=['A4','A5','A11','A13','A20','A26','A29','A35','A54','A58','A59','B6','B14','C33','E19']
POOLS={
 'three_plus':['A1','A6','A7','A9','A14','A17','A18','A19','A24','A30','A33','A37','B4','B27','E37'],
 'colloquial':['C4','C5','C6','C8','C11','C12','C14','C18','C20','C24','C25','C26','C33','C35','C36'],
 'clarification':['A4','A5','A11','A13','B5','A26','A29','A35','A54','B7','A59','B6','B14','C33','B17'],
 'controls':['B1','B2','B3','B8','B9','B10','B15','B16','B18','B19','B21','B22','B23','B24','B25']}


def original():
    return {r['id']:r for r in [json.loads(l) for l in
        (ROOT/'tests/data/eval_ch04.jsonl').read_text('utf-8').splitlines() if l.strip()]}


def fixtures(index):
    ids=[str(3100+index*7),str(6100+index*7),str(8100+index*7)]
    names=['智能猫砂盆','自动饮水机','自动喂食器']
    data={}
    for j,oid in enumerate(ids):
        data[oid]={'order_id':oid,'status':'已发货' if j==1 else '已签收',
            'amount':289+index*3+j*79,'created_at':'2026-11-02 10:00',
            'product':names[j], 'tracking_no':'SF'+oid.zfill(12)}
        if j!=1:data[oid]['delivered_at']='2026-11-02 15:00'
    logistics={x['tracking_no']:{'tracking_no':x['tracking_no'],
        'status':'运输中' if j==1 else '已签收',
        'current_city':['杭州','苏州','南京'][j],
        'trace':['隔离业务夹具，无真实包裹']} for j,x in enumerate(data.values())}
    return ids,data,logistics


def build_cases():
    src=original();out=[]
    for gi,group in enumerate(GROUPS):
      for i in range(15):
        index=gi*15+i
        (f,l,r),orders,logistics=fixtures(index)
        atom=POOLS.get(group,ATOMS)[i];q=src[atom]['query']
        case={'id':f'M{gi+1}-{i+1:02d}','group':group,'family_id':f'{group}:{i%5}',
            'source_question_ids':[], 'query':'','goals':[], 'orders':orders,'logistics':logistics,
            'actor':{'selection_targets':{},'cancel_selection':False},'followup':None,
            'fault':None,'forbidden_reads':[],'forbidden_ticket_types':[],
            'safety_required':False,'scenario_class':'normal'}
        def goal(kind,oid=None,*,outcome=None,explicit=True,source=None):
            defaults={'refund':'ticket','complaint':'ticket','order':'answer','logistics':'answer',
                      'knowledge':'answer','refuse':'refuse'}
            g={'id':f'g{len(case["goals"])+1}','kind':kind,'order_id':oid,
               'explicit_order':bool(oid and explicit),'expected_outcome':outcome or defaults[kind]}
            if source:
                s=copy.deepcopy(src[source]);g['source']=s
                g['required_points']=s['expect_points']
                if source=='B14':
                    g['required_points']=['6L','App 远程']
                    g['label_review']='原B14额外列出1080P摄像头，但本问只问远程投喂和容量，去掉未被询问的要点；原300题不改。'
                case['source_question_ids'].append(source)
            if kind=='refund':case['source_question_ids'].append('A2')
            if kind=='complaint':case['source_question_ids'].append('E30')
            if kind in ('order','logistics'):case['source_question_ids'].append('A15' if kind=='order' else 'E43')
            case['goals'].append(g);return g
        def knowledge(source=atom):return goal('knowledge',source=source)
        def complaint(outcome=None):return goal('complaint',outcome=outcome)
        def refund(oid=f,**kw):return goal('refund',oid,**kw)
        if group=='explicit_two':
            mode=i%5
            if mode==0:
                case['query']=f'请告诉我订单{f}的状态和实付金额；另外，{q}？'
                goal('order',f);knowledge()
            elif mode==1:
                case['query']=f'请查询订单{l}物流现在的位置和配送状态；另外，{q}？'
                goal('logistics',l);knowledge()
            elif mode==2:
                case['query']=f'订单{f}商品质量不好，请帮我申请退款；另外，{q}？'
                refund();knowledge()
            elif mode==3:
                case['query']=f'商家拒绝处理质量问题，我要投诉，请登记投诉工单；另外，{q}？'
                complaint();knowledge()
            else:
                case['query']=f'订单{f}有质量问题，请办理退款，同时登记商家推诿处理的投诉工单。'
                refund();complaint()
        elif group=='three_plus':
            mode=i%5
            if mode==0:
                case['query']=f'查订单{l}物流位置和状态，订单{f}申请质量问题退款，再登记商家拖延处理的投诉工单。'
                goal('logistics',l);refund();complaint()
            elif mode==1:
                case['query']=f'告诉我订单{f}状态和金额，给这笔订单申请质量问题退款，还要登记投诉商家的工单。'
                goal('order',f);refund();complaint()
            elif mode==2:
                case['query']=f'订单{f}商品坏了，请申请退款并登记投诉商家售后的工单；另外，{q}？'
                refund();complaint();knowledge()
            elif mode==3:
                case['query']=f'查{l}订单物流位置和状态，再告诉我{f}订单的状态和金额；最后解释一下：{q}？'
                goal('logistics',l);goal('order',f);knowledge()
            else:
                case['query']=f'先查{l}物流位置和状态，{f}商品坏了申请退款，登记投诉商家售后的工单，再回答：{q}？'
                goal('logistics',l);refund();complaint();knowledge()
        elif group=='colloquial':
            forms=[
                f'{f}这东西坏了，我要退掉把钱拿回来，店家一直推脱，请找专员登记投诉。',
                f'帮俺看看{l}那单快递跑哪了、啥状态，再给{f}这单坏掉的货办退钱。',
                f'{f}那单现在啥状态、我到底付了多少？还有那个事我想问：{q}？',
                f'店家一直不理我，帮我把投诉登记给人工；顺便说说{q}？',
                f'{f}买到的东西不能用，想退掉把款拿回来。顺带问一句，{q}？']
            case['query']=forms[i%5]
            if i%5==0:refund();complaint()
            elif i%5==1:goal('logistics',l);refund()
            elif i%5==2:goal('order',f);knowledge()
            elif i%5==3:complaint();knowledge()
            else:refund();knowledge()
        elif group=='binding':
            mode=i%5
            if mode==0:
                case['query']=f'订单{l}只查物流位置和状态；订单{f}因质量问题申请退款，顺便登记投诉工单。'
                goal('logistics',l);refund();complaint()
            elif mode==1:
                case['query']=f'订单{f}申请质量问题退款，订单{r}也申请质量问题退款。'
                refund();refund(r)
            elif mode==2:
                case['query']=f'{f}和{r}都因质量问题申请退款，{l}只查物流位置和状态。'
                refund();refund(r);goal('logistics',l)
            elif mode==3:
                case['query']=f'我有{10000+index}积分，想问{q}？另外订单{f}商品坏了申请退款。积分数字不是订单号。'
                knowledge();refund()
            else:
                case['query']=f'给{r}订单申请质量问题退款；{f}订单只查状态和金额，不要给{f}退款。'
                refund(r);goal('order',f)
        elif group=='clarification':
            mode=i%5
            if mode==0:
                case['query']=f'查询订单{l}物流位置和状态；另有一笔商品坏了要退款，我还没说退款订单号，也要登记投诉商家的工单。'
                goal('logistics',l);refund(explicit=False);complaint()
                case['actor']['selection_targets']={'refund':[f],'logistics':[l]}
            elif mode==1:
                case['query']=[
                    '一笔商品坏了想退款，订单号想不起来了；也请登记投诉商家推诿的工单。',
                    '想给有质量问题的那笔单办退款，但单号忘记了，请让我选；商家不理人，也请登记投诉工单。',
                    '东西坏了需要退钱，订单编号稍后补充；商家售后一直拖延，我还要提交投诉工单。'][i//5]
                refund(explicit=False);complaint();case['actor']['selection_targets']={'refund':[f]}
            elif mode==2:
                case['query']=f'订单{f}商品坏了，请申请退款；再帮我查另一笔订单物流位置和状态，物流订单请让我选择。'
                refund();goal('logistics',l,explicit=False)
                case['actor']['selection_targets']={'refund':[f],'logistics':[l]}
            elif mode==3:
                case['query']=f'订单{f}商品坏了，请申请退款，并登记投诉工单。订单号和诉求都在这句话里。'
                refund();complaint()
            else:
                case['query']=f'有笔质量问题订单想退款，订单稍后选择；还想知道{q}？'
                refund(explicit=False);knowledge();case['actor']['selection_targets']={'refund':[f]}
        elif group=='continuation':
            mode=i%5
            case['query']=f'订单{f}商品坏了，申请退款，并登记投诉商家推诿的工单。'
            if mode==0:
                refund(outcome='cancelled');complaint()
            elif mode==1:
                refund();complaint(outcome='cancelled')
            elif mode==2:
                case['query']=[
                    '商品坏了要退款，订单稍后选择；并登记投诉商家的工单。',
                    '我要给质量问题商品办理退款，请先让我选订单，再登记商家售后推诿的投诉工单。',
                    '有笔订单的货坏了想退钱，先让我选单；商家的服务问题还要登记投诉工单。'][i//5]
                refund(outcome='cancelled',explicit=False);complaint()
                case['actor']['cancel_selection']=True
            elif mode==3:
                refund(r);complaint()
                case['followup']=f'刚才退款订单说错了，只给订单{r}申请质量问题退款，订单{f}不要退款；投诉商家这件事继续，请登记投诉工单。'
            else:
                refund(outcome='cancelled');complaint()
                case['followup']='退款这项取消，不要提交任何退款申请；刚才投诉商家推诿处理的事情继续，请登记投诉工单。'
        elif group=='recovery':
            mode=i%3
            if mode==0:
                case['query']=[
                    '商品坏了要退款，订单号请让我选择；同时登记投诉商家售后的工单。',
                    '想申请质量问题退款，但忘了单号，请给我订单列表；商家的推诿问题还请登记投诉工单。',
                    '先让我选择需要退货退款的质量问题订单，然后登记商家不处理的投诉工单。',
                    '购买的东西坏了，申请退款的订单想点选；同时要登记商家售后处理差的投诉工单。',
                    '质量问题想办退款，单号记不清请让我选一笔；并提交投诉商家的人工工单。'][i//3]
                refund(explicit=False);complaint()
                case['actor']['selection_targets']={'refund':[f]};case['fault']='restart_connection'
            elif mode==1:
                case['query']=f'订单{f}申请质量问题退款，并登记投诉商家售后的工单。'
                refund();complaint();case['fault']='duplicate_submission'
            else:
                case['query']=f'查询订单{l}物流位置和状态，并登记投诉物流一直不更新的工单。'
                goal('logistics',l,outcome='unavailable');complaint()
                case['fault']='logistics_unavailable';case['scenario_class']='fault'
        else:
            case['scenario_class']='control';mode=i%5
            if mode==0:
                case['query']=f'只咨询规则：{q}？不要申请退款，也不要投诉或建工单。'
                knowledge();case['forbidden_ticket_types']=['退款','投诉','咨询','售后']
            elif mode==1:
                case['query']=f'订单{f}商品质量不好，要申请退款；不要投诉，也不要查询{l}的物流。'
                refund();case['forbidden_ticket_types']=['投诉'];case['forbidden_reads']=[l]
            elif mode==2:
                case['query']=f'登记投诉商家售后服务差的工单，不要退款。另请解释：{q}？'
                complaint();knowledge();case['forbidden_ticket_types']=['退款']
            elif mode==3:
                absent=['D4','D14','D22'][i//5]
                case['query']=f'请分别回答：{q}？以及{src[absent]["query"]}？如果某项没有资料请明确说明，不要影响另一项。不要建任何工单。'
                knowledge();goal('refuse',source=absent)
                case['forbidden_ticket_types']=['退款','投诉','咨询','售后']
            else:
                case['query']=f'有人说“我要退款并投诉”，这只是引用，不是我的要求。我现在只想知道订单{f}的状态和金额。'
                goal('order',f);case['forbidden_ticket_types']=['退款','投诉','咨询','售后']
        for g in case['goals']:
            if g['kind'] in ('order','logistics'):
                g['facts']=orders[g['order_id']] if g['kind']=='order' else logistics[orders[g['order_id']]['tracking_no']]
        case['source_question_ids']=sorted(set(case['source_question_ids']))
        case['source_mapping']={k:src[k]['query'] for k in case['source_question_ids']}
        case['derivation']='知识子问保留原题事实；退款/投诉/订单/物流主题扩展为带隔离业务对象和交互的任务。不是原题的原始标注。'
        out.append(case)
    return out
