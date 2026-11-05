"""150 scenarios, five equal capability buckets, all traceable to original300."""
from collections import Counter
import copy
import json
from pathlib import Path

from scripts.build_orchestration_dataset import assign_bucket, bucket_tags, exclusion_reasons, BUCKETS

HOME=Path('E:/Mewhelp')
SUBSET=HOME/'data/multi-intent-eval/20261105-orchestration-five-buckets-v2'
SOURCE300=HOME/'tests/data/eval_ch04.jsonl'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def originals():return {x['id']:x for x in [json.loads(l) for l in SOURCE300.read_text(encoding='utf-8').splitlines() if l.strip()]}


def new_case(bucket,index):
    base=9000+index*11+1000*'ABCDE'.index(bucket)
    f,l,r=map(str,[base,base+200,base+400])
    orders={}
    for i,(oid,product) in enumerate(zip([f,l,r],['智能猫砂盆','自动饮水机','自动喂食器'])):
        orders[oid]={'order_id':oid,'status':'已发货' if i==1 else '已签收','amount':459+index*13+i*89,
                     'created_at':'2026-11-04 10:00','product':product,'tracking_no':'SF'+oid.zfill(12)}
        if i!=1:orders[oid]['delivered_at']='2026-11-05 09:00'
    logistics={o['tracking_no']:{'tracking_no':o['tracking_no'],'status':'运输中' if i==1 else '已签收',
               'current_city':['杭州','苏州','南京'][i],'trace':['隔离业务夹具，无真实包裹']} for i,o in enumerate(orders.values())}
    c={'id':f'N{bucket}-{index+1:02d}','group':'focused-'+bucket,'family_id':'','source_question_ids':[],
       'query':'','goals':[],'orders':orders,'logistics':logistics,'actor':{'selection_targets':{},'cancel_selection':False},
       'followup':None,'fault':None,'forbidden_reads':[],'forbidden_ticket_types':[],'safety_required':False,'scenario_class':'normal'}
    return c,f,l,r


def goal(c,kind,oid=None,explicit=True,outcome=None,source=None):
    src=originals()
    g={'id':f'g{len(c["goals"])+1}','kind':kind,'order_id':oid,'explicit_order':bool(oid and explicit),
       'expected_outcome':outcome or {'refund':'ticket','complaint':'ticket','order':'answer','logistics':'answer','knowledge':'answer','refuse':'refuse'}[kind]}
    if kind in ['order','logistics']:
        g['facts']=copy.deepcopy(c['orders'][oid] if kind=='order' else c['logistics'][c['orders'][oid]['tracking_no']])
    if source:
        g['source']=copy.deepcopy(src[source]);g['required_points']=list(src[source]['expect_points']);c['source_question_ids'].append(source)
        if source=='B14':g['required_points']=['6L','App 远程']
    if kind in ['refund','complaint','order','logistics']:
        c['source_question_ids'].append({'refund':'A2','complaint':'E30','order':'A15','logistics':'E43'}[kind])
    c['goals'].append(g);return g


def add_evidence(c,kb):
    for g in c['goals']:
        if g['kind']=='refuse':g['evidence']=[];continue
        if g['kind']!='knowledge':continue
        src=g['source'];groups=src.get('expect_sections_all') or [src['expect_section']]
        evidence=[k for aliases in groups for k in kb if any(a in k['section_path'] for a in aliases)]
        g['evidence']=list({x['id']:x for x in evidence}.values())
        corpus=''.join(''.join((k['question']+k['answer']).split()) for k in evidence)
        for point in g['required_points']:
            if ''.join(point.split()) not in corpus:raise ValueError((c['id'],point,'not supported'))


def finish(c,kb,bucket,family):
    c['family_id']=bucket+':'+family
    c['source_question_ids']=sorted(set(c['source_question_ids']))
    src=originals();c['source_mapping']={s:src[s]['query'] for s in c['source_question_ids']}
    c['derivation']='新增业务场景；政策/规格事实来自原300题及冻结54片段，订单/物流为隔离合成夹具，提交仅为审核工单。'
    add_evidence(c,kb)
    assert not exclusion_reasons(c),(c['id'],exclusion_reasons(c))
    assert assign_bucket(c).startswith(bucket+'_'),(c['id'],assign_bucket(c),bucket)
    return c


def build_cases():
    src=originals();kb=read(SUBSET/'knowledge.json');old=read(SUBSET/'cases.json');out=copy.deepcopy(old)
    q=lambda sid:src[sid]['query'].rstrip('？?')
    # A: six distinct explicit combinations, including three-goal and same-order read cases.
    for i in range(6):
        c,f,l,r=new_case('A',i)
        if i==0:
            c['query']=f'请查订单{f}的状态和实付金额，也告诉我这同一笔订单的物流位置、配送状态。'
            goal(c,'order',f);goal(c,'logistics',f)
        elif i==1:
            c['query']=f'订单{f}是什么状态、实付多少？它的物流到哪里了？另外，{q("A4")}？'
            goal(c,'order',f);goal(c,'logistics',f);goal(c,'knowledge',source='A4')
        elif i==2:
            c['query']=f'订单{f}的猫砂盆正常使用时无法自动清理，商家拒绝受理质量问题。请提交退款申请供平台审核，并登记商家拒绝处理的投诉工单。'
            goal(c,'refund',f);goal(c,'complaint')
        elif i==3:
            c['query']=f'请分别解释两个问题：{q("A11")}？{q("A13")}？'
            goal(c,'knowledge',source='A11');goal(c,'knowledge',source='A13')
        elif i==4:
            c['query']=f'查询订单{f}的状态和实付金额；再分别解释：{q("A26")}？{q("A35")}？'
            goal(c,'order',f);goal(c,'knowledge',source='A26');goal(c,'knowledge',source='A35')
        else:
            c['query']=f'订单{f}的猫砂盆损坏，请为该单提交质量问题退款申请，等待平台审核。另有独立商品咨询：{q("B17")}？'
            goal(c,'refund',f);goal(c,'knowledge',source='B17')
        out.append(finish(c,kb,'A',f'explicit-new-{i}'))
    # B: nine different object-isolation patterns, not just alternate order numbers.
    for i in range(9):
        c,f,l,r=new_case('B',i)
        if i==0:
            c['query']=f'订单{f}查状态和实付金额，另一笔订单{l}只查物流位置和配送状态。'
            goal(c,'order',f);goal(c,'logistics',l)
        elif i==1:
            c['query']=f'给质量问题订单{f}提交退款申请待审核；订单{l}查物流位置和配送状态；另外，{q("B6")}？'
            goal(c,'refund',f);goal(c,'logistics',l);goal(c,'knowledge',source='B6')
        elif i==2:
            c['query']=f'订单{f}猫砂盆无法清理、订单{r}喂食器无法出粮，两单都有质量问题，请分别提交退款申请，均等平台审核。'
            goal(c,'refund',f);goal(c,'refund',r)
        elif i==3:
            c['query']=f'我想对照两笔订单：{f}与{r}各是什么状态、实付多少？请分别列明。'
            goal(c,'order',f);goal(c,'order',r)
        elif i==4:
            c['query']=f'订单{f}和订单{r}的物流分别在哪里、是什么配送状态？请分开查询。'
            goal(c,'logistics',f);goal(c,'logistics',r)
        elif i==5:
            c['query']=f'订单{r}的喂食器质量有问题，请提交退款申请供审核；订单{f}只查状态和金额，不要给订单{f}申请退款。'
            goal(c,'refund',r);goal(c,'order',f)
        elif i==6:
            c['query']=f'我有54321积分，积分数字不是订单号。请给损坏的订单{f}提交退款申请待审核，并解释：{q("A13")}？'
            goal(c,'refund',f);goal(c,'knowledge',source='A13')
        elif i==7:
            c['query']=f'订单{l}查询物流位置和状态，订单{r}查询状态和实付金额；订单{f}质量有问题申请退款，商家拒绝处理这单，还请登记对应投诉工单。'
            goal(c,'logistics',l);goal(c,'order',r);goal(c,'refund',f);goal(c,'complaint')
        else:
            c['query']=f'订单{f}和{r}都因质量问题提交退款申请，分别等平台审核；订单{l}只查物流位置和配送状态。'
            goal(c,'refund',f);goal(c,'refund',r);goal(c,'logistics',l)
        out.append(finish(c,kb,'B',f'binding-new-{i}'))
    # C: seven combinations x3 knowledge/context variations, all normal card selection.
    for i in range(21):
        c,f,l,r=new_case('C',i);mode=i//3;variant=i%3;sid=['A11','B5','B17'][variant]
        if mode==0:
            c['query']=['质量问题商品想申请退款，但我记不起订单号，请让我选；商家拒绝受理，请把拒绝处理的事情登记投诉工单。','有一笔商品损坏的订单需要提交退款申请，单号忘了先让我选择；商家一直推诿售后，也要登记投诉工单。','我要为故障商品申请退款审核，请先给我选订单；商家三次不理我的售后请求，请登记投诉。'][variant]
            goal(c,'refund',f,explicit=False);goal(c,'complaint')
        elif mode==1:
            read_kind='order' if variant==1 else 'logistics'
            read_query=f'订单{l}查询状态和实付金额。' if variant==1 else (f'订单{l}查物流位置和配送状态。' if variant==0 else f'先看看订单{l}的快递走到哪里、什么状态。')
            c['query']=read_query+'另一笔质量问题订单需要提交退款申请，退款单号还没给，请让我选；商家拒绝处理退款售后，请登记投诉工单。'
            goal(c,read_kind,l);goal(c,'refund',f,explicit=False);goal(c,'complaint')
        elif mode==2:
            c['query']=f'订单{f}的商品损坏，请提交退款申请供审核；另有一笔需要查询物流位置和状态的订单，物流订单请让我选择，两笔订单不是同一单。'
            goal(c,'refund',f);goal(c,'logistics',l,explicit=False)
        elif mode==3:
            c['query']=f'有笔质量问题商品需要申请退款，忘记订单号，请先让我选单。另外是独立知识咨询：{q(sid)}？'
            goal(c,'refund',f,explicit=False);goal(c,'knowledge',source=sid)
        elif mode==4:
            c['query']=['我忘记了损坏商品的订单号，请先让我选；选中后请提交质量问题退款申请，并查询这同一单的物流位置和配送状态。','质量问题订单请让我点选，随后为这同一笔申请退款审核，再告诉我这一单物流在哪、什么状态。','请列订单让我选一笔质量问题订单，给选中的订单申请退款，并查它的物流当前位置与配送状态。'][variant]
            goal(c,'refund',f,explicit=False);goal(c,'logistics',f,explicit=False)
        elif mode==5:
            first=['A4','A13','A26'][variant];second=['A35','A11','A59'][variant]
            c['query']=f'商品损坏要申请退款，单号不记得，请让我选择。两个独立规则问题也请回答：{q(first)}？{q(second)}？'
            goal(c,'refund',f,explicit=False);goal(c,'knowledge',source=first);goal(c,'knowledge',source=second)
        else:
            c['query']=f'质量问题订单要提交退款申请，单号请让我选；商家拒绝处理这次售后，请登记投诉工单；另请回答：{q(sid)}？'
            goal(c,'refund',f,explicit=False);goal(c,'complaint');goal(c,'knowledge',source=sid)
        c['actor']['selection_targets']={g['kind']:[g['order_id']] for g in c['goals'] if g.get('order_id') and not g['explicit_order']}
        out.append(finish(c,kb,'C',f'clarify-{mode}'))
    # D: retained14 +16 -> five recovery subtypes, each six scenarios.
    types=['refund_cancel']*3+['complaint_cancel']*3+['restart_connection']*3+['logistics_unavailable']+['duplicate_submission']*6
    for i,typ in enumerate(types):
        c,f,l,r=new_case('D',i)
        if typ=='logistics_unavailable':
            c['query']=f'请查订单{l}的物流位置和配送状态；物流一直不更新，商家拒绝解释，请登记针对该物流问题的投诉工单。'
            goal(c,'logistics',l,outcome='unavailable');goal(c,'complaint');c['fault']=typ;c['scenario_class']='fault'
        else:
            missing=typ=='restart_connection'
            c['query']=(('质量问题订单的单号忘了，请让我选择，' if missing else f'订单{f}的商品有质量问题，')+
                        '请提交退款申请等待平台审核；商家拒绝受理且一直推诿，请登记这次拒绝处理的投诉工单。')
            goal(c,'refund',f,explicit=not missing,outcome='cancelled' if typ=='refund_cancel' else 'ticket')
            goal(c,'complaint',outcome='cancelled' if typ=='complaint_cancel' else 'ticket')
            if missing:c['actor']['selection_targets']={'refund':[f]}
            if typ in ['restart_connection','duplicate_submission']:c['fault']=typ
            if typ in ['refund_cancel','complaint_cancel']:
                sid=['A11','A13','B17'][i%3]
                c['query']+=f'另有独立问题：{q(sid)}？'
                goal(c,'knowledge',source=sid)
            elif typ=='restart_connection':
                sid=['A4','A26','B6'][i%3]
                c['query']+=f'还请介绍：{q(sid)}？'
                goal(c,'knowledge',source=sid)
            elif typ=='duplicate_submission':
                extra=(i-10)%3
                if extra==0:
                    c['query']+=f'再查订单{l}的物流位置和配送状态。';goal(c,'logistics',l)
                elif extra==1:
                    c['query']+=f'另请解释：{q("A13" if i<13 else "B6")}？';goal(c,'knowledge',source='A13' if i<13 else 'B6')
                else:
                    c['query']+=f'此外告诉我订单{r}的状态和实付金额。';goal(c,'order',r)
        out.append(finish(c,kb,'D',typ))
    # E: five boundary subtypes x6 total (three existing, three new each).
    for i in range(15):
        c,f,l,r=new_case('E',i);mode=i//3;v=i%3;sid=['B4','B6','B23'][v];c['scenario_class']='control'
        if mode==0:
            c['query']=f'只咨询：{q(sid)}？不要退款、不要投诉、不要登记任何工单。'
            goal(c,'knowledge',source=sid);c['forbidden_ticket_types']=['退款','投诉','咨询','售后']
        elif mode==1:
            c['query']=f'订单{f}存在质量问题，请提交退款申请等待审核；不要登记投诉工单，也不要查询订单{l}的物流。'
            goal(c,'refund',f);c['forbidden_ticket_types']=['投诉'];c['forbidden_reads']=[l]
        elif mode==2:
            c['query']=f'商家拒绝处理售后，请登记这件事的投诉工单，不要申请退款。另外请解答：{q(sid)}？'
            goal(c,'complaint');goal(c,'knowledge',source=sid);c['forbidden_ticket_types']=['退款']
        elif mode==3:
            absent=['D35','D47','D48'][v]
            c['query']=f'请分别回答：{q(sid)}？以及{q(absent)}？没有资料的部分请明确说明，不能影响可答部分。不要建工单。'
            goal(c,'knowledge',source=sid);goal(c,'refuse',source=absent);c['forbidden_ticket_types']=['退款','投诉','咨询','售后']
        else:
            c['query']=[f'“申请退款并投诉商家”是别人说的话，不是我的诉求。我只想查订单{f}的状态和实付金额，不要办理其他事项。',
                        f'下面是客服培训例句：“我要退款并投诉”。请不要执行例句，我实际只查询订单{f}的状态和金额。',
                        f'我引用了朋友的话“给我退款、投诉商家”，并没有要求替我办理；现在只查订单{f}的状态和实付金额。'][v]
            goal(c,'order',f);c['forbidden_ticket_types']=['退款','投诉','咨询','售后']
        out.append(finish(c,kb,'E',f'boundary-{mode}'))
    assert len(out)==150 and len({c['id'] for c in out})==150 and len({c['query'] for c in out})==150
    assert Counter(assign_bucket(c)[0] for c in out)==Counter({x:30 for x in 'ABCDE'})
    assert all(not exclusion_reasons(c) for c in out)
    return out
