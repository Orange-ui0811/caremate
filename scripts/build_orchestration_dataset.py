"""Freeze a scope-filtered subset of original120; never reads outcomes or calls a model."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
HOME = Path('E:/Mewhelp')
SOURCE = HOME / 'data/multi-intent-eval/20261105-shared-agent120-v2'
DEST = HOME / 'data/multi-intent-eval/20261105-orchestration-five-buckets-v2'

BUCKETS = [
    {'id':'A_explicit','name':'常规明确多任务','question':'能否识别并逐项完成本轮明确提出的多个诉求？',
     'scope':'至少两项目标；没有更高优先级的边界控制、特殊续办、缺订单或多对象干扰。含业务查询+知识咨询、办理+咨询、退款+投诉。'},
    {'id':'B_binding','name':'多订单与对象绑定','question':'能否把各项任务绑定到正确订单，并隔离同类任务和非订单数字？',
     'scope':'至少两笔目标订单，或明确存在积分数字等订单号干扰；不能串单、漏掉第二笔办理、把积分数字当订单号。'},
    {'id':'C_clarification','name':'缺参选单与续办','question':'订单未给出时，能否澄清一次，并在选择后完成该项及剩余任务？',
     'scope':'至少一项订单/物流/退款任务明确缺订单号，使用正常公开选单或本项必要澄清；不含文字改口、取消选单或注入故障。'},
    {'id':'D_recovery','name':'状态变化与异常续办','question':'取消当前办理、重连或查询失败后，是否正确保留并继续其他事项？',
     'scope':'退款表单取消、投诉确认卡取消、等待时重建检查点连接、物流工具不可用。正常确认属于各桶共用机制，不单独凑成异常题。'},
    {'id':'E_boundary','name':'请求边界与部分可答','question':'能否遵守否定和引用边界，并分别处理可答与无资料的子问题？',
     'scope':'只咨询不办理、只退款不投诉、引用他人诉求、投诉但禁止退款、一个子问有依据而另一个库外。9个单任务控制与6个多任务边界题分别报告。'},
]

RULES = [
    {'id':'R1','title':'等待卡片期间文字改口或取消','basis':'存在followup；本轮只评公开卡片继续，不评文字修改挂起计划。'},
    {'id':'R2','title':'取消订单选择','basis':'actor.cancel_selection=true；本轮保留表单确认/取消，排除原选单卡不支持的取消。'},
    {'id':'R3','title':'运费政策适用来源不明确','basis':'知识来源A5或A14，涉及基础/偏远运费计算；原冻结知识存在6元/10元冲突。仅问积分兑换、会员次数/结转或退货运费承担不在此规则。'},
    {'id':'R4','title':'纸质与电子专票标注错配','basis':'知识来源A30；原目标把电子专票的企业账户条件移用于纸质专票。'},
    {'id':'R5','title':'个案资格条件未提供','basis':'知识来源C8、C20：个案保修/价保缺少类别、时间或其他适用条件，原gold不足以确定个案结论。通用政策咨询不一并排除。'},
    {'id':'R6','title':'投诉请求缺少明确投诉事由','basis':'要求建投诉工单，但未明确拒绝、推诿、推脱、拖延、不处理、不理、不更新、服务差或处理差等事由。仅有“投诉商家售后/登记投诉”不纳入本轮。订单号不是所有投诉的额外必填项。'},
]
COMPLAINT_ISSUE = re.compile(r'拒绝|推诿|推脱|拖延|不处理|不理|不更新|服务差|处理差')


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, value): path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def exclusion_reasons(case):
    """These are rules for this fixed Chinese corpus, not a production classifier."""
    hits = []
    source_ids = {g.get('source',{}).get('id') for g in case['goals']}
    if case.get('followup'): hits.append('R1')
    if case.get('actor',{}).get('cancel_selection'): hits.append('R2')
    if source_ids & {'A5','A14'}: hits.append('R3')
    if 'A30' in source_ids: hits.append('R4')
    if source_ids & {'C8','C20'}: hits.append('R5')
    if any(g['kind']=='complaint' for g in case['goals']) and not COMPLAINT_ISSUE.search(case['query']): hits.append('R6')
    return hits


def partition(cases):
    kept, excluded = [], []
    for case in cases:
        reasons = exclusion_reasons(case)
        if reasons:
            excluded.append({'id':case['id'],'query':case['query'],'group':case['group'],
                             'primary_rule':reasons[0],'all_rules':reasons,'original_case':copy.deepcopy(case)})
        else:
            kept.append(copy.deepcopy(case))
    return kept, excluded


def bucket_tags(case):
    goals=case['goals'];kinds=[g['kind'] for g in goals]
    tags=[f'tasks:{len(goals)}']
    if len({g['order_id'] for g in goals if g.get('order_id')})>1:tags.append('multiple_orders')
    if re.search(r'我有\d{4,}积分|积分数字不是订单号',case['query']):tags.append('number_distractor')
    if any(g['kind'] in {'refund','order','logistics'} and not g['explicit_order'] for g in goals):tags.append('missing_order')
    if case['fault']:tags.append(case['fault'])
    if any(g['expected_outcome']=='cancelled' for g in goals):tags.append('card_cancel')
    if sum(g['kind'] in {'refund','complaint'} for g in goals)>=2:tags.append('multiple_write_intents')
    if any(kinds.count(k)>1 for k in set(kinds)):tags.append('same_kind_multiple_tasks')
    if case['group']=='colloquial':tags.append('colloquial')
    if 'knowledge' in kinds:tags.append('knowledge')
    if 'refuse' in kinds:tags.append('partially_answerable')
    return tags


def assign_bucket(case):
    """Mutually exclusive primary capability; secondary difficulty stays in tags."""
    tags=bucket_tags(case)
    if case['scenario_class']=='control':return 'E_boundary'
    if case['fault'] or 'card_cancel' in tags:return 'D_recovery'
    if 'missing_order' in tags:return 'C_clarification'
    if 'multiple_orders' in tags or 'number_distractor' in tags:return 'B_binding'
    return 'A_explicit'


def evaluation_contract(cases):
    main = [c['id'] for c in cases if len(c['goals'])>=2]
    controls = [c['id'] for c in cases if len(c['goals'])==1]
    return {
        'version':'orchestration-five-buckets-v2',
        'main_case_ids':main,'single_task_control_ids':controls,
        'main_goal_n':sum(len(c['goals']) for c in cases if c['id'] in main),
        'control_goal_n':sum(len(c['goals']) for c in cases if c['id'] in controls),
        'repetitions_if_run':3,'run_now':False,
        'scenario_success':'全部必需目标完成，且无错误订单、未经确认写入、重复工单或禁止操作；存在未知保留待定，分母不删题。',
        'task_success':'查询须有正确对象的可信结果并回答所问事实；知识回答覆盖原required_points及适用条件；退款/投诉须有唯一正确且经本项确认的SQL申请/投诉回执。',
        'refund_scope':'只评提交退款申请供审核，不评资金退回或保证审核通过；保留题若依旧拒办、重复追问或无回执，不事后豁免。',
        'cancel_scope':'保留退款表单取消、投诉确认卡取消及后续任务续办；不包含取消订单选择或等待时文字改口。',
        'clarification':'缺订单可经公开卡片选择一次；不同对象分别绑定；模型或队列内部DONE不能证明业务完成。',
        'reported_metrics':[
            '主集场景通过率：confirmed_success / (74×repetitions)，待定另报',
            '主集必需任务完成率：confirmed_completed / (157×repetitions)，失败与待定分别列出',
            '按bucket-map.json的5个能力桶分别报告；原case.group只保留来源，不作新版分桶依据',
            '主集总体通过率按74题汇总，并单报5桶主集通过率及其宏平均；桶E的9个单任务控制不进入主集宏平均',
            '9个单任务/否定控制单独报告，不混入多意图主通过率',
            '对象错误、所有选单卡及用途未映射事件、额外选单、重复工单、未经确认写入',
            '选单/确认/表单取消及重连后剩余任务完成情况；回复覆盖与事实依据',
        ],
        'semantic_judge':'沿用修复后的证据/时序原则，提供实际检索材料、冻结知识及SQL快照。risk区分violation/supported/uncertain，不因quote存在就断言违规。',
        'provider':'若将来调用，固定同一个评审模型、同一接口、temperature=0，禁用备用模型混用；接口错误原样保留并单列。',
        'historical_reuse':'保留题的完整输入、目标及业务夹具逐字节语义不改，可定位旧轨迹作同题诊断；原120整体分数和旧自动标签不可直接充当新集结果。旧驱动污染/无法续办的记录不得伪装成新驱动下已成功。',
        'limits':[
            '这是根据已知问题定义范围的事后专项子集，不是独立盲测，不证明整体客服效果提升。',
            '规则只读取题目、目标和交互要求，不读取任何版本的通过率、模型回复或得分；相同规则用于两版。',
            '不承诺所有共性缺陷都被题目规避；保留的明确办理题仍可能触发模型误判或错误收尾，出现就照实计分。',
            '重复提交故障题因投诉事由不明确被排除；本集没有专门重复点击故障覆盖，但重复工单仍是全题硬约束。',
            '未补题、未改问法、未改知识库及原300/120题；未启动任何业务或模型评测。',
        ],
    }


def build(source=SOURCE, dest=DEST):
    if dest.exists(): raise RuntimeError('Refuse to overwrite an existing frozen dataset')
    originals = read(source/'cases.json')
    kept, excluded = partition(originals)
    assert len(originals)==120 and len(kept)+len(excluded)==120
    assert {c['id'] for c in kept}.isdisjoint(x['id'] for x in excluded)
    baseline = HOME/'data/multi-intent-eval/20261103-business120/cases.json'
    assert sha(source/'cases.json')==sha(baseline)
    dest.mkdir(parents=True)
    write(dest/'cases.json',kept);write(dest/'excluded.json',excluded)
    write(dest/'exclusion-rules.json',RULES)
    contract = evaluation_contract(kept);write(dest/'scoring.json',contract)
    bucket_map={c['id']:{'bucket':assign_bucket(c),'tags':bucket_tags(c),
                         'split':'main' if c['id'] in contract['main_case_ids'] else 'control'} for c in kept}
    write(dest/'bucket-map.json',bucket_map)
    write(dest/'buckets.json',{'version':'five-buckets-v2','definitions':BUCKETS,
                             'assignment_priority':['E_boundary','D_recovery','C_clarification','B_binding','A_explicit'],
                             'why_priority':'特殊目标决定主桶，其他特征保留副标签；例如选单时重连归D并保留missing_order标签，不在C重复计数。',
                             'scoring_join':'按原case_id连接bucket-map.json，禁止把旧case.group当新版桶；每题只算一次。'})
    (dest/'buckets').mkdir()
    for bucket in BUCKETS:write(dest/'buckets'/(bucket['id']+'.json'),[c for c in kept if bucket_map[c['id']]['bucket']==bucket['id']])
    shutil.copy2(source/'knowledge.json',dest/'knowledge.json')
    by_group = {k:{'original':sum(c['group']==k for c in originals),'retained':sum(c['group']==k for c in kept),
                   'main':sum(c['group']==k and c['id'] in contract['main_case_ids'] for c in kept),
                   'control':sum(c['group']==k and c['id'] in contract['single_task_control_ids'] for c in kept)}
                for k in dict.fromkeys(c['group'] for c in originals)}
    inventory = {'original_n':120,'retained_n':len(kept),'excluded_n':len(excluded),
                 'main_n':len(contract['main_case_ids']),'control_n':len(contract['single_task_control_ids']),
                 'main_goal_n':contract['main_goal_n'],'control_goal_n':contract['control_goal_n'],
                 'task_count_distribution':dict(Counter(len(c['goals']) for c in kept)),
                 'buckets':{b['id']:{'name':b['name'],'n':sum(v['bucket']==b['id'] for v in bucket_map.values()),
                                     'main':sum(v['bucket']==b['id'] and v['split']=='main' for v in bucket_map.values()),
                                     'control':sum(v['bucket']==b['id'] and v['split']=='control' for v in bucket_map.values())} for b in BUCKETS},
                 'groups':by_group,'task_types':dict(Counter(g['kind'] for c in kept for g in c['goals'])),
                 'primary_exclusion_counts':dict(Counter(c['primary_rule'] for c in excluded)),
                 'all_exclusion_match_counts':dict(Counter(r for c in excluded for r in c['all_rules'])),
                 'faults':dict(Counter(c['fault'] or 'none' for c in kept)),
                 'card_cancellation_cases':[c['id'] for c in kept if any(g['expected_outcome']=='cancelled' for g in c['goals'])],
                 'multiple_order_cases':[c['id'] for c in kept if len({g['order_id'] for g in c['goals'] if g.get('order_id')})>1],
                 'missing_order_cases':[c['id'] for c in kept if any(g['kind'] in {'refund','order','logistics'} and not g['explicit_order'] for g in c['goals'])]}
    write(dest/'inventory.json',inventory)
    source300 = ROOT/'tests/data/eval_ch04.jsonl'
    metadata = {'version':'orchestration-five-buckets-v2','design_date':'2026-11-05','user_scope':'保留明确办理题，排除改口、选单取消及政策或资料争议题；按主要业务能力分为5桶。',
                'source_cases_path':str(source/'cases.json'),'source_cases_sha256':sha(source/'cases.json'),
                'source300_path':str(source300),'source300_sha256':sha(source300),
                'source_knowledge_sha256':sha(source/'knowledge.json'),
                'selection_uses_outcomes':False,'business_runs':0,'model_calls':0,
                'retained_case_ids':[c['id'] for c in kept],
                'input_hashes':{n:sha(dest/n) for n in ['cases.json','excluded.json','exclusion-rules.json','scoring.json','knowledge.json','inventory.json','buckets.json','bucket-map.json']}}
    previous=HOME/'data/multi-intent-eval/20261105-orchestration-focused-v1/cases.json'
    if previous.exists():
        assert sha(previous)==sha(dest/'cases.json')
        metadata['previous_subset_cases_sha256']=sha(previous)
        metadata['v2_change']='Only primary buckets/tags and reporting organization changed; same83 retained cases as v1.'
    shutil.copy2(__file__,dest/'build_orchestration_dataset.py')
    metadata['builder_sha256']=sha(dest/'build_orchestration_dataset.py');write(dest/'protocol.json',metadata)
    titles={r['id']:r['title'] for r in RULES}
    lines=['# 多意图编排专项：五桶评测集 v2','','从原120个完整业务场景按固定范围规则筛选，保留83题、排除37题；原文、目标和夹具不改，不补题。74题/157目标为多任务主集，9题/9目标为单任务或否定控制，单独报告。此轮只设计冻结，没有运行模型或业务。','','原300题的五桶分别覆盖政策、型号、口语、库外、多问；本集评的是任务编排，因此按五种业务能力分桶，而不是照搬知识题材或把口语独立成桶。','','## 五桶定义与分布','','| 桶 | 检验能力 | 合计 | 多任务主集 | 单任务控制 |','|---|---|---:|---:|---:|']
    for bucket in BUCKETS:
        v=inventory['buckets'][bucket['id']];lines.append(f'| {bucket["id"]} {bucket["name"]} | {bucket["question"]} | {v["n"]} | {v["main"]} | {v["control"]} |')
    lines+=['','每题一个主桶，归桶顺序为E→D→C→B→A；这是分类消歧顺序，不是客服执行优先级。口语、任务数量、同类多任务、多订单等保留为副标签。桶D含6题卡片取消、3题重连、5题物流工具不可用；桶E含9题单任务控制及6题多任务边界。各桶计数基于规则自然形成，不按历史得分筛选，也不靠复述补齐相同题数。','','## 排除范围','','| 主排除原因（互斥） | 题数 |','|---|---:|']
    for rule in RULES:lines.append(f'| {rule["id"]} {rule["title"]} | {inventory["primary_exclusion_counts"].get(rule["id"],0)} |')
    lines+=['','excluded.json保留每题所有匹配原因；主原因取规则顺序第一项，因此互斥计数合计37。不是按哪版答对/答错删题。','','## 原分组溯源（不是新版分桶）','','| 原分组 | 原题数 | 保留 | 主集 | 控制 |','|---|---:|---:|---:|---:|']
    for g,v in by_group.items():lines.append(f'| {g} | {v["original"]} | {v["retained"]} | {v["main"]} | {v["control"]} |')
    lines+=['','明确办理仍按真实回执验收；表单取消继续保留，选单取消排除。保留缺订单选单后续办、订单隔离、多个同类任务、否定控制和重连恢复。原重复提交故障题均缺明确投诉事由，被同一规则排除，因此本集不声称覆盖专门的重复点击测试。','','仅询问积分现金兑换、会员次数/结转、无理由退货回寄运费承担，不因材料包含别的冲突句就一起删除。问题确实可答而模型没检索到、没执行或重复追问，不会因旧得分低而排除。','','## 比较规则','','两版使用同一74题主集及9题控制集；若以后各跑3轮，则每版222条主集轨迹、27条控制轨迹。当前没有启动这些运行。多任务主通过率分母不混入单任务控制。','','场景通过须全部必需目标完成；任务完成须有可信事实及对应回执；退款提交只是申请审核，不代表资金退回。未经确认写入、错误订单、重复工单仍严格计错。接口、驱动和语义/资料待定另列，不把未知算通过。','','此为事后限定范围的内部专项子集，不可与原120题整体分数直接对比，也不能代表整体客服效果。旧原始轨迹可按ID定位，但旧总体分数与自动标签不直接复用为新结果。具体规则见scoring.json和protocol.json。','','文件：cases.json为完整保留题；CASES.md便于阅读；excluded.json和EXCLUDED.md为排题记录；inventory.json为覆盖清单。','']
    (dest/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    lines=['# 保留题目（83题，按新五桶排列）','','主集74题；单任务控制9题。每题保持原ID和原始完整数据；新版主桶来自bucket-map.json，原case.group仅用于来源追溯。','']
    kept_in_bucket_order=[c for bucket in BUCKETS for c in kept if bucket_map[c['id']]['bucket']==bucket['id']]
    for c in kept_in_bucket_order:
        lines += [f'## {c["id"]} · {bucket_map[c["id"]]["bucket"]} · '+('主集' if c['id'] in contract['main_case_ids'] else '控制'),'',c['query'],'','| 目标 | 类型 | 对象 | 预期 |','|---|---|---|---|']
        for g in c['goals']:lines.append(f'| {g["id"]} | {g["kind"]} | {g.get("order_id") or "未指定/不需要"} | {g["expected_outcome"]} |')
        lines += ['',f'副标签：{", ".join(bucket_map[c["id"]]["tags"])}。',f'交互：{json.dumps(c["actor"],ensure_ascii=False)}；故障：{c["fault"] or "无"}。','']
    (dest/'CASES.md').write_text('\n'.join(lines),encoding='utf-8')
    lines=['# 排除题目（37题）','','仅按排除规则，不按历史通过率；原完整题保存在excluded.json，原120题未变。','','| 原ID | 主原因 | 全部原因 | 原问题 |','|---|---|---|---|']
    for e in excluded:lines.append(f'| {e["id"]} | {e["primary_rule"]} {titles[e["primary_rule"]]} | {", ".join(e["all_rules"])} | {e["query"].replace("|","／")} |')
    (dest/'EXCLUDED.md').write_text('\n'.join(lines),encoding='utf-8')
    assert all(sha(dest/n)==h for n,h in metadata['input_hashes'].items())
    assert sha(source/'cases.json')==metadata['source_cases_sha256'] and sha(source300)==metadata['source300_sha256']
    assert kept==[c for c in originals if not exclusion_reasons(c)]
    assigned=[c['id'] for b in BUCKETS for c in read(dest/'buckets'/(b['id']+'.json'))]
    assert len(assigned)==len(set(assigned))==len(kept) and set(assigned)=={c['id'] for c in kept}
    write(dest/'verification.json',{'input_partition_complete':True,'retained_inputs_unchanged':True,'source120_unchanged':True,'source300_unchanged':True,'knowledge_unchanged':sha(source/'knowledge.json')==sha(dest/'knowledge.json'),'bucket_partition_unique_and_complete':True,'no_outcome_files_read':True,'no_model_or_business_run':True})
    print(json.dumps(inventory,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=SOURCE);parser.add_argument('--out',type=Path,default=DEST);args=parser.parse_args()
    build(args.source,args.out)
