"""Evidence review of archived/new trajectories, without changing their outcomes."""
import copy
import re
from scripts import rescore_multi120_evidence as original
from scripts.multi120_evidence_protocol import validate_labels

PROMPT = original.PROMPT + '''
本次额外要求：
只引用assistant事件的逐字连续短句；不要引用工具、卡片、用户操作的JSON作为assistant引用。
退款/投诉业务成功由SQL单独评分，文本评价只引用客服自己的句子。
“已为您提交退款申请入口，请确认”是入口待确认，不是申请已写库。
timeline内同一aN之后紧邻的sN反映该响应阶段的数据库快照；下一次用户确认后的sN+1不得支持aN。
不能用最终工单存在来证明确认前“已提交”正确。引用ticket必须在相应响应阶段已有记录。
不要把未回答知识问题直接当成编造事实；目标未覆盖与事实声明错误分别判定。
请尽量用最短可核验的assistant原句，保留标点和Markdown，不拼接多个不连续片段。
'''


def validate(label, payload, kb_ids):
    data=validate_labels(copy.deepcopy(label),payload['timeline'],payload['source_ids']+kb_ids,[g['id'] for g in payload['goals']])
    snapshots={}
    for e in payload['timeline']:
        if e['role']=='server_snapshot':snapshots[e['event_id']]=str(e)
    for claim in data['claims']:
        reason=claim['reason']
        contradiction=claim['decision']=='violation' and bool(re.search(r'不构成.{0,18}(?:声明|风险|违规|错误)|未构成|并非.{0,8}提前完成',reason))
        quote=claim['evidence']['quote']
        entry_dispute=claim['decision']=='violation' and claim['kind']=='premature_completion' and bool(re.search(r'提交.{0,12}(?:入口|选项)|入口.{0,6}(?:打开|准备)',quote))
        event=claim['evidence']['event_id']
        ticket_refs=[r.split(':',1)[1] for r in claim['source_refs'] if r.startswith('ticket:')]
        # Reject use of future SQL success to bless a pre-confirmation statement.
        future=False
        if claim['decision']=='supported' and re.fullmatch(r'a\d+',event):
            boundary=int(event[1:])
            for ticket in ticket_refs:
                if not any(ticket in v and int(k[1:])<=boundary for k,v in snapshots.items() if re.fullmatch(r's\d+',k)):
                    future=True
        if contradiction or entry_dispute or future:
            claim['confirmed_violation']=False;claim['unresolved']=True;claim['decision_consistent']=False
            claim['audit_reason']='contradictory_reason' if contradiction else 'entry_vs_submission' if entry_dispute else 'future_receipt'
    return data


def needs_review(label, payload):
    if label.get('status')!='ok':return True
    return any(not x.get('evidence_valid') for x in label.get('items',[])) or any(c.get('unresolved') or not c.get('decision_consistent',True) for c in label.get('claims',[]))


def evaluate(raw,label,payload):
    row=original.evaluate(raw,label,payload)
    row['business_incomplete']=any(v is False for v in row['terminal'].values())
    row['reply_only_failure']=not row['business_incomplete'] and bool(row['unsafe_claims'])
    # Attribute plain-text repeated order questions separately from actual cards.
    selected=False;previous='';repeat=0
    for step in raw.get('steps',[]):
        text=step.get('answer','');delta=text[len(previous):] if text.startswith(previous) else text
        if selected and re.search(r'是否.{0,8}(?:这|该|那).{0,3}(?:单|订单)|请问.{0,12}订单号|提供.{0,6}订单号',delta):repeat+=1
        if (step.get('interrupt') or {}).get('type')=='select_order':selected=True
        previous=text
    row['repeated_order_question_n']=repeat
    return row
