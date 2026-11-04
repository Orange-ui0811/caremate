"""Compose customer replies from verified results, without another model/tool call."""
import hashlib
import json
import re


def signature(task, result):
    value={k:task.get(k) for k in ('status','step','outcome','order_id','receipt','retry_submission')}
    value['result']=result
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()


def fact(result, name):
    for receipt in reversed(result.get('receipts',[])):
        if receipt.get('name')!=name:continue
        try:value=json.loads(receipt['content'])
        except (ValueError,TypeError,KeyError):continue
        if isinstance(value,dict) and not value.get('error'):return value
    return {}


def safe(value):
    # Tool text is data, not Markdown or internal diagnostic material.
    return re.sub(r'([\\`*_{}\[\]<>|])',r'\\\1',str(value)).replace('\n',' ')


def knowledge_text(text):
    if re.search(r'本次事项处理结果|\b(?:PENDING|WAITING_USER|WAITING_DEPENDENCY|RUNNING|DONE|task_id|operation_id)\b',text):
        return '这条知识问题的答复暂时无法可靠呈现，需要人工进一步核实。'
    lines=text.rstrip().splitlines()
    while lines and (not lines[-1].strip() or re.match(
        r'^(?:需要我|还有其他(?:问题|需要)|有其他(?:问题|需要)|如需(?:其他|更多)帮助|如有其他|请问您还|随时告诉我)',lines[-1].strip())):
        lines.pop()
    return '\n'.join(lines)


def compose(state):
    tasks=state['task_plan']; results=state.get('subflow_results',{})
    seen=state.get('subflow_presented',{}); now=dict(seen)
    changed=[]
    for key,t in tasks.items():
        if t['kind']=='safety':continue
        digest=signature(t,results.get(key,{}));now[key]=digest
        if digest!=seen.get(key):changed.append((key,t))
    if not changed:return '',[],now
    parts=[]; groups={}; citations=[]
    for key,t in changed:
        kind=t['kind']; status=t['status']; oid=t.get('order_id')
        result=results.get(key,{})
        if kind in {'order','logistics'}:
            if status=='DONE':groups.setdefault(oid,[]).append((kind,result))
            elif status=='FAILED':parts.append(f'订单{safe(oid or "")}的查询暂时无法完成，请稍后重试。')
            elif status=='CANCELLED':parts.append('已取消这次订单选择。')
        elif kind in {'refund','complaint'}:
            label='退款审核' if kind=='refund' else '投诉'
            if t.get('receipt',{}).get('ticket_no'):
                parts.append(f"{label}工单已提交，工单号：{safe(t['receipt']['ticket_no'])}。")
            elif status=='CANCELLED':parts.append(f'已取消这次{label}申请，未创建工单。')
            elif status=='FAILED':parts.append(f'{label}暂时无法办理，尚未提交；其他可办理事项会继续处理。')
            elif t.get('outcome')=='unpaid_order':parts.append(f'订单{safe(oid)}目前待付款，无需申请退款。')
            elif t.get('step')=='confirm_submission':
                if t.get('retry_submission'):
                    parts.append(f'{label}的提交结果暂时无法确认。重试原操作会核对原回执，不会重复创建工单。')
                else:parts.append(f'{"订单"+safe(oid)+"的" if oid else ""}{label}申请已准备，尚未提交。')
        elif kind=='knowledge':
            if status=='DONE':
                text=knowledge_text(result.get('answer',''))
                local={}
                for c in t.get('citations',[]):
                    identity=(c.get('id'),c.get('answer'))
                    found=next((i+1 for i,x in enumerate(citations) if (x.get('id'),x.get('answer'))==identity),None)
                    if found is None:citations.append({**c,'n':len(citations)+1});found=len(citations)
                    local[str(c.get('n'))]=str(found)
                text=re.sub(r'\[(\d+)\]',lambda m:'['+local[m[1]]+']' if m[1] in local else '',text)
                if text:parts.append(text)
            elif status=='CANCELLED':parts.append('已取消本次商品资料查询。')
            elif status=='FAILED':
                parts.append('暂时未能取得该订单的商品资料，请稍后重试。' if t.get('outcome','').startswith('upstream_order:')
                             else '这条知识问题暂时缺少充分依据，需要人工进一步核实。')
        elif kind=='other':parts.append('部分诉求暂时无法自动办理，需要人工进一步处理。')
    for oid,items in groups.items():
        order={}; shipping={}
        for kind,result in items:
            record=fact(result,'query_order')
            if record.get('order_id')==oid:order=record
            if kind=='logistics':
                candidate=fact(result,'query_logistics')
                if order.get('tracking_no') and candidate.get('tracking_no')==order['tracking_no']:shipping=candidate
        if not order:
            parts.insert(0,f'订单{safe(oid)}的详情暂时无法核实。');continue
        lines=[f'**订单 {safe(oid)}'+(f"（{safe(order['product'])}）" if order.get('product') else '')+'**']
        if any(k=='order' for k,_ in items):
            for label,key in [('订单状态','status'),('金额','amount'),('下单时间','created_at')]:
                if order.get(key) is not None:lines.append(f'- {label}：{safe(order[key])}'+('元' if key=='amount' else ''))
        if any(k=='logistics' for k,_ in items):
            if order.get('status')=='待付款':lines.append('- 当前待付款，暂无可查询的物流进度。')
            elif shipping:
                for label,key in [('物流单号','tracking_no'),('物流状态','status'),('当前位置','current_city')]:
                    if shipping.get(key):lines.append(f'- {label}：{safe(shipping[key])}')
                trace=shipping.get('trace') or []
                public=[x for x in trace if isinstance(x,str) and not re.search(r'内部|状态码|PENDING|DONE|WAITING_USER',x)] if isinstance(trace,list) else []
                if public:lines.append('- 最新轨迹：'+safe(public[0]))
            else:lines.append('- 暂时未能核实物流进度。')
        parts.insert(0,'\n'.join(lines))
    waiting=[t for _,t in changed if t.get('step')=='select_order' and t['status']=='WAITING_USER']
    if waiting:
        labels={'order':'查询订单','logistics':'查询物流','refund':'申请退款审核','knowledge':'查询商品说明'}
        purposes=list(dict.fromkeys(labels.get(t['kind'],'办理售后') for t in waiting))
        parts.append('、'.join(purposes)+'前，需要先确认对应订单。')
    return '\n\n'.join(dict.fromkeys(parts)),citations,now
