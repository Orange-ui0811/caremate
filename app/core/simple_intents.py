"""Multi-label classification after the unchanged baseline reference resolver."""
import json
import hashlib
import re
from typing import Literal

from pydantic import BaseModel, Field
from app.core import llm, task_context
from app.core.model_pool import model_call
from app.config import settings
from app.tools import business


class Item(BaseModel):
    kind: Literal['order','logistics','return_advice','knowledge','refund','complaint','other']
    query: str = Field(min_length=1, max_length=500)
    source: str = Field(min_length=1, max_length=1000)
    order_ref: str = Field(min_length=1, max_length=80,
        description='必须给对象引用名，同一对象共用；对象未知也用同一个临时引用；无对象的通用问题用none')
    order_id: str | None = None
    product_specific: bool = False


class Intents(BaseModel):
    tasks: list[Item] = Field(max_length=8)


PROMPT = '''识别本轮请求中的所有实际诉求，按原文顺序列出任务。原话决定任务和授权，补全问题和历史只帮助理解对象。
kind: order订单商品/状态/金额；logistics物流；return_advice能否退货/退款资格咨询；knowledge商品说明或通用规则；
refund明确要求办理退货退款；complaint登记投诉/人工工单；other闲聊或回顾。
“它是什么商品/里面买的什么/订单商品名称”从订单工具查询，必须是order，不是knowledge。
“刚刚那个订单送到哪了，它是什么商品，我可以退货吗”完整输出logistics、order、return_advice三项。
“我可以退货吗/能退款吗/怎么申请”是咨询，不是refund。不要根据历史申请重新办理。
query保留这项诉求的问法；source必须逐字引用本轮原话，可引用整句以保留语义。
order_id只能来自可信候选；不明确为null。order_ref必须非空，同一对象共用同一引用，独立对象不同引用。
订单不明确时order_id=null，但order_ref仍填写如pending_order；通用咨询的order_ref用none。
“两笔订单中，那笔送到哪了，它是什么商品”两项都用pending_order，都保留order_id=null，不能拆成两个待选订单。
先理解整句话：如“刚刚那个订单送到哪了，它是什么商品，我可以退货吗”，三个任务共享同一订单引用，
不要求每一子句都重复说订单号。“刚刚这单我要退款并投诉”也共用对象，办理的确认由后续单独处理。
“查1001物流，另外一笔我要退款”两项对象不同，退款null并用独立引用，不能偷用1001。
“两笔订单中那笔”无法确定时保留null，不因最近提到某笔就猜。明确更正及当前指定优先于旧对象。
同一未确定订单的任务共用order_ref，选一次即可。独立退款不得借物流的号码。
product_specific只用于某笔订单商品的知识/资格咨询；通用政策和明确另一个商品咨询不关联订单。
取消、否定、引用、假设中的操作不规划。没有当前办理请求，就不能因为补全问题或历史有申请而新增办理。
最多8项；同一对象的订单状态/金额合为一项。不要输出安全任务，已有前置安全提醒。
'''


def normalize(raw, items, context):
    from app.core.order_binding import numbers, prohibited
    candidates={x['order_id'] for x in context['candidates'] if business.owns_order(context['user_id'],x['order_id'])}
    output=[]
    for index, item in enumerate(items):
        t=Item.model_validate(item).model_dump()
        if t['order_ref']=='none':t['order_ref']=''
        if t['source'] not in raw:raise ValueError('任务来源不在本轮原文')
        if prohibited(raw,t):continue
        # A question cannot acquire write authority merely through classification.
        if t['kind']=='refund' and (re.search(r'可以|能否|能不能|可不可以|怎么|如何|是否|能.{0,4}[吗么]',t['source'])
                and not re.search(r'请.{0,5}(?:办理|申请|提交)|我要(?:申请|办理)|帮我(?:申请|办理|提交)',t['source'])):
            t['kind']='return_advice';t['product_specific']=True
        if t['kind']=='refund' and not re.search(r'要|想|申请|办理|退掉|退钱|退回|退了|提交',t['source']):
            t['kind']='return_advice';t['product_specific']=True
        oid=t['order_id']
        if oid and oid not in candidates:oid=None
        local=numbers(t['source'])
        if len(local)==1 and local[0] in candidates and not re.search(r'另一|另外|不是|不要',t['source']):
            oid=local[0]
        # Explicitly independent/unspecified objects are never attached by proximity.
        independent=bool(re.search(r'(?:另外|另一|别的|其他).{0,8}(?:笔|订单)|订单我自己选择',t['source']))
        if independent and t['kind']=='refund':oid=None;t['order_ref']=f'independent:{index}'
        if (len(candidates)>1 and not numbers(raw) and re.search(r'两笔|几笔|多笔|这些订单',raw)
                and not re.search(r'都|分别|各自|每笔',raw)):
            # Reject a guessed member without turning the same singular
            # referent into one selection per candidate suggested by the model.
            oid=None;t['order_ref']='ambiguous:'+hashlib.sha256(t['source'].encode()).hexdigest()[:12]
        if t['kind']=='return_advice':t['product_specific']=bool(t['order_ref'] or oid)
        if t['kind']=='knowledge' and not t['product_specific']:
            oid=None;t['order_ref']=''
        needs=t['kind'] in {'order','logistics','refund'} or t['product_specific']
        if needs and not t['order_ref']:t['order_ref']=f'order:{oid}' if oid else f'unresolved:{index}'
        t.update(order_id=oid,status='pending',result={},step='',binding={'method':'simple_context','status':'resolved' if oid else 'missing'})
        if not any((x['kind'],x['order_ref'],x['order_id'],x['query'])==
                   (t['kind'],t['order_ref'],t['order_id'],t['query']) for x in output):output.append(t)
    # Conflicting known objects must not share future selections.
    for ref in {t['order_ref'] for t in output if t['order_ref']}:
        members=[t for t in output if t['order_ref']==ref]
        ids={t['order_id'] for t in members if t['order_id']}
        if len(ids)>1:
            for i,t in enumerate(members):t['order_ref']=f'{ref}:{t["order_id"] or i}'
    return output


@model_call('intent_classification')
async def classify(state):
    context=task_context.build(state)
    response=await llm.structured(Intents,model=settings.intent_model or None,temperature=0).ainvoke([
        {'role':'system','content':PROMPT},
        {'role':'user','content':json.dumps({'original':state['task_origin'],
            'resolved':state.get('resolved_query'),'context':context},ensure_ascii=False)}])
    proposals=[t.model_dump() for t in response.tasks]
    return normalize(state['task_origin'],proposals,context),context,proposals
