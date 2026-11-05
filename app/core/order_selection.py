"""Validate an explicit choice request without granting business authorization."""
import re

def requests(raw):
    spans=[]
    for part in re.split(r'[，,。；;！？!?]',raw):
        if re.search(r'不要|不用|不需要|无需|如果|假设|他说|她说|[“”"]',part):continue
        if re.search(r'让我.{0,8}(?:选|挑)|重新.{0,6}(?:选|挑)|(?:先|再).{0,4}选择.{0,6}订单',part):
            spans.append(part.strip())
    return spans

def groups(raw, proposals):
    spans=requests(raw);refs={};unmatched=[]
    order_tasks=[t for t in proposals if t.get('intent') in {'订单','物流','退款退货','售后'}]
    for span in spans:
        explicit={t['order_ref'] for t in order_tasks if t.get('selection_source') and t['selection_source'] in span}
        if not explicit:
            related=[t for t in order_tasks if t.get('source') and (span in t['source'] or t['source'] in span)]
            if not related:
                kinds={'退款退货'} if re.search('退款|退货',span) else {'物流'} if '物流' in span else set()
                related=[t for t in order_tasks if t.get('intent') in kinds] if kinds else order_tasks
            explicit={t['order_ref'] for t in related}
        if len(explicit)==1:refs[next(iter(explicit))]=span
        else:unmatched.append(span)
    return refs,unmatched
