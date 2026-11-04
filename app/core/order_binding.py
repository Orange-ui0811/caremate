"""Resolve order slots from task-local user text, never from a generated query."""
import re
from dataclasses import dataclass

ORDER_KINDS={'refund','order','logistics'}
ID=re.compile(r'(?<![A-Za-z\d_-])\d{4,20}(?![A-Za-z\d_-])')
BREAK=re.compile(r'[，,。；;！？!?]|(?=另外|另有|另一笔|另一个|再对|再给|再查|还要|还想|顺便|同时|然后|不过|而是|但|并(?:查|给|对))')
COREF=re.compile(r'这(?:一)?(?:笔订单|个订单|单)|该(?:笔)?订单|同一(?:笔)?订单|此(?:笔)?订单')
PARALLEL=re.compile(r'(?:订单\s*)?\d{4,20}(?:\s*(?:和|与|及|、)\s*(?:订单\s*)?\d{4,20})+\s*(?:都|均|分别|各自|一起)')
ANCHORS={
    'refund':r'退款|退货|退钱|退回|退给|退掉|拿回|款拿|钱.{0,6}退|款.{0,6}退',
    'logistics':r'物流|快递|配送|轨迹|运单|包裹',
    'order':r'状态|金额|多少钱|实付|付了多少|查(?:询|看)?(?:一下|下)?订单',
    'complaint':r'投诉|登记.*工单'}

PUNCT=str.maketrans({'，':',','；':';','：':':','？':'?','！':'!',
    '（':'(','）':')','“':'"','”':'"','「':'"','」':'"'})


def canonical(text):
    """Map presentation only, retaining digit-separating spaces and source offsets."""
    chars=[];positions=[]
    for i,ch in enumerate(text):
        if ch.isspace():
            before=text[i-1] if i else ''
            j=i+1
            while j<len(text) and text[j].isspace():j+=1
            if before.isdigit() and j<len(text) and text[j].isdigit():
                chars.append(' ');positions.append(i)
            continue
        chars.append(ch.translate(PUNCT));positions.append(i)
    return ''.join(chars),positions


def recover_source(raw,task):
    """Recover an actual raw span, never accept semantic paraphrases or changed IDs."""
    source=task['source']
    if source in raw:return source,None
    body,offsets=canonical(raw);needle,_=canonical(source)
    start=body.find(needle) if needle else -1
    if start>=0 and body.find(needle,start+1)<0:
        end=offsets[start+len(needle)-1]+1;begin=offsets[start]
        return raw[begin:end],{'method':'presentation_only_alignment','start':begin,'end':end,'model_source':source}
    # Models often project a coordinated request into individual source strings.
    # Accept only a grammar-checked projection of one explicit parallel clause.
    hits=[];kind=task['kind'];oid=task.get('order_id')
    if kind in ORDER_KINDS and oid:
      for c in clauses(raw):
        if not has_action(kind,c.text) or inactive(kind,c.text):continue
        group=PARALLEL.search(c.text)
        if not group or oid not in numbers(group.group()) or set(numbers(group.group()))!=set(numbers(c.text)):continue
        group_text=group.group();quantifier=re.search(r'(都|均|分别|各自|一起)$',group_text).group()
        prefix='订单' if group_text.startswith('订单') else ''
        for suffix in (quantifier,''):
            projected=c.text[:group.start()]+prefix+oid+suffix+c.text[group.end():]
            projected_key,_=canonical(projected)
            if needle and needle in projected_key:
                hits.append(c);break
    if len(hits)==1:
        c=hits[0]
        return c.text,{'method':'explicit_parallel_projection','start':c.start,'end':c.end,'model_source':source}
    raise ValueError('任务证据不是当前用户的可核验原文')


@dataclass(frozen=True)
class Clause:
    start:int
    end:int
    text:str


def clauses(raw):
    result=[];start=0
    for m in BREAK.finditer(raw):
        if m.start()==m.end() and re.search(r'(?:不要|不用|不必|别|不需要|不想|无需|取消|不再)\s*$',raw[max(0,m.start()-8):m.start()]):
            continue
        if m.start()>start:result.append(Clause(start,m.start(),raw[start:m.start()]))
        start=m.end()
    if start<len(raw):result.append(Clause(start,len(raw),raw[start:]))
    return result


def numbers(text):
    """Exclude amounts, points, phone/tracking/SKU labels and date fragments."""
    result=[]
    for m in ID.finditer(text):
        before=text[max(0,m.start()-10):m.start()];after=text[m.end():m.end()+8]
        if re.search(r'(?:积分|金额|价格|手机|电话|运单|快递单|货号|型号|验证码|账号|￥|¥)\s*(?:是|为)?\s*[:：]?\s*$',before):continue
        if re.match(r'\s*(?:积分|元|块|年|月|日|天|小时|件|kg|KG|毫升|mAh)',after):continue
        if m.group() not in result:result.append(m.group())
    return result


def has_action(kind,text):
    if not re.search(ANCHORS.get(kind,r'(?!)'),text):return False
    if kind=='order' and re.search(ANCHORS['logistics'],text):
        # A logistics status is not an independent order-status request.
        return bool(re.search(r'订单\s*\d{4,20}.{0,8}(?:金额|实付|多少钱)',text))
    return True


def inactive(kind,text):
    unquoted=re.sub(r'"[^"]*"|“[^”]*”|「[^」]*」','',text)
    if has_action(kind,text) and not has_action(kind,unquoted):return True
    if re.search(r'["“”「」]|如果|假设|比如|例如|他说|她说',unquoted):return True
    anchor=ANCHORS.get(kind,r'(?!)')
    return bool(re.search(r'(?:不要|不用|不必|别|不需要|不想|无需|取消|不再|不是要|并非要|不(?=退款|退货|退钱|投诉|查询)).{0,10}(?:'+anchor+r')',ID.sub('#',unquoted)))


def source_clauses(raw,source):
    spans=[];pos=0
    while True:
        start=raw.find(source,pos)
        if start<0:break
        spans.append((start,start+len(source)));pos=start+1
    cs=clauses(raw)
    return cs,[i for i,c in enumerate(cs) if any(c.start<end and c.end>start for start,end in spans)]


def prohibited(raw,task):
    cs,indices=source_clauses(raw,task['source'])
    relevant=[i for i in indices if has_action(task['kind'],cs[i].text)]
    def blocked(i):
        return inactive(task['kind'],cs[i].text) or (i>0 and
            not re.search(r'[。；;！？!?]',raw[cs[i-1].end:cs[i].start]) and
            bool(re.search(r'如果|假设|比如|例如',cs[i-1].text)))
    return bool(relevant) and all(blocked(i) for i in relevant)


def resolve(raw,task):
    """Return only positively supported local bindings, including explicit same-turn links."""
    kind=task['kind'];cs,indices=source_clauses(raw,task['source']);candidates=[];parallel=[]
    for index in indices:
        c=cs[index]
        # The model may quote the order subject rather than its immediately
        # following action. Verify the link in raw text instead of discarding it.
        if not has_action(kind,c.text) and not inactive(kind,c.text) and index+1<len(cs):
            following=cs[index+1];ids=numbers(c.text);following_ids=numbers(following.text)
            same_sentence=not re.search(r'[。；;！？!?]',raw[c.end:following.start])
            new_object=bool(re.match(r'\s*(?:另外|另有|另一笔|另一个|再对|再给|再查)',following.text))
            subject=bool(re.search(r'订单|这单|那单',c.text))
            if (len(ids)==1 and same_sentence and not new_object and has_action(kind,following.text)
                and not inactive(kind,following.text) and (not following_ids or following_ids==ids)
                and (COREF.search(following.text) or (subject and not any(has_action(k,c.text) for k in ANCHORS)))):
                candidates.append((ids[0],'adjacent_order_subject_action',[c,following]))
        if not has_action(kind,c.text) or inactive(kind,c.text):continue
        ids=numbers(c.text)
        if len(ids)==1:
            candidates.append((ids[0],'task_clause',[c]));continue
        if len(ids)>1:
            group=PARALLEL.search(c.text)
            if group and set(numbers(group.group()))==set(ids):
                parallel.append((ids,c))
            continue
        if not index:continue
        previous=cs[index-1];antecedents=numbers(previous.text)
        between=raw[previous.end:c.start]
        if len(antecedents)!=1 or re.search(r'[。；;！？!?]',between) or inactive(kind,previous.text):continue
        if COREF.search(c.text):
            # Explicit 'this order' may share an order across distinct task kinds.
            candidates.append((antecedents[0],'explicit_same_message_reference',[previous,c]))
        elif (not re.match(r'\s*(?:另外|另有|另一笔|另一个|再对|再给|再查|还要|还想|顺便)',c.text)
              and re.search(r'订单|这单|那单',previous.text)
              and not any(has_action(k,previous.text) for k in ANCHORS)):
            candidates.append((antecedents[0],'adjacent_order_subject',[previous,c]))
    if parallel and not candidates and len(parallel)==1:
        ids,c=parallel[0]
        return {'status':'resolved','method':'explicit_parallel_orders','order_ids':ids,
                'evidence':[{'start':c.start,'end':c.end,'text':c.text}]}
    unique={x[0] for x in candidates}
    # Repeated generic source phrases need an independently valid reference to disambiguate.
    if len(unique)>1:
        reference=task.get('order_reference','')
        hinted=numbers(reference) if reference and reference in raw else []
        if len(hinted)==1 and hinted[0] in unique:
            candidates=[x for x in candidates if x[0]==hinted[0]];unique={hinted[0]}
    if len(unique)==1:
        oid,method,proof=candidates[0]
        return {'status':'resolved','method':method,'order_ids':[oid],
                'evidence':[{'start':c.start,'end':c.end,'text':c.text} for c in proof]}
    return {'status':'ambiguous' if len(unique)>1 or parallel else 'missing',
            'method':'needs_user_selection','order_ids':[],
            'reason':'multiple_supported_orders' if len(unique)>1 or parallel else 'no_local_order_binding',
            'evidence':[]}
