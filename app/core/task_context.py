"""Source-backed input and object validation for the v2 single-call planner."""
import json
import re

from app.config import settings
from app.core import order_binding as binding, order_context
from app.core.context_policy import estimate
from app.tools import business

VERSION = 2


def build(state):
    from app.graph import nodes
    raw = state['task_origin']
    history = nodes._history_text(state, stage='coref')
    budget = settings.context_coref_budget
    # History uses the baseline selector. Preserve its head and recalled tail if
    # its combined window/supplement exceeds this planner's history allowance.
    original_size = estimate(history)
    while estimate(history) > budget:
        cut = max(1, len(history)//10)
        mid = len(history)//2
        history = history[:mid-cut] + '\n[历史预算裁剪]\n' + history[mid+cut:]
    candidates = {}
    uid = state['user_id']
    ctx = state.get('business_context') or {}
    if ctx.get('user_id') == uid and ctx.get('conversation_id') == state['conversation_id']:
        for oid in ctx.get('order_ids', []):
            if business.owns_order(uid, oid):
                candidates[oid] = {'order_id':oid, 'source':ctx.get('order_sources', {}).get(oid,
                    {'kind':'verified_workflow_context'}), 'focus':True}
    # A customer's positive, anchored order mention is evidence of a referent,
    # never proof of a completed operation. Assistant prose cannot create one.
    for message in state.get('messages', []):
        mid = str(getattr(message, 'id', '') or '')
        if message.type != 'human' or not mid.startswith('db-') or not mid[3:].isdigit():
            continue
        if int(mid[3:]) >= state['current_message_id']:
            continue
        text = message.content if isinstance(message.content, str) else ''
        for clause in binding.clauses(text):
            if not re.search('订单|这单|那单', clause.text) or re.search('不要|不用|别查|如果|假设|他说|“|"', clause.text):
                continue
            for oid in binding.numbers(clause.text):
                if business.owns_order(uid, oid) and oid not in candidates:
                    candidates[oid] = {'order_id':oid, 'source':{'kind':'user_message',
                        'message_id':int(mid[3:]),'text':clause.text},'focus':False}
    for oid in binding.numbers(raw):
        if business.owns_order(uid, oid):
            candidates[oid] = {'order_id':oid,'source':{'kind':'current_message',
                'message_id':state['current_message_id']},'focus':False}
    values = [{**v,'ref':'order:'+oid} for oid,v in candidates.items()]
    result = {'version':VERSION,'user_id':uid,'conversation_id':state['conversation_id'],
              'message_id':state['current_message_id'],'history':history,'candidates':values,
              'selection':{'policy':settings.context_policy,'legacy_recall':settings.context_legacy_recall_enabled,
                           'history_estimate_before':original_size,'history_estimate':estimate(history),'budget':budget}}
    if estimate(raw + json.dumps(result,ensure_ascii=False)) > settings.context_input_budget - 3000:
        raise ValueError('规划输入超出预算，不能丢弃当前任务或对象来源')
    return result


def normalize(raw, items, context):
    """Accept semantic links only with raw spans and owned, server-built candidates."""
    uid = context['user_id']
    candidates = {x['ref']:x for x in context['candidates'] if business.owns_order(uid,x['order_id'])}
    out=[]; seen=set()
    for item in items:
        t=dict(item)
        t['source'],alignment=binding.recover_source(raw,t)
        if alignment:t['source_alignment']=alignment
        if binding.prohibited(raw,t):continue
        span=t.get('reference_span') or ''
        if span and span not in raw and t['kind'] in {'order','logistics'}:
            literal=t.get('order_reference','')
            cs,indices=binding.source_clauses(raw,t['source'])
            if (literal and literal in raw and len(binding.numbers(literal))==1 and any(
                    literal in cs[i].text and binding.numbers(cs[i].text)==binding.numbers(literal)
                    and not binding.inactive(t['kind'],cs[i].text) for i in indices)):
                # A second, verbatim field independently proves the same local
                # order. Retain the rejected proposal and use that actual span.
                t['reference_alignment']={'method':'task_local_literal_reference','model_span':span}
                span=t['reference_span']=literal
        if span and span not in raw:
            # Models sometimes use a literal ellipsis between copied endpoints.
            # Recover only one continuous, single-clause, single-order span;
            # never accept arbitrary paraphrases or a link across two orders.
            parts=re.split(r'\.{3}|…+',span)
            matches=[]
            if len(parts)>1 and all(parts):
                pattern='.*?'.join(re.escape(p) for p in parts)
                for clause in binding.clauses(raw):
                    for match in re.finditer(pattern,clause.text):
                        text=match.group()
                        if (len(binding.numbers(text))==1 and not order_context.NEW_OBJECT.search(text)
                                and not binding.inactive(t['kind'],clause.text)):
                            matches.append(text)
            if len(matches)!=1:raise ValueError('对象引用不是本轮原文')
            t['reference_alignment']={'method':'literal_ellipsis_alignment','model_span':span}
            span=t['reference_span']=matches[0]
        proposal={k:t.get(k) for k in ('order_id','subject_ref','candidate_ref','binding_status','reference_span')}
        local=binding.resolve(raw,t) if t['kind'] in binding.ORDER_KINDS else None
        ids=local['order_ids'] if local and local['status']=='resolved' else []
        # An explicit local number on a colloquial read does not need a fixed
        # action keyword, but a write still needs its own positive local binding.
        if not ids and t['kind'] in {'order','logistics'}:
            numbers=binding.numbers(t['source'])
            if len(numbers)==1 and not re.search('不是|不要|不查',t['source']):ids=numbers
        candidate=candidates.get(t.get('candidate_ref'))
        error=None
        if candidate and not span:error='missing_reference_span'
        if t.get('candidate_ref') and not candidate:error='unsupported_candidate'
        if candidate and order_context.NEW_OBJECT.search(span) and not binding.numbers(span):error='new_object'
        if t.get('binding_status')=='ambiguous':error='ambiguous'
        if (len(candidates)>1 and not ids and not binding.numbers(raw)
                and re.search(r'(?:两|几|多|二|三|四)(?:笔|个)|这些订单|那几单',raw)
                and not order_context.PRODUCT.search(raw)):
            # An explicitly unnamed member of a plural set is not resolved by
            # recency alone, even if the model proposes the latest candidate.
            error='ambiguous'
        if t['kind']=='refund' and not ids and not (order_context.REFERENCE.search(span) or order_context.PRONOUN.search(span)):
            error='write_has_no_current_reference'
        if ids and candidate and candidate['order_id'] not in ids:error='current_order_conflict'
        if candidate and not ids and not error:ids=[candidate['order_id']]
        # The model's free-form order_id cannot bypass candidate/local validation.
        if not ids and t.get('order_id'):error=error or 'unproven_model_id'
        group=t.get('subject_ref') or ''
        if not span or (error and error!='ambiguous'):group=''
        if ids and any(not business.owns_order(uid,oid) for oid in ids):
            ids=[];error='unowned_order';group=''
        for oid in ids or [None]:
            final={**t,'order_id':oid if t['kind'] in binding.ORDER_KINDS else None,
                   'subject_ref':group,'reference_group':group,'plan_version':VERSION,
                   'binding_proposal':proposal,'binding_validation':error or 'accepted'}
            if t['kind']=='knowledge':
                final.pop('context_order_id',None)
                if t.get('knowledge_scope')=='product' and not error and oid:
                    final['context_order_id']=oid
                elif t.get('knowledge_scope')!='product':
                    final['subject_ref']=final['reference_group']=''
            if t['kind'] in binding.ORDER_KINDS:
                final['binding']={'status':'resolved' if oid else ('ambiguous' if error=='ambiguous' else 'missing'),
                    'method':'context_v2' if oid else 'needs_user_selection','order_ids':[oid] if oid else [],
                    'source':candidate['source'] if candidate and not error else {'kind':'current_message'},
                    'reason':error or ''}
            sig=(t['kind'],oid,group,t['source'])
            if sig not in seen:out.append(final);seen.add(sig)
    # A proposed group containing conflicting explicit objects is never shared.
    for group in {t['subject_ref'] for t in out if t['subject_ref']}:
        members=[t for t in out if t['subject_ref']==group]
        ids={t.get('order_id') or t.get('context_order_id') for t in members}-{None}
        if len(ids)>1:
            for t in members:t['subject_ref']=t['reference_group']='';t['binding_validation']='group_conflict'
    if len(out)>8:raise ValueError('独立任务超过上限')
    return out


def packet(state, task):
    return {'source':task['source'],'query':task['query'],'order_id':task.get('order_id') or task.get('context_order_id'),
            'binding':task.get('binding',{}),'subject_ref':task.get('subject_ref',''),
            'user_report':(state.get('business_context') or {}).get('user_reports',{}).get(
                task.get('order_id') or task.get('context_order_id'),{}),
            'message_id':state['current_message_id']}
