import pytest
from app.core.task_planner import normalize


def task(kind, source, oid=None, reference=''):
    return {'kind':kind,'query':source,'source':source,'order_id':oid,'order_reference':reference}


def test_orders_stay_bound_to_their_own_tasks():
    raw='查9284物流，对5017申请退款，还要投诉商家'
    result=normalize(raw,[task('logistics','查9284物流','9284','查9284物流'),
                          task('refund','对5017申请退款','5017','对5017申请退款')])
    assert [t['order_id'] for t in result]==['9284','5017']


def test_refund_cannot_borrow_the_only_number_from_logistics():
    raw='我要退款，还要投诉，另查9284物流'
    result=normalize(raw,[task('refund','我要退款','9284','另查9284物流')])
    assert result[0]['order_id'] is None


def test_bad_global_reference_is_repaired_only_from_the_local_refund_clause():
    raw='查9284物流，对5017申请退款'
    assert normalize(raw,[task('refund','对5017申请退款','9284',raw)])[0]['order_id']=='5017'


def test_short_source_cannot_revive_negated_query():
    raw='我要退款，不要再查9284物流'
    assert normalize(raw,[task('logistics','查9284物流','9284','查9284物流')])==[]


def test_unsupported_source_is_rejected():
    with pytest.raises(ValueError):
        normalize('我要退款',[task('refund','退款订单5017','5017','退款订单5017')])


@pytest.mark.parametrize('kind,raw,source,ref,oid',[
 ('logistics','请查订单9284的物流位置','请查订单9284的物流位置','订单9284','9284'),
 ('refund','订单5017商品坏了，请申请退款','订单5017商品坏了，请申请退款','订单5017','5017'),
 ('order','5017那单现在啥状态、付了多少','5017那单现在啥状态、付了多少','5017那单','5017'),
 ('refund','查订单5017的状态，给这笔订单申请退款','给这笔订单申请退款','这笔订单','5017'),
 ('refund','查9284物流，给这单申请退款','给这单申请退款','这单','9284'),
 ('refund','订单5017物流异常申请退款','订单5017物流异常申请退款','订单5017','5017')])
def test_short_reference_or_explicit_pronoun_uses_local_source(kind,raw,source,ref,oid):
    result=normalize(raw,[task(kind,source,None,ref)])
    assert result[0]['order_id']==oid
    assert result[0]['binding']['status']=='resolved'


@pytest.mark.parametrize('raw,source',[
 ('查9284物流；另一笔商品坏了要退款','另一笔商品坏了要退款'),
 ('我要退款并查9284物流','我要退款'),
 ('我有12000积分，帮我申请退款','帮我申请退款'),
 ('电话是13800138000，帮我申请退款','帮我申请退款'),
 ('2026-11-03申请退款','2026-11-03申请退款')])
def test_other_task_points_phone_or_date_are_not_refund_bindings(raw,source):
    assert normalize(raw,[task('refund',source,'9284','9284')])[0]['order_id'] is None


def test_broad_source_does_not_borrow_logistics_id_for_missing_refund():
    raw='我要退款，另外查9284物流'
    assert normalize(raw,[task('refund',raw,'9284',raw)])[0]['order_id'] is None


def test_parallel_refunds_expand_even_if_model_outputs_only_one_item():
    raw='订单5017和8765都申请退款，9284只查物流'
    result=normalize(raw,[task('refund','订单5017和8765都申请退款','5017',raw)])
    assert [t['order_id'] for t in result]==['5017','8765']
    assert all(t['binding']['method']=='explicit_parallel_orders' for t in result)


def test_parallel_model_items_and_repeated_same_order_do_not_duplicate_writes():
    raw='订单5017和8765都申请退款'
    result=normalize(raw,[task('refund',raw,oid,raw) for oid in ('5017','8765')])
    assert [t['order_id'] for t in result]==['5017','8765']


def test_repeated_generic_source_is_disambiguated_by_valid_local_reference():
    raw='订单5017申请退款，订单8765申请退款'
    result=normalize(raw,[task('refund','申请退款',oid,'订单'+oid) for oid in ('5017','8765')])
    assert [t['order_id'] for t in result]==['5017','8765']


@pytest.mark.parametrize('raw,source,kind',[
 ('不要查询订单12345678901234567890的物流','查询订单12345678901234567890的物流','logistics'),
 ('如果订单5017坏了，申请退款','申请退款','refund'),
 ('订单5017不要退款','退款','refund')])
def test_short_quote_cannot_revive_negation_or_hypothetical(raw,source,kind):
    assert normalize(raw,[task(kind,source,'5017','订单5017')])==[]


def test_negated_sibling_does_not_cancel_positive_refund():
    raw='不要投诉，但订单5017申请退款'
    assert normalize(raw,[task('refund','订单5017申请退款','5017','订单5017')])[0]['order_id']=='5017'


def test_expansion_over_capacity_is_not_silently_truncated():
    raw='订单'+ '和'.join(str(x) for x in range(5000,5009))+'都申请退款'
    with pytest.raises(ValueError,match='上限'):
        normalize(raw,[task('refund',raw,'5000',raw)])


def test_quoted_product_name_does_not_erase_an_actual_refund_binding():
    raw='订单5017的“智能猫盆”坏了，请申请退款'
    assert normalize(raw,[task('refund',raw,'5017','订单5017')])[0]['order_id']=='5017'


def test_quoted_refund_request_is_not_a_current_action():
    raw='他说“我要退款”，我现在只问订单5017的金额'
    assert normalize(raw,[task('refund','我要退款','5017','订单5017')])==[]


def test_half_and_full_width_comma_recover_the_actual_raw_span():
    raw='订单5017申请退款；另外，我用啥付钱都行吗,微信支付宝能刷不？'
    r=normalize(raw,[task('knowledge','我用啥付钱都行吗，微信支付宝能刷不')])
    assert r[0]['source']=='我用啥付钱都行吗,微信支付宝能刷不'
    assert r[0]['source_alignment']['method']=='presentation_only_alignment'


def test_presentation_alignment_does_not_merge_separate_order_numbers():
    raw='订单5017 8765申请退款'
    with pytest.raises(ValueError):normalize(raw,[task('refund','订单50178765申请退款','50178765')])


def test_changed_order_number_or_semantic_paraphrase_is_not_recovered():
    for source in ['订单5018申请退款','订单5017申请换货']:
        with pytest.raises(ValueError):normalize('订单5017申请退款',[task('refund',source,'5017')])


def test_parallel_projected_sources_are_reanchored_to_full_original_clause():
    raw='订单5017和8765都因质量问题申请退款，9284只查物流'
    r=normalize(raw,[task('refund','5017都因质量问题申请退款','5017'),
                     task('refund','8765都因质量问题申请退款','8765')])
    assert [t['order_id'] for t in r]==['5017','8765']
    assert all(t['source']=='订单5017和8765都因质量问题申请退款' for t in r)


def test_parallel_projection_cannot_invent_order_or_revive_negated_request():
    for raw,oid in [('订单5017和8765都申请退款','9999'),('不要给订单5017和8765都申请退款','5017')]:
        with pytest.raises(ValueError):normalize(raw,[task('refund',oid+'都申请退款',oid)])


def test_source_can_quote_subject_with_action_in_next_local_clause():
    raw='订单5017商品坏了，请申请退款，另查9284物流'
    r=normalize(raw,[task('refund','订单5017商品坏了','5017','订单5017')])
    assert r[0]['order_id']=='5017'
    assert r[0]['binding']['method']=='adjacent_order_subject_action'


def test_subject_quote_does_not_borrow_other_query_order_for_another_refund():
    raw='查订单9284物流，另有一笔商品坏了申请退款'
    r=normalize(raw,[task('refund','查订单9284物流','9284','订单9284')])
    assert r[0]['order_id'] is None
