"""第一道评分：最高候选分，不受排列、重复、分差和关键词影响。"""
import pytest

from app.core.confidence import SCORING_METHOD, compute_evidence_confidence, snapshot_from_hits


def _hit(score, q="退货运费谁出", a="满99包邮,退货运费买家承担"):
    return {"question": q, "answer": a, "rerank_score": score,
            "section_path": "售后 / 退货", "id": 1, "content_type": "faq"}


def test_empty_hits_zero_confidence():
    r = compute_evidence_confidence([])
    assert r.score == 0.0
    assert r.signals == {"scoring_method": SCORING_METHOD, "max_rerank_score": 0.0,
                         "candidate_count": 0, "invalid_score_count": 0}


def test_strong_evidence_scores_high():
    r = compute_evidence_confidence([_hit(0.95), _hit(0.40), _hit(0.35)])
    assert r.score == 0.95
    assert r.signals["max_rerank_score"] == 0.95
    assert r.signals["candidate_count"] == 3


def test_weak_flat_evidence_scores_low():
    hits = [_hit(0.22, q="猫粮口味", a="三文鱼味与鸡肉味"),
            _hit(0.21, q="猫粮口味", a="三文鱼味与鸡肉味")]
    r = compute_evidence_confidence(hits)
    assert r.score == .22


def test_adding_related_or_duplicate_hits_cannot_reduce_score():
    r = compute_evidence_confidence([_hit(0.8)])
    assert r.score == compute_evidence_confidence([_hit(.8), _hit(.79), _hit(.8)]).score


def test_round_robin_order_does_not_change_max_score():
    scores = [.1, .92, .3, .8, .2, .4, .6, .5, .7, .9]
    for order in (scores, scores[::-1], sorted(scores)):
        assert compute_evidence_confidence([_hit(s) for s in order]).score == .92


def test_policy_keyword_cannot_raise_score():
    assert compute_evidence_confidence([_hit(.2)]).score == compute_evidence_confidence(
        [_hit(.2, q="猫粮口味", a="三文鱼味")]).score


def test_score_preserves_threshold_precision():
    value = .2260993940477108
    assert compute_evidence_confidence([_hit(value)]).score == value


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), None, "bad"])
def test_invalid_scores_cannot_open_gate(value):
    r = compute_evidence_confidence([_hit(value), _hit(.1)])
    assert r.score == .1
    assert r.signals["invalid_score_count"] == 1
    assert compute_evidence_confidence([_hit(value)]).score == 0


def test_score_monotonic_in_top1():
    low = compute_evidence_confidence([_hit(0.3), _hit(0.2)])
    high = compute_evidence_confidence([_hit(0.9), _hit(0.2)])
    assert high.score > low.score


def test_score_bounded_zero_one():
    assert compute_evidence_confidence([_hit(2.0), _hit(-1)]).score == 1
    assert compute_evidence_confidence([_hit(-1)]).score == 0


def test_snapshot_from_hits_top3_shape():
    hits = [_hit(0.9), _hit(0.8), _hit(0.7), _hit(0.6)]
    snap = snapshot_from_hits(hits)
    assert len(snap) == 3
    assert snap[0] == {"question": "退货运费谁出", "answer": "满99包邮,退货运费买家承担",
                       "rerank_score": 0.9, "section_path": "售后 / 退货"}


def test_snapshot_empty_hits():
    assert snapshot_from_hits([]) == []
