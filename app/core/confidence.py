"""第一道相关性粗筛：取最终候选中的最高重排分，不依赖合并顺序。

分数不是答案正确概率，也不证明复合问题已完整覆盖；通过后仍须执行证据充分性检查。
保留原始浮点精度，与阈值校准时的 >= 比较保持一致。
"""
import math
from dataclasses import dataclass

SCORING_METHOD = "max_rerank_v1"


@dataclass
class EvidenceConfidence:
    score: float    # 0-1 最高相关分
    signals: dict   # 各原始信号,落池 reason 与 trace 留痕用


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def compute_evidence_confidence(hits: list[dict]) -> EvidenceConfidence:
    scores, invalid_count = [], 0
    for h in hits:
        try:
            value = float(h.get("rerank_score", 0.0))
        except (TypeError, ValueError, OverflowError):
            invalid_count += 1
            continue
        if not math.isfinite(value):
            invalid_count += 1
            continue
        scores.append(_clip01(value))
    score = max(scores, default=0.0)
    return EvidenceConfidence(score, {"scoring_method": SCORING_METHOD,
                                     "max_rerank_score": score,
                                     "candidate_count": len(hits),
                                     "invalid_score_count": invalid_count})


def snapshot_from_hits(hits: list[dict], top_n: int = 3) -> list[dict]:
    """落池召回快照的统一出口:Top N 的原文与得分(审核页给人看的那份)。"""
    return [{"question": h.get("question", ""), "answer": h.get("answer", ""),
             "rerank_score": float(h.get("rerank_score", 0.0)),
             "section_path": h.get("section_path", "")}
            for h in hits[:top_n]]
