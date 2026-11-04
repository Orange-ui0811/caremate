import logging
from typing import Literal

from pydantic import BaseModel, Field

from app.config import settings
from app.core import llm
from app.core.llm import get_chat_model
from app.core.prompts import INTENT_CLASSIFY_PROMPT
from app.core.model_pool import model_call
from app.core.context_runtime import ContextInputLimit

INTENTS = ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "人工", "闲聊", "其他")
logger = logging.getLogger(__name__)


class _Intent(BaseModel):
    intent: Literal["物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "人工", "闲聊", "其他"] = Field(
        description="九类意图之一")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="判断把握 0-1")


@model_call("intent_classification")
async def classify(query: str, history: str = "") -> dict:
    """九类意图 + confidence(ch08 增「人工」:明确要求建工单/转人工)。扁平字段避开 glm 嵌套 502;
    解析失败/越界 → 归「其他」(最保守兜底,替换 ch05 的回退闲聊)。
    模型默认走 settings.intent_model(空则回落 chat_model)——先求准。"""
    model = llm.structured(_Intent, model=settings.intent_model or None)
    try:
        r: _Intent = await (INTENT_CLASSIFY_PROMPT | model).ainvoke(
            {"query": query, "history": history or "(无)"})
    except ContextInputLimit:
        raise
    except Exception as exc:
        logger.warning("intent classification unavailable: %s", type(exc).__name__)
        return {"intent": "其他", "confidence": 0.0, "classification_error": type(exc).__name__}
    intent = r.intent if r.intent in INTENTS else "其他"
    return {"intent": intent, "confidence": float(r.confidence)}
