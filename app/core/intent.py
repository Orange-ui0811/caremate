import logging
from typing import Literal

from pydantic import BaseModel, Field

from app.config import settings
from app.core import llm
from app.core.llm import get_chat_model
from app.core.prompts import INTENT_CLASSIFY_PROMPT
from app.core.model_pool import model_call
from app.core.context_runtime import ContextInputLimit


class TaskIntent(BaseModel):
    intent: Literal["物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "人工", "闲聊", "其他"]
    query: str = Field(min_length=1, description="只包含这一项诉求，保留咨询或办理原意，补全已知对象")
    source: str = Field(min_length=1, description="本轮原话中逐字引用的诉求片段")
    order_id: str | None = None
    order_ref: str = Field(min_length=1, description="必填对象引用，例如ref1；即使缺订单号也不能为空。同一对象共用，独立事项用不同引用")
    action: Literal["consult", "request"] = "consult"


class TaskIntents(BaseModel):
    tasks: list[TaskIntent] = Field(min_length=1, max_length=8)


@model_call("intent_classification")
async def classify_tasks(raw: str, query: str, history: str, candidates: list, *, feedback=None) -> list[dict]:
    """Same nine labels and model as classify(); only the output is a task list."""
    import json
    from app.core.prompts import INTENT_CLASSIFY_SYSTEM
    prompt=INTENT_CLASSIFY_SYSTEM.partition('## 输出要求')[0]+'''
本轮识别所有实际诉求，按原文顺序输出tasks，使用上述同一套九类意图。
只拆当前原话，历史仅解释对象；否定、假设、引用的操作不创建任务。
每项query只表达该诉求，source逐字引用原话；不要把整句复制给每个query。
商品是什么/订单金额为订单；送到哪了为物流；能否退货为退款退货且action=consult。
保修维修为售后；商品规格/说明为商品咨询。明确申请退款action=request。
明确登记投诉工单/要求人工办理为人工且action=request，query保留具体投诉内容。
只有抱怨或一般投诉表达而没有要求登记/跟进时用投诉，沿用原安抚和用户自选入口。
同一对象的任务共用order_ref，不能要求每个子句重复订单号。
订单号未知也必须填写order_ref；例如“商品坏了，申请退款并登记投诉”两项都用ref1，选择订单后共用。
通用咨询也分配一个独立引用；不允许空字符串。不要把“未知订单”误当成“没有对象关系”。
order_id只能来自提供的可信候选。明确指定优先；两笔订单指代不明时留空。
独立退款、另一笔订单不能借旁边物流任务的订单，使用不同order_ref。
咨询不升级为办理，无法确定时action=consult。商品咨询query保留已知商品名，不推断型号。
合并同一对象同一意图的诉求，不重复创建相同任务。
“给这单办退钱”“提交退款申请，等平台审核”是request，不是查询申请方法或审核进度。
只有询问“是否可以退/如何申请/进度如何”才是consult。query不得改变原话中的办理动词或时态。
人工客服在线时间、服务规则属于商品咨询，需要知识检索，不是人工建单。
source必须是original内连续原文，不能复制resolved中的改写。两笔订单分别建任务，使用不同引用。
'''
    result=await llm.structured(TaskIntents,model=settings.intent_model or None).ainvoke([
        {"role":"system","content":prompt},
        {"role":"user","content":json.dumps({'original':raw,'resolved':query,'history':history,'candidates':candidates,
            **({'correction':feedback} if feedback else {})},ensure_ascii=False)}])
    return [t.model_dump() for t in result.tasks]

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
