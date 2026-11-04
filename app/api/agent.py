import asyncio
import logging
import uuid

from fastapi import APIRouter, HTTPException
from langchain_core.messages import AIMessage, ToolMessage
from sqlalchemy.exc import SQLAlchemyError

from app.graph import runtime
from app.graph.nodes import resolve_answer
from app.config import settings
from app.core.model_pool import get_store
from app.schemas.agent import AgentRequest, AgentResponse, ToolCallView, ToolResultView

logger = logging.getLogger(__name__)
router = APIRouter()


def _views_from_state(state):
    """从终态 messages 重建工具调用/结果轨迹(供评估/单测看模型选了什么工具)。"""
    calls, results = [], []
    for m in state.get("messages", []):
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                calls.append(ToolCallView(id=tc["id"], name=tc["name"], args=tc["args"]))
        elif isinstance(m, ToolMessage):
            results.append(ToolResultView(
                tool_call_id=m.tool_call_id, name=m.name or "", ok=(m.status != "error"),
                content=m.content))
    return calls, results


@router.post("/api/agent", response_model=AgentResponse)
async def run_agent(req: AgentRequest) -> AgentResponse:
    turn_store, turn_owner = get_store(), uuid.uuid4().hex
    if req.conversation_id is not None:
        acquired = await asyncio.to_thread(turn_store.acquire_turn, req.conversation_id, turn_owner)
        if not acquired:
            raise HTTPException(status_code=409, detail="当前会话还有请求正在处理，请等待该轮结束后重试")
    try:
        async with asyncio.timeout(settings.model_pool_turn_timeout):
            options = {'multi_intent_enabled': req.multi_intent_enabled} if req.multi_intent_enabled is not None else {}
            out = await runtime.run_turn(req.user_id, req.message, req.conversation_id, **options)
    except runtime.ConversationNotFound:
        raise HTTPException(status_code=404, detail="会话不存在")
    except runtime.ResumeRejected as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except SQLAlchemyError:
        logger.exception("数据库错误 user_id=%s", req.user_id)
        raise HTTPException(status_code=503, detail="数据库暂时不可用,请稍后重试")
    except TimeoutError:
        raise HTTPException(status_code=504, detail="本轮处理已超时，请先查看会话记录再重试")
    except Exception:
        logger.exception("图编排失败 user_id=%s", req.user_id)
        raise HTTPException(status_code=502, detail="上游模型暂时不可用,请稍后重试")
    finally:
        if req.conversation_id is not None:
            await asyncio.shield(asyncio.to_thread(turn_store.release_turn, req.conversation_id, turn_owner))

    state = out["state"]
    calls, results = _views_from_state(state)
    return AgentResponse(
        task_plan=state.get("task_plan", {}),
        conversation_id=out["conversation_id"],
        answer=resolve_answer(state),
        tool_calls=calls,
        tool_results=results,
        suggested_actions=runtime.dedup_actions(state.get("suggested_actions", [])),
        interrupt=out.get("interrupt"),
    )
