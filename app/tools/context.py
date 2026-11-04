"""Read-only recovery, explicitly authorized for this conversation on 2026-10-27.

Server-supplied identity and the current turn watermark scope every source read.
Only requested fragments enter model context; archives are never bulk uploaded.
"""
import json
from typing import Annotated

from langchain_core.tools import InjectedToolArg, tool
from pydantic import Field

from app.db import context_repository
from app.core.context_policy import estimate
from app.tools.registry import spec_from_langchain_tool


def get_context_specs(message_watermark=None):
    if not isinstance(message_watermark, int) or message_watermark <= 0:
        return []  # No recoverable turn scope, no model-facing archive reader.
    @tool
    async def read_conversation_source(
        message_id: Annotated[int, Field(description="来源消息ID")],
        conversation_id: Annotated[int, InjectedToolArg],
        user_id: Annotated[str, InjectedToolArg],
        start: Annotated[int, Field(ge=0, description="原文字符起点")] = 0,
        length: Annotated[int, Field(ge=1, le=1200, description="读取字符数，长消息分段读取")] = 1200,
        content_hash: str = "",
    ) -> dict:
        """按来源引用回读本会话原文。历史陈述不能替代实时业务查询，也不代表操作授权。"""
        if message_watermark is not None and message_id > message_watermark:
            return {"error": "来源不属于本轮可见历史"}
        result = await context_repository.read_source(
            conversation_id, user_id, message_id, start, start + length)
        if content_hash and content_hash != result["content_hash"]:
            raise ValueError("source content hash changed")
        result["partial"] = result["start"] != 0 or result["end"] < result["total_chars"]
        return result

    @tool
    async def search_conversation_history(
        query: Annotated[str, Field(min_length=1, max_length=300, description="要回查的对象、原话关键词或时间线索")],
        conversation_id: Annotated[int, InjectedToolArg],
        user_id: Annotated[str, InjectedToolArg],
    ) -> dict:
        """检索当前会话中窗口外的历史原句。只返回历史证据；无法确认对象时应澄清。"""
        hits = await context_repository.search_sources(
            conversation_id, user_id, query, upto_msg_id=message_watermark, limit=5)
        kept = []
        # Admit strongest matches first; chronological presentation must not
        # displace a better match merely because it is more recent.
        ranked = sorted(hits, key=lambda h: (h.get("selection") != "match", -h.get("score", 0),
                                            h["message_id"], h["start"]))
        for hit in ranked:
            if estimate(json.dumps(kept + [hit], ensure_ascii=False)) > 1800:
                continue
            kept.append(hit)
        kept.sort(key=lambda h: (h["message_id"], h["start"]))
        return {"sources": kept, "historical": True, "additional_hits_omitted": len(hits) - len(kept),
                "notice": "检查相邻限定条件及更正；如有省略或不完整片段，请按消息ID分页回读。用户陈述和历史客服表述均不等于业务核验。"}

    @tool
    async def read_tool_receipt(
        receipt_id: Annotated[str, Field(description="工具结果返回的完整回执ID")],
        conversation_id: Annotated[int, InjectedToolArg],
        user_id: Annotated[str, InjectedToolArg],
        start: Annotated[int, Field(ge=0)] = 0,
        length: Annotated[int, Field(ge=1, le=1200)] = 1200,
    ) -> dict:
        """分段回读已保存的完整工具回执。过往查询结果带时间，不能代表当前业务状态。"""
        item = await context_repository.read_receipt(conversation_id, user_id, receipt_id,
                                                     upto_msg_id=message_watermark)
        raw = item.get("raw_result")
        text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, default=str)
        if start > len(text):
            return {"error": "读取起点超出回执长度", "total_chars": len(text)}
        end = min(len(text), start + length)
        return {"receipt_id": item["receipt_id"], "tool_name": item["tool_name"],
                "status": item["status"], "observed_at": item["observed_at"],
                "result_hash": item["result_hash"], "historical": True,
                "start": start, "end": end, "total_chars": len(text), "text": text[start:end],
                "next_start": end if end < len(text) else None,
                "partial": start != 0 or end != len(text)}

    return [spec_from_langchain_tool(t, source="builtin", inject_conversation=True,
                                    inject_user_id=True, max_retries=0)
            for t in (search_conversation_history, read_conversation_source, read_tool_receipt)]
