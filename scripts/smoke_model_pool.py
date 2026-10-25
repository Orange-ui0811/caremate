"""Small real-provider smoke with synthetic inputs only; never prints credentials."""
import asyncio
import json
from pathlib import Path
import time
import uuid

from pydantic import BaseModel

from app.config import settings
from app.core import embeddings, llm, rerank
from app.core.model_pool import call_context, get_store
from app.core.model_transport import close_clients


class Probe(BaseModel):
    order_id: str


async def main():
    report = {"synthetic_inputs_only": True, "tests": [], "model": settings.chat_model,
              "pool_enabled": settings.model_pool_enabled}
    async def check(name, function):
        started, trace = time.monotonic(), "pool-smoke-" + uuid.uuid4().hex
        try:
            with call_context(trace):
                value = await function()
            detail = value if isinstance(value, dict) else {"ok": bool(value)}
            report["tests"].append({"name": name, **detail, "seconds": round(time.monotonic()-started, 3), "request_id": trace})
        except Exception as exc:
            report["tests"].append({"name": name, "ok": False, "error_type": type(exc).__name__,
                                    "seconds": round(time.monotonic()-started, 3), "request_id": trace})
    async def plain():
        message = await llm.get_chat_model(temperature=0).bind(max_tokens=256).ainvoke("只输出两个大写英文字母：OK。不要输出其他文字、解释或标点。")
        return {"ok": message.content.strip() == "OK", "synthetic_response": message.content,
                "finish_reason": message.response_metadata.get("finish_reason"), "usage": message.usage_metadata,
                "max_tokens": 256}
    async def stream():
        aggregate = None
        async for chunk in llm.get_chat_model(streaming=True, temperature=0).bind(max_tokens=256).astream("只输出两个大写英文字母：OK。不要输出其他文字、解释或标点。"):
            aggregate = chunk if aggregate is None else aggregate + chunk
        return {"ok": aggregate.content.strip() == "OK", "synthetic_response": aggregate.content,
                "finish_reason": aggregate.response_metadata.get("finish_reason"), "usage": aggregate.usage_metadata,
                "max_tokens": 256}
    async def structured():
        return (await llm.structured(Probe, temperature=0).ainvoke("订单号1001。仅提取订单号。")).order_id == "1001"
    async def embedding():
        values = await embeddings.embed_texts(["清洁前请先断电。"])
        return len(values) == 1 and len(values[0]) == 1024
    async def ranking():
        values = await rerank.rerank("清洁前需要做什么？", ["清洁前请先断电。", "快递发货后提供运单号。"], top_n=2)
        return len(values) == 2 and values[0][0] == 0
    try:
        for name, fn in (("plain_chat", plain), ("streaming_chat", stream), ("function_calling", structured),
                         ("embedding_1024", embedding), ("rerank", ranking)):
            await check(name, fn)
    finally:
        await close_clients()
        embeddings._CLIENT = None
    report["all_passed"] = all(r["ok"] for r in report["tests"])
    report["remaining_leases"] = len(get_store().snapshot()["running"])
    path = Path("data/verification/model-pool-provider-smoke.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
