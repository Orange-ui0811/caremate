"""Two synthetic requests on the configured real provider; not a throughput test."""
import asyncio
import json
from pathlib import Path
import time
import uuid

from app.core.llm import get_chat_model
from app.core.model_pool import call_context, get_store
from app.core.model_transport import close_clients


async def main():
    store, trace = get_store(), "pool-pair-" + uuid.uuid4().hex
    peak, results = 0, []
    model = get_chat_model(streaming=True, temperature=0).bind(max_tokens=256)
    started = time.monotonic()
    async def request():
        try:
            message = await model.ainvoke("只输出两个大写英文字母：OK。不要输出其他文字、解释或标点。")
            return {"ok": message.content.strip() == "OK", "text": message.content, "usage": message.usage_metadata}
        except Exception as exc:
            return {"ok": False, "error_type": type(exc).__name__}
    try:
        with call_context(trace):
            tasks = [asyncio.create_task(request()) for _ in range(2)]
            while not all(task.done() for task in tasks):
                state = await asyncio.to_thread(store.snapshot)
                peak = max(peak, sum(r["request_id"] == trace for r in state["running"]))
                await asyncio.sleep(.05)
            results = await asyncio.gather(*tasks)
    finally:
        await close_clients()
    report = {"scope": "Two real-provider synthetic streaming requests; not a sustained load or quota test",
              "requests": 2, "request_id": trace, "peak_registered_inflight": peak,
              "elapsed_seconds": round(time.monotonic()-started, 3), "policy": store.policy().model_dump(),
              "results": results, "all_passed": len(results) == 2 and all(r["ok"] for r in results),
              "remaining_smoke_leases": sum(r["request_id"] == trace for r in store.snapshot()["running"])}
    path = Path("data/verification/model-pool-real-concurrency.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
