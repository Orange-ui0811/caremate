"""Real application /api/chat benchmark, via ASGI (excludes network socket overhead).

Creates clearly named benchmark conversations, deletes only those conversations afterwards.
No actual orders/tickets are changed. Uses the real graph, retrieval and upstream models.
"""
import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
import uuid

import httpx

from scripts.benchmark_model_pool import load_keys, percentile
from app.config import settings
from app.core import model_pool as pool


async def main(args):
    keys = load_keys(args.env_file)
    for channel in ("chat", "embed", "rerank"):
        setattr(settings, channel + "_api_key", keys[0])
    settings.model_pool_enabled = args.mode != "direct"
    settings.langfuse_public_key = ""  # Keep benchmark traces out of business usage dashboards.
    settings.model_pool_turn_timeout = 180
    settings.model_pool_db_path = str(Path(args.output).with_suffix("."+uuid.uuid4().hex[:8]+".sqlite3"))
    settings.checkpointer_db_path = settings.model_pool_db_path + ".checkpoints"
    settings.model_pool_chat_keys = keys[0]
    store = pool.get_store()
    if args.mode == "pool":
        store.set_credentials([{"api_key": k, "channels": ["chat", "embed", "rerank"]} for k in keys])
    policy = pool.Policy(global_limit=args.global_limit, key_limit=args.key_limit,
                         background_limit=max(1,args.global_limit//2), queue_timeout=args.queue_timeout,
                         call_timeout=90, read_timeout=60, max_queue=512, max_attempts=2)
    store.set_policy(policy)
    from app.main import app
    from app.db import repository
    logging.getLogger("app").setLevel(logging.CRITICAL)
    logging.getLogger("httpx").setLevel(logging.CRITICAL)
    logging.getLogger("langchain_openai").setLevel(logging.CRITICAL)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    run = "pool-bench-" + uuid.uuid4().hex[:12]
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "mode": args.mode,
              "scope": "Real /api/chat through ASGI, new knowledge-question conversations, includes graph/RAG/DB; excludes socket overhead",
              "turn_timeout": settings.model_pool_turn_timeout, "policy": policy.model_dump(), "stages": []}
    questions = ["MH-LP100 的废砂盒多久倒一次", "智能猫砂盆 Pro MH-LP100 有没有 App 远程监控",
                 "MH-LP50 和 Pro 版有什么区别", "MH-LP200 适合多大的猫,几天倒一次废砂盒"]
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        for concurrency in args.concurrency:
            gate = asyncio.Semaphore(concurrency)
            async def one(index):
                async with gate:
                    user = f"{run}-{concurrency}-{index}"
                    started = time.monotonic()
                    try:
                        async with asyncio.timeout(185):
                            response = await client.post("/api/chat", json={"user_id": user,
                                "request_id": user, "message": questions[index % len(questions)]})
                        events = []
                        for line in response.text.splitlines():
                            if line.startswith("data: ") and line != "data: [DONE]":
                                try: events.append(json.loads(line[6:]))
                                except ValueError: pass
                        ok = response.status_code == 200 and "data: [DONE]" in response.text and "event: error" not in response.text
                        return {"ok": ok, "seconds": time.monotonic()-started,
                                "status": "complete" if ok else "stream_error",
                                "answer_chars": sum(len(e.get("delta", "")) for e in events),
                                "activities": [e.get("name") for e in events if e.get("event") == "activity"],
                                "citations": sum(len(e.get("items",[])) for e in events if e.get("event") == "citations")}
                    except Exception as exc:
                        return {"ok": False, "seconds": time.monotonic()-started, "status": type(exc).__name__}
                    finally:
                        # Limit cleanup to this invocation's unique user, including interrupted turns.
                        for conversation in await repository.list_conversations(user):
                            cid = conversation["id"] if isinstance(conversation, dict) else conversation.id
                            await repository.delete_conversation(cid, user)
            started = time.monotonic()
            samples = await asyncio.gather(*(one(i) for i in range(concurrency * args.waves)))
            elapsed = time.monotonic()-started
            good = [s for s in samples if s["ok"]]
            stage = {"concurrency": concurrency, "turns": len(samples), "successes": len(good),
                     "statuses": dict(Counter(s["status"] for s in samples)), "elapsed_seconds": round(elapsed,3),
                     "successful_turns_per_second": round(len(good)/elapsed,3),
                     "p50_seconds": percentile([s["seconds"] for s in good], .5),
                     "p95_seconds": percentile([s["seconds"] for s in good], .95), "samples": samples}
            report["stages"].append(stage)
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({k:v for k,v in stage.items() if k != "samples"}, ensure_ascii=False), flush=True)
            if len(good) != len(samples) or (stage["p95_seconds"] or 180) > 60:
                break
            await asyncio.sleep(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["direct", "pool", "old-pool"], default="pool")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[4,8,16,32])
    parser.add_argument("--waves", type=int, default=1)
    parser.add_argument("--global-limit", type=int, default=64)
    parser.add_argument("--key-limit", type=int, default=64)
    parser.add_argument("--queue-timeout", type=float, default=30)
    asyncio.run(main(parser.parse_args()))
