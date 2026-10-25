"""Bounded real-provider benchmark. Reports contain no credentials or response bodies.

Run with PYTHONPATH=. python scripts/benchmark_model_pool.py --help.
Uses an isolated pool DB; never changes the running service's policy.
"""
import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
import uuid

import httpx
from dotenv import dotenv_values

from app.config import settings
from app.core import model_pool as pool
from app.core.model_transport import AsyncPoolTransport


def load_keys(path):
    values = dotenv_values(path)
    keys = [values.get(f"GITEE_AI_KEY_{i}") for i in range(1, 11)]
    if not all(keys) or len(set(keys)) != 10:
        raise ValueError("Expected ten distinct configured Gitee keys")
    return keys


def payload(channel, index=0):
    if channel == "embed":
        return {"model": settings.embed_model, "input": ["订单签收七天内可以申请无理由退货吗？"]}
    if channel == "rerank":
        return {"model": settings.rerank_model, "query": "签收七天内可以退货吗？",
                "documents": ["七天无理由退货需保持商品完好。", "订单发货后可查询物流。"], "top_n": 2}
    return {"model": settings.chat_model, "messages": [
        {"role": "system", "content": "你是电商售后客服。仅根据给定政策简短回答，不编造订单信息。"},
        {"role": "user", "content": f"测试编号{index}。政策：商品完好且签收七天内支持无理由退货，需先申请。问题：刚签收两天的完好商品如何退货？用三句话回答。"}],
        "temperature": 0, "max_tokens": 128, "thinking": {"type": "disabled"}, "stream": False}


def percentile(values, q):
    return round(sorted(values)[max(0, math.ceil(len(values) * q) - 1)], 3) if values else None


async def run_batch(mode, keys, concurrency, count, channels, policy=None):
    clients = {}
    endpoints = {"chat": "chat/completions", "embed": "embeddings", "rerank": "rerank"}
    for channel in set(channels):
        clients[channel] = httpx.AsyncClient(
            transport=AsyncPoolTransport(channel) if mode == "pool" else httpx.AsyncHTTPTransport(
                retries=0, limits=httpx.Limits(max_connections=512, max_keepalive_connections=64)),
            timeout=httpx.Timeout(90, read=60), limits=httpx.Limits(max_connections=256, max_keepalive_connections=64))
    semaphore = asyncio.Semaphore(concurrency)
    async def one(index):
        async with semaphore:
            start = time.monotonic()
            statuses, tokens = [], 0
            try:
                async with asyncio.timeout(90):
                    for channel in channels:
                        key = keys[index % len(keys)]
                        response = await clients[channel].post(
                            getattr(settings, channel + "_base_url").rstrip("/") + "/" + endpoints[channel],
                            headers={"Authorization": "Bearer " + key, "X-Failover-Enabled": "false"},
                            json=payload(channel, index))
                        statuses.append(response.status_code)
                        if response.status_code != 200:
                            return {"ok": False, "status": str(response.status_code), "seconds": time.monotonic()-start}
                        data = response.json()
                        valid = bool(data.get({"chat": "choices", "embed": "data", "rerank": "results"}[channel]))
                        if not valid:
                            return {"ok": False, "status": "invalid_body", "seconds": time.monotonic()-start}
                        tokens += (data.get("usage") or {}).get("total_tokens", 0)
                return {"ok": True, "status": "200", "seconds": time.monotonic()-start, "tokens": tokens}
            except Exception as exc:
                return {"ok": False, "status": type(exc).__name__, "seconds": time.monotonic()-start}
    start = time.monotonic()
    try:
        rows = await asyncio.gather(*(one(i) for i in range(count)))
    finally:
        for client in clients.values():
            await client.aclose()
    elapsed = time.monotonic()-start
    good = [r for r in rows if r["ok"]]
    result = {"mode": mode, "key_count": len(keys), "concurrency": concurrency,
              "requests": count, "channels_per_request": channels, "successes": len(good),
              "statuses": dict(Counter(r["status"] for r in rows)), "elapsed_seconds": round(elapsed, 3),
              "successful_requests_per_second": round(len(good)/elapsed, 3),
              "p50_seconds": percentile([r["seconds"] for r in good], .5),
              "p95_seconds": percentile([r["seconds"] for r in good], .95),
              "reported_total_tokens": sum(r.get("tokens", 0) for r in rows), "samples": rows}
    if policy:
        result["policy"] = policy.model_dump()
    return result


async def main(args):
    keys = load_keys(args.env_file)
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "model": settings.chat_model,
              "scope": "Real upstream; short nonstreaming chat (max_tokens=128); no customer data",
              "stages": []}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    def save(result):
        report["stages"].append(result)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k != "samples"}, ensure_ascii=False), flush=True)
    if args.health:
        for i, key in enumerate(keys, 1):
            result = await run_batch("direct", [key], 1, 1, ["chat", "embed", "rerank"])
            result["key_number"] = i
            save(result)
        return
    selected = keys[:args.keys]
    # Use exactly the production adapter with isolated coordination, never the live DB.
    settings.model_pool_db_path = str(output.with_suffix("." + uuid.uuid4().hex[:8] + ".sqlite3"))
    settings.model_pool_chat_keys = ",".join(selected)
    store = pool.get_store()
    if hasattr(store, "set_credentials"):
        store.set_credentials([{"api_key": k, "channels": ["chat", "embed", "rerank"]} for k in selected])
    policy = pool.Policy(global_limit=args.global_limit, key_limit=args.key_limit,
                         background_limit=args.background_limit, max_queue=512,
                         queue_timeout=args.queue_timeout, call_timeout=90, read_timeout=60, max_attempts=args.attempts)
    store.set_policy(policy)
    for concurrency in args.concurrency:
        result = await run_batch(args.mode, selected, concurrency, max(concurrency, args.waves*concurrency),
                                 ["embed", "rerank", "chat"] if args.mixed else ["chat"], policy if args.mode == "pool" else None)
        save(result)
        # Stop escalation at first error or slow p95. Never hammer a failed quota.
        if result["successes"] != result["requests"] or (result["p95_seconds"] or 90) > 45:
            break
        await asyncio.sleep(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--health", action="store_true")
    parser.add_argument("--mode", choices=["direct", "pool"], default="direct")
    parser.add_argument("--keys", type=int, choices=range(1, 11), default=1)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--waves", type=int, default=2)
    parser.add_argument("--global-limit", type=int, default=80)
    parser.add_argument("--key-limit", type=int, default=16)
    parser.add_argument("--background-limit", type=int, default=40)
    parser.add_argument("--queue-timeout", type=float, default=10)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--mixed", action="store_true")
    asyncio.run(main(parser.parse_args()))
