"""Read-only BGE compatibility checks using synthetic text and locally held credentials.

Checks Gitee with CHAT_API_KEY and the original providers with their own keys.
No settings file, database or vector collection is modified; raw keys/vectors stay out of logs.
"""
import argparse
import asyncio
import json
import math
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from openai import AsyncOpenAI

from app.config import settings

TEXTS = ["退货需要哪些凭证？", "申请退货时请保留订单记录、购买凭证和完整配件。", "猫爬架需要按照说明自行安装。"]
QUERY = "自动喂食器应该怎么清洁？"
DOCS = ["自动喂食器清洁前先断电，食盆可拆下清洗，主机用软布擦拭。",
        "发货后可凭运单号查询快递轨迹。", "猫爬架可按说明安装，使用前检查连接件。"]


def cosine(a, b):
    if len(a) != len(b):
        return None
    na, nb = math.sqrt(sum(x*x for x in a)), math.sqrt(sum(x*x for x in b))
    return sum(x*y for x, y in zip(a, b)) / (na*nb) if na and nb else None


async def embedding(base, key, model):
    started = time.monotonic()
    result = {"host": urlsplit(base).hostname, "model": model}
    client = AsyncOpenAI(base_url=base, api_key=key, timeout=40, max_retries=0,
                         default_headers={"X-Failover-Enabled": "false"})
    vectors = None
    try:
        response = await client.embeddings.create(model=model, input=TEXTS)
        vectors = [item.embedding for item in response.data]
        dimensions = [len(v) for v in vectors]
        finite = all(math.isfinite(x) for v in vectors for x in v)
        result |= {"ok": len(vectors) == len(TEXTS) and finite and all(d == 1024 for d in dimensions),
                   "count": len(vectors), "dimensions": dimensions, "finite": finite,
                   "norms": [math.sqrt(sum(x*x for x in v)) for v in vectors],
                   "relevant_similarity": cosine(vectors[0], vectors[1]),
                   "irrelevant_similarity": cosine(vectors[0], vectors[2]),
                   "usage": response.usage.model_dump() if response.usage else None}
    except Exception as exc:
        result |= {"ok": False, "error_type": type(exc).__name__, "status": getattr(exc, "status_code", None)}
    finally:
        await client.close()
    return result | {"seconds": round(time.monotonic() - started, 3)}, vectors


async def rerank(base, key, model):
    started = time.monotonic()
    base = base.rstrip("/")
    result = {"host": urlsplit(base).hostname, "model": model}
    try:
        async with httpx.AsyncClient(timeout=40) as client:
            response = await client.post(base + "/rerank", json={"model": model, "query": QUERY,
                "documents": DOCS, "top_n": 3}, headers={"Authorization": "Bearer " + key, "X-Failover-Enabled": "false"})
        result["status"] = response.status_code
        response.raise_for_status()
        payload = response.json()
        values = sorted([(int(r["index"]), float(r["relevance_score"])) for r in payload["results"]], key=lambda r:r[1], reverse=True)
        result |= {"ok": len(values) == 3 and values[0][0] == 0 and all(math.isfinite(v[1]) for v in values),
                   "ranked": values, "score_range_0_1": all(0 <= v[1] <= 1 for v in values), "usage": payload.get("usage")}
    except Exception as exc:
        result |= {"ok": False, "error_type": type(exc).__name__, "status": result.get("status", getattr(exc, "status_code", None))}
    return result | {"seconds": round(time.monotonic() - started, 3)}


async def main(args):
    new_embed, old_embed, new_rank, old_rank = await asyncio.gather(
        embedding(settings.chat_base_url, settings.chat_api_key, "bge-m3"),
        embedding(settings.embed_base_url, settings.embed_api_key, settings.embed_model),
        rerank(settings.chat_base_url, settings.chat_api_key, "bge-reranker-v2-m3"),
        rerank(settings.rerank_base_url, settings.rerank_api_key, settings.rerank_model))
    report = {"gitee_embedding": new_embed[0], "original_embedding": old_embed[0],
              "gitee_rerank": new_rank, "original_rerank": old_rank,
              "settings_modified": False, "credentials_exposed": False,
              "scope": "Three synthetic texts and three rerank candidates; functional compatibility, not retrieval-quality calibration."}
    if new_embed[1] is not None and old_embed[1] is not None:
        report["same_text_cross_provider_cosines"] = [cosine(a, b) for a, b in zip(new_embed[1], old_embed[1])]
    if new_rank.get("ok") and old_rank.get("ok"):
        report["same_query_rerank_max_absolute_score_difference"] = max(abs(dict(new_rank["ranked"])[i] - score) for i, score in old_rank["ranked"])
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/provider-setup/retrieval-compatibility.json")
    args = parser.parse_args()
    asyncio.run(main(args))
