"""Compare Gitee BGE embeddings to existing Milvus vectors and smoke the real retrieval helpers.

Read-only: no re-vectorization, upsert, DB write, production setting file change or key logging.
"""
import asyncio
import hashlib
import json
from pathlib import Path

from openai import AsyncOpenAI

from app.config import settings
from app.core import embeddings, rerank, retrieval
from app.kb import milvus_client
from scripts.check_retrieval_provider import cosine


async def main():
    def read():
        client = milvus_client.get_client()
        if not client.has_collection(milvus_client.COLLECTION):
            raise RuntimeError("Existing collection unavailable; no collection will be created")
        rows = client.query(milvus_client.COLLECTION, filter="id >= 0", output_fields=["id", "dense", "text"], limit=8)
        return sorted(rows, key=lambda r: r["id"])
    rows = await milvus_client.acall(read)
    original = {name: getattr(settings, name) for name in
                ("embed_base_url", "embed_api_key", "embed_model", "rerank_base_url", "rerank_api_key", "rerank_model")}
    original_client = embeddings._CLIENT
    original_post = rerank._post
    report = {"settings_file_modified": False, "vector_collection_modified": False,
              "sample_size": len(rows), "rows": [], "retrieval_smokes": [], "rerank_requests": []}
    try:
        settings.embed_base_url = settings.rerank_base_url = settings.chat_base_url
        settings.embed_api_key = settings.rerank_api_key = settings.chat_api_key
        settings.embed_model, settings.rerank_model = "bge-m3", "bge-reranker-v2-m3"
        embeddings._CLIENT = AsyncOpenAI(base_url=settings.chat_base_url, api_key=settings.chat_api_key,
            timeout=50, max_retries=0, default_headers={"X-Failover-Enabled": "false"})
        async def fixed_post(url, json, headers, timeout):
            response = await original_post(url, json, {**headers, "X-Failover-Enabled": "false"}, timeout)
            diagnostic = {"url": url, "model": json.get("model"), "documents": len(json.get("documents", [])),
                          "max_document_chars": max((len(d) for d in json.get("documents", [])), default=0),
                          "status": response.status_code}
            if response.status_code != 200:
                try:
                    body = response.json()
                    diagnostic["error"] = body.get("message", body.get("error", body.get("detail")))
                except ValueError:
                    diagnostic["error"] = "non-JSON error response"
            report["rerank_requests"].append(diagnostic)
            return response
        rerank._post = fixed_post
        new_vectors = await embeddings.embed_texts([r["text"] for r in rows])
        for row, vector in zip(rows, new_vectors):
            def search():
                return milvus_client.dense_search(milvus_client.get_client(), vector, 3)
            hits = await milvus_client.acall(search)
            report["rows"].append({"id": row["id"], "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
                "dimension": len(vector), "cosine_to_stored_vector": cosine(row["dense"], vector),
                "self_in_top3": row["id"] in [h["id"] for h in hits],
                "top3_ids": [h["id"] for h in hits]})
        for query in ("退货需要满足什么条件？", "自动喂食器怎么清洗？", "保修需要提供什么凭证？"):
            try:
                hits = await retrieval.search_knowledge(query, strategy="hybrid_rerank", top_k=3)
                report["retrieval_smokes"].append({"query": query, "ok": bool(hits),
                    "hits": [{"id": h["id"], "question": h["question"], "rerank_score": h["rerank_score"]} for h in hits]})
            except Exception as exc:
                response = getattr(exc, "response", None)
                report["retrieval_smokes"].append({"query": query, "ok": False, "error_type": type(exc).__name__,
                    "status": response.status_code if response is not None else None})
    finally:
        if embeddings._CLIENT is not original_client:
            await embeddings._CLIENT.close()
        embeddings._CLIENT = original_client
        rerank._post = original_post
        for name, value in original.items():
            setattr(settings, name, value)
    scores = [r["cosine_to_stored_vector"] for r in report["rows"]]
    report["minimum_cosine_to_stored"] = min(scores) if scores else None
    report["compatible_smoke"] = (len(rows) == len(report["rows"]) and len(rows) > 0
        and all(r["dimension"] == 1024 and r["cosine_to_stored_vector"] >= .995 for r in report["rows"])
        and all(r["ok"] for r in report["retrieval_smokes"]))
    path = Path("data/provider-setup/existing-vectors-gitee.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
