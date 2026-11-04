"""Validate the configured chat provider without exposing credentials or customer records."""
import argparse
import asyncio
import json
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from app.config import settings
from app.core import llm
from scripts.eval_context_engineering import tracked


class Probe(BaseModel):
    order_id: str


async def main(args):
    host = urlsplit(settings.chat_base_url).hostname
    result = {"host": host, "configured_model": settings.chat_model, "checks": {}}
    headers = {"Authorization": "Bearer " + settings.chat_api_key}
    async with httpx.AsyncClient(timeout=25) as client:
        try:
            response = await client.get(settings.chat_base_url.rstrip("/") + "/models", headers=headers)
            result["catalog_status"] = response.status_code
            if response.status_code == 200:
                models = [item["id"] for item in response.json().get("data", [])]
                wanted = settings.chat_model.rsplit("/", 1)[-1].casefold()
                matches = [m for m in models if m.rsplit("/", 1)[-1].casefold() == wanted]
                result["exact_model_listed"] = settings.chat_model in models
                result["same_model_id_candidates"] = matches
            else:
                result["same_model_id_candidates"] = []
        except Exception as error:
            result["catalog_error"] = type(error).__name__
    if not args.catalog_only:
        model = args.model or settings.chat_model
        result["requested_model"] = model
        async def plain():
            msg = await llm.get_chat_model(model=model, temperature=0).bind(max_tokens=32).ainvoke("Reply OK only.")
            return msg.content
        result["checks"]["plain"] = await tracked(plain, 40)
        if result["checks"]["plain"].get("calls"):
            async def structured():
                obj = await llm.structured(Probe, model=model, temperature=0).ainvoke([
                    HumanMessage("Extract the order_id literally: order 610010.")])
                return obj.model_dump()
            result["checks"]["structured_streaming"] = await tracked(structured, 40)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/provider-setup/chat-check.json")
    parser.add_argument("--catalog-only", action="store_true")
    parser.add_argument("--model")
    args = parser.parse_args()
    asyncio.run(main(args))
