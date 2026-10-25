"""Model governance status exposes only fingerprints and request metadata."""
import asyncio
import ipaddress
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.config import settings
from app.core.model_pool import Policy, get_store, providers

router = APIRouter(prefix="/api/model-pool", tags=["model-pool"])


@router.get("/runtime")
async def runtime():
    store = get_store()
    snapshot, policy = await asyncio.gather(asyncio.to_thread(store.snapshot), asyncio.to_thread(store.policy))
    slots = {s["resource"]: s for s in snapshot["slots"]}
    configured = {}
    managed = await asyncio.to_thread(store.credential_metadata)
    for channel, model in (("chat", settings.chat_model), ("embed", settings.embed_model), ("rerank", settings.rerank_model)):
        for provider in await asyncio.to_thread(providers, channel, model):
            row = configured.setdefault(provider.fingerprint, {"resource": provider.fingerprint, "channels": [],
                                       "fallback": provider.fallback, "managed": provider.fingerprint in managed,
                                       **slots.get(provider.fingerprint, {})})
            row["channels"].append(channel)
    snapshot["slots"] = list(configured.values())
    return {"enabled": settings.model_pool_enabled, "coordination": "single-host-multi-process",
            "policy": policy.model_dump(), "turn_timeout": settings.model_pool_turn_timeout,
            "replay_ttl": settings.model_pool_replay_ttl, **snapshot}


def require_local_key_admin(request):
    # Existing demo admin has no login. Do not expose credential mutation to LAN or cross-site forms.
    try:
        local = ipaddress.ip_address(request.client.host).is_loopback
    except (ValueError, AttributeError):
        local = False
    if not local:
        raise HTTPException(403, "请在运行服务的本机管理 Key")
    origin = request.headers.get("origin")
    if origin:
        parsed = urlsplit(origin)
        if parsed.scheme != request.url.scheme or parsed.netloc != request.url.netloc:
            raise HTTPException(403, "不允许跨站修改 Key")


@router.post("/keys")
async def add_keys(request: Request):
    require_local_key_admin(request)
    # Deliberately avoid validation responses containing Pydantic's original secret input.
    if request.headers.get("content-type", "").split(";")[0] != "application/json":
        raise HTTPException(415, "请使用 JSON 提交")
    raw = await request.body()
    if len(raw) > 65536:
        raise HTTPException(400, "单次最多添加 50 个 Key")
    try:
        import json
        body = json.loads(raw)
        keys, channels = body["keys"], body["channels"]
        if (not isinstance(keys, list) or not 1 <= len(keys) <= 50
                or any(not isinstance(k, str) or not 8 <= len(k.strip()) <= 4096
                       or any(c.isspace() for c in k.strip()) for k in keys)
                or not isinstance(channels, list) or not channels
                or any(ch not in ("chat", "embed", "rerank") for ch in channels)):
            raise ValueError()
        fingerprints = await asyncio.to_thread(get_store().add_credentials,
                                                list(dict.fromkeys(k.strip() for k in keys)), channels)
    except (ValueError, KeyError, TypeError):
        raise HTTPException(400, "Key 或用途无效；共享 Key 的用途必须属于同一厂商") from None
    return {"saved": True, "resources": fingerprints}


@router.delete("/keys/{fingerprint}")
async def remove_key(fingerprint: str, request: Request):
    require_local_key_admin(request)
    removed = await asyncio.to_thread(get_store().remove_credential, fingerprint)
    if not removed:
        raise HTTPException(404, "此 Key 来自环境配置，请停用，或先在管理页添加后再移除")
    return {"saved": True}


@router.put("/policy")
async def update_policy(policy: Policy):
    await asyncio.to_thread(get_store().set_policy, policy)
    return {"saved": True, "policy": policy.model_dump()}


class SlotControl(BaseModel):
    disabled: bool = False


@router.post("/slots/{fingerprint}")
async def control(fingerprint: str, body: SlotControl):
    allowed = {p.fingerprint for kind, model in (("chat", settings.chat_model), ("embed", settings.embed_model),
                                                ("rerank", settings.rerank_model)) for p in providers(kind, model)}
    if fingerprint not in allowed:
        raise HTTPException(404, "未配置该Key槽位")
    await asyncio.to_thread(get_store().control, fingerprint, body.disabled)
    return {"saved": True}
