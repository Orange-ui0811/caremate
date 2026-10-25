"""httpx adapters preserve LangChain tools/SSE while governing actual HTTP calls.

Retries happen only before a successful response body is exposed. A partially
delivered stream is never restarted. SDK retries must be disabled by the caller.
"""
import asyncio
from contextlib import suppress
from email.utils import parsedate_to_datetime
import json
import threading
import time
import uuid
import weakref

import httpx

from app.core import model_pool as pool


def retry_after(response):
    value = response.headers.get("retry-after", "1")
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        except (ValueError, TypeError, OverflowError):
            seconds = 1
    return max(1, min(seconds, 86400))


def outbound(request, provider, channel, body, policy, remaining):
    payload = json.loads(body)
    if provider.model:
        payload["model"] = provider.model
    headers = dict(request.headers)
    for key in ("host", "content-length", "authorization"):
        headers.pop(key, None)
    headers["authorization"] = "Bearer " + provider.api_key
    # A configured fallback is explicit; vendor-side invisible model routing stays off.
    endpoint = {"chat": "chat/completions", "embed": "embeddings", "rerank": "rerank"}[channel]
    extensions = dict(request.extensions)
    extensions["timeout"] = {"connect": min(5, remaining), "read": min(policy.read_timeout, remaining),
                             "write": min(5, remaining), "pool": min(policy.queue_timeout, remaining)}
    return httpx.Request(request.method, provider.base_url.rstrip("/") + "/" + endpoint,
                         headers=headers, content=json.dumps(payload).encode(), extensions=extensions)


def transient(status):
    return status in {401, 403, 408, 429, 500, 502, 503, 504}


def mark_failure(store, provider, response):
    if response.status_code in {401, 403}:
        store.mark(provider.fingerprint, invalid=True)
    elif response.status_code == 429:
        store.mark(provider.fingerprint, cooldown=retry_after(response))
    elif response.status_code >= 500:
        store.mark(provider.fingerprint, cooldown=.5)


class AsyncBody(httpx.AsyncByteStream):
    def __init__(self, response, store, token, deadline):
        self.response, self.store, self.token, self.deadline = response, store, token, deadline
        self.closed = False
        self.completed, self.tail = False, b"\n"

    async def finish(self, outcome):
        if self.closed:
            return
        self.closed = True
        if outcome == "CANCELLED" and self.completed:
            outcome = "SUCCEEDED"
        try:
            await self.response.aclose()
        finally:
            await asyncio.shield(asyncio.to_thread(self.store.finish, self.token, outcome, self.response.status_code))

    async def __aiter__(self):
        try:
            async with asyncio.timeout(max(.001, self.deadline - time.monotonic())):
                async for chunk in self.response.stream:
                    self.completed = self.completed or b"\ndata: [DONE]" in self.tail + chunk
                    self.tail = (self.tail + chunk)[-32:]
                    yield chunk
        except (TimeoutError, httpx.TimeoutException):
            await self.finish("TIMEOUT")
            raise httpx.ReadTimeout("模型调用超过时间预算") from None
        except asyncio.CancelledError:
            await self.finish("CANCELLED")
            raise
        except BaseException:
            await self.finish("FAILED")
            raise
        else:
            await self.finish("SUCCEEDED" if self.response.status_code < 400 else "FAILED")

    async def aclose(self):
        await self.finish("CANCELLED")


class AsyncPoolTransport(httpx.AsyncBaseTransport):
    def __init__(self, channel="chat", inner=None):
        self.channel = channel
        self.inner = inner or httpx.AsyncHTTPTransport(retries=0, limits=httpx.Limits(max_connections=512, max_keepalive_connections=64))

    async def handle_async_request(self, request):
        store = pool.get_store()
        policy = await asyncio.to_thread(store.policy)
        body = await request.aread()
        model = json.loads(body).get("model", "")
        choices = await asyncio.to_thread(pool.providers, self.channel, model)
        trace, priority = pool.request_id(), pool.priority()
        # Queueing, retries and reading all share this request budget.
        deadline = time.monotonic() + policy.call_timeout
        for attempt in range(1, policy.max_attempts + 1):
            token, queued_at = uuid.uuid4().hex, time.time()
            try:
                registration = asyncio.create_task(asyncio.to_thread(store.enqueue, token, priority, policy))
                try:
                    await asyncio.shield(registration)
                except asyncio.CancelledError:
                    await asyncio.shield(registration)
                    raise
                queue_deadline = min(deadline, time.monotonic() + policy.queue_timeout)
                while True:
                    acquisition = asyncio.create_task(asyncio.to_thread(store.acquire, token, choices, policy, self.channel,
                                                                         model, priority, trace, attempt, queued_at))
                    try:
                        spec = await asyncio.shield(acquisition)
                    except asyncio.CancelledError:
                        await asyncio.shield(acquisition)
                        await asyncio.shield(asyncio.to_thread(store.finish, token, "CANCELLED"))
                        raise
                    if spec:
                        break
                    if time.monotonic() >= queue_deadline:
                        raise pool.PoolBusy("模型服务繁忙，等待已超时")
                    await asyncio.sleep(.025)
            except pool.PoolBusy:
                await asyncio.to_thread(store.reject, token, trace, self.channel, model, priority, attempt, queued_at)
                raise
            finally:
                await asyncio.shield(asyncio.to_thread(store.dequeue, token))
            try:
                remaining = max(.001, deadline - time.monotonic())
                response = await asyncio.wait_for(self.inner.handle_async_request(
                    outbound(request, spec, self.channel, body, policy, remaining)), remaining)
            except BaseException as exc:
                outcome = "CANCELLED" if isinstance(exc, asyncio.CancelledError) else "TIMEOUT" if isinstance(exc, (TimeoutError, httpx.TimeoutException)) else "FAILED"
                await asyncio.shield(asyncio.to_thread(store.finish, token, outcome))
                if (isinstance(exc, (httpx.TransportError, TimeoutError)) and attempt < policy.max_attempts
                        and time.monotonic() + .1 < deadline):
                    continue
                raise
            if transient(response.status_code):
                await asyncio.to_thread(mark_failure, store, spec, response)
                if attempt < policy.max_attempts and time.monotonic() + .1 < deadline:
                    await response.aclose()
                    await asyncio.to_thread(store.finish, token, "RETRY", response.status_code)
                    continue
            return httpx.Response(response.status_code, headers=response.headers,
                                  stream=AsyncBody(response, store, token, deadline), extensions=response.extensions)

    async def aclose(self):
        await self.inner.aclose()


class SyncBody(httpx.SyncByteStream):
    def __init__(self, response, store, token, deadline):
        self.response, self.store, self.token, self.deadline = response, store, token, deadline
        self.closed = False
        self.completed, self.tail = False, b"\n"

    def finish(self, outcome):
        if not self.closed:
            self.closed = True
            if outcome == "CANCELLED" and self.completed:
                outcome = "SUCCEEDED"
            try:
                self.response.close()
            finally:
                self.store.finish(self.token, outcome, self.response.status_code)

    def __iter__(self):
        try:
            for chunk in self.response.stream:
                if time.monotonic() >= self.deadline:
                    raise httpx.ReadTimeout("模型调用超过时间预算")
                self.completed = self.completed or b"\ndata: [DONE]" in self.tail + chunk
                self.tail = (self.tail + chunk)[-32:]
                yield chunk
        except httpx.TimeoutException:
            self.finish("TIMEOUT")
            raise
        except BaseException:
            self.finish("FAILED")
            raise
        else:
            self.finish("SUCCEEDED" if self.response.status_code < 400 else "FAILED")

    def close(self):
        self.finish("CANCELLED")


class SyncPoolTransport(httpx.BaseTransport):
    def __init__(self, channel="chat", inner=None):
        self.channel = channel
        self.inner = inner or httpx.HTTPTransport(retries=0)

    def handle_request(self, request):
        store = pool.get_store()
        policy = store.policy()
        body = request.read()
        model = json.loads(body).get("model", "")
        choices, trace, priority = pool.providers(self.channel, model), pool.request_id(), pool.priority()
        deadline = time.monotonic() + policy.call_timeout
        for attempt in range(1, policy.max_attempts + 1):
            token, queued_at = uuid.uuid4().hex, time.time()
            try:
                store.enqueue(token, priority, policy)
                queue_deadline = min(deadline, time.monotonic()+policy.queue_timeout)
                while True:
                    spec = store.acquire(token, choices, policy, self.channel, model, priority, trace, attempt, queued_at)
                    if spec:
                        break
                    if time.monotonic() >= queue_deadline:
                        raise pool.PoolBusy("模型服务繁忙，等待已超时")
                    time.sleep(.025)
            except pool.PoolBusy:
                store.reject(token, trace, self.channel, model, priority, attempt, queued_at)
                raise
            finally:
                store.dequeue(token)
            try:
                response = self.inner.handle_request(outbound(request, spec, self.channel, body, policy, max(.001, deadline-time.monotonic())))
            except BaseException as exc:
                store.finish(token, "TIMEOUT" if isinstance(exc, httpx.TimeoutException) else "FAILED")
                if isinstance(exc, httpx.TransportError) and attempt < policy.max_attempts and time.monotonic() < deadline:
                    continue
                raise
            if transient(response.status_code):
                mark_failure(store, spec, response)
                if attempt < policy.max_attempts and time.monotonic() < deadline:
                    response.close()
                    store.finish(token, "RETRY", response.status_code)
                    continue
            return httpx.Response(response.status_code, headers=response.headers,
                                  stream=SyncBody(response, store, token, deadline), extensions=response.extensions)

    def close(self):
        self.inner.close()


_SYNC = {}
_ASYNC = weakref.WeakKeyDictionary()
_LOCK = threading.Lock()


def clients(channel="chat"):
    with _LOCK:
        sync = _SYNC.get(channel)
        if sync is None or sync.is_closed:
            sync = _SYNC[channel] = httpx.Client(transport=SyncPoolTransport(channel))
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # A model built in synchronous code may be awaited later on any loop.
            return sync, httpx.AsyncClient(transport=AsyncPoolTransport(channel))
        by_channel = _ASYNC.setdefault(loop, {})
        client = by_channel.get(channel)
        if client is None or client.is_closed:
            client = by_channel[channel] = httpx.AsyncClient(transport=AsyncPoolTransport(channel))
        return sync, client


async def close_clients():
    loop = asyncio.get_running_loop()
    with _LOCK:
        current = _ASYNC.pop(loop, {})
        sync = list(_SYNC.values())
        _SYNC.clear()
    for client in current.values():
        with suppress(Exception):
            await client.aclose()
    for client in sync:
        await asyncio.to_thread(client.close)
