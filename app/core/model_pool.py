"""Bounded provider admission and durable request replay on one shared local host.

SQLite WAL transactions coordinate API workers and CLI jobs; this is not a
multi-host Redis semaphore. No prompt, credential or exception body is logged.
Only explicitly identified, completed chat requests are replayed.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
from urllib.parse import urlsplit
import uuid

from cryptography.fernet import Fernet
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.config import settings

PRIORITY = ContextVar("model_pool_priority", default="foreground")
REQUEST_ID = ContextVar("model_pool_request_id", default="")


@contextmanager
def call_context(request_id="", priority="foreground"):
    a, b = REQUEST_ID.set(request_id), PRIORITY.set(priority)
    try:
        yield
    finally:
        REQUEST_ID.reset(a)
        PRIORITY.reset(b)


class PoolBusy(RuntimeError):
    pass


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    global_limit: int = Field(default=4, ge=1, le=512)
    key_limit: int = Field(default=2, ge=1, le=128)
    background_limit: int = Field(default=1, ge=0, le=511)
    max_queue: int = Field(default=64, ge=1, le=4096)
    queue_timeout: float = Field(default=3, ge=.05, le=120)
    call_timeout: float = Field(default=30, ge=.1, le=600)
    read_timeout: float = Field(default=15, ge=.05, le=300)
    max_attempts: int = Field(default=2, ge=1, le=5)

    @model_validator(mode="after")
    def bounded(self):
        if self.background_limit >= self.global_limit:
            raise ValueError("后台并发必须小于总并发，保留至少一个前台槽位")
        if self.read_timeout > self.call_timeout:
            raise ValueError("读取超时不能大于单次调用总期限")
        return self


@dataclass(frozen=True)
class Provider:
    base_url: str
    api_key: str = field(repr=False)
    model: str = ""
    fallback: bool = False

    @property
    def fingerprint(self):
        host = urlsplit(self.base_url).hostname or ""
        if host in {"api.moark.com", "ai.gitee.com"}:
            host = "gitee"
        return hashlib.sha256((host + "\0" + self.api_key).encode()).hexdigest()[:16]


def environment_providers(channel, model=""):
    if channel == "chat":
        keys = list(dict.fromkeys(k.strip() for k in re.split(r"[,;\n]", settings.model_pool_chat_keys) if k.strip()))
        result = [Provider(settings.chat_base_url, k, model) for k in (keys or [settings.chat_api_key])]
        if (settings.model_pool_fallback_base_url and settings.model_pool_fallback_api_key
                and settings.model_pool_fallback_model and model == settings.chat_model):
            result.append(Provider(settings.model_pool_fallback_base_url,
                                   settings.model_pool_fallback_api_key,
                                   settings.model_pool_fallback_model, True))
        return result
    return [Provider(getattr(settings, channel + "_base_url"),
                     getattr(settings, channel + "_api_key"), model)]


def providers(channel, model=""):
    configured = get_store().credential_providers(channel, model)
    if configured is None:
        return environment_providers(channel, model)
    # Keep explicitly configured cross-provider fallback separate from managed primary keys.
    return configured + [p for p in environment_providers(channel, model) if p.fallback]


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self.connect() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript("""
            CREATE TABLE IF NOT EXISTS policy (id INTEGER PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS leases (
              id TEXT PRIMARY KEY, resource TEXT, priority TEXT, expires REAL,
              request_id TEXT, channel TEXT, model TEXT);
            CREATE TABLE IF NOT EXISTS waiters (id TEXT PRIMARY KEY, priority TEXT, created REAL, expires REAL);
            CREATE TABLE IF NOT EXISTS slots (
              resource TEXT PRIMARY KEY, disabled INTEGER DEFAULT 0, invalid INTEGER DEFAULT 0,
              cooldown REAL DEFAULT 0, last_used REAL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS attempts (
              id TEXT PRIMARY KEY, request_id TEXT, resource TEXT, channel TEXT, model TEXT,
              priority TEXT, attempt INTEGER, started REAL, finished REAL, queue_ms REAL,
              outcome TEXT, http_status INTEGER);
            CREATE INDEX IF NOT EXISTS attempts_started ON attempts(started);
            CREATE INDEX IF NOT EXISTS attempts_finished ON attempts(finished);
            CREATE TABLE IF NOT EXISTS replay (
              id TEXT PRIMARY KEY, digest TEXT, status TEXT, response TEXT, expires REAL);
            CREATE TABLE IF NOT EXISTS turn_locks (id INTEGER PRIMARY KEY, owner TEXT, expires REAL);
            CREATE INDEX IF NOT EXISTS replay_expires ON replay(expires);
            CREATE TABLE IF NOT EXISTS credentials (
              resource TEXT PRIMARY KEY, secret TEXT NOT NULL, channels TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS credential_channels (channel TEXT PRIMARY KEY);
            """)

    def _cipher(self, *, create=False):
        # Separate key file: copying the status DB alone must not expose API keys.
        # Creation runs under the database write transaction (also across processes).
        path = self.path.with_suffix(self.path.suffix + ".key")
        if create and not path.exists():
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(Fernet.generate_key())
        return Fernet(path.read_bytes())

    def credential_providers(self, channel, model):
        with self.connect() as c:
            if not c.execute("SELECT 1 FROM credential_channels WHERE channel=?", (channel,)).fetchone():
                return None
            rows = c.execute("SELECT * FROM credentials ORDER BY resource").fetchall()
        rows = [r for r in rows if channel in json.loads(r["channels"])]
        if not rows:
            return []  # Never silently resurrect an environment key after deletion.
        cipher = self._cipher()
        return [Provider(getattr(settings, channel + "_base_url"), cipher.decrypt(r["secret"].encode()).decode(), model)
                for r in rows]

    def _put_credential(self, c, cipher, api_key, channels):
        # A key can only be shared between channels of the same provider host.
        specs = [Provider(getattr(settings, ch + "_base_url"), api_key) for ch in channels]
        if len({p.fingerprint for p in specs}) != 1:
            raise ValueError("所选用途必须属于同一厂商")
        fp = specs[0].fingerprint
        previous = c.execute("SELECT channels FROM credentials WHERE resource=?", (fp,)).fetchone()
        merged = sorted(set(channels) | (set(json.loads(previous[0])) if previous else set()))
        c.execute("INSERT OR REPLACE INTO credentials VALUES(?,?,?)",
                  (fp, cipher.encrypt(api_key.encode()).decode(), json.dumps(merged)))
        c.executemany("INSERT OR IGNORE INTO credential_channels VALUES(?)", [(ch,) for ch in channels])
        return fp

    def set_credentials(self, entries):
        """Atomic explicit replacement, used by local import and isolated benchmarks."""
        with self.transaction() as c:
            cipher = self._cipher(create=True)
            c.execute("DELETE FROM credentials")
            c.execute("DELETE FROM credential_channels")
            for entry in entries:
                self._put_credential(c, cipher, entry["api_key"], entry["channels"])

    def add_credentials(self, keys, channels):
        with self.transaction() as c:
            cipher = self._cipher(create=True)
            for channel in channels:
                if not c.execute("SELECT 1 FROM credential_channels WHERE channel=?", (channel,)).fetchone():
                    for spec in environment_providers(channel):
                        if not spec.fallback:
                            self._put_credential(c, cipher, spec.api_key, [channel])
            return [self._put_credential(c, cipher, key, channels) for key in keys]

    def remove_credential(self, fingerprint):
        with self.transaction() as c:
            return bool(c.execute("DELETE FROM credentials WHERE resource=?", (fingerprint,)).rowcount)

    def credential_metadata(self):
        with self.connect() as c:
            return {r[0]: json.loads(r[1]) for r in c.execute("SELECT resource,channels FROM credentials")}

    @contextmanager
    def connect(self):
        c = sqlite3.connect(str(self.path), timeout=2, isolation_level=None)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA busy_timeout=2000")
        try:
            yield c
        finally:
            c.close()

    @contextmanager
    def transaction(self):
        # SQLite has one writer: serialize local writers before the DB lock so
        # a burst does not fill the executor with competing BEGIN IMMEDIATEs.
        # Independent processes still coordinate through SQLite's transaction.
        with self._lock, self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            try:
                yield c
                c.commit()
            except BaseException:
                c.rollback()
                raise

    def policy(self):
        with self.connect() as c:
            row = c.execute("SELECT data FROM policy WHERE id=1").fetchone()
        if row:
            return Policy.model_validate_json(row[0])
        default = Policy(**{k: getattr(settings, "model_pool_" + k) for k in Policy.model_fields})
        # First use freezes a shared policy; another worker's environment must
        # not silently apply a different "global" capacity to the same file.
        with self.transaction() as c:
            c.execute("INSERT OR IGNORE INTO policy VALUES(1,?)", (default.model_dump_json(),))
            selected = c.execute("SELECT data FROM policy WHERE id=1").fetchone()[0]
        return Policy.model_validate_json(selected)

    def set_policy(self, policy):
        with self.transaction() as c:
            c.execute("INSERT OR REPLACE INTO policy VALUES(1,?)", (policy.model_dump_json(),))

    @staticmethod
    def cleanup(c, now):
        c.execute("UPDATE attempts SET outcome='EXPIRED',finished=? WHERE outcome='RUNNING' AND id IN (SELECT id FROM leases WHERE expires<=?)", (now, now))
        c.execute("DELETE FROM leases WHERE expires<=?", (now,))
        c.execute("DELETE FROM waiters WHERE expires<=?", (now,))
        c.execute("DELETE FROM attempts WHERE finished<?", (now - 7 * 86400,))
        c.execute("DELETE FROM replay WHERE expires<? AND status!='RUNNING'", (now,))

    def enqueue(self, token, priority, policy):
        now = time.time()
        with self.transaction() as c:
            self.cleanup(c, now)
            if c.execute("SELECT COUNT(*) FROM waiters").fetchone()[0] >= policy.max_queue:
                raise PoolBusy("模型等待队列已满，请稍后重试")
            c.execute("INSERT INTO waiters VALUES(?,?,?,?)", (token, priority, now, now + policy.queue_timeout + 5))

    def dequeue(self, token):
        with self.connect() as c:
            c.execute("DELETE FROM waiters WHERE id=?", (token,))

    def acquire(self, token, choices, policy, channel, model, priority, request_id, attempt, queued_at):
        now = time.time()
        with self.transaction() as c:
            self.cleanup(c, now)
            first = c.execute("SELECT id FROM waiters ORDER BY CASE priority WHEN 'foreground' THEN 0 ELSE 1 END,created,id LIMIT 1").fetchone()
            if not first or first[0] != token:
                return None
            running = c.execute("SELECT COUNT(*) FROM leases").fetchone()[0]
            if running >= policy.global_limit:
                return None
            if priority == "background":
                background = c.execute("SELECT COUNT(*) FROM leases WHERE priority='background'").fetchone()[0]
                if background >= policy.background_limit or running >= policy.global_limit - 1:
                    return None
            candidates = []
            for spec in choices:
                fp = spec.fingerprint
                c.execute("INSERT OR IGNORE INTO slots(resource) VALUES(?)", (fp,))
                state = c.execute("SELECT * FROM slots WHERE resource=?", (fp,)).fetchone()
                active = c.execute("SELECT COUNT(*) FROM leases WHERE resource=?", (fp,)).fetchone()[0]
                if not state["disabled"] and not state["invalid"] and state["cooldown"] <= now and active < policy.key_limit:
                    candidates.append((spec.fallback, active, state["last_used"], fp, spec))
            if not candidates:
                return None
            spec = min(candidates, key=lambda r: r[:4])[-1]
            fp = spec.fingerprint
            # A crashed worker's lease expires; live streams have a finite call deadline.
            c.execute("INSERT INTO leases VALUES(?,?,?,?,?,?,?)", (token, fp, priority, now + policy.call_timeout + 10, request_id, channel, model))
            c.execute("UPDATE slots SET last_used=? WHERE resource=?", (now, fp))
            c.execute("DELETE FROM waiters WHERE id=?", (token,))
            c.execute("INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (token, request_id, fp, channel, spec.model, priority, attempt, now, None, (now-queued_at)*1000, "RUNNING", None))
            return spec

    def finish(self, token, outcome, status=None):
        with self.transaction() as c:
            c.execute("DELETE FROM leases WHERE id=?", (token,))
            c.execute("UPDATE attempts SET outcome=?,http_status=?,finished=? WHERE id=? AND outcome='RUNNING'", (outcome, status, time.time(), token))

    def reject(self, token, trace, channel, model, priority, attempt, queued_at):
        now = time.time()
        with self.transaction() as c:
            c.execute("INSERT OR IGNORE INTO attempts VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                      (token, trace, None, channel, model, priority, attempt, queued_at, now,
                       (now-queued_at)*1000, "REJECTED", 503))

    def mark(self, resource, *, invalid=False, cooldown=0):
        with self.transaction() as c:
            c.execute("UPDATE slots SET invalid=MAX(invalid,?),cooldown=MAX(cooldown,?) WHERE resource=?", (int(invalid), time.time()+cooldown, resource))

    def control(self, resource, disabled=False):
        with self.transaction() as c:
            c.execute("INSERT OR IGNORE INTO slots(resource) VALUES(?)", (resource,))
            c.execute("UPDATE slots SET disabled=?,invalid=0,cooldown=0 WHERE resource=?", (int(disabled), resource))

    def snapshot(self):
        with self.transaction() as c:
            self.cleanup(c, time.time())
            return {
                "running": [dict(r) for r in c.execute("SELECT * FROM leases ORDER BY expires")],
                "waiting": c.execute("SELECT COUNT(*) FROM waiters").fetchone()[0],
                "slots": [dict(r) for r in c.execute("SELECT * FROM slots ORDER BY resource")],
                "recent_calls": [dict(r) for r in c.execute("SELECT * FROM attempts ORDER BY started DESC LIMIT 100")],
                "replay_counts": {r[0]: r[1] for r in c.execute("SELECT status,COUNT(*) FROM replay GROUP BY status")},
            }

    def begin_replay(self, key, digest):
        now = time.time()
        with self.transaction() as c:
            self.cleanup(c, now)
            row = c.execute("SELECT * FROM replay WHERE id=?", (key,)).fetchone()
            if row:
                if row["digest"] != digest:
                    return "CONFLICT", None
                if row["status"] == "RUNNING" and row["expires"] <= now:
                    c.execute("UPDATE replay SET status='INTERRUPTED',expires=? WHERE id=?", (now+settings.model_pool_replay_ttl, key))
                    return "INTERRUPTED", None
                if row["status"] == "REJECTED":
                    c.execute("UPDATE replay SET status='RUNNING',expires=? WHERE id=?",
                              (now+settings.model_pool_turn_timeout+10, key))
                    return "NEW", None
                return row["status"], row["response"]
            c.execute("INSERT INTO replay VALUES(?,?,'RUNNING',NULL,?)", (key, digest, now+settings.model_pool_turn_timeout+10))
            return "NEW", None

    def finish_replay(self, key, status, response=None):
        with self.connect() as c:
            c.execute("UPDATE replay SET status=?,response=?,expires=? WHERE id=? AND status='RUNNING'", (status, response, time.time()+settings.model_pool_replay_ttl, key))

    def acquire_turn(self, conversation_id, owner):
        with self.transaction() as c:
            c.execute("DELETE FROM turn_locks WHERE expires<=?", (time.time(),))
            cursor = c.execute("INSERT OR IGNORE INTO turn_locks VALUES(?,?,?)",
                               (conversation_id, owner, time.time()+settings.model_pool_turn_timeout+10))
            return cursor.rowcount == 1

    def release_turn(self, conversation_id, owner):
        with self.connect() as c:
            c.execute("DELETE FROM turn_locks WHERE id=? AND owner=?", (conversation_id, owner))


@lru_cache(maxsize=8)
def _store(path):
    return Store(path)


def get_store():
    path = Path(settings.model_pool_db_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return _store(str(path))


def priority():
    return "background" if PRIORITY.get() == "background" or os.environ.get("MODEL_POOL_PRIORITY") == "background" else "foreground"


def request_id():
    return REQUEST_ID.get() or uuid.uuid4().hex
