"""Request-local workflow choice; never mutate process-wide settings per user."""
from contextlib import contextmanager
from contextvars import ContextVar

from app.config import settings

_mode = ContextVar('multi_intent_mode', default=None)


def enabled() -> bool:
    value = _mode.get()
    return settings.multi_intent_enabled if value is None else value


@contextmanager
def use(enabled: bool):
    token = _mode.set(enabled)
    try:
        yield
    finally:
        _mode.reset(token)
