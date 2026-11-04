"""Small additive migrations for existing local installations."""
from sqlalchemy import inspect, text

from app.db import base


async def ensure_conversation_deletion_column() -> None:
    async with base.engine.begin() as conn:
        def needs_column(sync_conn):
            inspector = inspect(sync_conn)
            return (inspector.has_table("conversations") and
                    "deleted_at" not in {c["name"] for c in inspector.get_columns("conversations")})
        if await conn.run_sync(needs_column):
            await conn.execute(text("ALTER TABLE conversations ADD COLUMN deleted_at DATETIME NULL"))


async def ensure_message_activities_column() -> None:
    async with base.engine.begin() as conn:
        def needs_column(sync_conn):
            inspector = inspect(sync_conn)
            return (inspector.has_table("messages") and
                    "activities" not in {c["name"] for c in inspector.get_columns("messages")})
        if await conn.run_sync(needs_column):
            await conn.execute(text("ALTER TABLE messages ADD COLUMN activities JSON NULL"))
