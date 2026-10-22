from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL
from sqlalchemy.exc import SQLAlchemyError

from umai.config.settings import Settings


class ConversationRepositoryError(RuntimeError):
    """Raised when a conversation cannot be loaded or saved."""


class ConversationNotFoundError(ConversationRepositoryError):
    """Raised when a conversation does not belong to the given user/site."""


@dataclass(frozen=True)
class Conversation:
    id: int
    uuid: str
    user_id: int
    site_id: int


class ConversationRepository:
    def __init__(
        self,
        engine: Engine,
        ttl_hours: int,
        history_limit: int,
    ):
        self.engine = engine
        self.ttl_hours = ttl_hours
        self.history_limit = history_limit

    @classmethod
    def from_settings(cls, settings: Settings) -> "ConversationRepository":
        settings.validate_database()
        url = URL.create(
            drivername="mysql+pymysql",
            username=settings.db_user,
            password=settings.db_password,
            host=settings.db_host,
            port=settings.db_port,
            database=settings.db_name,
        )
        engine = create_engine(
            url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout,
            pool_recycle=settings.db_pool_recycle,
            pool_pre_ping=True,
        )
        return cls(
            engine,
            ttl_hours=settings.umai_conversation_ttl_hours,
            history_limit=settings.umai_conversation_history_limit,
        )

    def get_or_create(
        self,
        conversation_id: str | None,
        user_id: int,
        site_id: int,
    ) -> Conversation:
        if conversation_id:
            return self.get(conversation_id, user_id, site_id)
        return self.create(user_id, site_id)

    def create(self, user_id: int, site_id: int) -> Conversation:
        conversation_uuid = str(uuid4())
        now = _utcnow()
        expires_at = now + timedelta(hours=self.ttl_hours)
        query = text(
            """
            INSERT INTO ai_conversations
                (uuid, user_id, site_id, expires_at, created_at, updated_at)
            VALUES
                (:uuid, :user_id, :site_id, :expires_at, :now, :now)
            """
        )
        try:
            with self.engine.begin() as connection:
                result = connection.execute(
                    query,
                    {
                        "uuid": conversation_uuid,
                        "user_id": user_id,
                        "site_id": site_id,
                        "expires_at": expires_at,
                        "now": now,
                    },
                )
                conversation_pk = int(result.lastrowid)
        except SQLAlchemyError as exc:
            raise ConversationRepositoryError(
                f"Could not create conversation: {exc}"
            ) from exc

        return Conversation(
            id=conversation_pk,
            uuid=conversation_uuid,
            user_id=user_id,
            site_id=site_id,
        )

    def get(self, conversation_id: str, user_id: int, site_id: int) -> Conversation:
        try:
            conversation_uuid = str(UUID(conversation_id))
        except ValueError as exc:
            raise ConversationNotFoundError("Conversation not found.") from exc

        query = text(
            """
            SELECT id, uuid, user_id, site_id
            FROM ai_conversations
            WHERE uuid = :uuid
                AND user_id = :user_id
                AND site_id = :site_id
                AND (expires_at IS NULL OR expires_at > :now)
            LIMIT 1
            """
        )
        try:
            with self.engine.connect() as connection:
                row = (
                    connection.execute(
                        query,
                        {
                            "uuid": conversation_uuid,
                            "user_id": user_id,
                            "site_id": site_id,
                            "now": _utcnow(),
                        },
                    )
                    .mappings()
                    .first()
                )
        except SQLAlchemyError as exc:
            raise ConversationRepositoryError(
                f"Could not load conversation: {exc}"
            ) from exc

        if row is None:
            raise ConversationNotFoundError("Conversation not found.")

        return Conversation(
            id=int(row["id"]),
            uuid=str(row["uuid"]),
            user_id=int(row["user_id"]),
            site_id=int(row["site_id"]),
        )

    def get_shared(self, conversation_id: str, site_id: int) -> Conversation:
        """Load a site-scoped persistent troubleshooting conversation."""
        try:
            conversation_uuid = str(UUID(conversation_id))
        except ValueError as exc:
            raise ConversationNotFoundError("Conversation not found.") from exc
        with self.engine.connect() as connection:
            row = connection.execute(text("SELECT id, uuid, user_id, site_id FROM ai_conversations WHERE uuid=:uuid AND site_id=:site_id LIMIT 1"), {"uuid": conversation_uuid, "site_id": site_id}).mappings().first()
        if row is None:
            raise ConversationNotFoundError("Conversation not found.")
        return Conversation(id=int(row["id"]), uuid=str(row["uuid"]), user_id=int(row["user_id"]), site_id=int(row["site_id"]))

    def load_history(self, conversation: Conversation) -> list[dict[str, str]]:
        query = text(
            """
            SELECT role, content
            FROM ai_messages
            WHERE ai_conversation_id = :conversation_id
            ORDER BY created_at DESC, id DESC
            LIMIT :limit
            """
        )
        try:
            with self.engine.connect() as connection:
                rows = [
                    dict(row)
                    for row in connection.execute(
                        query,
                        {
                            "conversation_id": conversation.id,
                            "limit": self.history_limit,
                        },
                    ).mappings()
                ]
        except SQLAlchemyError as exc:
            raise ConversationRepositoryError(
                f"Could not load history: {exc}"
            ) from exc

        rows.reverse()
        return [
            {"role": str(row["role"]), "content": str(row["content"])}
            for row in rows
        ]

    def append_message(
        self,
        conversation: Conversation,
        role: str,
        content: str,
        reasoning_effort: str | None = None,
        info: dict[str, Any] | None = None,
    ) -> int:
        if role not in {"user", "assistant"}:
            raise ConversationRepositoryError(f"Unsupported message role: {role}")
        if role != "assistant":
            reasoning_effort = None
            info = None

        now = _utcnow()
        expires_at = now + timedelta(hours=self.ttl_hours)
        try:
            with self.engine.begin() as connection:
                result = connection.execute(
                    text(
                        """
                        INSERT INTO ai_messages
                            (
                                ai_conversation_id,
                                role,
                                content,
                                reasoning_effort,
                                info,
                                created_at,
                                updated_at
                            )
                        VALUES
                            (
                                :conversation_id,
                                :role,
                                :content,
                                :reasoning_effort,
                                :info,
                                :now,
                                :now
                            )
                        """
                    ),
                    {
                        "conversation_id": conversation.id,
                        "role": role,
                        "content": content,
                        "reasoning_effort": reasoning_effort,
                        "info": _json_dumps(info) if info is not None else None,
                        "now": now,
                    },
                )
                message_id = int(result.lastrowid)
                connection.execute(
                    text(
                        """
                        UPDATE ai_conversations
                        SET updated_at = :now,
                            expires_at = CASE WHEN expires_at IS NULL THEN NULL ELSE :expires_at END
                        WHERE id = :conversation_id
                        """
                    ),
                    {
                        "conversation_id": conversation.id,
                        "now": now,
                        "expires_at": expires_at,
                    },
                )
        except SQLAlchemyError as exc:
            raise ConversationRepositoryError(
                f"Could not append message: {exc}"
            ) from exc

        return message_id


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False, separators=(",", ":"))
