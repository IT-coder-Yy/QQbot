from datetime import datetime
from uuid import uuid4

from qqbot.schema import (
    ContextMessage,
    ContextRole,
    Conversation,
    MessageReceived,
    Session,
)
from qqbot.settings import Settings

from .store import InMemoryContextStore


class ContextManager:
    def __init__(self, store: InMemoryContextStore, *, settings: Settings | None = None) -> None:
        self.store = store
        self.settings = settings or Settings()
        if min(self.settings.context_max_messages, self.settings.context_max_chars) <= 0:
            raise ValueError("上下文容量必须大于 0")

    def get_or_create_context(self, message: MessageReceived) -> tuple[Conversation, Session]:
        conversation = self.store.get_conversation(message.conversation_id)

        if conversation is None:
            conversation = Conversation(
                conversation_id=message.conversation_id,
                conversation_type=message.conversation_type,
                created_at=message.timestamp,
                updated_at=message.timestamp,
            )

            self.store.save_conversation(conversation)

        session = None

        if conversation.active_session_id:
            session = self.store.get_session(conversation.active_session_id)

        if session is None:
            session = self._create_session(
                conversation=conversation,
                timestamp=message.timestamp,
            )

        return (
            conversation,
            session,
        )

    def record_user_message(self, message: MessageReceived) -> tuple[Conversation, Session, ContextMessage]:
        conversation, session = self.get_or_create_context(message)

        context_message = ContextMessage(
            id=uuid4().hex,
            role=ContextRole.USER,
            content=message.text,
            timestamp=message.timestamp,
            sender_id=message.sender.id,
            sender_name=message.sender.display_name,
            source_message_id=message.message_id,
        )

        self._append_message(session, context_message)

        session.updated_at = message.timestamp

        conversation.updated_at = message.timestamp

        self.save_session(session)

        self.store.save_conversation(conversation)

        return conversation, session, context_message

    def record_assistant_message(self, *, conversation: Conversation, session: Session, content: str, timestamp: datetime, source_message_id: int | None = None) -> ContextMessage:
        context_message = ContextMessage(
            id=uuid4().hex,
            role=ContextRole.ASSISTANT,
            content=content,
            timestamp=timestamp,
            source_message_id=source_message_id,
        )

        self._append_message(session, context_message)

        session.updated_at = timestamp

        conversation.updated_at = timestamp

        self.save_session(session)

        self.store.save_conversation(conversation)

        return context_message

    @staticmethod
    def _message_chars(message: ContextMessage) -> int:
        return len(message.content) + len(message.sender_name or "")

    def _append_message(self, session: Session, message: ContextMessage) -> None:
        # 超长消息仍完整传给本轮 Agent/Jev，但不长期保留，也不清空已有历史。
        if self._message_chars(message) <= self.settings.context_max_chars:
            session.recent_messages.append(message)

    def save_session(self, session: Session) -> None:
        """
        Compactor 修改 Session 后，
        通过 ContextManager 写回 Store。
        """

        chars = 0
        keep = 0
        for message in reversed(session.recent_messages[-self.settings.context_max_messages :]):
            chars += self._message_chars(message)
            if chars > self.settings.context_max_chars:
                break
            keep += 1
        del session.recent_messages[: len(session.recent_messages) - keep]
        self.store.save_session(session)

    def _create_session(self, *, conversation: Conversation, timestamp: datetime) -> Session:
        session = Session(
            session_id=uuid4().hex,
            conversation_id=conversation.conversation_id,
            created_at=timestamp,
            updated_at=timestamp,
        )

        conversation.active_session_id = session.session_id

        self.store.save_session(session)

        self.store.save_conversation(conversation)

        return session
