from datetime import datetime
from uuid import uuid4

from qqbot.schema import (
    ContextMessage,
    ContextRole,
    Conversation,
    MessageReceived,
    Session,
)

from .store import InMemoryContextStore


class ContextManager:
    def __init__(self, store: InMemoryContextStore) -> None:
        self.store = store

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

        session.recent_messages.append(context_message)

        session.updated_at = message.timestamp

        conversation.updated_at = message.timestamp

        self.store.save_session(session)

        self.store.save_conversation(conversation)

        return conversation, session, context_message

    def record_assistant_message(self, *, conversation: Conversation, session: Session, content: str, timestamp: datetime) -> ContextMessage:
        context_message = ContextMessage(
            id=uuid4().hex,
            role=ContextRole.ASSISTANT,
            content=content,
            timestamp=timestamp,
        )

        session.recent_messages.append(context_message)

        session.updated_at = timestamp

        conversation.updated_at = timestamp

        self.store.save_session(session)

        self.store.save_conversation(conversation)

        return context_message

    def save_session(self, session: Session) -> None:
        """
        Compactor 修改 Session 后，
        通过 ContextManager 写回 Store。
        """

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
