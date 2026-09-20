import asyncio
from typing import Literal, TypedDict


class HistoryMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class ConversationStore:
    def __init__(
        self,
        max_messages: int = 20,
    ) -> None:
        self.max_messages = max_messages

        self._histories: dict[
            str,
            list[HistoryMessage],
        ] = {}

        self._locks: dict[
            str,
            asyncio.Lock,
        ] = {}

    def get(
        self,
        conversation_id: str,
    ) -> list[HistoryMessage]:
        return list(
            self._histories.get(
                conversation_id,
                [],
            )
        )

    def add_round(
        self,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        history = self._histories.setdefault(
            conversation_id,
            [],
        )

        history.extend(
            [
                {
                    "role": "user",
                    "content": user_message,
                },
                {
                    "role": "assistant",
                    "content": assistant_message,
                },
            ]
        )

        if len(history) > self.max_messages:
            self._histories[conversation_id] = (
                history[-self.max_messages :]
            )

    def clear(
        self,
        conversation_id: str,
    ) -> None:
        self._histories.pop(
            conversation_id,
            None,
        )

    def lock(
        self,
        conversation_id: str,
    ) -> asyncio.Lock:
        return self._locks.setdefault(
            conversation_id,
            asyncio.Lock(),
        )