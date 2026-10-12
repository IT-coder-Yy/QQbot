from __future__ import annotations

from pathlib import Path
from typing import Any

from typesafe_sdk import (
    AsyncTypeSafeClient,
    Noul,
)

from .schema import (
    ContextMessage,
    ContextRole,
    IntentResult,
    Session,
)


PROMPTS_DIR = Path(__file__).parent / "prompts"

INTENT_PROMPT_PATH = PROMPTS_DIR / "intent.md"


class IntentDetector:
    """
    Jev 群聊回复意图检测器。

    不维护自己的 Context Store。

    它直接复用：

        Session.recent_messages

    只是在调用 Jev 时，
    从中抽取最近一小段消息。
    """

    def __init__(
        self,
        *,
        client: AsyncTypeSafeClient,
        reply_threshold: float,
        recent_message_limit: int = 12,
    ) -> None:
        if not 0.0 <= reply_threshold <= 1.0:
            raise ValueError("reply_threshold 必须在 0~1 之间")

        if recent_message_limit <= 0:
            raise ValueError("recent_message_limit 必须大于 0")

        self.client = client

        self.reply_threshold = reply_threshold

        self.recent_message_limit = recent_message_limit

        self.instructions = INTENT_PROMPT_PATH.read_text(encoding="utf-8").strip()

    async def detect(
        self,
        *,
        session: Session,
        current_turn: ContextMessage,
    ) -> IntentResult:
        state = self._build_state(
            session=session,
            current_turn=current_turn,
        )

        response = await self.client.system_one(
            state=state,
            questions={
                "should_reply": Noul(
                    instructions=(self.instructions),
                ),
            },
        )

        probability = float(response.nouls["should_reply"].noul)

        model = getattr(
            response,
            "model",
            None,
        )

        return IntentResult(
            should_reply=(probability >= self.reply_threshold),
            probability=probability,
            model=model,
        )

    def _build_state(
        self,
        *,
        session: Session,
        current_turn: ContextMessage,
    ) -> dict[str, Any]:
        """
        从 Agent 已经使用的 Session Context 中，
        截取最近消息给 Jev。

        current_turn 单独提供，
        避免在 recent_messages 中重复。
        """

        history = [message for message in session.recent_messages if (message.id != current_turn.id and not message.recalled)]

        recent_messages = history[-self.recent_message_limit :]

        return {
            "context": ("This is a QQ group chat. The AI bot is one participant in the group."),
            "recent_messages": [self._render_message(message) for message in recent_messages],
            "current_message": (self._render_message(current_turn)),
        }

    @staticmethod
    def _render_message(
        message: ContextMessage,
    ) -> dict[str, str]:
        if message.role is ContextRole.ASSISTANT:
            return {
                "speaker": "AI bot",
                "role": "assistant",
                "text": message.content,
            }

        sender = message.sender_name or (str(message.sender_id) if message.sender_id is not None else "unknown")

        return {
            "speaker": sender,
            "role": "user",
            "text": message.content,
        }
