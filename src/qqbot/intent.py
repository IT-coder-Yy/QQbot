from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from time import monotonic
from typing import Any

from typesafe_sdk import (
    AsyncTypeSafeClient,
    Noul,
)

from .schema import (
    ContextMessage,
    IntentResult,
    Session,
    MessageReceived,
)
from .chat_context import DEFAULT_BOT_NAMES, build_intent_state, direct_reply_decision

logger = logging.getLogger(__name__)


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
        bot_names: tuple[str, ...] = DEFAULT_BOT_NAMES,
    ) -> None:
        if not 0.0 <= reply_threshold <= 1.0:
            raise ValueError("reply_threshold 必须在 0~1 之间")

        if recent_message_limit <= 0:
            raise ValueError("recent_message_limit 必须大于 0")

        self.client = client

        self.reply_threshold = reply_threshold

        self.recent_message_limit = recent_message_limit
        self.bot_names = bot_names

        self.instructions = INTENT_PROMPT_PATH.read_text(encoding="utf-8").strip()

    async def detect(
        self,
        *,
        session: Session,
        current_turn: ContextMessage,
        message: MessageReceived | None = None,
    ) -> IntentResult:
        state = self._build_state(
            session=session,
            current_turn=current_turn,
            message=message,
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
        message: MessageReceived | None = None,
    ) -> dict[str, Any]:
        """
        从 Agent 已经使用的 Session Context 中，
        截取最近消息给 Jev。

        current_turn 单独提供，
        避免在 recent_messages 中重复。
        """

        return build_intent_state(session=session, current_turn=current_turn, message=message,
                                  bot_names=self.bot_names, recent_limit=self.recent_message_limit)

    async def fallback(self, *, client, model: str, session: Session, current_turn: ContextMessage,
                       message: MessageReceived | None = None) -> bool:
        """JEV 故障时复用主模型做一次纯判断，不运行工具，不发送消息。"""
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": self.instructions + '\n只输出 JSON 对象 {"should_reply": true} 或 {"should_reply": false}，不输出其他文字。'},
                {"role": "user", "content": json.dumps(self._build_state(session=session, current_turn=current_turn, message=message), ensure_ascii=False)},
            ],
            max_tokens=128,
        )
        result = json.loads(response.choices[0].message.content or "")
        if not isinstance(result, dict) or type(result.get("should_reply")) is not bool:
            raise ValueError("接话兜底未返回布尔 should_reply")
        return result["should_reply"]


async def should_respond(*, message: MessageReceived, session: Session, current_turn: ContextMessage,
                         detector: IntentDetector | None, agent, timeout: float,
                         fallback_timeout: float = 20.0, store=None) -> bool:
    """规则 → JEV → 仅故障时主模型兜底；外层任务期限与取消始终生效。"""
    if message.reply_to_message_id is not None and getattr(store, "persistent", False):
        if not any(item.source_message_id == message.reply_to_message_id for item in session.recent_messages):
            quoted = next((item for item in reversed(store.history(message.conversation_id)) if item.get("source_message_id") == message.reply_to_message_id), None)
            if quoted is not None:
                session = session.model_copy(update={"recent_messages": [ContextMessage.model_validate(quoted), *session.recent_messages]})
    decision = direct_reply_decision(message, session, bot_names=getattr(detector, "bot_names", DEFAULT_BOT_NAMES))
    if decision is not None:
        logger.info("接话判断 source=rule reply=%s conversation=%s message=%s", decision, message.conversation_id, message.message_id)
        return decision
    if detector is None:
        return False
    started = monotonic()
    try:
        async with asyncio.timeout(timeout):
            result = await detector.detect(session=session, current_turn=current_turn, message=message)
    except Exception as exc:
        logger.warning("接话判断 source=jev error=%s elapsed_ms=%.0f conversation=%s message=%s，转主模型兜底",
                       type(exc).__name__, (monotonic()-started)*1000, message.conversation_id, message.message_id)
    else:
        logger.info("接话判断 source=jev reply=%s probability=%s threshold=%s model=%s elapsed_ms=%.0f conversation=%s message=%s",
                    result.should_reply, getattr(result, "probability", None), getattr(detector, "reply_threshold", None),
                    getattr(result, "model", None), (monotonic()-started)*1000, message.conversation_id, message.message_id)
        return result.should_reply
    started = monotonic()
    try:
        async with asyncio.timeout(fallback_timeout):
            decision = await detector.fallback(client=agent.client, model=agent.model, session=session,
                                               current_turn=current_turn, message=message)
    except Exception as exc:
        logger.warning("接话判断 source=fallback error=%s elapsed_ms=%.0f conversation=%s message=%s，保持沉默",
                       type(exc).__name__, (monotonic()-started)*1000, message.conversation_id, message.message_id)
        return False
    logger.info("接话判断 source=fallback reply=%s model=%s elapsed_ms=%.0f conversation=%s message=%s",
                decision, agent.model, (monotonic()-started)*1000, message.conversation_id, message.message_id)
    return decision
