from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI

from qqbot.schema import (
    ContextMessage,
    ContextRole,
    Session,
    SessionSummary,
)

from .token_counter import TokenCounter


class ContextCompactor:
    """
    Context 压缩器。

    当前第一版只实现：

        Recent Messages
              ↓
        Session Summary

    暂时不处理：
        - Long-term Memory
        - Tool Result
        - Session Summary 二次压缩
    """

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        token_counter: TokenCounter,
        prompt_path: Path,
        max_output_tokens: int,
        min_recent_messages: int = 6,
    ) -> None:
        self.client = client
        self.model = model

        self.token_counter = token_counter

        self.max_output_tokens = max_output_tokens

        self.min_recent_messages = min_recent_messages

        self.system_prompt = prompt_path.read_text(encoding="utf-8").strip()

    async def compact_session(
        self,
        session: Session,
        *,
        current_turn_id: str,
        overflow_tokens: int,
        max_source_tokens: int,
    ) -> bool:
        """
        压缩 Session 中最老的一批消息。

        返回：
            True:
                确实发生了压缩

            False:
                已经没有可以继续压缩的消息

        current_turn 永远不会被压缩。
        """

        candidates = [message for message in session.recent_messages if (message.id != current_turn_id and not message.recalled)]

        # 始终尽量保留最近的一批原始消息。
        if len(candidates) <= self.min_recent_messages:
            return False

        compressible = candidates[: -self.min_recent_messages]

        batch = self._select_batch(
            compressible,
            overflow_tokens=overflow_tokens,
            max_source_tokens=(max_source_tokens),
        )

        if not batch:
            return False

        new_summary = await self._summarize(
            old_summary=session.summary,
            messages=batch,
        )

        compressed_ids = {message.id for message in batch}

        # 真正删除已经进入 Summary 的旧消息。
        session.recent_messages = [message for message in session.recent_messages if message.id not in compressed_ids]

        new_summary.compressed_message_count = session.summary.compressed_message_count + len(batch)

        new_summary.updated_at = datetime.now(tz=session.updated_at.tzinfo)

        session.summary = new_summary

        return True

    def _select_batch(
        self,
        messages: list[ContextMessage],
        *,
        overflow_tokens: int,
        max_source_tokens: int,
    ) -> list[ContextMessage]:
        """
        从最老的消息开始选择一批。

        目标：
        只压缩解决当前 overflow 所需的消息，
        而不是一次把整个 Session 压掉。
        """

        selected: list[ContextMessage] = []

        selected_tokens = 0

        target_tokens = max(
            1,
            overflow_tokens,
        )

        for message in messages:
            rendered = self._render_message(message)

            message_tokens = self.token_counter.count_text(rendered)

            # 已经选了一些消息，
            # 再加入这一条会让压缩请求本身过大，
            # 就先停止。
            if selected and selected_tokens + message_tokens > max_source_tokens:
                break

            selected.append(message)

            selected_tokens += message_tokens

            # 已经覆盖当前 overflow，
            # 就不继续无意义地压更多历史。
            if selected_tokens >= target_tokens:
                break

        return selected

    async def _summarize(
        self,
        *,
        old_summary: SessionSummary,
        messages: list[ContextMessage],
    ) -> SessionSummary:
        """
        使用模型：

        旧 SessionSummary
        +
        新的一批历史消息

        → 新 SessionSummary
        """

        old_summary_data = {
            "narrative": (old_summary.narrative),
            "goals": old_summary.goals,
            "facts": old_summary.facts,
            "decisions": (old_summary.decisions),
            "unresolved_questions": (old_summary.unresolved_questions),
            "important_entities": (old_summary.important_entities),
        }

        rendered_messages = "\n".join(self._render_message(message) for message in messages)

        user_prompt = (
            "下面是当前已有的 Session Summary：\n\n"
            + json.dumps(
                old_summary_data,
                ensure_ascii=False,
                indent=2,
            )
            + "\n\n"
            + "下面是新的旧消息，这些消息即将从 "
            "Recent Messages 中删除：\n\n" + rendered_messages + "\n\n" + "请把两部分合并成新的 Session Summary。"
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (self.system_prompt),
                },
                {
                    "role": "user",
                    "content": (user_prompt),
                },
            ],
            max_tokens=(self.max_output_tokens),
        )

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError("Session Compactor 没有返回内容")

        data = self._parse_json(content)

        return SessionSummary(
            narrative=self._as_string(data.get("narrative")),
            goals=self._as_string_list(data.get("goals")),
            facts=self._as_string_list(data.get("facts")),
            decisions=(self._as_string_list(data.get("decisions"))),
            unresolved_questions=(self._as_string_list(data.get("unresolved_questions"))),
            important_entities=(self._as_string_list(data.get("important_entities"))),
        )

    @staticmethod
    def _render_message(
        message: ContextMessage,
    ) -> str:
        timestamp = message.timestamp.strftime("%Y-%m-%d %H:%M:%S")

        if message.role is ContextRole.ASSISTANT:
            speaker = "assistant"

        else:
            speaker = message.sender_name or str(message.sender_id or "unknown")

        return f"[{timestamp}] [{speaker}] {message.content}"

    @staticmethod
    def _parse_json(
        content: str,
    ) -> dict:
        """
        模型偶尔可能返回 ```json ... ```，
        这里仅做最小清理。
        """

        text = content.strip()

        if text.startswith("```"):
            lines = text.splitlines()

            if lines:
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            text = "\n".join(lines).strip()

        try:
            data = json.loads(text)

        except json.JSONDecodeError as exc:
            raise RuntimeError("Session Compactor 返回的不是合法 JSON") from exc

        if not isinstance(data, dict):
            raise RuntimeError("Session Compactor 必须返回 JSON Object")

        return data

    @staticmethod
    def _as_string(
        value: object,
    ) -> str:
        if isinstance(value, str):
            return value.strip()

        return ""

    @staticmethod
    def _as_string_list(
        value: object,
    ) -> list[str]:
        if not isinstance(
            value,
            list,
        ):
            return []

        return [item.strip() for item in value if (isinstance(item, str) and item.strip())]
