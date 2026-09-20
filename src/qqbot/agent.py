import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from .context import ConversationStore
from .schema import MessageReceived


load_dotenv()


SYSTEM_PROMPT_PATH = (
    Path(__file__).parent
    / "system.md"
)


class Agent:
    def __init__(self) -> None:
        self.api_key = self._get_required_env(
            "LLM_API_KEY"
        )
        self.base_url = self._get_required_env(
            "LLM_BASE_URL"
        )
        self.model = self._get_required_env(
            "LLM_MODEL"
        )

        self.system_prompt = (
            SYSTEM_PROMPT_PATH.read_text(
                encoding="utf-8"
            ).strip()
        )

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        self.context = ConversationStore(
            max_messages=20
        )

    async def run(
        self,
        message: MessageReceived,
    ) -> str:
        conversation_id = (
            message.conversation_id
        )

        async with self.context.lock(
            conversation_id
        ):
            history = self.context.get(
                conversation_id
            )

            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                *history,
                {
                    "role": "user",
                    "content": message.text,
                },
            ]

            response = (
                await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
            )

            content = (
                response
                .choices[0]
                .message
                .content
            )

            if not content:
                return "模型没有返回内容。"

            content = content.strip()

            self.context.add_round(
                conversation_id=conversation_id,
                user_message=message.text,
                assistant_message=content,
            )

            return content

    @staticmethod
    def _get_required_env(
        name: str,
    ) -> str:
        value = os.getenv(name)

        if not value:
            raise RuntimeError(
                f"缺少环境变量: {name}"
            )

        return value