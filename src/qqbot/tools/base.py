from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, cast

from openai.types.chat import ChatCompletionToolParam
from pydantic import BaseModel, ConfigDict, ValidationError


class ToolArgs(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )


ArgsT = TypeVar(
    "ArgsT",
    bound=ToolArgs,
)


class BaseTool(
    ABC,
    Generic[ArgsT],
):
    """
    所有 Agent Tool 的统一抽象。
    """

    name: str
    description: str
    args_model: type[ArgsT]

    @property
    def definition(self) -> ChatCompletionToolParam:
        """
        转换成 OpenAI / DeepSeek
        function calling schema。
        """

        return cast(
            ChatCompletionToolParam,
            {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": self.args_model.model_json_schema(),
                },
            },
        )

    async def execute(self, raw_arguments: str) -> str:
        """
        统一完成：

        JSON 解析
        ↓
        Pydantic 校验
        ↓
        Tool 执行
        ↓
        返回字符串 observation
        """

        try:
            data = json.loads(raw_arguments or "{}")

        except json.JSONDecodeError:
            return json.dumps(
                {
                    "error": ("invalid_arguments"),
                    "message": ("工具参数不是合法 JSON"),
                },
                ensure_ascii=False,
            )

        if not isinstance(data, dict):
            return json.dumps(
                {
                    "error": ("invalid_arguments"),
                    "message": ("工具参数必须是 JSON Object"),
                },
                ensure_ascii=False,
            )

        try:
            args = self.args_model.model_validate(data)

        except ValidationError as exc:
            return json.dumps(
                {
                    "error": ("validation_error"),
                    "message": str(exc),
                },
                ensure_ascii=False,
            )

        try:
            return await self.run(args)

        except Exception as exc:
            return json.dumps(
                {
                    "error": ("tool_execution_failed"),
                    "tool": self.name,
                    "message": str(exc),
                },
                ensure_ascii=False,
            )

    @abstractmethod
    async def run(self, args: ArgsT) -> str:
        raise NotImplementedError
