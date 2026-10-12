from __future__ import annotations

from openai.types.chat import (
    ChatCompletionToolParam,
)

from .base import BaseTool


class ToolRegistry:
    """
    Agent 可用工具注册中心。
    """

    def __init__(self, tools: list[BaseTool]) -> None:
        self._tools: dict[
            str,
            BaseTool,
        ] = {}

        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"重复的 Tool 名称: {tool.name}")

            self._tools[tool.name] = tool

    @property
    def definitions(self) -> list[ChatCompletionToolParam]:
        return [tool.definition for tool in self._tools.values()]

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())

    async def execute(self, *, name: str, arguments: str) -> str:
        tool = self._tools.get(name)

        if tool is None:
            return f'{{"error": "unknown_tool", "tool": "{name}"}}'

        return await tool.execute(arguments)
