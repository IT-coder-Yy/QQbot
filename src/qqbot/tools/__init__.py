from __future__ import annotations

import os
import logging

from .calculator import (
    CalculatorTool,
)
from .getdatetime import (
    CurrentDateTimeTool,
)
from .registry import (
    ToolRegistry,
)
from .weather import (
    WeatherTool,
)
from .web_search import (
    WebSearchTool,
)


def build_default_tool_registry() -> ToolRegistry:
    tools = [
        CalculatorTool(),
        CurrentDateTimeTool(),
        WeatherTool(),
    ]

    tavily_api_key = os.getenv("TAVILY_SEARCH_API_KEY", "").strip()

    if tavily_api_key:
        tools.append(WebSearchTool(api_key=tavily_api_key))

    else:
        logging.getLogger(__name__).info("TAVILY_SEARCH_API_KEY 未配置，web_search 未启用")

    return ToolRegistry(tools)


__all__ = [
    "ToolRegistry",
    "build_default_tool_registry",
]
