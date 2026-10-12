from __future__ import annotations

import json
from typing import Literal

import httpx
from pydantic import Field

from .base import BaseTool, ToolArgs


class WebSearchArgs(ToolArgs):
    query: str = Field(
        min_length=1,
        max_length=1000,
        description="需要搜索的关键词或问题。例如：'Python 3.15 最新进展'、'OpenAI 最近有什么新闻'",
    )

    max_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="最多返回多少条搜索结果，范围 1~10",
    )

    topic: Literal[
        "general",
        "news",
    ] = Field(
        default="general",
        description="搜索类型。general 用于普通网页搜索，news 用于新闻和近期事件。",
    )

    search_depth: Literal[
        "basic",
        "advanced",
    ] = Field(
        default="basic",
        description="搜索深度。basic 更快、更省 credits；advanced 会返回更相关、更丰富的网页内容。",
    )

    days: int | None = Field(
        default=None,
        ge=1,
        le=3650,
        description="限制搜索最近多少天的内容。通常只在新闻、近期事件、最新信息查询时使用。",
    )


class WebSearchTool(BaseTool[WebSearchArgs]):
    name = "web_search"

    description = "搜索互联网以获取最新或外部信息。当问题涉及新闻、当前事件、最新软件版本、近期文档、实时变化的信息，或模型自身知识不足时使用。"

    args_model = WebSearchArgs

    SEARCH_URL = "https://api.tavily.com/search"

    def __init__(self, *, api_key: str) -> None:
        self.api_key = api_key

    async def run(self, args: WebSearchArgs) -> str:
        payload = {
            "query": args.query,
            "topic": args.topic,
            "search_depth": args.search_depth,
            "max_results": args.max_results,
            # 不让 Tavily 再生成一遍最终答案。
            # 最终答案由我们的 DeepSeek Agent 生成。
            "include_answer": False,
            # 当前只需要适合 LLM 使用的内容片段，
            # 不需要整个网页正文。
            "include_raw_content": False,
            "include_images": False,
        }

        if args.days is not None:
            payload["days"] = args.days

        async with httpx.AsyncClient(
            timeout=20.0,
        ) as client:
            response = await client.post(
                self.SEARCH_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

            response.raise_for_status()

            data = response.json()

        results = []

        for item in data.get(
            "results",
            [],
        ):
            results.append(
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "content": item.get("content"),
                    "score": item.get("score"),
                    "published_date": item.get("published_date"),
                }
            )

        return json.dumps(
            {
                "query": args.query,
                "topic": args.topic,
                "search_depth": args.search_depth,
                "result_count": len(results),
                "results": results,
            },
            ensure_ascii=False,
        )
