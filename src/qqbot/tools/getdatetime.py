from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import (
    ZoneInfo,
    ZoneInfoNotFoundError,
)

from pydantic import Field

from .base import (
    BaseTool,
    ToolArgs,
)


class CurrentDateTimeArgs(ToolArgs):
    timezone: str = Field(
        default="Asia/Shanghai",
        description=("IANA 时区名称，例如 Asia/Shanghai、Asia/Tokyo、America/New_York"),
    )


class CurrentDateTimeTool(BaseTool[CurrentDateTimeArgs]):
    name = "get_current_datetime"

    description = "获取指定时区当前准确的日期、时间和星期。涉及今天、明天、今晚、星期几或当前时间时使用。"

    args_model = CurrentDateTimeArgs

    async def run(
        self,
        args: CurrentDateTimeArgs,
    ) -> str:
        try:
            timezone = ZoneInfo(args.timezone)

        except ZoneInfoNotFoundError:
            raise ValueError(f"未知时区: {args.timezone}")

        now = datetime.now(timezone)

        weekday_names = [
            "星期一",
            "星期二",
            "星期三",
            "星期四",
            "星期五",
            "星期六",
            "星期日",
        ]

        return json.dumps(
            {
                "timezone": (args.timezone),
                "iso": (now.isoformat()),
                "date": (now.strftime("%Y-%m-%d")),
                "time": (now.strftime("%H:%M:%S")),
                "weekday": (weekday_names[now.weekday()]),
            },
            ensure_ascii=False,
        )
