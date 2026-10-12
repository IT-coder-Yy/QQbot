from __future__ import annotations

import json

import httpx
from pydantic import Field

from .base import (
    BaseTool,
    ToolArgs,
)


class WeatherArgs(ToolArgs):
    location: str = Field(
        min_length=1,
        max_length=100,
        description=("城市或地点，例如：北京、郑州、上海、东京"),
    )

    forecast_days: int = Field(
        default=3,
        ge=1,
        le=7,
        description=("需要未来多少天的天气预报，范围 1~7"),
    )


class WeatherTool(BaseTool[WeatherArgs]):
    name = "get_weather"

    description = "查询指定地点的当前天气和未来几天天气预报。涉及实时天气、温度、降雨、风速或未来天气时使用。"

    args_model = WeatherArgs

    GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    async def run(
        self,
        args: WeatherArgs,
    ) -> str:
        async with httpx.AsyncClient(
            timeout=15.0,
        ) as client:
            location = await self._resolve_location(
                client,
                args.location,
            )

            response = await client.get(
                self.FORECAST_URL,
                params={
                    "latitude": (location["latitude"]),
                    "longitude": (location["longitude"]),
                    "timezone": "auto",
                    "forecast_days": (args.forecast_days),
                    "current": ("temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,weather_code,wind_speed_10m"),
                    "daily": ("weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,sunrise,sunset"),
                },
            )

            response.raise_for_status()

            data = response.json()

        current = data.get(
            "current",
            {},
        )

        daily = data.get(
            "daily",
            {},
        )

        days = []

        dates = daily.get(
            "time",
            [],
        )

        for index, date in enumerate(dates):
            code = self._item(
                daily,
                "weather_code",
                index,
            )

            days.append(
                {
                    "date": date,
                    "condition": (self._weather_description(code)),
                    "temperature_max_c": (
                        self._item(
                            daily,
                            "temperature_2m_max",
                            index,
                        )
                    ),
                    "temperature_min_c": (
                        self._item(
                            daily,
                            "temperature_2m_min",
                            index,
                        )
                    ),
                    "precipitation_probability_max_percent": (
                        self._item(
                            daily,
                            "precipitation_probability_max",
                            index,
                        )
                    ),
                    "sunrise": (
                        self._item(
                            daily,
                            "sunrise",
                            index,
                        )
                    ),
                    "sunset": (
                        self._item(
                            daily,
                            "sunset",
                            index,
                        )
                    ),
                }
            )

        current_code = current.get("weather_code")

        result = {
            "location": {
                "name": (location["name"]),
                "admin1": (location.get("admin1")),
                "country": (location.get("country")),
                "latitude": (location["latitude"]),
                "longitude": (location["longitude"]),
            },
            "timezone": (data.get("timezone")),
            "current": {
                "time": (current.get("time")),
                "condition": (self._weather_description(current_code)),
                "temperature_c": (current.get("temperature_2m")),
                "apparent_temperature_c": (current.get("apparent_temperature")),
                "humidity_percent": (current.get("relative_humidity_2m")),
                "precipitation_mm": (current.get("precipitation")),
                "wind_speed_kmh": (current.get("wind_speed_10m")),
            },
            "forecast": days,
        }

        return json.dumps(
            result,
            ensure_ascii=False,
        )

    async def _resolve_location(
        self,
        client: httpx.AsyncClient,
        location: str,
    ) -> dict:
        response = await client.get(
            self.GEOCODING_URL,
            params={
                "name": location,
                "count": 1,
                "language": "zh",
                "format": "json",
            },
        )

        response.raise_for_status()

        data = response.json()

        results = data.get(
            "results",
            [],
        )

        if not results:
            raise ValueError(f"没有找到地点: {location}")

        return results[0]

    @staticmethod
    def _item(
        data: dict,
        key: str,
        index: int,
    ):
        values = data.get(
            key,
            [],
        )

        if not isinstance(
            values,
            list,
        ) or index >= len(values):
            return None

        return values[index]

    @staticmethod
    def _weather_description(
        code,
    ) -> str:
        descriptions = {
            0: "晴",
            1: "大部晴朗",
            2: "局部多云",
            3: "阴",
            45: "雾",
            48: "雾凇",
            51: "小毛毛雨",
            53: "中等毛毛雨",
            55: "强毛毛雨",
            56: "轻微冻毛毛雨",
            57: "强冻毛毛雨",
            61: "小雨",
            63: "中雨",
            65: "大雨",
            66: "轻微冻雨",
            67: "强冻雨",
            71: "小雪",
            73: "中雪",
            75: "大雪",
            77: "雪粒",
            80: "小阵雨",
            81: "中等阵雨",
            82: "强阵雨",
            85: "小阵雪",
            86: "强阵雪",
            95: "雷暴",
            96: "雷暴伴小冰雹",
            99: "雷暴伴强冰雹",
        }

        return descriptions.get(
            code,
            f"未知天气代码 {code}",
        )
