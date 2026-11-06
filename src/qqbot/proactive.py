"""全部群的主动机会调度；发不发、聊什么及下次时间由主模型决定。"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictStr, TypeAdapter, ValidationInfo, field_validator

from .schema import ProactiveEvent
from .settings import Settings

logger = logging.getLogger(__name__)


class ProactiveDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reply: StrictStr
    next_check_at: AwareDatetime
    reason: StrictStr

    @field_validator("next_check_at")
    @classmethod
    def future_time(cls, value: datetime, info: ValidationInfo) -> datetime:
        if value <= info.context["now"]:
            raise ValueError("next_check_at 必须晚于当前时间")
        return value


class BotInfo(BaseModel):
    user_id: int = Field(strict=True, gt=0)


class GroupInfo(BaseModel):
    group_id: int = Field(strict=True, gt=0)


def next_active_time(when: datetime, settings: Settings) -> datetime:
    local = when.astimezone(ZoneInfo(settings.proactive_timezone))
    day = local.replace(hour=0, minute=0, second=0, microsecond=0)
    start = day + timedelta(hours=settings.proactive_start_hour)
    end = day + timedelta(hours=settings.proactive_end_hour)
    if local < start:
        return start
    if local >= end:
        return start + timedelta(days=1)
    return local


async def run_proactive(settings, store, transport, dispatcher):
    connection = None
    groups = []
    bot_id = None
    refresh_at = 0.0
    while True:
        try:
            socket = await transport.wait_connected()
            now = datetime.now(timezone.utc)
            if socket is not connection or now.timestamp() >= refresh_at:
                if socket is not connection:
                    bot_id = BotInfo.model_validate(await transport.request("get_login_info", {})).user_id
                groups = TypeAdapter(list[GroupInfo]).validate_python(await transport.request("get_group_list", {"no_cache": True}))
                connection = socket
                refresh_at = now.timestamp() + settings.proactive_refresh_seconds
                logger.info("主动聊天群清单已更新 groups=%s", len(groups))

            now = datetime.now(timezone.utc)
            for group in groups:
                event = ProactiveEvent(bot_id=bot_id, group_id=group.group_id, timestamp=now)
                due = store.next_proactive_time(event.conversation_id)
                active_due = next_active_time(due or now, settings)
                if due != active_due:
                    store.schedule_proactive(event.conversation_id, active_due)
                due = active_due
                if due <= now:
                    # 过期计划按现在重评，休息时段不补发。
                    due = next_active_time(now, settings)
                    if due > now:
                        store.schedule_proactive(event.conversation_id, due)
                    elif dispatcher.submit_proactive(event):
                        # 生成失败、断线或被真实消息打断时仍有下一次机会；成功结果会覆盖此时间。
                        due = next_active_time(now + timedelta(seconds=settings.proactive_retry_seconds), settings)
                        store.schedule_proactive(event.conversation_id, due)
            # 轻量扫描只读本地计划，不调用模型；及时拾取模型刚保存的更早时间。
            await asyncio.sleep(1)
        except Exception:
            logger.exception("主动聊天调度失败，将重新评估")
            await asyncio.sleep(settings.proactive_retry_seconds)
