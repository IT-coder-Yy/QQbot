import os
import math
from dataclasses import dataclass
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class Settings:
    ws_url: str = "ws://127.0.0.1:3001"
    token: str = ""
    # 时间单位为秒；存储、多模态超时等可通过 from_env 中列出的配置调整。
    concurrency: int = 4
    conversation_queue_limit: int = 20
    queue_limit: int = 100
    task_timeout: float = 600.0
    action_timeout: float = 10.0
    llm_timeout: float = 300.0
    reply_interval: float = 0.8
    input_debounce_seconds: float = 1.2
    proactive_idle_seconds: float = 900.0
    proactive_timezone: str = "Asia/Shanghai"
    proactive_start_hour: int = 10
    proactive_end_hour: int = 23
    proactive_refresh_seconds: float = 300.0
    proactive_retry_seconds: float = 900.0
    intent_timeout: float = 10.0
    dedup_ttl: float = 600.0
    dedup_capacity: int = 10_000
    notice_cooldown: float = 30.0
    reconnect_delay: float = 1.0
    reconnect_max_delay: float = 30.0
    # 合成会话实测结果见 docs/evals；切换 JEV 模型或提示词后需重新评估。
    intent_threshold: float = 0.5
    intent_recent_limit: int = 12
    intent_fallback_timeout: float = 20.0
    bot_names: tuple[str, ...] = ("糖糖", "村头洋哥")
    context_max_messages: int = 200
    context_max_chars: int = 64_000
    context_max_conversations: int = 256
    context_idle_timeout: float = 24 * 60 * 60
    context_cleanup_interval: float = 60.0
    data_dir: str = "data"
    conversation_bytes: int = 50_000_000
    attachment_bytes: int = 50_000_000
    media_timeout: float = 900.0
    media_api_timeout: float = 120.0
    soffice: str = "soffice"
    napcat_file_root: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        defaults = cls()
        token = os.getenv("NAPCAT_TOKEN", "").strip()
        if not token:
            raise ValueError("缺少环境变量: NAPCAT_TOKEN")
        ws_url = os.getenv("NAPCAT_WS_URL", defaults.ws_url).strip()
        if not ws_url.startswith(("ws://", "wss://")):
            raise ValueError("NAPCAT_WS_URL 必须使用 ws:// 或 wss://")
        values = dict(ws_url=ws_url, token=token)
        zone = os.getenv("CHAT_PROACTIVE_TIMEZONE", defaults.proactive_timezone).strip()
        ZoneInfo(zone)
        start = int(os.getenv("CHAT_PROACTIVE_START_HOUR", str(defaults.proactive_start_hour)))
        end = int(os.getenv("CHAT_PROACTIVE_END_HOUR", str(defaults.proactive_end_hour)))
        if not 0 <= start < end <= 24:
            raise ValueError("主动聊天时段必须满足 0 <= 开始小时 < 结束小时 <= 24")
        values.update(proactive_timezone=zone, proactive_start_hour=start, proactive_end_hour=end)
        for field, env in (('reply_interval', 'CHAT_REPLY_INTERVAL'), ('input_debounce_seconds', 'CHAT_INPUT_DEBOUNCE_SECONDS'), ('proactive_idle_seconds', 'CHAT_PROACTIVE_IDLE_SECONDS')):
            raw = os.getenv(env, "").strip()
            if raw:
                value = float(raw)
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"{env} 必须是有限非负数")
                values[field] = value
        if os.getenv("BOT_NAMES", "").strip():
            names = tuple(name.strip() for name in os.environ["BOT_NAMES"].replace("，", ",").split(",") if name.strip())
            if not names:
                raise ValueError("BOT_NAMES 至少包含一个称呼")
            values["bot_names"] = names
        for field, env, cast in (
            ("data_dir", "BOT_DATA_DIR", str),
            ("conversation_bytes", "CONVERSATION_STORAGE_BYTES", int),
            ("attachment_bytes", "ATTACHMENT_STORAGE_BYTES", int),
            ("context_cleanup_interval", "STORAGE_CLEANUP_INTERVAL", float),
            ("media_timeout", "MEDIA_TASK_TIMEOUT", float),
            ("media_api_timeout", "MEDIA_API_TIMEOUT", float),
            ("task_timeout", "CHAT_TASK_TIMEOUT", float),
            ("llm_timeout", "LLM_REQUEST_TIMEOUT", float),
            ("proactive_refresh_seconds", "CHAT_PROACTIVE_GROUP_REFRESH_SECONDS", float),
            ("proactive_retry_seconds", "CHAT_PROACTIVE_RETRY_SECONDS", float),
            ("soffice", "LIBREOFFICE_BIN", str),
            ("napcat_file_root", "NAPCAT_FILE_ROOT", str),
        ):
            raw = os.getenv(env, "").strip()
            if raw:
                value = cast(raw)
                if cast is not str and (not math.isfinite(value) or value <= 0):
                    raise ValueError(f"{env} 必须是有限正数")
                values[field] = value
        return cls(**values)
