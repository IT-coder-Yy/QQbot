import os
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    ws_url: str = "ws://127.0.0.1:3001"
    token: str = ""
    # 时间单位为秒；存储、多模态超时等可通过 from_env 中列出的配置调整。
    concurrency: int = 4
    conversation_queue_limit: int = 20
    queue_limit: int = 100
    task_timeout: float = 120.0
    action_timeout: float = 10.0
    llm_timeout: float = 60.0
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
        for field, env, cast in (
            ("data_dir", "BOT_DATA_DIR", str),
            ("conversation_bytes", "CONVERSATION_STORAGE_BYTES", int),
            ("attachment_bytes", "ATTACHMENT_STORAGE_BYTES", int),
            ("context_cleanup_interval", "STORAGE_CLEANUP_INTERVAL", float),
        ):
            raw = os.getenv(env, "").strip()
            if raw:
                value = cast(raw)
                if cast is not str and (not math.isfinite(value) or value <= 0):
                    raise ValueError(f"{env} 必须是有限正数")
                values[field] = value
        return cls(**values)
