import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    ws_url: str = "ws://127.0.0.1:3001"
    token: str = ""
    # 运行策略在代码中调整，不读取环境变量；时间单位为秒。
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
    intent_threshold: float = 0.75
    intent_recent_limit: int = 12
    context_max_messages: int = 200
    context_max_chars: int = 64_000
    context_max_conversations: int = 256
    context_idle_timeout: float = 24 * 60 * 60
    context_cleanup_interval: float = 60.0

    @classmethod
    def from_env(cls) -> "Settings":
        defaults = cls()
        token = os.getenv("NAPCAT_TOKEN", "").strip()
        if not token:
            raise ValueError("缺少环境变量: NAPCAT_TOKEN")
        ws_url = os.getenv("NAPCAT_WS_URL", defaults.ws_url).strip()
        if not ws_url.startswith(("ws://", "wss://")):
            raise ValueError("NAPCAT_WS_URL 必须使用 ws:// 或 wss://")
        return cls(ws_url=ws_url, token=token)
