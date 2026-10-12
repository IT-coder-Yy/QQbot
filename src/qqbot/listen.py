from __future__ import annotations

import asyncio
import json
import os

import websockets
from dotenv import load_dotenv
from typesafe_sdk import AsyncTypeSafeClient

from .agent import Agent
from .context import ContextBuilder, ContextManager, InMemoryContextStore
from .intent import IntentDetector
from .normalizer import normalize_event
from .router import RouteDecision, decide_route
from .schema import ConversationType, MessageReceived
from .transport import NapCatTransport

from .tools import build_default_tool_registry


load_dotenv()


WS_URL = os.getenv(
    "NAPCAT_WS_URL",
    "ws://127.0.0.1:3001",
)

TOKEN = os.environ["NAPCAT_TOKEN"]


JEV_REPLY_THRESHOLD = float(
    os.getenv(
        "JEV_REPLY_THRESHOLD",
        "0.75",
    )
)

JEV_RECENT_MESSAGE_LIMIT = int(
    os.getenv(
        "JEV_RECENT_MESSAGE_LIMIT",
        "12",
    )
)


async def main() -> None:
    # =====================================
    # Context Harness
    # =====================================

    store = InMemoryContextStore()

    context_manager = ContextManager(store=store)

    context_builder = ContextBuilder()

    # =====================================
    # Main Agent
    # =====================================

    tool_registry = build_default_tool_registry()

    print("[Tools] 已加载: " + ", ".join(tool_registry.names))

    agent = Agent(
        context_manager=context_manager,
        context_builder=context_builder,
        tool_registry=tool_registry,
    )

    # =====================================
    # NapCat
    # =====================================

    headers = {"Authorization": f"Bearer {TOKEN}"}

    # TypeSafe 官方提供 AsyncTypeSafeClient，
    # 使用 async context manager 管理连接生命周期。
    async with AsyncTypeSafeClient() as jev_client:
        intent_detector = IntentDetector(
            client=jev_client,
            reply_threshold=(JEV_REPLY_THRESHOLD),
            recent_message_limit=(JEV_RECENT_MESSAGE_LIMIT),
        )

        async with websockets.connect(
            WS_URL,
            additional_headers=headers,
        ) as ws:
            print("已连接 NapCat")

            transport = NapCatTransport(ws)

            async for raw in ws:
                try:
                    await handle_event(
                        raw=raw,
                        context_manager=(context_manager),
                        agent=agent,
                        intent_detector=(intent_detector),
                        transport=transport,
                    )

                except Exception as exc:
                    print(f"[Event Error] {exc!r}")


async def handle_event(
    *,
    raw: str,
    context_manager: ContextManager,
    agent: Agent,
    intent_detector: IntentDetector,
    transport: NapCatTransport,
) -> None:
    raw_event = json.loads(raw)

    domain_event = normalize_event(raw_event)

    if not isinstance(
        domain_event,
        MessageReceived,
    ):
        return

    # =====================================
    # 1. 所有消息先进入统一 Context
    # =====================================

    (
        conversation,
        session,
        current_turn,
    ) = context_manager.record_user_message(domain_event)

    # =====================================
    # 2. Router 硬规则
    # =====================================

    route = decide_route(domain_event)

    if route is RouteDecision.IGNORE:
        return

    # =====================================
    # 3. 未 @ 的群消息
    #    使用 Jev 判断
    # =====================================

    if route is RouteDecision.INTENT:
        try:
            intent = await intent_detector.detect(
                session=session,
                current_turn=(current_turn),
            )

        except Exception as exc:
            # Intent Detection 失败时
            # 采用 fail-closed：
            #
            # 没 @Bot 就不要乱插嘴。
            print(f"[Intent Error] {exc!r}")

            return

        print(
            "[Intent] "
            f"sender="
            f"{current_turn.sender_name} "
            f"text="
            f"{current_turn.content!r} "
            f"probability="
            f"{intent.probability:.3f} "
            f"threshold="
            f"{JEV_REPLY_THRESHOLD:.3f} "
            f"reply="
            f"{intent.should_reply} "
            f"model="
            f"{intent.model}"
        )

        if not intent.should_reply:
            return

    # =====================================
    # 4. 到这里一定需要回复
    #
    #    私聊
    #    或 @Bot
    #    或 Jev=True
    # =====================================

    try:
        reply = await agent.run(
            message=domain_event,
            conversation=conversation,
            session=session,
            current_turn=current_turn,
        )

        # 群聊统一引用触发机器人的消息。
        #
        # 这样即使没有 @Bot，
        # 大家也知道机器人在回复谁。
        quote = domain_event.conversation_type is ConversationType.GROUP

        await transport.reply_text(
            domain_event,
            reply,
            quote=quote,
        )

    except Exception as exc:
        print(f"[Agent Error] {exc!r}")


if __name__ == "__main__":
    asyncio.run(main())
