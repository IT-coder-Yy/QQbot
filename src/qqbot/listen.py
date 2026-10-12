from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import AsyncExitStack, aclosing
from datetime import datetime

import websockets
from dotenv import load_dotenv
from typesafe_sdk import AsyncTypeSafeClient
from websockets.client import backoff
from websockets.exceptions import ConnectionClosed

from .agent import Agent
from .context import ContextBuilder, ContextManager, InMemoryContextStore
from .dispatcher import MessageDispatcher, MessageJob
from .intent import IntentDetector
from .normalizer import normalize_event
from .router import RouteDecision, decide_route
from .schema import ConversationType, MessageReceived
from .settings import Settings
from .tools import build_default_tool_registry
from .transport import NapCatTransport

logger = logging.getLogger(__name__)


async def handle_message(
    job: MessageJob,
    *,
    context_manager: ContextManager,
    agent: Agent,
    intent_detector: IntentDetector | None,
    transport: NapCatTransport,
    intent_timeout: float,
) -> None:
    message = job.message
    route = decide_route(message)
    if route is RouteDecision.IGNORE:
        return
    # 断线时保留任务，且不继续消耗模型额度；外层总期限仍然计时。
    await transport.wait_connected()
    conversation, session, current_turn = context_manager.record_user_message(message)
    if route is RouteDecision.INTENT:
        if intent_detector is None:
            return
        try:
            async with asyncio.timeout(intent_timeout):
                intent = await intent_detector.detect(session=session, current_turn=current_turn)
        except Exception:
            logger.exception("意图判断失败，保持沉默 conversation=%s message=%s", message.conversation_id, message.message_id)
            return
        if not intent.should_reply:
            return

    reply = await agent.run(message=message, conversation=conversation, session=session, current_turn=current_turn)
    if not reply.strip():
        raise RuntimeError("模型返回了空白回复")
    if asyncio.get_running_loop().time() >= job.deadline:
        raise TimeoutError
    # 从这里开始超时不能再补发“失败”提示：原回复可能已经送达。
    job.reply_started = True
    message_id = await transport.reply_text(
        message,
        reply,
        quote=message.conversation_type is ConversationType.GROUP,
        deadline=job.deadline,
    )
    context_manager.record_assistant_message(
        conversation=conversation,
        session=session,
        content=reply,
        timestamp=datetime.now(tz=message.timestamp.tzinfo),
        source_message_id=message_id,
    )


async def receive_events(websocket, transport: NapCatTransport, dispatcher: MessageDispatcher) -> None:
    async for raw in websocket:
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("OneBot 报文必须是 JSON Object")
            if transport.handle_response(payload):
                continue
            event = normalize_event(payload)
            if isinstance(event, MessageReceived):
                dispatcher.submit(event)
        except Exception:
            logger.exception("忽略无效的 OneBot 事件")


async def serve(settings: Settings, transport: NapCatTransport, dispatcher: MessageDispatcher) -> None:
    connector = websockets.connect(
        settings.ws_url,
        additional_headers={"Authorization": f"Bearer {settings.token}"},
        open_timeout=settings.action_timeout,
        close_timeout=settings.action_timeout,
        reconnect_delays=lambda: backoff(
            initial_delay=settings.reconnect_delay,
            min_delay=settings.reconnect_delay,
            max_delay=settings.reconnect_max_delay,
        ),
    )
    delay = settings.reconnect_delay
    async with aclosing(connector.__aiter__()) as connections:
        async for websocket in connections:
            transport.attach(websocket)
            started = asyncio.get_running_loop().time()
            logger.info("已连接 NapCat")
            try:
                await receive_events(websocket, transport, dispatcher)
            except ConnectionClosed:
                logger.warning("NapCat 连接中断")
            finally:
                transport.detach(websocket)
            if asyncio.get_running_loop().time() - started >= settings.reconnect_max_delay:
                delay = settings.reconnect_delay
            # 连接建立后立即关闭也需要退避，避免反复快速重连。
            logger.info("将在 %.1f 秒后重连 NapCat", delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, settings.reconnect_max_delay)


async def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env()
    context_manager = ContextManager(store=InMemoryContextStore())
    tool_registry = build_default_tool_registry()
    logger.info("已加载工具: %s", ", ".join(tool_registry.names))

    async with AsyncExitStack() as stack:
        agent = Agent(
            context_manager=context_manager,
            context_builder=ContextBuilder(),
            tool_registry=tool_registry,
            request_timeout=settings.llm_timeout,
        )
        stack.push_async_callback(agent.client.close)
        intent_detector = None
        if os.getenv("TYPESAFE_API_KEY", "").strip():
            jev_client = await stack.enter_async_context(AsyncTypeSafeClient(timeout=settings.intent_timeout))
            intent_detector = IntentDetector(
                client=jev_client,
                reply_threshold=settings.intent_threshold,
                recent_message_limit=settings.intent_recent_limit,
            )
        else:
            logger.info("TYPESAFE_API_KEY 未配置，仅响应私聊和 @Bot")

        transport = NapCatTransport(action_timeout=settings.action_timeout)
        stack.callback(transport.close)

        async def process(job: MessageJob) -> None:
            await handle_message(
                job,
                context_manager=context_manager,
                agent=agent,
                intent_detector=intent_detector,
                transport=transport,
                intent_timeout=settings.intent_timeout,
            )

        async def send_notice(message: MessageReceived, text: str) -> None:
            # 故障提示只在当前可发送时尝试一次，不跨重连积压。
            if transport.connected:
                await transport.reply_text(message, text, quote=message.conversation_type is ConversationType.GROUP)

        dispatcher = MessageDispatcher(settings, process, send_notice)
        stack.push_async_callback(dispatcher.close)
        await serve(settings, transport, dispatcher)


if __name__ == "__main__":
    from . import main as run

    run()
