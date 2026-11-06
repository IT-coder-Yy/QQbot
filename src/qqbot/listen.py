from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from contextlib import AsyncExitStack, aclosing
from datetime import datetime, timezone
from pathlib import Path
from contextlib import nullcontext

import websockets
from dotenv import load_dotenv
from typesafe_sdk import AsyncTypeSafeClient
from websockets.client import backoff
from websockets.exceptions import ConnectionClosed

from .agent import Agent
from .context import ContextBuilder, ContextManager
from .context.sqlite_store import SQLiteContextStore, StorageError
from .media import MediaService
from .dispatcher import MessageDispatcher, MessageJob
from .intent import IntentDetector, should_respond
from .normalizer import normalize_event
from .proactive import ProactiveDecision, next_active_time, run_proactive
from .reply import SILENT_MARKER, parse_reply
from .router import RouteDecision, decide_route
from .schema import ContextRole, ConversationType, MessageReceived, FileUploaded, MentionContent, ProactiveEvent
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
    media: MediaService | None = None,
) -> None:
    message = job.message
    autonomous = isinstance(message, ProactiveEvent)
    route = RouteDecision.REPLY if autonomous else decide_route(message)
    if route is RouteDecision.IGNORE:
        return
    store = context_manager.store
    settings = context_manager.settings
    if autonomous:
        now = datetime.now(timezone.utc)
        active_at = next_active_time(now, settings)
        if active_at > now:
            store.schedule_proactive(message.conversation_id, active_at)
            return
    if job.proactive and not transport.connected:
        return
    if not job.proactive and getattr(store, "persistent", False) and store.has_message(message.conversation_id, message.message_id):
        return
    refs = media.register_message(message) if media and not job.proactive else []
    quoted_media = False
    if media and route is RouteDecision.REPLY and not job.proactive:
        known = {ref.id for ref in refs}
        quoted = media.referenced(message)
        quoted_media = bool(quoted and message.text and not any(isinstance(p, MentionContent) and p.target_id for p in message.content))
        refs.extend(ref for ref in quoted if ref.id not in known)
        message.metadata["attachments"] = [ref.model_dump(mode="json") for ref in refs]
    # 断线时保留任务，且不继续消耗模型额度；外层总期限仍然计时。
    await transport.wait_connected()
    if job.previous_messages:
        _, session = context_manager.get_or_create_context(message)
        known_ids = {item.source_message_id for item in session.recent_messages if item.role is ContextRole.USER}
        for previous in job.previous_messages:
            if (store.has_message(previous.conversation_id, previous.message_id) if getattr(store, "persistent", False)
                    else previous.message_id in known_ids):
                continue
            context_manager.record_user_message(previous)
        logger.info("合并连发 conversation=%s messages=%s", message.conversation_id,
                    [item.message_id for item in (*job.previous_messages, message)])
    # 模型、工具和发送回执等待期间，其他会话不能淘汰正在使用的上下文。
    with (store.pin([ref.id for ref in refs]) if media else nullcontext()), context_manager.processing(message, proactive=job.proactive) as (conversation, session, current_turn):
        if job.previous_messages:
            current_turn.metadata["batch_message_ids"] = [item.message_id for item in (*job.previous_messages, message)]
        # 引用也可能是在对另一个群友说话，不能仅凭有引用就跳过受话者判断。
        if not job.proactive and message.conversation_type is ConversationType.GROUP and not any(item.mentioned_bot for item in (*job.previous_messages, message)) and not quoted_media:
            if not await should_respond(message=message, session=session, current_turn=current_turn,
                                        detector=intent_detector, agent=agent, timeout=intent_timeout,
                                        fallback_timeout=context_manager.settings.intent_fallback_timeout, store=store):
                return

        quote_ids = {item.source_message_id for item in session.recent_messages
                     if item.role is ContextRole.USER and item.source_message_id is not None and not item.recalled
                     and message.conversation_type is ConversationType.GROUP}
        if message.conversation_type is ConversationType.GROUP and current_turn.source_message_id is not None:
            quote_ids.add(current_turn.source_message_id)
        reply = None
        # 是否接话已经确定；自然接话与 @ 回复使用相同的附件能力。
        if media and refs:
            try:
                prepared = [await media.prepare(message.conversation_id, ref.id) for ref in refs]
                if not message.text and all(row["kind"] == "document" for row in prepared):
                    partial = [row for row in prepared if row["status"] != "ready"]
                    if partial:
                        has_text = any(p.get("text", "").strip() for row in prepared for p in json.loads(row["parsed_content_json"] or "[]"))
                        reply = ("已读取部分内容，尚有未完成的识别：" if has_text else "文件已保存，但文字识别尚未完成：") + str(partial[0]["error"])
                    else:
                        reply = "已读取，可以继续提问。"
            except StorageError as exc:
                reply = str(exc)
        if reply is None:
            reply = await agent.run(message=message, conversation=conversation, session=session, current_turn=current_turn)
        if job.superseded:
            return
        if autonomous:
            decision = ProactiveDecision.model_validate_json(reply, context={"now": datetime.now(timezone.utc)})
            logger.info("主动决策 conversation=%s next=%s reason=%s", message.conversation_id, decision.next_check_at, decision.reason)
            reply = decision.reply
            if not reply.strip():
                store.schedule_proactive(message.conversation_id, next_active_time(decision.next_check_at, settings))
                return
        if job.proactive and reply.strip() == SILENT_MARKER:
            return
        replies = parse_reply(reply, current_message_id=None if autonomous else message.message_id, quote_ids=quote_ids)
        if not replies:
            raise RuntimeError("模型返回了空白回复")
        for index, (reply, quote_id) in enumerate(replies):
            if index:
                await asyncio.sleep(context_manager.settings.reply_interval)
            if job.superseded:
                return
            if job.proactive and not transport.connected:
                return
            if autonomous:
                now = datetime.now(timezone.utc)
                active_at = next_active_time(now, settings)
                if active_at > now:
                    store.schedule_proactive(message.conversation_id, active_at)
                    return
            if asyncio.get_running_loop().time() >= job.deadline:
                raise TimeoutError
            # 每条独立确认并入库；中途失败时保留已经送达的部分。
            job.reply_started = True
            if autonomous:
                message_id = await transport.send_group_text(message.group_id, reply, reply_to_message_id=quote_id, deadline=job.deadline)
            else:
                message_id = await transport.reply_text(message, reply, reply_to_message_id=quote_id, deadline=job.deadline)
            context_manager.record_assistant_message(
                conversation=conversation, session=session, content=reply,
                timestamp=datetime.now(tz=message.timestamp.tzinfo), source_message_id=message_id,
                reply_to_message_id=quote_id,
                proactive=job.proactive,
            )
        if autonomous:
            store.schedule_proactive(message.conversation_id, next_active_time(decision.next_check_at, settings))


async def cleanup_contexts(context_manager: ContextManager) -> None:
    while True:
        await asyncio.sleep(context_manager.settings.context_cleanup_interval)
        try:
            context_manager.cleanup_expired()
        except Exception:
            logger.exception("本轮存储清理失败，将在下次重试")


async def receive_events(websocket, transport: NapCatTransport, dispatcher: MessageDispatcher, media: MediaService | None = None) -> None:
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
            elif isinstance(event, FileUploaded) and media:
                media.register_upload(event)
        except Exception:
            logger.exception("忽略无效的 OneBot 事件")


async def serve(settings: Settings, transport: NapCatTransport, dispatcher: MessageDispatcher, media: MediaService | None = None) -> None:
    connector = websockets.connect(
        settings.ws_url,
        additional_headers={"Authorization": f"Bearer {settings.token}"},
        open_timeout=settings.action_timeout,
        close_timeout=settings.action_timeout,
        max_size=settings.attachment_bytes * 4 // 3 + 2_000_000,
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
                await receive_events(websocket, transport, dispatcher, media)
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    settings = Settings.from_env()
    tool_registry = build_default_tool_registry()
    logger.info("已加载工具: %s", ", ".join(tool_registry.names))

    async with AsyncExitStack() as stack:
        data_dir = Path(settings.data_dir)
        store = SQLiteContextStore(data_dir / "qqbot.sqlite3", data_dir / "attachments",
                                   conversation_bytes=settings.conversation_bytes, attachment_bytes=settings.attachment_bytes)
        stack.callback(store.close)
        context_manager = ContextManager(store=store, settings=settings)
        transport = NapCatTransport(action_timeout=settings.action_timeout)
        stack.callback(transport.close)
        media = MediaService(store, transport, settings)
        stack.push_async_callback(media.close)
        agent = Agent(
            context_manager=context_manager,
            context_builder=ContextBuilder(),
            tool_registry=tool_registry,
            request_timeout=settings.llm_timeout,
            media=media,
        )
        stack.push_async_callback(agent.client.close)
        intent_detector = None
        if os.getenv("TYPESAFE_API_KEY", "").strip():
            jev_client = await stack.enter_async_context(AsyncTypeSafeClient(timeout=settings.intent_timeout))
            intent_detector = IntentDetector(
                client=jev_client,
                reply_threshold=settings.intent_threshold,
                recent_message_limit=settings.intent_recent_limit,
                bot_names=settings.bot_names,
            )
        else:
            logger.info("TYPESAFE_API_KEY 未配置，仅响应私聊、@Bot 及明确引用机器人的消息")

        async def process(job: MessageJob) -> None:
            await handle_message(
                job,
                context_manager=context_manager,
                agent=agent,
                intent_detector=intent_detector,
                transport=transport,
                intent_timeout=settings.intent_timeout,
                media=media,
            )

        async def send_notice(message: MessageReceived, text: str) -> None:
            # 故障提示只在当前可发送时尝试一次，不跨重连积压。
            if transport.connected:
                await transport.reply_text(message, text)

        dispatcher = MessageDispatcher(settings, process, send_notice)
        stack.push_async_callback(dispatcher.close)
        cleanup_task = asyncio.create_task(cleanup_contexts(context_manager), name="context-cleanup")
        proactive_task = asyncio.create_task(run_proactive(settings, store, transport, dispatcher), name="group-proactive")
        try:
            await serve(settings, transport, dispatcher, media)
        finally:
            cleanup_task.cancel()
            proactive_task.cancel()
            await asyncio.gather(cleanup_task, proactive_task, return_exceptions=True)


if __name__ == "__main__":
    from . import main as run

    run()
