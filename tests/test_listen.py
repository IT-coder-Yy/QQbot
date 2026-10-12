import asyncio
import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import websockets
from websockets.datastructures import Headers
from websockets.exceptions import InvalidStatus
from websockets.http11 import Response

from qqbot.context import ContextManager, InMemoryContextStore
from qqbot.dispatcher import MessageDispatcher, MessageJob
from qqbot.listen import handle_message, receive_events, serve
from qqbot.schema import ContextRole
from qqbot.settings import Settings
from qqbot.transport import NapCatTransport
from tests.helpers import FakeSocket, eventually, message, raw_message, success


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.settings = replace(Settings(), task_timeout=1, action_timeout=0.2)
        self.manager = ContextManager(InMemoryContextStore())
        self.agent = SimpleNamespace(run=AsyncMock(return_value="答案"))
        self.intent = None
        self.ws = FakeSocket()
        self.transport = NapCatTransport(self.ws, action_timeout=self.settings.action_timeout)
        self.notices = AsyncMock()

        async def handle(job):
            await handle_message(job, context_manager=self.manager, agent=self.agent, intent_detector=self.intent, transport=self.transport, intent_timeout=0.01)

        self.dispatcher = MessageDispatcher(self.settings, handle, self.notices)
        self.receiver = asyncio.create_task(receive_events(self.ws, self.transport, self.dispatcher))

    async def asyncTearDown(self):
        self.receiver.cancel()
        await asyncio.gather(self.receiver, return_exceptions=True)
        await self.dispatcher.close()
        self.transport.close()

    def history(self):
        return [entry for session in self.manager.store.sessions.values() for entry in session.recent_messages]

    async def feed(self, payload):
        await self.ws.incoming.put(json.dumps(payload))

    async def test_history_written_only_after_successful_ack_and_duplicates_ignored(self):
        await self.feed(raw_message())
        await eventually(lambda: self.ws.sent)
        self.assertEqual([entry.role for entry in self.history()], [ContextRole.USER])
        await self.feed(raw_message())
        await self.feed(success(self.ws.sent[0], 501))
        await eventually(lambda: not self.dispatcher._workers)
        self.assertEqual([entry.role for entry in self.history()], [ContextRole.USER, ContextRole.ASSISTANT])
        self.assertEqual(self.history()[-1].source_message_id, 501)
        self.agent.run.assert_awaited_once()
        self.assertEqual(len(self.ws.sent), 1)

    async def test_full_queue_does_not_block_acknowledgement(self):
        self.dispatcher.settings = replace(self.settings, queue_limit=1, conversation_queue_limit=1)
        await self.feed(raw_message(1))
        await eventually(lambda: self.ws.sent)
        await self.feed(raw_message(2))
        await self.feed(raw_message(3))
        await self.feed(success(self.ws.sent[0]))
        await eventually(lambda: len(self.ws.sent) == 2)
        self.assertEqual(self.agent.run.await_count, 2)
        await self.feed(success(self.ws.sent[1], 502))
        await eventually(lambda: not self.dispatcher._workers)
        self.assertEqual(len(self.history()), 4)
        self.notices.assert_awaited_once()

    async def test_missing_ack_leaves_no_assistant_history_or_failure_notice(self):
        self.transport.action_timeout = 0.01
        await self.feed(raw_message())
        await eventually(lambda: self.ws.sent)
        await eventually(lambda: not self.dispatcher._workers)
        self.assertEqual([entry.role for entry in self.history()], [ContextRole.USER])
        self.notices.assert_not_awaited()
        self.assertEqual(len(self.ws.sent), 1)

    async def test_total_deadline_during_ack_does_not_send_failure_notice(self):
        self.dispatcher.settings = replace(self.settings, task_timeout=0.01)
        await self.feed(raw_message())
        await eventually(lambda: self.ws.sent)
        await eventually(lambda: not self.dispatcher._workers)
        self.assertFalse(self.transport._pending)
        self.assertEqual(len(self.history()), 1)
        self.assertEqual(len(self.ws.sent), 1)
        self.notices.assert_not_awaited()

    async def test_model_failure_reports_once_and_next_message_can_succeed(self):
        self.agent.run.side_effect = [RuntimeError("模拟模型失败"), "恢复了"]
        await self.feed(raw_message(1))
        await eventually(lambda: self.notices.await_count == 1)
        await self.feed(raw_message(2))
        await eventually(lambda: self.ws.sent)
        await self.feed(success(self.ws.sent[0]))
        await eventually(lambda: not self.dispatcher._workers)
        self.assertEqual(len(self.history()), 3)
        self.assertEqual(self.history()[-1].content, "恢复了")

    async def test_self_and_invalid_events_do_not_pollute_context_or_stop_receiver(self):
        await self.ws.incoming.put("invalid json")
        await self.feed([])
        await self.feed(raw_message(user_id=10))
        await self.feed(raw_message())
        await eventually(lambda: self.ws.sent)
        self.assertEqual(len(self.history()), 1)
        self.assertFalse(self.receiver.done())

    async def test_intent_timeout_stays_silent_and_mention_bypasses_intent(self):
        async def wait_forever(**kwargs):
            await asyncio.Event().wait()

        self.intent = SimpleNamespace(detect=AsyncMock(side_effect=wait_forever))
        await self.feed(raw_message(1, group_id=30))
        await eventually(lambda: self.intent.detect.await_count == 1)
        await eventually(lambda: not self.dispatcher._workers)
        self.agent.run.assert_not_awaited()
        self.notices.assert_not_awaited()
        await self.feed(raw_message(2, group_id=30, mentioned=True))
        await eventually(lambda: self.ws.sent)
        self.assertEqual(self.intent.detect.await_count, 1)
        self.agent.run.assert_awaited_once()

    async def test_missing_intent_configuration_keeps_group_context_without_reply(self):
        await self.feed(raw_message(1, group_id=30))
        await eventually(lambda: self.history())
        self.agent.run.assert_not_awaited()
        self.assertFalse(self.ws.sent)

    async def test_silent_group_traffic_keeps_only_bounded_recent_history(self):
        for intent in [None, SimpleNamespace(detect=AsyncMock(return_value=SimpleNamespace(should_reply=False)))]:
            with self.subTest(intent_enabled=intent is not None):
                self.manager = ContextManager(InMemoryContextStore(), settings=replace(self.settings, context_max_messages=5))
                for message_id in range(300):
                    incoming = message(message_id, group_id=30)
                    await handle_message(MessageJob(incoming, asyncio.get_running_loop().time() + 1), context_manager=self.manager, agent=self.agent, intent_detector=intent, transport=self.transport, intent_timeout=0.1)
                    if message_id % 50 == 0:
                        await asyncio.sleep(0)
                self.assertEqual([entry.source_message_id for entry in self.history()], list(range(295, 300)))
                self.assertFalse(self.manager._active)
                self.agent.run.assert_not_awaited()
                self.assertFalse(self.ws.sent)

    async def test_context_is_protected_until_acknowledgement_arrives(self):
        self.manager = ContextManager(InMemoryContextStore(), settings=replace(self.settings, context_idle_timeout=10))
        with patch("qqbot.context.manager.monotonic", return_value=0) as clock:
            await self.feed(raw_message())
            await eventually(lambda: self.ws.sent)
            clock.return_value = 20
            self.manager.cleanup_expired()
            self.assertEqual(len(self.history()), 1)
            await self.feed(success(self.ws.sent[0]))
            await eventually(lambda: not self.dispatcher._workers)
            self.manager.cleanup_expired()
            self.assertEqual(len(self.history()), 2)
            clock.return_value = 30
            self.manager.cleanup_expired()
            self.assertFalse(self.manager.store.conversations)
            self.assertFalse(self.manager.store.sessions)

    async def test_cancelled_model_call_releases_context_for_eviction(self):
        self.manager = ContextManager(InMemoryContextStore(), settings=replace(self.settings, context_max_conversations=1))

        async def wait_forever(**kwargs):
            await asyncio.Event().wait()

        self.agent.run.side_effect = wait_forever
        await self.feed(raw_message())
        await eventually(lambda: self.agent.run.await_count == 1)
        original_id = message().conversation_id
        await self.dispatcher.close()
        self.manager.record_user_message(message(user_id=21))
        self.assertNotIn(original_id, self.manager.store.conversations)
        self.assertFalse(self.manager._active)

    async def test_reconnect_keeps_generated_unsent_reply_without_rerunning_model(self):
        gate = asyncio.Event()

        async def generate(**kwargs):
            await gate.wait()
            return "答案"

        self.agent.run.side_effect = generate
        await self.feed(raw_message())
        await eventually(lambda: self.agent.run.await_count == 1)
        self.transport.detach(self.ws)
        gate.set()
        await asyncio.sleep(0)
        self.assertFalse(self.ws.sent)
        new_ws = FakeSocket()
        self.transport.attach(new_ws)
        await eventually(lambda: new_ws.sent)
        self.transport.handle_response(success(new_ws.sent[0], 600))
        await eventually(lambda: not self.dispatcher._workers)
        self.agent.run.assert_awaited_once()
        self.assertEqual(self.history()[-1].source_message_id, 600)

    async def test_task_expiring_offline_never_replays_after_reconnect(self):
        self.dispatcher.settings = replace(self.settings, task_timeout=0.01)
        self.transport.detach(self.ws)
        self.dispatcher.submit(message())
        await eventually(lambda: not self.dispatcher._workers)
        self.transport.attach(self.ws)
        self.agent.run.assert_not_awaited()
        self.assertFalse(self.history())
        self.assertFalse(self.ws.sent)


class ConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_library_retries_initial_network_failure_and_recovers_normal_close(self):
        settings = replace(Settings(), reconnect_delay=0.005, reconnect_max_delay=0.01)
        first, second = FakeSocket(), FakeSocket()
        transport = NapCatTransport()
        dispatcher = SimpleNamespace(submit=lambda message: None)
        enter = AsyncMock(side_effect=[OSError("模拟连接失败"), first, second])
        leave = AsyncMock(return_value=False)
        with patch.object(websockets.connect, "__aenter__", enter), patch.object(websockets.connect, "__aexit__", leave):
            task = asyncio.create_task(serve(settings, transport, dispatcher))
            try:
                await eventually(lambda: transport.websocket is first)
                await first.incoming.put(None)
                await eventually(lambda: transport.websocket is second)
                self.assertEqual(enter.await_count, 3)
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assertFalse(transport.connected)
        self.assertEqual(leave.await_count, 2)

    async def test_authentication_failure_stops_instead_of_retrying(self):
        error = InvalidStatus(Response(401, "Unauthorized", Headers()))
        enter = AsyncMock(side_effect=error)
        with patch.object(websockets.connect, "__aenter__", enter):
            with self.assertRaises(InvalidStatus):
                await serve(Settings(), NapCatTransport(), SimpleNamespace(submit=lambda message: None))
        enter.assert_awaited_once()
