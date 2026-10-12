import asyncio
import unittest

from qqbot.transport import DeliveryError, DeliveryUnknownError, NapCatTransport
from tests.helpers import FakeSocket, eventually, message, success


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.ws = FakeSocket()
        self.transport = NapCatTransport(self.ws, action_timeout=0.1)
        self.addCleanup(self.transport.close)

    async def start_send(self, **kwargs):
        count = len(self.ws.sent)
        task = asyncio.create_task(self.transport.reply_text(message(group_id=30), "回复", quote=True, **kwargs))
        await eventually(lambda: len(self.ws.sent) > count)
        return task

    async def test_success_matches_echo_and_keeps_quote(self):
        task = await self.start_send()
        action = self.ws.sent[-1]
        self.assertFalse(task.done())
        self.transport.handle_response({**success(action), "echo": "unrelated"})
        self.assertFalse(task.done())
        self.transport.handle_response(success(action))
        self.assertEqual(await task, 101)
        self.assertEqual(action["params"]["group_id"], 30)
        self.assertEqual(action["params"]["message"][0], {"type": "reply", "data": {"id": "1"}})
        self.assertFalse(self.transport._pending)

    async def test_concurrent_sends_match_out_of_order_responses(self):
        first = await self.start_send()
        second = await self.start_send()
        self.transport.handle_response(success(self.ws.sent[1], 202))
        self.transport.handle_response(success(self.ws.sent[0], 201))
        self.assertEqual(await first, 201)
        self.assertEqual(await second, 202)

    async def test_rejected_response_is_known_failure(self):
        task = await self.start_send()
        self.transport.handle_response({"echo": self.ws.sent[-1]["echo"], "status": "failed", "retcode": 100})
        with self.assertRaises(DeliveryError) as caught:
            await task
        self.assertNotIsInstance(caught.exception, DeliveryUnknownError)

    async def test_async_or_incomplete_response_is_unknown(self):
        for fields in [
            {"status": "async", "retcode": 1},
            {"status": "ok", "retcode": 0, "data": {}},
            {"status": "ok", "retcode": False, "data": {"message_id": 1}},
        ]:
            with self.subTest(fields=fields):
                task = await self.start_send()
                self.transport.handle_response({"echo": self.ws.sent[-1]["echo"], **fields})
                with self.assertRaises(DeliveryUnknownError):
                    await task

    async def test_missing_ack_is_unknown_and_not_retried(self):
        task = await self.start_send()
        with self.assertRaises(DeliveryUnknownError):
            await task
        self.assertEqual(len(self.ws.sent), 1)
        self.assertFalse(self.transport._pending)
        self.transport.handle_response(success(self.ws.sent[-1]))

    async def test_disconnect_fails_pending_and_new_connection_does_not_resend(self):
        task = await self.start_send()
        self.transport.detach(self.ws)
        with self.assertRaises(DeliveryUnknownError):
            await task
        new_ws = FakeSocket()
        self.transport.attach(new_ws)
        self.transport.detach(self.ws)
        self.assertTrue(self.transport.connected)
        self.assertFalse(new_ws.sent)
        self.assertFalse(self.transport._pending)

    async def test_unsent_request_waits_for_reconnect_with_original_deadline(self):
        self.transport.detach(self.ws)
        deadline = asyncio.get_running_loop().time() + 1
        task = asyncio.create_task(self.transport.reply_text(message(), "回复", deadline=deadline))
        await asyncio.sleep(0)
        self.assertFalse(self.ws.sent)
        self.transport.attach(self.ws)
        await eventually(lambda: self.ws.sent)
        self.transport.handle_response(success(self.ws.sent[-1]))
        self.assertEqual(await task, 101)

    async def test_expired_unsent_request_never_sends_after_reconnect(self):
        self.transport.detach(self.ws)
        with self.assertRaises(TimeoutError):
            await self.transport.reply_text(message(), "回复", deadline=asyncio.get_running_loop().time() + 0.01)
        self.transport.attach(self.ws)
        self.assertFalse(self.ws.sent)

    async def test_cancellation_cleans_pending(self):
        task = await self.start_send()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(self.transport._pending)

    async def test_deadline_already_passed_does_not_send(self):
        with self.assertRaises(TimeoutError):
            await self.transport.reply_text(message(), "回复", deadline=asyncio.get_running_loop().time() - 1)
        self.assertFalse(self.ws.sent)
