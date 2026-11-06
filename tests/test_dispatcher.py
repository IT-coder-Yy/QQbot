import asyncio
import unittest
from dataclasses import replace
from unittest.mock import AsyncMock

from qqbot.dispatcher import MessageDispatcher
from qqbot.schema import ImageContent
from qqbot.settings import Settings
from tests.helpers import eventually, message


class DispatcherTests(unittest.IsolatedAsyncioTestCase):
    def create_dispatcher(self, handler, **overrides):
        overrides.setdefault('input_debounce_seconds', 0)
        notices = AsyncMock()
        dispatcher = MessageDispatcher(replace(Settings(), **overrides), handler, notices)
        self.addAsyncCleanup(dispatcher.close)
        return dispatcher, notices

    async def test_slow_conversation_does_not_block_others_or_reorder_same_conversation(self):
        gate = asyncio.Event()
        started = []
        finished = []

        async def handle(job):
            started.append(job.message.message_id)
            if job.message.message_id == 1:
                await gate.wait()
            finished.append(job.message.message_id)

        dispatcher, _ = self.create_dispatcher(handle, concurrency=2)
        dispatcher.submit(message(1))
        dispatcher.submit(message(2))
        dispatcher.submit(message(3, user_id=21))
        await eventually(lambda: 3 in finished)
        self.assertEqual(started, [1, 3])
        gate.set()
        await eventually(lambda: not dispatcher._workers)
        self.assertEqual(finished, [3, 1, 2])
        self.assertEqual(dispatcher._waiting, 0)

    async def test_consecutive_texts_merge_without_crossing_other_speakers(self):
        jobs = []
        async def handle(job):
            jobs.append(job)
        dispatcher, _ = self.create_dispatcher(handle, input_debounce_seconds=.02)
        for incoming in [message(1, group_id=30), message(2, group_id=30),
                         message(3, user_id=21, group_id=30), message(4, group_id=30)]:
            dispatcher.submit(incoming)
        await eventually(lambda: not dispatcher._workers)
        self.assertEqual([job.message.message_id for job in jobs], [2, 3, 4])
        self.assertEqual([[m.message_id for m in job.previous_messages] for job in jobs], [[1], [], []])
        self.assertEqual(dispatcher._waiting, 0)
        self.assertFalse(dispatcher._inflight)

    async def test_text_debounce_does_not_hold_a_slot_or_merge_images(self):
        seen = []
        async def handle(job):
            seen.append(job.message.message_id)
            self.assertFalse(job.previous_messages)
        dispatcher, _ = self.create_dispatcher(handle, input_debounce_seconds=.05, concurrency=1)
        dispatcher.submit(message(1, group_id=30))
        picture = message(2, group_id=31)
        picture.content = [ImageContent(file='synthetic.png')]
        dispatcher.submit(picture)
        await eventually(lambda: seen)
        self.assertEqual(seen, [2])
        same_group_picture = picture.model_copy(update={'conversation_id':message(group_id=30).conversation_id, 'group_id':30, 'message_id':3})
        dispatcher.submit(same_group_picture)
        await eventually(lambda: not dispatcher._workers)
        self.assertEqual(seen, [2, 1, 3])

    async def test_debounce_deadline_and_close_leave_no_queued_ids(self):
        handler = AsyncMock()
        dispatcher, notices = self.create_dispatcher(handler, input_debounce_seconds=.2, task_timeout=.01)
        dispatcher.submit(message(1))
        dispatcher.submit(message(2))
        await eventually(lambda: not dispatcher._workers)
        handler.assert_not_awaited()
        self.assertFalse(dispatcher._inflight)
        self.assertEqual(dispatcher._waiting, 0)
        dispatcher.settings = replace(dispatcher.settings, task_timeout=1)
        dispatcher.submit(message(3))
        await asyncio.sleep(0)
        await dispatcher.close()
        self.assertFalse(dispatcher._running)
        self.assertFalse(dispatcher._inflight)
        self.assertEqual(dispatcher._waiting, 0)

    async def test_concurrency_and_queue_limits_reject_new_work(self):
        gate = asyncio.Event()
        started = []

        async def handle(job):
            started.append(job.message.message_id)
            await gate.wait()

        dispatcher, notices = self.create_dispatcher(handle, concurrency=1, queue_limit=2, conversation_queue_limit=1)
        self.assertTrue(dispatcher.submit(message(1)))
        await eventually(lambda: started)
        self.assertTrue(dispatcher.submit(message(2)))
        self.assertFalse(dispatcher.submit(message(3)))
        self.assertTrue(dispatcher.submit(message(4, user_id=21)))
        self.assertFalse(dispatcher.submit(message(5, user_id=22)))
        await eventually(lambda: notices.await_count > 0)
        self.assertEqual(started, [1])
        self.assertEqual(dispatcher._waiting, 2)
        gate.set()
        await eventually(lambda: not dispatcher._workers)
        self.assertCountEqual(started, [1, 2, 4])

    async def test_duplicate_events_are_filtered_even_while_first_is_running(self):
        gate = asyncio.Event()
        handler = AsyncMock(side_effect=lambda job: None)

        async def handle(job):
            await handler(job)
            await gate.wait()

        dispatcher, _ = self.create_dispatcher(handle, dedup_capacity=1)
        self.assertTrue(dispatcher.submit(message(1)))
        await eventually(lambda: handler.await_count == 1)
        dispatcher.submit(message(2, user_id=21))
        self.assertFalse(dispatcher.submit(message(1)))
        self.assertEqual(len(dispatcher._seen), 1)
        gate.set()
        await eventually(lambda: not dispatcher._workers)
        self.assertEqual(handler.await_count, 2)

    async def test_dedup_expires_and_is_scoped_to_conversation(self):
        handler = AsyncMock()
        dispatcher, _ = self.create_dispatcher(handler, dedup_ttl=0.01)
        self.assertTrue(dispatcher.submit(message(1)))
        self.assertTrue(dispatcher.submit(message(1, user_id=21)))
        self.assertFalse(dispatcher.submit(message(1)))
        await eventually(lambda: not dispatcher._workers)
        await asyncio.sleep(0.02)
        self.assertTrue(dispatcher.submit(message(1)))
        await eventually(lambda: not dispatcher._workers)
        self.assertEqual(handler.await_count, 3)

    async def test_deadline_includes_waiting_for_concurrency_slot(self):
        gate = asyncio.Event()
        started = []

        async def handle(job):
            started.append(job.message.message_id)
            await gate.wait()

        dispatcher, notices = self.create_dispatcher(handle, concurrency=1, task_timeout=0.02)
        dispatcher.submit(message(1))
        await eventually(lambda: started)
        dispatcher.submit(message(2, user_id=21))
        queued = dispatcher._queues[message(user_id=21).conversation_id][0]
        queued.deadline = asyncio.get_running_loop().time() + 0.005
        await eventually(lambda: not dispatcher._workers)
        self.assertEqual(started, [1])
        self.assertEqual(dispatcher._waiting, 0)
        await eventually(lambda: notices.await_count == 2)

    async def test_notice_tasks_and_cooldown_are_bounded(self):
        gate = asyncio.Event()
        dispatcher, _ = self.create_dispatcher(AsyncMock(), concurrency=1)
        notice = AsyncMock(side_effect=lambda *args: None)

        async def hold_notice(*args):
            await notice(*args)
            await gate.wait()

        dispatcher.send_notice = hold_notice
        dispatcher._notify(message(), "忙")
        dispatcher._notify(message(2), "忙")
        dispatcher._notify(message(3, user_id=21), "忙")
        dispatcher._notify(message(4, group_id=30), "忙")
        self.assertEqual(len(dispatcher._notifications), 1)
        await eventually(lambda: notice.await_count == 1)
        gate.set()
        await eventually(lambda: not dispatcher._notifications)
        dispatcher._notify(message(5), "忙")
        self.assertFalse(dispatcher._notifications)

    async def test_close_cancels_active_and_queued_work(self):
        handler = AsyncMock(side_effect=asyncio.Event().wait)

        async def hold(job):
            await handler()

        dispatcher, _ = self.create_dispatcher(hold, concurrency=1)
        dispatcher.submit(message(1))
        dispatcher.submit(message(2))
        dispatcher.submit(message(3, user_id=21))
        await eventually(lambda: handler.await_count == 1)
        await dispatcher.close()
        self.assertFalse(dispatcher._workers)
        self.assertFalse(dispatcher._inflight)
        self.assertEqual(dispatcher._waiting, 0)
        self.assertFalse(dispatcher.submit(message(4)))

    async def test_self_messages_are_ignored_before_processing(self):
        handler = AsyncMock()
        dispatcher, notices = self.create_dispatcher(handler)
        self.assertFalse(dispatcher.submit(message(user_id=10)))
        handler.assert_not_awaited()
        notices.assert_not_awaited()
