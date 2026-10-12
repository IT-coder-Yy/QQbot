import asyncio
import unittest
from dataclasses import replace
from unittest.mock import patch

from qqbot.context import ContextBuilder, ContextManager, InMemoryContextStore
from qqbot.listen import cleanup_contexts
from qqbot.schema import AgentRun, TextContent
from qqbot.settings import Settings
from tests.helpers import eventually, message


class ContextRetentionTests(unittest.TestCase):
    def manager(self, **limits):
        return ContextManager(InMemoryContextStore(), settings=replace(Settings(), **limits))

    def test_large_message_stream_keeps_only_latest_history_and_existing_summary(self):
        manager = self.manager(context_max_messages=20)
        _, session, _ = manager.record_user_message(message(0, group_id=30))
        session.summary.narrative = "已经生成的摘要"
        for message_id in range(1, 10_001):
            manager.record_user_message(message(message_id, group_id=30))
        self.assertEqual([item.source_message_id for item in session.recent_messages], list(range(9981, 10_001)))
        self.assertEqual(session.summary.narrative, "已经生成的摘要")

    def test_character_budget_includes_sender_names_and_assistant_replies(self):
        manager = self.manager(context_max_chars=20)
        incoming = message().model_copy(update={"content": [TextContent(text="甲" * 6)]})
        conversation, session, _ = manager.record_user_message(incoming)
        manager.record_user_message(incoming.model_copy(update={"message_id": 2}))
        self.assertEqual(len(session.recent_messages), 2)  # 每条 6 字正文 + 4 字昵称。
        manager.record_assistant_message(conversation=conversation, session=session, content="乙" * 10, timestamp=incoming.timestamp, source_message_id=3)
        self.assertEqual([item.source_message_id for item in session.recent_messages], [2, 3])
        manager.record_assistant_message(conversation=conversation, session=session, content="丙" * 11, timestamp=incoming.timestamp, source_message_id=4)
        self.assertEqual([item.source_message_id for item in session.recent_messages], [4])

    def test_oversized_turn_is_complete_for_model_but_not_retained(self):
        manager = self.manager(context_max_chars=20)
        manager.record_user_message(message(1))
        incoming = message(2).model_copy(update={"content": [TextContent(text="长" * 100)]})
        conversation, session, turn = manager.record_user_message(incoming)
        run = AgentRun(run_id="test", conversation_id=conversation.conversation_id, session_id=session.session_id, current_turn=turn, started_at=incoming.timestamp)
        context = ContextBuilder().build(system_prompt="测试", conversation=conversation, session=session, run=run)
        self.assertEqual(context.current_turn.content, "长" * 100)
        self.assertEqual([item.source_message_id for item in context.recent_messages], [1])
        reply = manager.record_assistant_message(conversation=conversation, session=session, content="答" * 100, timestamp=incoming.timestamp)
        self.assertEqual(reply.content, "答" * 100)
        self.assertEqual([item.source_message_id for item in session.recent_messages], [1])

    def test_capacity_evicts_least_recently_used_conversation_and_related_state(self):
        manager = self.manager(context_max_conversations=2)
        first, first_session, first_turn = manager.record_user_message(message(user_id=20))
        second, second_session, second_turn = manager.record_user_message(message(user_id=21))
        for conversation, session, turn in [(first, first_session, first_turn), (second, second_session, second_turn)]:
            manager.store.save_run(AgentRun(run_id=conversation.conversation_id, conversation_id=conversation.conversation_id, session_id=session.session_id, current_turn=turn, started_at=turn.timestamp))
        manager.record_user_message(message(2, user_id=20))
        third, _, _ = manager.record_user_message(message(user_id=22))
        self.assertEqual(set(manager.store.conversations), {first.conversation_id, third.conversation_id})
        self.assertNotIn(second_session.session_id, manager.store.sessions)
        self.assertEqual(set(manager.store.runs), {first.conversation_id})
        self.assertIs(manager.store.get_session(first_session.session_id), first_session)
        for user_id in range(30, 1030):
            manager.record_user_message(message(user_id=user_id))
        self.assertEqual(len(manager.store.conversations), 2)
        self.assertEqual(len(manager.store.sessions), 2)
        self.assertEqual(len(manager._last_used), 2)
        self.assertFalse(manager.store.runs)

    def test_idle_expiry_uses_local_clock_and_returns_fresh_context(self):
        manager = self.manager(context_idle_timeout=10)
        with patch("qqbot.context.manager.monotonic", return_value=0) as clock:
            conversation, session, _ = manager.record_user_message(message())
            session.summary.narrative = "过期后删除"
            clock.return_value = 9
            manager.cleanup_expired()
            self.assertIn(conversation.conversation_id, manager.store.conversations)
            clock.return_value = 10
            manager.cleanup_expired()
            self.assertFalse(manager.store.conversations)
            self.assertFalse(manager.store.sessions)
            self.assertFalse(manager._last_used)
            _, fresh_session, _ = manager.record_user_message(message(2))
            self.assertNotEqual(fresh_session.session_id, session.session_id)
            self.assertEqual(fresh_session.summary.narrative, "")

    def test_busy_context_survives_expiry_and_capacity_pressure(self):
        manager = self.manager(context_max_conversations=2, context_idle_timeout=10)
        with patch("qqbot.context.manager.monotonic", return_value=0) as clock:
            with manager.processing(message(user_id=20)) as (active, session, _):
                idle, _, _ = manager.record_user_message(message(user_id=21))
                manager.record_user_message(message(user_id=22))
                self.assertNotIn(idle.conversation_id, manager.store.conversations)
                clock.return_value = 20
                manager.cleanup_expired()
                self.assertEqual(list(manager.store.conversations), [active.conversation_id])
                self.assertIs(manager.store.get_session(session.session_id), session)
            clock.return_value = 29
            manager.cleanup_expired()
            self.assertIn(active.conversation_id, manager.store.conversations)
            clock.return_value = 30
            manager.cleanup_expired()
            self.assertFalse(manager.store.conversations)

    def test_full_busy_cache_rejects_new_context_and_exception_releases_protection(self):
        manager = self.manager(context_max_conversations=1)
        with self.assertRaisesRegex(ValueError, "模拟失败"):
            with manager.processing(message()) as (conversation, _, _):
                with self.assertRaisesRegex(RuntimeError, "上下文容量已满"):
                    manager.record_user_message(message(user_id=21))
                self.assertEqual(list(manager.store.conversations), [conversation.conversation_id])
                raise ValueError("模拟失败")
        manager.record_user_message(message(user_id=21))
        self.assertNotIn(conversation.conversation_id, manager.store.conversations)
        self.assertFalse(manager._active)

    def test_nonpositive_retention_limits_are_rejected(self):
        for name in ["context_max_messages", "context_max_chars", "context_max_conversations", "context_idle_timeout", "context_cleanup_interval"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.manager(**{name: 0})


class ContextCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_idle_context_is_cleaned_without_new_messages(self):
        manager = ContextManager(InMemoryContextStore(), settings=replace(Settings(), context_idle_timeout=10, context_cleanup_interval=0.001))
        with patch("qqbot.context.manager.monotonic", return_value=0) as clock:
            manager.record_user_message(message())
            task = asyncio.create_task(cleanup_contexts(manager))
            try:
                clock.return_value = 10
                await eventually(lambda: not manager.store.conversations)
                self.assertFalse(manager.store.sessions)
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            self.assertTrue(task.cancelled())
