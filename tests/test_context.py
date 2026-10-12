import unittest
from dataclasses import replace

from qqbot.context import ContextBuilder, ContextManager, InMemoryContextStore
from qqbot.schema import AgentRun, TextContent
from qqbot.settings import Settings
from tests.helpers import message


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


    def test_nonpositive_retention_limits_are_rejected(self):
        for name in ["context_max_messages", "context_max_chars"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.manager(**{name: 0})
