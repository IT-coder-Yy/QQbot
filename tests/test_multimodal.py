import tempfile
import unittest
from pathlib import Path


from qqbot.context import ContextManager
from qqbot.context.sqlite_store import SQLiteContextStore
from qqbot.schema import TextContent
from tests.helpers import message


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.store = SQLiteContextStore(self.root / 'bot.db', self.root / 'files')
        self.addCleanup(lambda: self.store.close())
        self.manager = ContextManager(self.store)

    def test_restart_preserves_archives_and_summary_boundary(self):
        incoming = message(1)
        for i in range(1, 5):
            conv, session, turn = self.manager.record_user_message(message(i))
        session.summary.narrative = '旧消息的摘要'
        session.summary_through_seq = 2
        session.recent_messages = session.recent_messages[2:]
        self.manager.save_session(session)
        self.assertEqual(len(self.store.history(incoming.conversation_id)), 4)
        self.store.close()
        self.store = SQLiteContextStore(self.root / 'bot.db', self.root / 'files')
        restored = self.store.get_session(incoming.conversation_id)
        self.assertEqual([m.seq for m in restored.recent_messages], [3, 4])
        self.assertEqual(restored.summary.narrative, '旧消息的摘要')
        manager = ContextManager(self.store)
        _, session, last = manager.record_user_message(message(5))
        self.assertEqual(last.seq, 5)
        self.assertEqual(len(self.store.history(incoming.conversation_id)), 5)
        self.assertTrue(self.store.has_message(incoming.conversation_id, 1))
        tables = {r[0] for r in self.store.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertEqual(tables, {'conversations', 'attachments'})

    def test_global_quota_cleans_old_conversation_not_idle_ttl(self):
        for user in (20, 21):
            for index in range(3):
                incoming = message(index, user_id=user)
                incoming.content = [TextContent(text='保留历史' * 60)]
                self.manager.record_user_message(incoming)
            self.store.release(message(user_id=user).conversation_id)
        before = self.store.db.execute('SELECT sum(storage_bytes) FROM conversations').fetchone()[0]
        self.store.conversation_limit = before - 1000
        self.store.cleanup()
        self.assertLessEqual(self.store.db.execute('SELECT sum(storage_bytes) FROM conversations').fetchone()[0], self.store.conversation_limit)
        self.assertEqual(len(self.store.history(message(user_id=21).conversation_id)), 3)
        self.manager.cleanup_expired()
        self.assertEqual(len(self.store.history(message(user_id=21).conversation_id)), 3)

    def test_one_active_conversation_can_evict_own_old_messages(self):
        self.store.conversation_limit = 2200
        for i in range(20):
            incoming = message(i)
            incoming.content = [TextContent(text='中' * 60)]
            with self.manager.processing(incoming):
                pass
        history = self.store.history(message().conversation_id)
        self.assertEqual(history[-1]['source_message_id'], 19)
        self.assertLess(len(history), 20)
        self.assertLessEqual(self.store.db.execute('SELECT sum(storage_bytes) FROM conversations').fetchone()[0], 2200)
        self.assertFalse(self.store.sessions)
