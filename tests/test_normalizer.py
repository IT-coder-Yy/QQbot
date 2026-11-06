import unittest

from qqbot.chat_context import build_intent_state
from qqbot.context import ContextManager, InMemoryContextStore
from qqbot.normalizer import normalize_event
from tests.helpers import raw_message


class EmojiTests(unittest.TestCase):
    def test_face_names_and_unicode_keep_order_in_intent_context(self):
        event = raw_message(group_id=30)
        event['message'] = [
            {'type': 'text', 'data': {'text': '😂'}},
            {'type': 'face', 'data': {'id': '20', 'raw': {'faceText': '/偷笑'}}},
            {'type': 'text', 'data': {'text': '👨‍👩‍👧‍👦👍🏽'}},
            {'type': 'face', 'data': {'id': '4', 'raw': {'faceText': '/得意'}}},
        ]
        incoming = normalize_event(event)
        expected = '😂[QQ表情：/偷笑]👨‍👩‍👧‍👦👍🏽[QQ表情：/得意]'
        self.assertEqual(incoming.text, expected)
        self.assertNotIn('unhandled_segments', incoming.metadata)
        _, session, turn = ContextManager(InMemoryContextStore()).record_user_message(incoming)
        self.assertEqual(build_intent_state(session=session, current_turn=turn)['current_message']['text'], expected)

    def test_face_without_name_is_preserved_without_guessing(self):
        for raw in (None, {}, {'faceText': ''}):
            with self.subTest(raw=raw):
                event = raw_message()
                event['message'] = [{'type': 'face', 'data': {'id': '9999', 'raw': raw}}]
                self.assertEqual(normalize_event(event).text, '[QQ表情 ID:9999，名称未知]')
