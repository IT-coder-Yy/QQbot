import asyncio
import contextlib
import io
import os
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from qqbot.agent import Agent
from qqbot.context import ContextBuilder, ContextManager, InMemoryContextStore
from qqbot.dispatcher import MessageJob
from qqbot.listen import handle_message
from qqbot.reply import MESSAGE_BREAK, QUOTE_MARKER
from qqbot.schema import ContextRole, ReplyContent
from qqbot.settings import Settings
from qqbot.tools import ToolRegistry
from qqbot.tools.calculator import CalculatorTool
from qqbot.transport import DeliveryError, NapCatTransport
from tests.helpers import FakeSocket, eventually, message, success


def response(content=None, calls=None):
    return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(
        content=content, tool_calls=calls or [], reasoning_content=None,
    ))])


class ReplyPolicyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager = ContextManager(InMemoryContextStore(), settings=replace(Settings(), reply_interval=0))
        self.completion = AsyncMock()
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.completion)))
        env = {
            "LLM_API_KEY": "test", "LLM_BASE_URL": "https://example.invalid", "LLM_MODEL": "test",
            "LLM_CONTEXT_WINDOW": "16000", "LLM_MAX_OUTPUT_TOKENS": "1000",
            "LLM_CONTEXT_SAFETY_MARGIN": "1000",
        }
        with patch.dict(os.environ, env, clear=True), patch("qqbot.agent.AsyncOpenAI", return_value=client):
            self.agent = Agent(context_manager=self.manager, context_builder=ContextBuilder(),
                               tool_registry=ToolRegistry([CalculatorTool()]))
        self.ws = FakeSocket()
        self.transport = NapCatTransport(self.ws)
        self.addCleanup(self.transport.close)

    async def handle(self, incoming, *, proactive=False):
        await handle_message(
            MessageJob(incoming, asyncio.get_running_loop().time() + 2, proactive=proactive),
            context_manager=self.manager, agent=self.agent, intent_detector=None,
            transport=self.transport, intent_timeout=.1,
        )

    async def deliver(self, incoming, *, proactive=False):
        sent_count = len(self.ws.sent)
        with contextlib.redirect_stdout(io.StringIO()):
            task = asyncio.create_task(self.handle(incoming, proactive=proactive))
            try:
                while not task.done():
                    await eventually(lambda: len(self.ws.sent) > sent_count or task.done())
                    if len(self.ws.sent) > sent_count:
                        action = self.ws.sent[sent_count]
                        self.transport.handle_response(success(action, 500 + sent_count))
                        sent_count += 1
                    await asyncio.sleep(0)
                await task
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        return self.ws.sent[-1]["params"]["message"] if self.ws.sent else []

    def replies(self):
        return [entry.content for session in self.manager.store.sessions.values()
                for entry in session.recent_messages if entry.role is ContextRole.ASSISTANT]









    async def test_group_plain_answer_does_not_quote_despite_history_mention_and_incoming_quote(self):
        for index in range(1, 6):
            self.manager.record_user_message(message(index, user_id=20 + index, group_id=30))
        incoming = message(6, group_id=30, mentioned=True)
        incoming.content.insert(0, ReplyContent(message_id=1))
        self.completion.return_value = response(" 好呀，一起去。 ")
        segments = await self.deliver(incoming)
        self.assertEqual(segments, [{"type": "text", "data": {"text": "好呀，一起去。"}}])
        self.completion.assert_awaited_once()
        request = self.completion.await_args.kwargs["messages"]
        self.assertIn("<qqbot-quote:", request[0]["content"])
        self.assertIn("默认直接回复，不引用", request[0]["content"])
        self.assertIn("QQ:21", request[1]["content"])
        self.assertIn("引用消息ID=1", request[-1]["content"])
        self.assertEqual(self.replies(), ["好呀，一起去。"])




    async def test_model_can_quote_current_message_after_tool_call_without_storing_marker(self):
        tool_call = SimpleNamespace(id="call-1", function=SimpleNamespace(
            name="calculator", arguments='{"expression":"2+3"}',
        ))
        self.completion.side_effect = [response(calls=[tool_call]), response(f"{QUOTE_MARKER}\n答案是 5")]
        incoming = message(12, group_id=30, mentioned=True)
        incoming.content.insert(0, ReplyContent(message_id=999))
        segments = await self.deliver(incoming)
        self.assertEqual(segments, [
            {"type": "reply", "data": {"id": "12"}},
            {"type": "text", "data": {"text": "答案是 5"}},
        ])
        self.assertEqual(self.completion.await_count, 2)
        self.assertEqual(self.completion.await_args.kwargs["messages"][-1]["role"], "tool")
        self.assertEqual(self.replies(), ["答案是 5"])

    async def test_each_turn_uses_its_own_model_decision(self):
        self.completion.return_value = response(f"{QUOTE_MARKER}\n周六见。")
        quoted = await self.deliver(message(1, group_id=30, mentioned=True))
        self.completion.return_value = response("好呀。")
        plain = await self.deliver(message(2, group_id=30, mentioned=True))
        self.assertEqual(quoted[0]["type"], "reply")
        self.assertEqual(plain, [{"type": "text", "data": {"text": "好呀。"}}])
        self.assertEqual(self.replies(), ["周六见。", "好呀。"])


    async def test_private_reply_never_quotes_even_if_model_emits_marker(self):
        self.completion.return_value = response(f"{QUOTE_MARKER}\n你好。")
        segments = await self.deliver(message())
        self.assertEqual(segments, [{"type": "text", "data": {"text": "你好。"}}])
        self.assertNotIn(QUOTE_MARKER, self.completion.await_args.kwargs["messages"][0]["content"])
        self.assertEqual(self.replies(), ["你好。"])

    async def test_marker_inside_body_is_preserved_as_text(self):
        answer = f"示例标记是 `{QUOTE_MARKER}` 和 [消息ID:123]。\n```text\n[消息ID:456] 示例\n```\n~~~text\n[消息ID:789] 示例\n~~~"
        self.completion.return_value = response(answer)
        segments = await self.deliver(message(group_id=30, mentioned=True))
        self.assertEqual(segments, [{"type": "text", "data": {"text": answer}}])

    async def test_marker_without_answer_is_never_sent_or_recorded(self):
        for index, answer in enumerate((f"{QUOTE_MARKER}\n  ", "[消息ID:9000400]", f"{QUOTE_MARKER}\n[消息ID:500]")):
            with self.subTest(answer=answer):
                self.completion.return_value = response(answer)
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "空白回复"):
                    await self.handle(message(index + 1, group_id=30, mentioned=True))
                self.assertFalse(self.ws.sent)
                self.assertEqual(self.replies(), [])

    async def test_generated_message_ids_are_removed_before_send_and_history(self):
        self.manager.record_user_message(message(21, user_id=21, group_id=30))
        self.manager.record_user_message(message(22, user_id=22, group_id=30))
        self.completion.return_value = response(MESSAGE_BREAK.join([
            "[消息ID:9000400] 嗯，等你",
            "<qqbot-quote:21/>\n[消息ID:9010400] [引用消息ID:21] 周六见。",
            "[消息ID:-9010401] <qqbot-quote:22/>\n在公园见吧。",
        ]))
        await self.deliver(message(23, group_id=30, mentioned=True))
        self.assertEqual([action['params']['message'] for action in self.ws.sent], [
            [{'type': 'text', 'data': {'text': '嗯，等你'}}],
            [{'type': 'reply', 'data': {'id': '21'}}, {'type': 'text', 'data': {'text': '周六见。'}}],
            [{'type': 'reply', 'data': {'id': '22'}}, {'type': 'text', 'data': {'text': '在公园见吧。'}}],
        ])
        self.assertEqual(self.replies(), ['嗯，等你', '周六见。', '在公园见吧。'])

    async def test_generated_message_ids_after_ordinary_newlines_are_removed(self):
        self.completion.return_value = response('[消息ID:9010300] 年糕这个名字好可爱\n\n[消息ID:9010301] 又软又黏的')
        segments = await self.deliver(message(group_id=30, mentioned=True))
        answer = '年糕这个名字好可爱\n\n又软又黏的'
        self.assertEqual(segments, [{'type': 'text', 'data': {'text': '年糕这个名字好可爱\n又软又黏的'}}])
        self.assertEqual(self.replies(), [answer])

    async def test_each_bubble_can_quote_its_matching_user_message(self):
        self.manager.record_user_message(message(21, user_id=21, group_id=30))
        self.manager.record_user_message(message(22, user_id=22, group_id=30))
        self.completion.return_value = response(MESSAGE_BREAK.join([
            "刚好我也想聊这个。", "<qqbot-quote:21/>\n周六有空。", "<qqbot-quote:22/>\n在公园见吧。",
        ]))
        await self.deliver(message(23, group_id=30, mentioned=True))
        self.assertEqual([action['params']['message'] for action in self.ws.sent], [
            [{'type': 'text', 'data': {'text': '刚好我也想聊这个。'}}],
            [{'type': 'reply', 'data': {'id': '21'}}, {'type': 'text', 'data': {'text': '周六有空。'}}],
            [{'type': 'reply', 'data': {'id': '22'}}, {'type': 'text', 'data': {'text': '在公园见吧。'}}],
        ])
        self.assertEqual(self.replies(), ['刚好我也想聊这个。', '周六有空。', '在公园见吧。'])
        history = next(iter(self.manager.store.sessions.values())).recent_messages
        self.assertEqual([entry.metadata.get('reply_to_message_id') for entry in history if entry.role is ContextRole.ASSISTANT], [None, 21, 22])
        self.assertIn('引用消息ID:22', self.agent._context_message_to_llm(history[-1])['content'])
        self.completion.assert_awaited_once()

    async def test_unknown_or_other_conversation_quote_targets_are_ignored(self):
        self.manager.record_user_message(message(80, group_id=31))
        self.completion.return_value = response(f"<qqbot-quote:80/>\n好呀。{MESSAGE_BREAK}<qqbot-quote:无效/>\n出发吧。")
        await self.deliver(message(group_id=30, mentioned=True))
        self.assertTrue(all([part['type'] for part in action['params']['message']] == ['text'] for action in self.ws.sent))
        self.assertEqual(self.replies(), ['好呀。', '出发吧。'])

    async def test_ordinary_newlines_and_code_are_not_split(self):
        answer = '步骤如下：\n1. 打开设置\n2. 点击保存\n```python\nprint(1)\nprint(2)\n```'
        self.completion.return_value = response(answer)
        await self.deliver(message())
        self.assertEqual(len(self.ws.sent), 1)
        self.assertEqual(self.replies(), [answer])

    async def test_failed_second_bubble_keeps_first_and_stops_remaining_send(self):
        self.completion.return_value = response(MESSAGE_BREAK.join(['第一句。', '第二句。', '第三句。']))
        with contextlib.redirect_stdout(io.StringIO()):
            task = asyncio.create_task(self.handle(message()))
            await eventually(lambda: len(self.ws.sent) == 1)
            self.transport.handle_response(success(self.ws.sent[0], 501))
            await eventually(lambda: len(self.ws.sent) == 2)
            self.transport.handle_response({'echo': self.ws.sent[1]['echo'], 'status': 'failed', 'retcode': 100})
            with self.assertRaises(DeliveryError):
                await task
        self.assertEqual(len(self.ws.sent), 2)
        self.assertEqual(self.replies(), ['第一句。'])
