import asyncio
import contextlib
import io
import json
import os
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from pydantic import ValidationError

from qqbot.agent import Agent
from qqbot.context import ContextBuilder, ContextManager
from qqbot.context.sqlite_store import SQLiteContextStore
from qqbot.dispatcher import MessageDispatcher, MessageJob
from qqbot.listen import handle_message
from qqbot.proactive import ProactiveDecision, next_active_time, run_proactive
from qqbot.reply import MESSAGE_BREAK
from qqbot.schema import ProactiveEvent
from qqbot.settings import Settings
from qqbot.tools import ToolRegistry
from qqbot.tools.web_search import WebSearchTool
from qqbot.transport import DeliveryUnknownError
from tests.helpers import eventually, message
from tests.test_reply import response


class Clock:
    value = datetime.fromisoformat('2026-10-27T10:00:00+08:00')

    @classmethod
    def now(cls, tz=None):
        return cls.value.astimezone(tz)


def event(group_id=30):
    return ProactiveEvent(bot_id=10, group_id=group_id, timestamp=Clock.value)


def decision(reply='', next_at=None):
    return json.dumps({'reply': reply, 'next_check_at': (next_at or Clock.value + timedelta(minutes=5)).isoformat(),
                       'reason': '有新话题再来看看'}, ensure_ascii=False)


class ProactiveTimeTests(unittest.TestCase):
    def test_activity_window_boundaries_and_utc_input(self):
        cases = [('2026-10-27T09:59:59+08:00', '2026-10-27T10:00:00+08:00'),
                 ('2026-10-27T10:00:00+08:00', '2026-10-27T10:00:00+08:00'),
                 ('2026-10-27T22:59:59+08:00', '2026-10-27T22:59:59+08:00'),
                 ('2026-10-27T15:00:00+00:00', '2026-10-28T10:00:00+08:00')]
        for source, expected in cases:
            with self.subTest(source=source):
                self.assertEqual(next_active_time(datetime.fromisoformat(source), Settings()).isoformat(), expected)
        all_day = replace(Settings(), proactive_start_hour=0, proactive_end_hour=24)
        late = datetime.fromisoformat('2026-10-27T23:59:00+08:00')
        self.assertEqual(next_active_time(late, all_day), late)

    def test_decision_requires_future_zoned_time_and_text_fields(self):
        now = datetime.fromisoformat('2026-10-27T10:00:00+08:00')
        valid = {'reply': '', 'next_check_at': '2026-10-27T10:05:00+08:00', 'reason': '先看看'}
        self.assertEqual(ProactiveDecision.model_validate(valid, context={'now': now}).reply, '')
        for change in ({'reply': None}, {'reply': 1}, {'next_check_at': '2026-10-27T10:05:00'},
                       {'next_check_at': now.isoformat()}, {'next_check_at': '2026-10-26T10:00:00+08:00'},
                       {'unexpected': True}):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                ProactiveDecision.model_validate({**valid, **change}, context={'now': now})

    def test_environment_controls_window_refresh_and_retry(self):
        env = {'NAPCAT_TOKEN': 'test', 'CHAT_PROACTIVE_TIMEZONE': 'UTC', 'CHAT_PROACTIVE_START_HOUR': '0',
               'CHAT_PROACTIVE_END_HOUR': '24', 'CHAT_PROACTIVE_GROUP_REFRESH_SECONDS': '60', 'CHAT_PROACTIVE_RETRY_SECONDS': '120'}
        with patch.dict(os.environ, env, clear=True):
            settings = Settings.from_env()
        self.assertEqual((settings.proactive_timezone, settings.proactive_start_hour, settings.proactive_end_hour), ('UTC', 0, 24))
        self.assertEqual((settings.proactive_refresh_seconds, settings.proactive_retry_seconds), (60, 120))
        for key, value in [('CHAT_PROACTIVE_TIMEZONE', 'not/a-zone'), ('CHAT_PROACTIVE_START_HOUR', '24'),
                           ('CHAT_PROACTIVE_END_HOUR', '0'), ('CHAT_PROACTIVE_GROUP_REFRESH_SECONDS', '0'),
                           ('CHAT_PROACTIVE_RETRY_SECONDS', 'nan')]:
            with self.subTest(key=key), patch.dict(os.environ, {**env, key: value}, clear=True), self.assertRaises((ValueError, KeyError)):
                Settings.from_env()

    def test_version_one_migration_preserves_history_and_schedule_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SQLiteContextStore(root / 'bot.db', root / 'files')
            manager = ContextManager(store)
            incoming = message(group_id=30)
            manager.record_user_message(incoming)
            original = store.history(incoming.conversation_id)
            last = store.db.execute('SELECT last_message_at FROM conversations').fetchone()[0]
            store.db.execute('ALTER TABLE conversations DROP COLUMN next_proactive_at')
            store.db.execute('PRAGMA user_version=1')
            store.close()
            store = SQLiteContextStore(root / 'bot.db', root / 'files')
            due = datetime.fromisoformat('2026-10-28T12:30:00+08:00')
            store.schedule_proactive(incoming.conversation_id, due)
            self.assertEqual(store.history(incoming.conversation_id), original)
            self.assertEqual(store.db.execute('SELECT last_message_at FROM conversations').fetchone()[0], last)
            store.close()
            store = SQLiteContextStore(root / 'bot.db', root / 'files')
            try:
                self.assertEqual(store.next_proactive_time(incoming.conversation_id), due)
                self.assertEqual(store.db.execute('PRAGMA user_version').fetchone()[0], 2)
                self.assertEqual(store.history(incoming.conversation_id), original)
            finally:
                store.close()


class ProactiveFlowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        Clock.value = datetime.fromisoformat('2026-10-27T10:00:00+08:00')
        self.enterContext(patch('qqbot.listen.datetime', Clock))
        self.enterContext(patch('qqbot.proactive.datetime', Clock))
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.store = SQLiteContextStore(root / 'bot.db', root / 'files')
        self.addCleanup(self.store.close)
        self.settings = replace(Settings(), reply_interval=0, input_debounce_seconds=0)
        self.manager = ContextManager(self.store, settings=self.settings)
        self.agent = SimpleNamespace(run=AsyncMock(return_value=decision()))
        self.transport = SimpleNamespace(connected=True, wait_connected=AsyncMock(return_value=object()),
                                         send_group_text=AsyncMock(return_value=501), reply_text=AsyncMock(return_value=502))

    async def handle(self, incoming=None, *, proactive=True):
        await handle_message(MessageJob(incoming or event(), asyncio.get_running_loop().time() + 2, proactive=proactive),
                             context_manager=self.manager, agent=self.agent, intent_detector=None,
                             transport=self.transport, intent_timeout=.1)

    async def test_silent_new_group_schedules_next_check_without_fabricated_messages(self):
        await self.handle()
        self.transport.send_group_text.assert_not_awaited()
        self.assertEqual(self.store.history(event().conversation_id), [])
        self.assertEqual(self.store.next_proactive_time(event().conversation_id), Clock.value + timedelta(minutes=5))
        turn = self.agent.run.await_args.kwargs['current_turn']
        self.assertIsNone(turn.source_message_id)
        self.assertIsNone(turn.sender_id)
        self.assertTrue(turn.metadata['autonomous'])
        self.assertIsNone(turn.metadata['last_message_at'])

    async def test_no_reply_allows_another_topic_with_no_daily_limit(self):
        answers = iter(['想聊电影。', '又发现一个小项目。', ''])
        self.agent.run.side_effect = lambda **kwargs: decision(next(answers))
        for _ in range(3):
            await self.handle()
            Clock.value += timedelta(minutes=5)
        self.assertEqual(self.agent.run.await_count, 3)
        self.assertEqual([row['content'] for row in self.store.history(event().conversation_id)], ['想聊电影。', '又发现一个小项目。'])
        self.assertEqual(self.transport.send_group_text.await_count, 2)
        self.assertTrue(all(row['role'] == 'assistant' and row['metadata']['proactive_reply'] for row in self.store.history(event().conversation_id)))
        self.assertEqual(self.store.next_proactive_time(event().conversation_id), datetime.fromisoformat('2026-10-27T10:15:00+08:00'))

    async def test_quote_only_own_group_history_and_keep_stop_request_as_context(self):
        incoming = message(12, group_id=30)
        incoming.content[0].text = '别聊了'
        self.manager.record_user_message(incoming)
        self.manager.record_user_message(message(99, group_id=31))
        self.agent.run.return_value = decision('<qqbot-quote:12/>我还有个想法。' + MESSAGE_BREAK + '<qqbot-quote:99/>稍后再聊。')
        await self.handle()
        calls = self.transport.send_group_text.await_args_list
        self.assertEqual([call.kwargs['reply_to_message_id'] for call in calls], [12, None])
        self.assertEqual(self.agent.run.await_count, 1)
        self.assertEqual([row['role'] for row in self.store.history(event().conversation_id)], ['user', 'assistant', 'assistant'])
        self.assertEqual(len(self.store.history(event(31).conversation_id)), 1)

    async def test_night_skips_proactive_but_normal_reply_still_works(self):
        Clock.value = datetime.fromisoformat('2026-10-27T23:00:00+08:00')
        await self.handle()
        self.agent.run.assert_not_awaited()
        self.assertEqual(self.store.next_proactive_time(event().conversation_id), datetime.fromisoformat('2026-10-28T10:00:00+08:00'))
        self.agent.run.return_value = '还醒着呢。'
        await self.handle(message(group_id=30, mentioned=True), proactive=False)
        self.transport.reply_text.assert_awaited_once()

    async def test_generation_crossing_23_stops_actual_send(self):
        Clock.value = datetime.fromisoformat('2026-10-27T22:59:59+08:00')
        async def generate(**kwargs):
            Clock.value += timedelta(seconds=1)
            return decision('想起个事。')
        self.agent.run.side_effect = generate
        await self.handle()
        self.transport.send_group_text.assert_not_awaited()
        self.assertEqual(self.store.history(event().conversation_id), [])
        self.assertEqual(self.store.next_proactive_time(event().conversation_id), datetime.fromisoformat('2026-10-28T10:00:00+08:00'))

    async def test_split_reply_stops_at_window_and_keeps_only_confirmed_first_part(self):
        Clock.value = datetime.fromisoformat('2026-10-27T22:59:59+08:00')
        self.agent.run.return_value = decision('第一句。' + MESSAGE_BREAK + '第二句。')
        async def send(*args, **kwargs):
            Clock.value += timedelta(seconds=1)
            return 501
        self.transport.send_group_text.side_effect = send
        await self.handle()
        self.transport.send_group_text.assert_awaited_once()
        self.assertEqual([row['content'] for row in self.store.history(event().conversation_id)], ['第一句。'])

    async def test_invalid_output_or_unknown_delivery_keeps_retry_without_fake_reply(self):
        retry = Clock.value + timedelta(minutes=15)
        self.store.schedule_proactive(event().conversation_id, retry)
        self.agent.run.return_value = '刚刚没能答上来，能再问我一次吗？'
        with self.assertRaises(ValidationError):
            await self.handle()
        self.transport.send_group_text.assert_not_awaited()
        self.agent.run.return_value = decision('聊个新项目。')
        self.transport.send_group_text.side_effect = DeliveryUnknownError('未确认')
        with self.assertRaises(DeliveryUnknownError):
            await self.handle()
        self.assertEqual(self.store.history(event().conversation_id), [])
        self.assertEqual(self.store.next_proactive_time(event().conversation_id), retry)

    async def test_disconnect_skips_model_and_stale_generated_draft(self):
        self.transport.connected = False
        await self.handle()
        self.agent.run.assert_not_awaited()
        self.transport.connected = True
        async def generate(**kwargs):
            self.transport.connected = False
            return decision('这条不应发出去。')
        self.agent.run.side_effect = generate
        await self.handle()
        self.transport.send_group_text.assert_not_awaited()
        self.assertEqual(self.store.history(event().conversation_id), [])

    async def test_scheduler_discovers_all_groups_refreshes_membership_and_restores_future_plan(self):
        groups = [30, 31]
        socket = object()
        async def request(action, params):
            return {'user_id': 10} if action == 'get_login_info' else [{'group_id': i} for i in groups]
        self.transport.request = AsyncMock(side_effect=request)
        self.transport.wait_connected.side_effect = lambda: socket
        dispatcher = SimpleNamespace(submit_proactive=Mock(return_value=True))
        self.store.schedule_proactive(event(31).conversation_id, Clock.value + timedelta(hours=1))
        steps = 0
        async def tick(seconds):
            nonlocal steps, socket
            steps += 1
            calls = [call.args[0].group_id for call in dispatcher.submit_proactive.call_args_list]
            if steps == 1:
                self.assertEqual(calls, [30])
                self.assertEqual(self.store.history(event().conversation_id), [])
                groups[:] = [31, 32]
                Clock.value += timedelta(minutes=5)
            elif steps == 2:
                self.assertEqual(calls, [30, 32])
                Clock.value += timedelta(minutes=55)
                socket = object()
            else:
                self.assertEqual(calls, [30, 32, 31, 32])
                self.assertEqual(sum(call.args[0] == 'get_login_info' for call in self.transport.request.await_args_list), 2)
                raise asyncio.CancelledError
        with patch('qqbot.proactive.asyncio', SimpleNamespace(sleep=tick)), self.assertRaises(asyncio.CancelledError):
            await run_proactive(self.settings, self.store, self.transport, dispatcher)

    async def test_scheduler_honors_model_reschedule_and_silence_with_real_dispatcher(self):
        self.transport.request = AsyncMock(side_effect=lambda action, params: {'user_id': 10} if action == 'get_login_info' else [{'group_id': 30}])
        self.agent.run.side_effect = lambda **kwargs: decision('新话题。' if self.agent.run.await_count == 1 else '', Clock.value + timedelta(seconds=2))
        async def handle(job):
            await self.handle(job.message)
        dispatcher = MessageDispatcher(self.settings, handle, AsyncMock())
        self.addAsyncCleanup(dispatcher.close)
        steps = 0
        async def tick(seconds):
            nonlocal steps
            await eventually(lambda: not dispatcher._workers)
            steps += 1
            if steps == 1:
                self.assertEqual(self.agent.run.await_count, 1)
                Clock.value += timedelta(seconds=1)
            elif steps == 2:
                self.assertEqual(self.agent.run.await_count, 1)
                Clock.value += timedelta(seconds=1)
            else:
                self.assertEqual(self.agent.run.await_count, 2)
                self.transport.send_group_text.assert_awaited_once()
                self.assertEqual(self.store.next_proactive_time(event().conversation_id), Clock.value + timedelta(seconds=2))
                raise asyncio.CancelledError
        with patch('qqbot.proactive.asyncio', SimpleNamespace(sleep=tick)), self.assertRaises(asyncio.CancelledError):
            await run_proactive(self.settings, self.store, self.transport, dispatcher)

    async def test_scheduler_keeps_failure_retry_without_qq_notice(self):
        self.transport.request = AsyncMock(side_effect=lambda action, params: {'user_id': 10} if action == 'get_login_info' else [{'group_id': 30}])
        self.agent.run.side_effect = RuntimeError('模拟主模型错误')
        notices = AsyncMock()
        async def handle(job):
            await self.handle(job.message)
        dispatcher = MessageDispatcher(self.settings, handle, notices)
        self.addAsyncCleanup(dispatcher.close)
        async def tick(seconds):
            await eventually(lambda: not dispatcher._workers)
            self.assertEqual(self.store.next_proactive_time(event().conversation_id), Clock.value + timedelta(seconds=self.settings.proactive_retry_seconds))
            self.assertEqual(self.store.history(event().conversation_id), [])
            self.transport.send_group_text.assert_not_awaited()
            notices.assert_not_awaited()
            raise asyncio.CancelledError
        with patch('qqbot.proactive.asyncio', SimpleNamespace(sleep=tick)), self.assertLogs('qqbot.dispatcher', level='ERROR'), self.assertRaises(asyncio.CancelledError):
            await run_proactive(self.settings, self.store, self.transport, dispatcher)

    async def test_scheduler_resting_window_and_rejected_busy_queue_do_not_advance_plan(self):
        self.transport.request = AsyncMock(side_effect=lambda action, params: {'user_id': 10} if action == 'get_login_info' else [{'group_id': 30}])
        dispatcher = SimpleNamespace(submit_proactive=Mock(return_value=False))
        Clock.value = datetime.fromisoformat('2026-10-27T09:59:00+08:00')
        async def tick(seconds):
            dispatcher.submit_proactive.assert_not_called()
            self.assertEqual(self.store.next_proactive_time(event().conversation_id), Clock.value + timedelta(minutes=1))
            raise asyncio.CancelledError
        with patch('qqbot.proactive.asyncio', SimpleNamespace(sleep=tick)), self.assertRaises(asyncio.CancelledError):
            await run_proactive(self.settings, self.store, self.transport, dispatcher)
        Clock.value += timedelta(minutes=1)
        async def busy_tick(seconds):
            dispatcher.submit_proactive.assert_called_once()
            self.assertEqual(self.store.next_proactive_time(event().conversation_id), Clock.value)
            raise asyncio.CancelledError
        with patch('qqbot.proactive.asyncio', SimpleNamespace(sleep=busy_tick)), self.assertRaises(asyncio.CancelledError):
            await run_proactive(self.settings, self.store, self.transport, dispatcher)

    async def test_real_agent_uses_search_then_structured_result_with_source_link(self):
        completion = AsyncMock(side_effect=[response(calls=[SimpleNamespace(id='search-1', function=SimpleNamespace(
            name='web_search', arguments='{"query":"新的开源项目","topic":"news","days":7}'))]),
            response('内部分析占位\n</think>\n' + decision('这个项目挺有意思：https://github.com/example/project'))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))
        env = {'LLM_API_KEY': 'test', 'LLM_BASE_URL': 'https://example.invalid', 'LLM_MODEL': 'test',
               'LLM_CONTEXT_WINDOW': '16000', 'LLM_MAX_OUTPUT_TOKENS': '1000', 'LLM_CONTEXT_SAFETY_MARGIN': '1000'}
        search = WebSearchTool(api_key='test')
        with patch.dict(os.environ, env, clear=True), patch('qqbot.agent.AsyncOpenAI', return_value=client):
            self.agent = Agent(context_manager=self.manager, context_builder=ContextBuilder(), tool_registry=ToolRegistry([search]))
        with patch.object(search, 'run', new_callable=AsyncMock, return_value='{"results":[{"url":"https://github.com/example/project"}]}') as call, contextlib.redirect_stdout(io.StringIO()):
            await self.handle()
        call.assert_awaited_once()
        self.assertEqual(call.await_args.args[0].days, 7)
        prompt = completion.await_args_list[0].kwargs['messages']
        self.assertEqual(prompt[-1]['role'], 'system')
        self.assertIn('next_check_at', prompt[-1]['content'])
        self.assertIn('尚无聊天记录', prompt[-1]['content'])
        self.assertIn('糖糖', prompt[0]['content'])
        self.assertEqual(completion.await_args.kwargs['messages'][-1]['role'], 'tool')
        self.assertIn('https://github.com/example/project', self.transport.send_group_text.await_args.args[1])
