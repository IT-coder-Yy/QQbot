import asyncio
import base64
import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from PIL import Image
from docx import Document
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference

from qqbot import documents
from qqbot.agent import Agent
from qqbot.context import ContextBuilder, ContextManager, InMemoryContextStore
from qqbot.context.sqlite_store import SQLiteContextStore, StorageError
from qqbot.context.token_counter import ApproxTokenCounter
from qqbot.dispatcher import MessageJob
from qqbot.listen import handle_message
from qqbot.media import MediaService
from qqbot.normalizer import normalize_event
from qqbot.schema import AudioContent, FileContent, ImageContent, TextContent, ReplyContent
from qqbot.settings import Settings
from qqbot.tools import ToolRegistry
from tests.helpers import message, raw_message


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


class MediaTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.store = SQLiteContextStore(self.root / 'bot.db', self.root / 'files')
        self.addCleanup(self.store.close)
        self.transport = SimpleNamespace(request=AsyncMock(), wait_connected=AsyncMock(), reply_text=AsyncMock(return_value=901))
        self.media = MediaService(self.store, self.transport, Settings())
        self.addAsyncCleanup(self.media.close)

    def register(self, name='sample.txt', data=b'hello', user=20, mid=1):
        incoming = message(mid, user_id=user)
        incoming.content = [FileContent(file_id='file-' + str(mid), name=name)]
        ref = self.media.register_message(incoming)[0]
        self.transport.request.return_value = {'base64': base64.b64encode(data).decode()}
        return incoming, ref.id

    async def test_document_only_ack_and_restart_file_followup(self):
        incoming, identifier = self.register()
        manager = ContextManager(self.store)
        agent = SimpleNamespace(run=AsyncMock())
        await handle_message(MessageJob(incoming, asyncio.get_running_loop().time()+10), context_manager=manager,
                             agent=agent, intent_detector=None, transport=self.transport, intent_timeout=1, media=self.media)
        self.assertEqual(self.transport.reply_text.call_args.args[1], '已读取，可以继续提问。')
        agent.run.assert_not_awaited()
        await self.media.close()
        self.store.close()
        self.store = SQLiteContextStore(self.root / 'bot.db', self.root / 'files')
        self.addCleanup(self.store.close)
        self.media = MediaService(self.store, self.transport, Settings())
        self.addAsyncCleanup(self.media.close)
        result, images = await self.media.execute('read_attachment', json.dumps({'attachment_id': identifier}), incoming.conversation_id)
        self.assertIn('hello', result)
        self.assertEqual(images, [])
        other, _ = await self.media.execute('read_attachment', json.dumps({'attachment_id': identifier}), message(user_id=21).conversation_id)
        self.assertIn('不属于当前会话', other)
        self.assertEqual(self.transport.request.await_count, 1)

    async def test_oversized_input_keeps_old_attachment(self):
        incoming, old_id = self.register(data=b'old')
        old = await self.media.prepare(incoming.conversation_id, old_id)
        self.store.attachment_limit = 2500
        incoming, new_id = self.register(data=b'x' * 3000, mid=2)
        with self.assertRaisesRegex(StorageError, '文件太大'):
            await self.media.prepare(incoming.conversation_id, new_id)
        self.assertTrue(self.store.file_path(old['local_path']).exists())
        self.assertIsNone(self.store.attachment(incoming.conversation_id, new_id)['local_path'])

    async def test_attachment_lru_and_conversation_independence(self):
        incoming, first_id = self.register(data=b'a' * 150, mid=1)
        first = await self.media.prepare(incoming.conversation_id, first_id)
        incoming, second_id = self.register(data=b'b' * 150, mid=2)
        second = await self.media.prepare(incoming.conversation_id, second_id)
        first['last_used_at'] = second['last_used_at'] + 10
        self.store.save_attachment(first)
        total = self.store.db.execute('SELECT sum(storage_bytes) FROM attachments').fetchone()[0]
        self.store.attachment_limit = total + 100
        incoming, third_id = self.register(data=b'c' * 150, mid=3)
        await self.media.prepare(incoming.conversation_id, third_id)
        self.assertIsNotNone(self.store.attachment(incoming.conversation_id, first_id))
        self.assertIsNone(self.store.attachment(incoming.conversation_id, second_id))
        self.assertFalse(self.store.file_path(second['local_path']).exists())
        self.store.db.execute('DELETE FROM conversations')
        self.store.db.commit()
        self.assertIsNotNone(self.store.attachment(incoming.conversation_id, first_id))

    async def test_silent_group_media_only_registers(self):
        incoming = message(1, group_id=30)
        incoming.content = [ImageContent(file='image')]
        manager = ContextManager(self.store)
        await handle_message(MessageJob(incoming, asyncio.get_running_loop().time()+10), context_manager=manager,
                             agent=SimpleNamespace(run=AsyncMock()), intent_detector=None, transport=self.transport,
                             intent_timeout=1, media=self.media)
        self.transport.request.assert_not_awaited()
        row = self.store.attachments(incoming.conversation_id)[0]
        self.assertEqual(row['status'], 'registered')
        other = message(2, group_id=30, user_id=21)
        self.assertEqual(other.conversation_id, incoming.conversation_id)
        self.assertIsNotNone(self.store.attachment(other.conversation_id, row['id']))

    async def test_audio_uses_record_conversion_and_multipart(self):
        incoming = message(1)
        incoming.content = [AudioContent(file='voice')]
        identifier = self.media.register_message(incoming)[0].id
        self.transport.request.return_value = {'base64': base64.b64encode(b'mp3-data').decode()}
        self.media.api_key = 'fake'
        requests = []
        def responder(request):
            requests.append(request)
            return httpx.Response(200, json={'text': '你好'})
        await self.media.client.aclose()
        self.media.client = httpx.AsyncClient(transport=httpx.MockTransport(responder))
        row = await self.media.prepare(incoming.conversation_id, identifier)
        self.assertIn('你好', row['parsed_content_json'])
        self.assertEqual(self.transport.request.call_args.args[0], 'get_record')
        self.assertEqual(self.transport.request.call_args.args[1]['out_format'], 'mp3')
        self.assertIn(b'FunAudioLLM/SenseVoiceSmall', requests[0].content)
        self.assertIn(b'mp3-data', requests[0].content)

    async def test_scanned_pdf_ocr_and_text_pdf_no_ocr(self):
        image = Image.new('RGB', (200, 100), 'white')
        out = io.BytesIO()
        image.save(out, format='PDF')
        incoming, identifier = self.register(name='scan.pdf', data=out.getvalue())
        with patch.object(self.media, 'ocr', AsyncMock(return_value='扫描文字')) as ocr:
            row = await self.media.prepare(incoming.conversation_id, identifier)
        ocr.assert_awaited_once()
        self.assertEqual(row['status'], 'ready')
        self.assertIn('扫描文字', row['parsed_content_json'])
        result, images = await self.media.execute('read_attachment', json.dumps({'attachment_id':identifier, 'visual':True}), incoming.conversation_id)
        self.assertEqual(len(images), 1)
        self.assertIn('total_pages', result)

    async def test_partial_ocr_never_claims_complete(self):
        out = io.BytesIO()
        Image.new('RGB', (100, 100), 'white').save(out, format='PDF')
        incoming, identifier = self.register(name='scan.pdf', data=out.getvalue())
        with patch.object(self.media, 'ocr', AsyncMock(side_effect=StorageError('限流'))):
            row = await self.media.prepare(incoming.conversation_id, identifier)
        self.assertEqual(row['status'], 'partial')
        self.assertIn('限流', row['error'])

    async def test_normalizer_audio_and_private_file(self):
        raw = raw_message()
        raw['message'] = [{'type':'record','data':{'file':'voice'}}, {'type':'file','data':{'file_id':'abc','name':'x.xls','file_size':120}}]
        result = normalize_event(raw)
        self.assertIsInstance(result.content[0], AudioContent)
        self.assertIsInstance(result.content[1], FileContent)

    async def test_local_file_requires_explicit_shared_root(self):
        path = self.root / 'secret'
        path.write_text('must not read')
        with self.assertRaises(StorageError):
            await self.media._copy_payload({'file':str(path)}, self.root / 'out')


    async def test_ntqq_image_host_is_allowed_but_redirects_still_checked(self):
        requests = []
        def responder(request):
            requests.append(request)
            if request.url.path == '/redirect':
                return httpx.Response(302, headers={'location': 'https://untrusted.invalid/image'})
            return httpx.Response(200, content=b'picture')
        await self.media.client.aclose()
        self.media.client = httpx.AsyncClient(transport=httpx.MockTransport(responder))
        output = self.root / 'download'
        await self.media._copy_payload({'url': 'https://multimedia.nt.qq.com.cn/download'}, output)
        self.assertEqual(output.read_bytes(), b'picture')
        for url in ('https://multimedia.nt.qq.com.cn.evil.invalid/download',
                    'https://multimedia.nt.qq.com.cn/redirect'):
            with self.subTest(url=url), self.assertRaisesRegex(StorageError, '域名'):
                await self.media._copy_payload({'url': url}, output)
        self.assertEqual([r.url.host for r in requests], ['multimedia.nt.qq.com.cn'] * 2)

    async def test_long_single_block_is_readable_without_truncation(self):
        doc = Document()
        expected = '长段落内容' * 10000
        doc.add_paragraph(expected)
        data = io.BytesIO()
        doc.save(data)
        incoming, identifier = self.register(name='long.docx', data=data.getvalue())
        start, offset, recovered = 1, 0, ''
        while start is not None:
            result, _ = await self.media.execute('read_attachment', json.dumps({'attachment_id':identifier, 'start':start,
                                               'end':start, 'char_offset':offset}), incoming.conversation_id)
            result = json.loads(result)
            recovered += ''.join(p['text'] for p in result['parts'])
            start, offset = result['next_start'], result['next_char_offset']
        self.assertEqual(recovered, expected)

    async def test_accepted_group_picture_is_prepared_without_mention(self):
        incoming = message(1, group_id=30)
        incoming.content = [ImageContent(file='picture'), TextContent(text='看看这张图')]
        png = io.BytesIO()
        Image.new('RGB', (10, 10), 'red').save(png, 'PNG')
        self.transport.request.return_value = {'base64': base64.b64encode(png.getvalue()).decode()}
        agent = SimpleNamespace(run=AsyncMock(return_value='红色图片'))
        detector = SimpleNamespace(detect=AsyncMock(return_value=SimpleNamespace(should_reply=True)))
        await handle_message(MessageJob(incoming, asyncio.get_running_loop().time()+10),
                             context_manager=ContextManager(self.store), agent=agent, intent_detector=detector,
                             transport=self.transport, intent_timeout=1, media=self.media)
        self.assertFalse(incoming.mentioned_bot)
        detector.detect.assert_awaited_once()
        self.transport.request.assert_awaited_once()
        current = agent.run.await_args.kwargs['current_turn']
        parts = self.media.render_refs(current, incoming.conversation_id, images=True, ordered=True)
        self.assertTrue(any(part['type'] == 'image_url' for part in parts))

    async def test_picture_then_unmentioned_question_reads_image_through_agent(self):
        incoming = message(1, group_id=30)
        incoming.content = [ImageContent(file='picture')]
        manager = ContextManager(self.store)
        detector = SimpleNamespace(detect=AsyncMock(return_value=SimpleNamespace(should_reply=False)))
        await handle_message(MessageJob(incoming, asyncio.get_running_loop().time()+10), context_manager=manager,
                             agent=SimpleNamespace(run=AsyncMock()), intent_detector=detector,
                             transport=self.transport, intent_timeout=1, media=self.media)
        self.transport.request.assert_not_awaited()
        self.transport.reply_text.assert_not_awaited()
        stored = self.store.attachments(incoming.conversation_id)[0]
        identifier = stored['id']
        # 复现旧版本已经留下下载失败记录的图片，后续自然追问仍可重试。
        self.store.save_attachment({**stored, 'status': 'failed', 'error': '附件下载地址不在允许的 QQ/NapCat 域名中。'})
        png = io.BytesIO()
        Image.new('RGB', (10, 10), 'red').save(png, 'PNG')
        self.transport.request.return_value = {'base64': base64.b64encode(png.getvalue()).decode()}
        call = SimpleNamespace(id='read-picture', function=SimpleNamespace(
            name='read_attachment', arguments=json.dumps({'attachment_id': identifier})))
        def response(content=None, calls=None):
            return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(
                content=content, tool_calls=calls or [], reasoning_content=None))])
        completion = AsyncMock(side_effect=[response(calls=[call]), response(content='这是一张红色图片')])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))
        env = dict(LLM_API_KEY='fake', LLM_BASE_URL='https://example.invalid', LLM_MODEL='qwen3.7-plus')
        with patch.dict(os.environ, env, clear=True), patch('qqbot.agent.AsyncOpenAI', return_value=client):
            agent = Agent(context_manager=manager, context_builder=ContextBuilder(),
                          tool_registry=ToolRegistry([]), media=self.media)
        question = message(2, group_id=30)
        question.content = [TextContent(text='这图片有什么')]
        detector.detect.return_value.should_reply = True
        with contextlib.redirect_stdout(io.StringIO()):
            await handle_message(MessageJob(question, asyncio.get_running_loop().time()+10), context_manager=manager,
                                 agent=agent, intent_detector=detector, transport=self.transport,
                                 intent_timeout=1, media=self.media)
        self.assertFalse(question.mentioned_bot)
        self.assertIsNone(question.reply_to_message_id)
        self.transport.request.assert_awaited_once()
        self.assertEqual(self.store.attachment(question.conversation_id, identifier)['status'], 'ready')
        sent = completion.call_args.kwargs['messages']
        self.assertEqual([m['role'] for m in sent[-3:]], ['assistant', 'tool', 'user'])
        self.assertEqual(sent[-1]['content'][-1]['type'], 'image_url')
        self.assertNotIn('error', json.loads(sent[-2]['content']))
        self.assertEqual(self.transport.reply_text.await_args.args[1], '这是一张红色图片')

    async def test_empty_response_recovers_and_reads_saved_images_after_restart(self):
        manager = ContextManager(self.store)
        identifiers = []
        for index, color in enumerate(('red', 'blue'), 1):
            incoming = message(index, group_id=30)
            incoming.content = [ImageContent(file=f'image-{index}')]
            ref = self.media.register_message(incoming)[0]
            manager.record_user_message(incoming)
            png = io.BytesIO()
            Image.new('RGB', (10, 10), color).save(png, 'PNG')
            self.transport.request.return_value = {'base64': base64.b64encode(png.getvalue()).decode()}
            await self.media.prepare(incoming.conversation_id, ref.id)
            identifiers.append(ref.id)
        await self.media.close()
        self.store.close()
        self.store = SQLiteContextStore(self.root / 'bot.db', self.root / 'files')
        self.addCleanup(self.store.close)
        self.media = MediaService(self.store, self.transport, Settings())
        self.addAsyncCleanup(self.media.close)
        self.transport.request.reset_mock()
        manager = ContextManager(self.store)

        def response(content=None, calls=None):
            return SimpleNamespace(usage=None, choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(
                content=content, tool_calls=calls or [], reasoning_content=None))])

        find = SimpleNamespace(id='find', function=SimpleNamespace(name='find_history', arguments='{"query":"图片"}'))
        reads = [SimpleNamespace(id=f'read-{index}', function=SimpleNamespace(name='read_attachment', arguments=json.dumps({'attachment_id': identifier})))
                 for index, identifier in enumerate(identifiers)]
        completion = AsyncMock(side_effect=[response(), response(calls=[find]), response(calls=reads), response('第一张是红色，第二张是蓝色。')])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))
        env = dict(LLM_API_KEY='fake', LLM_BASE_URL='https://example.invalid', LLM_MODEL='qwen3.7-plus')
        with patch.dict(os.environ, env, clear=True), patch('qqbot.agent.AsyncOpenAI', return_value=client):
            agent = Agent(context_manager=manager, context_builder=ContextBuilder(), tool_registry=ToolRegistry([]), media=self.media)
        question = message(3, group_id=30, mentioned=True)
        question.content.append(TextContent(text='给我说一下前面的几个图片内容'))
        with contextlib.redirect_stdout(io.StringIO()), self.assertLogs('qqbot.agent', level='WARNING'):
            await handle_message(MessageJob(question, asyncio.get_running_loop().time()+10), context_manager=manager,
                                 agent=agent, intent_detector=None, transport=self.transport, intent_timeout=1, media=self.media)
        self.assertEqual(completion.await_count, 4)
        self.assertEqual(completion.await_args_list[0].kwargs['messages'], completion.await_args_list[1].kwargs['messages'])
        final_input = completion.await_args.kwargs['messages']
        self.assertEqual([part['type'] for part in final_input[-1]['content']].count('image_url'), 2)
        self.assertTrue(all(not json.loads(item['content']).get('error') for item in final_input if item['role'] == 'tool'))
        self.transport.request.assert_not_awaited()
        self.transport.reply_text.assert_awaited_once()
        self.assertEqual(self.transport.reply_text.await_args.args[1], '第一张是红色，第二张是蓝色。')
        self.assertEqual(len(self.store.history(question.conversation_id)), 4)

    async def test_quoted_group_picture_triggers_download_and_keeps_order(self):
        incoming = message(1, group_id=30)
        incoming.content = [TextContent(text='前'), ImageContent(file='picture'), TextContent(text='后')]
        manager = ContextManager(self.store)
        agent = SimpleNamespace(run=AsyncMock(return_value='这是一张红色图片'))
        async def handle(item):
            await handle_message(MessageJob(item, asyncio.get_running_loop().time()+10), context_manager=manager,
                                 agent=agent, intent_detector=None, transport=self.transport, intent_timeout=1, media=self.media)
        await handle(incoming)
        self.transport.request.assert_not_awaited()
        png = io.BytesIO()
        Image.new('RGB', (10, 10), 'red').save(png, 'PNG')
        self.transport.request.return_value = {'base64':base64.b64encode(png.getvalue()).decode()}
        quote = message(2, group_id=30, user_id=21)
        quote.content = [ReplyContent(message_id=1), TextContent(text='这张图是什么？')]
        await handle(quote)
        self.transport.request.assert_awaited_once()
        agent.run.assert_awaited_once()
        original = self.store.get_session(incoming.conversation_id).recent_messages[0]
        rendered = self.media.render_refs(original, incoming.conversation_id, images=True, ordered=True)
        self.assertEqual([p['type'] for p in rendered], ['text', 'text', 'image_url', 'text'])
        self.assertEqual(rendered[0]['text'], '前')
        self.assertEqual(rendered[-1]['text'], '后')

    async def test_restart_repairs_orphan_missing_file_and_lower_quota(self):
        incoming, identifier = self.register()
        row = await self.media.prepare(incoming.conversation_id, identifier)
        self.store.file_path(row['local_path']).unlink()
        orphan = self.store.files / ('0' * 32 + '.blob')
        orphan.write_bytes(b'orphan')
        self.store.reconcile()
        self.assertFalse(orphan.exists())
        self.assertEqual(self.store.attachment(incoming.conversation_id, identifier)['status'], 'failed')
        await self.media.prepare(incoming.conversation_id, identifier)
        self.store.attachment_limit = 1
        self.store.cleanup()
        self.assertEqual(self.store.attachments(incoming.conversation_id), [])
        self.assertFalse(list(self.store.files.glob('*.blob')))

    async def test_malformed_pdf_returns_useful_error(self):
        incoming, identifier = self.register(name='bad.pdf', data=b'not pdf')
        with self.assertRaisesRegex(StorageError, '解析失败'):
            await self.media.prepare(incoming.conversation_id, identifier)
        self.assertEqual(self.store.attachment(incoming.conversation_id, identifier)['status'], 'failed')

    async def test_partial_ocr_can_be_retried(self):
        out = io.BytesIO()
        Image.new('RGB', (100, 100), 'white').save(out, format='PDF')
        incoming, identifier = self.register(name='scan.pdf', data=out.getvalue())
        with patch.object(self.media, 'ocr', AsyncMock(side_effect=StorageError('限流'))):
            await self.media.prepare(incoming.conversation_id, identifier)
        with patch.object(self.media, 'ocr', AsyncMock(return_value='补全的文字')):
            result, _ = await self.media.execute('read_attachment', json.dumps({'attachment_id':identifier, 'ocr':True}), incoming.conversation_id)
        self.assertIn('补全的文字', result)
        self.assertEqual(self.store.attachment(incoming.conversation_id, identifier)['status'], 'ready')

    async def test_unread_scan_ack_does_not_claim_partial_text(self):
        out = io.BytesIO()
        Image.new('RGB', (100, 100), 'white').save(out, format='PDF')
        incoming, _ = self.register(name='scan.pdf', data=out.getvalue())
        with patch.object(self.media, 'ocr', AsyncMock(side_effect=StorageError('未配置识别密钥'))):
            await handle_message(MessageJob(incoming, asyncio.get_running_loop().time()+10), context_manager=ContextManager(self.store),
                                 agent=SimpleNamespace(run=AsyncMock()), intent_detector=None, transport=self.transport,
                                 intent_timeout=1, media=self.media)
        reply = self.transport.reply_text.call_args.args[1]
        self.assertIn('文字识别尚未完成', reply)
        self.assertNotIn('已读取部分内容', reply)

    async def test_interrupted_ocr_keeps_completed_pages(self):
        out = io.BytesIO()
        page = Image.new('RGB', (100, 100), 'white')
        page.save(out, format='PDF', save_all=True, append_images=[page])
        incoming, identifier = self.register(name='scan.pdf', data=out.getvalue())
        started = asyncio.Event()
        calls = 0
        async def recognize(image):
            nonlocal calls
            calls += 1
            if calls == 1:
                return '第一页已识别'
            started.set()
            await asyncio.Future()
        with patch.object(self.media, 'ocr', recognize):
            task = asyncio.create_task(self.media.prepare(incoming.conversation_id, identifier))
            await asyncio.wait_for(started.wait(), 5)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        row = self.store.attachment(incoming.conversation_id, identifier)
        self.assertEqual(row['status'], 'partial')
        self.assertIn('第一页已识别', row['parsed_content_json'])
        self.assertIn('超时或中断', row['error'])
        self.assertTrue(self.store.file_path(row['local_path']).exists())


class DocumentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))

    async def test_docx_and_excel_keep_locations_and_formulas(self):
        doc = Document()
        doc.add_paragraph('合同正文')
        doc.add_table(rows=1, cols=1).cell(0, 0).text = '条款表格'
        path = self.root / 'sample.docx'
        doc.save(path)
        parts = documents.extract(path, path.name, 100000)
        self.assertEqual([p['text'] for p in parts], ['合同正文', '条款表格'])
        book = Workbook()
        book.active['A1'] = '第一张表'
        book.active['B1'] = '=1+1'
        book.create_sheet('第二张')['C2'] = 123
        path = self.root / 'sample.xlsx'
        book.save(path)
        parts = documents.extract(path, path.name, 100000)
        self.assertIn('无缓存值', parts[0]['text'])
        self.assertEqual(parts[1]['sheet'], '第二张')
        self.assertIn('C2=123', parts[1]['text'])

    @unittest.skipUnless(shutil.which('soffice'), '未安装 LibreOffice')
    async def test_legacy_formats_and_office_visual_render(self):
        doc = Document()
        doc.add_paragraph('Document conversion fixture')
        original = self.root / 'fixture.docx'
        doc.save(original)
        work = self.root / 'legacy'
        work.mkdir()
        legacy = await documents.convert_office(original, original.name, 'doc', work, binary='soffice', timeout=30, limit=5000000)
        modern_work = self.root / 'modern'
        modern_work.mkdir()
        modern = await documents.convert_office(legacy, legacy.name, 'docx', modern_work, binary='soffice', timeout=30, limit=5000000)
        self.assertIn('fixture', documents.extract(modern, modern.name, 5000000)[0]['text'])
        pdf_work = self.root / 'pdf'
        pdf_work.mkdir()
        pdf = await documents.convert_office(legacy, legacy.name, 'pdf', pdf_work, binary='soffice', timeout=30, limit=5000000)
        self.assertFalse(documents.extract(pdf, pdf.name, 5000000)[0]['needs_ocr'])
        image, total = documents.render_pdf(pdf, 1, [0, 0, 0.5, 0.5])
        self.assertEqual(total, 1)
        self.assertTrue(image['image_url']['url'].startswith('data:image/jpeg'))

    @unittest.skipUnless(shutil.which('soffice'), '未安装 LibreOffice')
    async def test_xls_and_chart_render_include_hidden_sheet_outside_print_area(self):
        book = Workbook()
        sheet = book.active
        sheet.append(['Month', 'Value'])
        sheet.append(['Jan', 100])
        sheet.append(['Feb', 200])
        sheet['B4'] = '=SUM(B2:B3)'
        sheet['Z30'] = 'Outside print area'
        sheet.print_area = 'A1:B4'
        chart = BarChart()
        chart.add_data(Reference(sheet, min_col=2, min_row=1, max_row=3), titles_from_data=True)
        sheet.add_chart(chart, 'D2')
        hidden = book.create_sheet('Hidden')
        hidden['A1'] = 'Hidden sheet content'
        hidden.sheet_state = 'hidden'
        original = self.root / 'fixture.xlsx'
        book.save(original)
        async def convert(source, target, folder):
            work = self.root / folder
            work.mkdir()
            return await documents.convert_office(source, source.name, target, work, binary='soffice', timeout=30, limit=5000000)
        legacy = await convert(original, 'xls', 'legacy')
        modern = await convert(legacy, 'xlsx', 'modern')
        parts = documents.extract(modern, modern.name, 5000000)
        self.assertIn('缓存值=300', parts[0]['text'])
        self.assertEqual(parts[-1]['sheet'], 'Hidden')
        pdf = await convert(legacy, 'pdf', 'pdf')
        pages = documents.extract(pdf, pdf.name, 5000000)
        self.assertEqual(len(pages), 2)
        self.assertIn('Outside print area', pages[0]['text'])
        self.assertIn('Hidden sheet content', pages[1]['text'])
        image, total = documents.render_pdf(pdf, 1)
        self.assertEqual(total, 2)
        self.assertIn('image_url', image)


class ModelTests(unittest.IsolatedAsyncioTestCase):
    async def test_qwen_limits_and_image_tool_loop(self):
        def response(content=None, calls=None):
            return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls or [], reasoning_content=None))])
        call = SimpleNamespace(id='1', function=SimpleNamespace(name='read_attachment', arguments='{"attachment_id":"id"}'))
        completion = AsyncMock(side_effect=[response(calls=[call]), response(content='内部分析占位\n</think>\n图表分析')])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))
        media = SimpleNamespace(definitions=[], tools={'read_attachment':None}, execute=AsyncMock(return_value=('图像资料', [{'type':'image_url','image_url':{'url':'data:image/png;base64,AAAA'}}])))
        manager = ContextManager(InMemoryContextStore())
        env = dict(LLM_API_KEY='fake', LLM_BASE_URL='https://example.invalid', LLM_MODEL='qwen3.7-plus', LLM_CONTEXT_WINDOW='16000', LLM_MAX_OUTPUT_TOKENS='1000')
        with patch.dict(os.environ, env, clear=True), patch('qqbot.agent.AsyncOpenAI', return_value=client):
            agent = Agent(context_manager=manager, context_builder=ContextBuilder(), tool_registry=ToolRegistry([]), media=media)
        self.assertEqual(agent.context_window, 1000000)
        self.assertEqual(agent.max_output_tokens, 131072)
        incoming = message()
        conv, session, turn = manager.record_user_message(incoming)
        with contextlib.redirect_stdout(io.StringIO()):
            answer = await agent.run(message=incoming, conversation=conv, session=session, current_turn=turn)
        self.assertEqual(answer, '图表分析')
        sent = completion.call_args.kwargs['messages']
        self.assertEqual([m['role'] for m in sent[-3:]], ['assistant', 'tool', 'user'])
        self.assertEqual(sent[-1]['content'][-1]['type'], 'image_url')
        self.assertEqual(media.execute.call_args.args[-1], incoming.conversation_id)

    def test_base64_does_not_count_as_text_tokens(self):
        messages = [{'role':'user','content':[{'type':'text','text':'看图'}, {'type':'image_url','image_url':{'url':'data:image/png;base64,'+'A'*100000}}]}]
        self.assertLess(ApproxTokenCounter().count_messages(messages), 9000)
