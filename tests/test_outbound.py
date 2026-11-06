import asyncio
import base64
import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from PIL import Image
from edge_tts.exceptions import NoAudioReceived

from qqbot.agent import Agent
from qqbot.context import ContextBuilder, ContextManager
from qqbot.context.sqlite_store import SQLiteContextStore, StorageError
from qqbot.dispatcher import MessageJob
from qqbot.listen import handle_message
from qqbot.media import MediaService
from qqbot.outbound import OutboundTurn
from qqbot.reply import MESSAGE_BREAK, parse_reply
from qqbot.schema import ImageContent, ReplyContent, TextContent
from qqbot.settings import Settings
from qqbot.tools import ToolRegistry
from qqbot.transport import DeliveryUnknownError, NapCatTransport
from tests.helpers import FakeSocket, eventually, message, success


class Speech:
    def __init__(self, text, voice):
        self.text, self.voice = text, voice

    async def stream(self):
        yield {"type": "WordBoundary", "text": self.text}
        yield {"type": "audio", "data": b"ID3-test-audio"}


class OutboundTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.settings = replace(Settings(), reply_interval=0, tts_timeout=0.05, tts_voice="zh-CN-YunxiNeural")
        self.store = SQLiteContextStore(self.root / "bot.db", self.root / "files")
        self.addCleanup(self.store.close)
        self.transport = SimpleNamespace(request=AsyncMock(), connected=True, wait_connected=AsyncMock(),
                                         reply_text=AsyncMock(return_value=900), send_group_text=AsyncMock(return_value=900),
                                         send_segments=AsyncMock(side_effect=range(901, 921)))
        self.media = MediaService(self.store, self.transport, self.settings)
        self.addAsyncCleanup(self.media.close)
        self.manager = ContextManager(self.store, settings=self.settings)
        self.incoming = message()
        out = io.BytesIO()
        Image.new("RGB", (8, 8), "red").save(out, format="GIF", save_all=True,
            append_images=[Image.new("RGB", (8, 8), "blue")], duration=100, loop=0)
        self.gif = out.getvalue()
        self.md5 = hashlib.md5(self.gif).hexdigest()

    async def register(self, incoming=None):
        incoming = incoming or self.incoming
        incoming.content = [ImageContent(file="gif", sub_type=1)]
        ref = self.media.register_message(incoming)[0]
        self.transport.request.return_value = {"base64": base64.b64encode(self.gif).decode()}
        await self.media.prepare(incoming.conversation_id, ref.id)
        self.transport.request.reset_mock()
        return ref.id

    async def prepare(self, turn, tool_name, **args):
        result, _ = await turn.execute(tool_name, json.dumps(args))
        data = json.loads(result)
        self.assertNotIn("error", data, data)
        if "marker" in data:
            action, _ = turn.expand([(data["marker"], None)])[0]
            return data, action
        return data, None

    def favorite(self):
        return {"md5": self.md5.upper(), "url": "https://qpic.cn/sticker.gif", "emoId": 17,
                "resId": "resource-17", "desc": "开心", "modifyWord": "哈哈", "ocrWord": "笑"}

    async def test_gif_original_bytes_private_scope_and_image_subtype(self):
        identifier = await self.register()
        with OutboundTurn(self.media, self.incoming.conversation_id) as turn:
            result, action = await self.prepare(turn, "prepare_image", attachment_id=identifier, as_sticker=True)
            self.transport.send_segments.assert_not_awaited()
            segments, text, refs = await turn.materialize(action, deadline=asyncio.get_running_loop().time()+10)
            self.assertEqual(segments[0]["data"]["sub_type"], 1)
            raw = base64.b64decode(segments[0]["data"]["file"].removeprefix("base64://"))
            self.assertEqual(raw, self.gif)
            self.assertEqual(Image.open(io.BytesIO(raw)).n_frames, 2)
            self.assertEqual(refs[0].id, identifier)
            self.assertIn(identifier, self.store.protected_attachments)
            expanded = turn.expand(parse_reply("前文" + MESSAGE_BREAK + result["marker"] + "后文", current_message_id=None, quote_ids=set()))
            self.assertEqual([item[0] for item in expanded], ["前文", action, "后文"])
            self.assertEqual(turn.expand([(f"`{result['marker']}`", None)])[0][0], f"`{result['marker']}`")
            with self.assertRaises(StorageError):
                turn.expand([("<qqbot-media:invented/>", None)])
        self.assertFalse(self.store.protected_attachments)
        with OutboundTurn(self.media, message(user_id=21).conversation_id) as other:
            result, _ = await other.execute("prepare_image", json.dumps({"attachment_id": identifier}))
            self.assertIn("不属于当前会话", result)

    async def test_voice_uses_config_and_keeps_transcript_without_asr(self):
        with OutboundTurn(self.media, self.incoming.conversation_id) as turn, patch("qqbot.outbound.edge_tts.Communicate", side_effect=Speech) as tts:
            _, action = await self.prepare(turn, "prepare_voice", text="晚安")
            tts.assert_called_once_with("晚安", "zh-CN-YunxiNeural")
            segments, content, refs = await turn.materialize(action, deadline=asyncio.get_running_loop().time()+10)
            self.assertEqual(segments[0]["type"], "record")
            self.assertIn("晚安", content)
            row = await self.media.prepare(self.incoming.conversation_id, refs[0].id)
            self.assertEqual(json.loads(row["parsed_content_json"])[0], {"number": 1, "text": "晚安", "method": "tts"})
            self.transport.request.assert_not_awaited()
        self.assertFalse(list(self.media.temp.iterdir()))

    async def test_tts_failure_and_timeout_prepare_text_without_sending(self):
        class BrokenSpeech(Speech):
            async def stream(self):
                raise NoAudioReceived("模拟服务失败")
                yield
        class SlowSpeech(Speech):
            async def stream(self):
                await asyncio.sleep(1)
                yield
        for speech in (BrokenSpeech, SlowSpeech):
            with OutboundTurn(self.media, self.incoming.conversation_id) as turn, patch("qqbot.outbound.edge_tts.Communicate", speech):
                _, action = await self.prepare(turn, "prepare_voice", text="晚安")
                self.assertEqual((action.kind, action.text), ("text", "晚安"))
        self.transport.send_segments.assert_not_awaited()
        self.assertFalse(self.store.attachments(self.incoming.conversation_id))
        self.assertFalse(list(self.media.temp.iterdir()))

    async def test_external_images_keep_download_boundary(self):
        with OutboundTurn(self.media, self.incoming.conversation_id) as turn:
            for url in ("file:///etc/passwd", "http://169.254.169.254/secret", "https://untrusted.invalid/a.png"):
                result, _ = await turn.execute("prepare_image", json.dumps({"url": url}))
                self.assertIn("error", json.loads(result))
            result, _ = await turn.execute("prepare_face", '{"name":"不存在的表情"}')
            self.assertIn("error", json.loads(result))





    async def test_real_agent_tool_loop_sends_image_voice_and_face_and_keeps_history(self):
        identifier = await self.register()
        first = True
        async def complete(**kwargs):
            nonlocal first
            self.assertIn("prepare_voice", [t["function"]["name"] for t in kwargs["tools"]])
            if first:
                first = False
                calls = [SimpleNamespace(id=str(i), function=SimpleNamespace(name=n, arguments=json.dumps(a))) for i, (n, a) in enumerate([
                    ("prepare_image", {"attachment_id": identifier, "as_sticker": True}),
                    ("prepare_voice", {"text": "晚安"}), ("prepare_face", {"name": "赞"})])]
                return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=calls, reasoning_content=None))])
            markers = [json.loads(m["content"])["marker"] for m in kwargs["messages"] if m["role"] == "tool"]
            return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content=MESSAGE_BREAK.join(markers), tool_calls=[]))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
        env = {"LLM_API_KEY":"fake", "LLM_BASE_URL":"https://example.invalid", "LLM_MODEL":"qwen3.7-plus"}
        with patch.dict(os.environ, env), patch("qqbot.agent.AsyncOpenAI", return_value=client), patch("qqbot.outbound.edge_tts.Communicate", Speech), contextlib.redirect_stdout(io.StringIO()):
            agent = Agent(context_manager=self.manager, context_builder=ContextBuilder(), tool_registry=ToolRegistry([]), media=self.media)
            await handle_message(MessageJob(self.incoming, asyncio.get_running_loop().time()+10), context_manager=self.manager,
                                 agent=agent, intent_detector=None, transport=self.transport, intent_timeout=1, media=self.media)
        self.assertEqual([c.args[1][0]["type"] for c in self.transport.send_segments.await_args_list], ["image", "record", "face"])
        history = self.store.history(self.incoming.conversation_id)
        self.assertEqual([m["source_message_id"] for m in history], [1, 901, 902, 903])
        self.assertEqual(history[1]["attachments"][0]["id"], identifier)
        self.assertEqual(history[2]["attachments"][0]["type"], "audio")
        self.assertIn("晚安", history[2]["content"])
        quoted = message(2)
        quoted.content = [ReplyContent(message_id=901), TextContent(text="再看看这张")]
        self.assertEqual(self.media.referenced(quoted)[0].id, identifier)
        self.assertFalse(self.store.protected_attachments)

    async def test_media_zero_ack_does_not_retry_or_erase_previous_success(self):
        ws = FakeSocket()
        transport = NapCatTransport(ws, action_timeout=1)
        self.addCleanup(transport.close)
        self.media.transport = transport
        async def generate(**kwargs):
            await self.prepare(kwargs["outbound"], "prepare_face", name="赞")
            data, _ = await self.prepare(kwargs["outbound"], "prepare_voice", text="晚安")
            return "前文" + MESSAGE_BREAK + data["marker"]
        with patch("qqbot.outbound.edge_tts.Communicate", Speech):
            task = asyncio.create_task(handle_message(MessageJob(self.incoming, asyncio.get_running_loop().time()+10),
                context_manager=self.manager, agent=SimpleNamespace(run=generate), intent_detector=None,
                transport=transport, intent_timeout=1, media=self.media))
            await eventually(lambda: len(ws.sent) == 1)
            transport.handle_response(success(ws.sent[0], 501))
            await eventually(lambda: len(ws.sent) == 2)
            self.assertEqual(len(self.store.history(self.incoming.conversation_id)), 2)
            self.assertEqual(ws.sent[1]["params"]["message"][0]["type"], "record")
            transport.handle_response(success(ws.sent[1], 0))
            with self.assertRaises(DeliveryUnknownError):
                await task
        self.assertEqual(len(ws.sent), 2)
        self.assertEqual([m["source_message_id"] for m in self.store.history(self.incoming.conversation_id)], [1, 501])
