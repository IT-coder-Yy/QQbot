import asyncio
import base64
import hashlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from PIL import Image

from qqbot.context import ContextManager
from qqbot.context.sqlite_store import SQLiteContextStore, StorageError
from qqbot.media import MediaService
from qqbot.outbound import OutboundTurn
from qqbot.reply import MESSAGE_BREAK, parse_reply
from qqbot.schema import ImageContent
from qqbot.settings import Settings
from tests.helpers import message




class OutboundTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.settings = replace(Settings(), reply_interval=0)
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



    async def test_external_images_keep_download_boundary(self):
        with OutboundTurn(self.media, self.incoming.conversation_id) as turn:
            for url in ("file:///etc/passwd", "http://169.254.169.254/secret", "https://untrusted.invalid/a.png"):
                result, _ = await turn.execute("prepare_image", json.dumps({"url": url}))
                self.assertIn("error", json.loads(result))
            result, _ = await turn.execute("prepare_face", '{"name":"不存在的表情"}')
            self.assertIn("error", json.loads(result))
