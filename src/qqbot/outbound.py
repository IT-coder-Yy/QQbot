"""本轮媒体准备和延迟执行；复用附件存储，不直接在模型工具里发送或收藏。"""
from __future__ import annotations

import asyncio
import base64
import logging
import re
import tempfile
import time
import uuid
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import edge_tts
import httpx
from aiohttp import ClientError
from edge_tts.exceptions import EdgeTTSException
from pydantic import Field

from . import documents
from .context.sqlite_store import StorageError, TOO_LARGE, dumps
from .media import MediaService
from .schema import ContextAttachment
from .tools.base import ToolArgs
from .transport import DeliveryError

logger = logging.getLogger(__name__)

# 来自 NapCat packages/napcat-core/external/face_config.json 的 QSid。
FACES = {"微笑": 14, "呲牙": 13, "流泪": 5, "大哭": 9, "鼓掌": 99,
         "捂脸": 264, "爱心": 66, "赞": 76, "抱拳": 118}

INSTRUCTIONS = """你可以自然选择文字、语音、图片、GIF表情包或QQ原生表情，不必每次用媒体。
用户要求语音时用 prepare_voice；图片或表情使用当前会话附件、允许下载的图片URL。GIF发送原件。
先调用准备工具，再把返回的 <qqbot-media:.../> 标记单独放在最终回复中，程序会按顺序执行并移除标记；语音内容不要再用文字重复发送。不编造标记、路径、表情编号或已发送的结果。
准备工具报错时根据实际结果选择文字回应；没有准备好的资源不能靠输出CQ码、Base64或本地路径发送。"""


class VoiceArgs(ToolArgs):
    text: str = Field(min_length=1, max_length=2000)


class ImageArgs(ToolArgs):
    attachment_id: str | None = None
    url: str | None = None
    as_sticker: bool = False


class FaceArgs(ToolArgs):
    name: Literal["微笑", "呲牙", "流泪", "大哭", "鼓掌", "捂脸", "爱心", "赞", "抱拳"]








@dataclass
class PreparedAction:
    kind: str
    text: str
    attachment: ContextAttachment | None = None
    face_id: int | None = None
    description: str = ""


class OutboundTurn:
    tools = {
        "prepare_voice": (VoiceArgs, "准备语音。返回标记和实际要说的文本，最终回复使用标记才发送；合成失败可回退为文字。"),
        "prepare_image": (ImageArgs, "准备图片/GIF。attachment_id 与 url 二选一；优先用当前会话附件。as_sticker=true作为表情包发送。返回预览与最终发送标记。"),
        "prepare_face": (FaceArgs, "准备一个已核对名称的QQ原生表情，最终回复使用返回的标记才发送。"),
    }
    definitions = [{"type": "function", "function": {"name": name, "description": desc,
                    "parameters": args.model_json_schema()}} for name, (args, desc) in tools.items()]

    def __init__(self, media: MediaService, conversation_id: str):
        self.media, self.cid = media, conversation_id
        self.actions: dict[str, PreparedAction] = {}
        self.pins = ExitStack()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return self.pins.__exit__(*exc)


    async def image(self, identifier):
        row = self.media.store.attachment(self.cid, identifier)
        if row is None or row["kind"] != "image":
            raise StorageError("没有可用的图片附件，或附件不属于当前会话。")
        self.pins.enter_context(self.media.store.pin([identifier]))
        return await self.media.prepare(self.cid, identifier)

    async def execute(self, name, arguments):
        try:
            args = self.tools[name][0].model_validate_json(arguments or "{}")
            images = []
            if name == "prepare_voice":
                text = args.text.strip()
                if not text:
                    raise ValueError("语音文本不能为空。")
                with tempfile.TemporaryDirectory(dir=self.media.temp) as folder:
                    raw = Path(folder) / "voice.mp3"
                    try:
                        async with asyncio.timeout(self.media.settings.tts_timeout):
                            with raw.open("wb") as output:
                                async for chunk in edge_tts.Communicate(text, self.media.settings.tts_voice).stream():
                                    if chunk["type"] == "audio":
                                        output.write(chunk["data"])
                                        if output.tell() > self.media.store.attachment_limit:
                                            raise StorageError(TOO_LARGE)
                        if not raw.stat().st_size:
                            raise ValueError("语音服务未返回音频。")
                    except (TimeoutError, ClientError, EdgeTTSException, OSError, ValueError, StorageError) as exc:
                        logger.warning("语音合成失败，准备文字回退：%s", type(exc).__name__)
                        action = PreparedAction("text", text)
                    else:
                        row = self.media._register(self.cid, "tts:" + uuid.uuid4().hex, None, None, "audio",
                                                   "语音.mp3", {"voice": self.media.settings.tts_voice}, int(time.time() * 1000))
                        self.pins.enter_context(self.media.store.pin([row["id"]]))
                        path = self.media.store.file_path(uuid.uuid4().hex + ".mp3")
                        raw.replace(path)
                        try:
                            row = self.media.store.save_attachment({**row, "local_path": path.name, "file_bytes": path.stat().st_size,
                                "status": "ready", "parsed_content_json": dumps([{"number": 1, "text": text, "method": "tts"}])})
                        except Exception:
                            path.unlink(missing_ok=True)
                            raise
                        action = PreparedAction("record", "[语音] " + text, self.media.reference(row))
            elif name == "prepare_face":
                action = PreparedAction("face", "[QQ表情：" + args.name + "]", face_id=FACES[args.name])
            else:
                identifier = args.attachment_id
                if name == "prepare_image":
                    if bool(identifier) == bool(args.url):
                        raise ValueError("attachment_id 与 url 必须二选一。")
                    if args.url:
                        if not args.url.startswith(("https://", "http://")):
                            raise ValueError("只支持允许下载主机的 HTTP(S) 图片地址。")
                        row = self.media._register(self.cid, "url:" + args.url, None, None, "image", "图片",
                                                   {"url": args.url}, int(time.time() * 1000))
                        identifier = row["id"]
                row = await self.image(identifier)
                ref = self.media.reference(row)
                ref.metadata["sub_type"] = 1 if args.as_sticker else 0
                action = PreparedAction("image", "[表情包]" if args.as_sticker else "[图片]", ref)
                images = [await documents.blocking(documents.image_part, self.media.store.file_path(row["local_path"]))]
            identifier = uuid.uuid4().hex[:12]
            self.actions[identifier] = action
            return dumps({"status": "待执行", "marker": f"<qqbot-media:{identifier}/>", "kind": action.kind,
                          "text": action.text, "note": "最终回复使用此标记才执行；当前尚未发送或收藏。"}), images
        except (StorageError, ValueError, KeyError, TypeError, OSError, httpx.HTTPError, DeliveryError, TimeoutError) as exc:
            return dumps({"error": str(exc)}), []

    def expand(self, replies):
        """媒体标记拆成独立消息，代码中的字面量不解释；全量校验后才开始发送。"""
        result = []
        for text, quote_id in replies:
            start = 0
            for match in re.finditer(r"```.*?```|~~~.*?~~~|`[^`\n]*`|<qqbot-media:(?P<id>[^<>]*)/>", text, re.S):
                if match["id"] is None:
                    continue
                if match["id"] not in self.actions:
                    raise StorageError("模型引用了未准备的媒体，未执行此次回复。")
                if text[start:match.start()].strip():
                    result.append((text[start:match.start()].strip(), quote_id))
                result.append((self.actions[match["id"]], quote_id))
                start = match.end()
            if text[start:].strip():
                result.append((text[start:].strip(), quote_id))
        return result

    async def materialize(self, action: PreparedAction, *, deadline):
        if action.kind == "text":
            return [{"type": "text", "data": {"text": action.text}}], action.text, []
        if action.kind == "face":
            return [{"type": "face", "data": {"id": str(action.face_id)}}], action.text, []
        row = self.media.store.attachment(self.cid, action.attachment.id)
        data = {"file": "base64://" + base64.b64encode(self.media.store.file_path(row["local_path"]).read_bytes()).decode()}
        if action.kind == "image":
            data["sub_type"] = action.attachment.metadata["sub_type"]
        return [{"type": action.kind, "data": data}], action.text, [action.attachment]
