import asyncio
import json
import logging
import re
import uuid
from typing import Any

from websockets.exceptions import ConnectionClosed

from .schema import ConversationType, MessageReceived

logger = logging.getLogger(__name__)


class DeliveryError(RuntimeError):
    """OneBot 明确拒绝了发送。"""


class DeliveryUnknownError(DeliveryError):
    """可能已发送，不能安全地自动重发。"""


class NapCatTransport:
    """
    负责：
    我们的程序 -> OneBot Action -> NapCat -> QQ
    """

    def __init__(self, websocket: Any = None, *, action_timeout: float = 10) -> None:
        self.websocket = websocket
        self.action_timeout = action_timeout
        self._connected = asyncio.Event()
        self._closed = False
        self._pending: dict[str, asyncio.Future] = {}
        if websocket is not None:
            self._connected.set()

    @property
    def connected(self) -> bool:
        return self.websocket is not None

    def attach(self, websocket: Any) -> None:
        self.websocket = websocket
        self._connected.set()

    def detach(self, websocket: Any) -> None:
        if self.websocket is not websocket:
            return
        self.websocket = None
        self._connected.clear()
        for future in self._pending.values():
            if not future.done():
                future.set_exception(DeliveryUnknownError("等待发送回执期间连接中断"))

    def close(self) -> None:
        self.detach(self.websocket)
        self._closed = True
        self._connected.set()

    async def wait_connected(self) -> Any:
        while not self.connected:
            if self._closed:
                raise RuntimeError("发送器已关闭")
            await self._connected.wait()
        return self.websocket

    def handle_response(self, payload: dict[str, Any]) -> bool:
        if "post_type" in payload or "echo" not in payload:
            return False
        echo = payload["echo"]
        future = self._pending.get(echo) if isinstance(echo, str) else None
        if future is not None and not future.done():
            future.set_result(payload)
        return True

    async def send_private_text(self, user_id: int, text: str, *, reply_to_message_id: int | None = None, deadline: float | None = None) -> int:
        message = self._build_text_message(
            text=text,
            reply_to_message_id=reply_to_message_id,
        )

        return await self._send_action(
            action="send_msg",
            params={
                "message_type": "private",
                "user_id": user_id,
                "message": message,
            },
            deadline=deadline,
        )

    async def send_group_text(self, group_id: int, text: str, *, reply_to_message_id: int | None = None, deadline: float | None = None) -> int:
        message = self._build_text_message(
            text=text,
            reply_to_message_id=reply_to_message_id,
        )

        return await self._send_action(
            action="send_msg",
            params={
                "message_type": "group",
                "group_id": group_id,
                "message": message,
            },
            deadline=deadline,
        )

    async def reply_text(self, message: MessageReceived, text: str, *, quote: bool = False, deadline: float | None = None) -> int:
        """
        对收到的 MessageReceived 进行回复。

        quote=True：
        使用 QQ 的引用回复。
        """

        reply_to_message_id = message.message_id if quote else None

        if message.conversation_type is ConversationType.PRIVATE:
            return await self.send_private_text(
                user_id=message.sender.id,
                text=text,
                reply_to_message_id=reply_to_message_id,
                deadline=deadline,
            )

        if message.conversation_type is ConversationType.GROUP:
            if message.group_id is None:
                raise ValueError("群聊消息缺少 group_id")

            return await self.send_group_text(
                group_id=message.group_id,
                text=text,
                reply_to_message_id=reply_to_message_id,
                deadline=deadline,
            )

        raise ValueError(f"不支持的会话类型: {message.conversation_type}")

    async def _send_action(self, action: str, params: dict[str, Any], *, deadline: float | None = None) -> int:
        loop = asyncio.get_running_loop()
        if deadline is None:
            deadline = loop.time() + self.action_timeout
        echo = str(uuid.uuid4())
        payload = {
            "action": action,
            "params": params,
            "echo": echo,
        }
        future = None
        attempted = False
        try:
            async with asyncio.timeout_at(deadline):
                websocket = await self.wait_connected()
                if loop.time() >= deadline:
                    raise TimeoutError
                future = loop.create_future()
                self._pending[echo] = future
                async with asyncio.timeout(self.action_timeout):
                    attempted = True
                    await websocket.send(json.dumps(payload, ensure_ascii=False))
                    response = await future
        except TimeoutError as exc:
            if attempted:
                raise DeliveryUnknownError("发送回执超时，结果未知，不自动重发") from exc
            raise
        except (ConnectionClosed, OSError) as exc:
            raise DeliveryUnknownError("发送时连接中断，结果未知，不自动重发") from exc
        except asyncio.CancelledError:
            if attempted:
                logger.warning("发送确认被取消，结果未知 echo=%s", echo)
            raise
        finally:
            self._pending.pop(echo, None)
            if future is not None:
                if not future.done():
                    future.cancel()
                elif not future.cancelled():
                    future.exception()

        if response.get("status") == "failed":
            raise DeliveryError(f"OneBot 拒绝发送，retcode={response.get('retcode')}")
        data = response.get("data")
        message_id = data.get("message_id") if isinstance(data, dict) else None
        if response.get("status") != "ok" or type(response.get("retcode")) is not int or response["retcode"] != 0 or type(message_id) is not int:
            raise DeliveryUnknownError("未收到完整的同步成功回执，结果未知")
        return message_id

    @staticmethod
    def _normalize_qq_text(text: str) -> str:
        # 连续两个及以上换行压成一个换行
        return re.sub(r"\n\s*\n+", "\n", text.strip())

    def _build_text_message(self, *, text: str, reply_to_message_id: int | None) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []

        if reply_to_message_id is not None:
            segments.append(
                {
                    "type": "reply",
                    "data": {
                        "id": str(reply_to_message_id),
                    },
                }
            )

        text = self._normalize_qq_text(text)

        segments.append(
            {
                "type": "text",
                "data": {
                    "text": text,
                },
            }
        )

        return segments
