import json
import uuid
from typing import Any

from .schema import ConversationType, MessageReceived


class NapCatTransport:
    """
    负责：
    我们的程序 -> OneBot Action -> NapCat -> QQ
    """

    def __init__(self, websocket: Any) -> None:
        self.websocket = websocket

    async def send_private_text(
        self,
        user_id: int,
        text: str,
        *,
        reply_to_message_id: int | None = None,
    ) -> None:
        message = self._build_text_message(
            text=text,
            reply_to_message_id=reply_to_message_id,
        )

        await self._send_action(
            action="send_msg",
            params={
                "message_type": "private",
                "user_id": user_id,
                "message": message,
            },
        )

    async def send_group_text(
        self,
        group_id: int,
        text: str,
        *,
        reply_to_message_id: int | None = None,
    ) -> None:
        message = self._build_text_message(
            text=text,
            reply_to_message_id=reply_to_message_id,
        )

        await self._send_action(
            action="send_msg",
            params={
                "message_type": "group",
                "group_id": group_id,
                "message": message,
            },
        )

    async def reply_text(
        self,
        message: MessageReceived,
        text: str,
        *,
        quote: bool = False,
    ) -> None:
        """
        对收到的 MessageReceived 进行回复。

        quote=True：
        使用 QQ 的引用回复。
        """

        reply_to_message_id = (
            message.message_id
            if quote
            else None
        )

        if message.conversation_type is ConversationType.PRIVATE:
            await self.send_private_text(
                user_id=message.sender.id,
                text=text,
                reply_to_message_id=reply_to_message_id,
            )
            return

        if message.conversation_type is ConversationType.GROUP:
            if message.group_id is None:
                raise ValueError(
                    "群聊消息缺少 group_id"
                )

            await self.send_group_text(
                group_id=message.group_id,
                text=text,
                reply_to_message_id=reply_to_message_id,
            )
            return

        raise ValueError(
            f"不支持的会话类型: {message.conversation_type}"
        )

    async def _send_action(
        self,
        action: str,
        params: dict[str, Any],
    ) -> None:
        payload = {
            "action": action,
            "params": params,
            "echo": str(uuid.uuid4()),
        }

        await self.websocket.send(
            json.dumps(
                payload,
                ensure_ascii=False,
            )
        )

    @staticmethod
    def _build_text_message(
        *,
        text: str,
        reply_to_message_id: int | None,
    ) -> list[dict[str, Any]]:
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

        segments.append(
            {
                "type": "text",
                "data": {
                    "text": text,
                },
            }
        )

        return segments