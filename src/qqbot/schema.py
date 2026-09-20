from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


class SchemaModel(BaseModel):
    """
    QQBot 内部统一 Schema 基类。
    extra="forbid":
    如果代码错误地给内部模型传入未定义字段，
    直接报错，而不是悄悄吞掉。
    """

    model_config = ConfigDict(extra="forbid")


# ============================================================
# Enum
# ============================================================


class ConversationType(str, Enum):
    PRIVATE = "private"
    GROUP = "group"


class SenderRole(str, Enum):
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    UNKNOWN = "unknown"


# ============================================================
# Sender
# ============================================================


class Sender(SchemaModel):
    id: int

    nickname: str | None = None
    card: str | None = None

    role: SenderRole = SenderRole.UNKNOWN

    @property
    def display_name(self) -> str:
        """
        群聊优先使用群名片，否则使用 QQ 昵称。
        """

        return self.card or self.nickname or str(self.id)


# ============================================================
# Message Content
# ============================================================


class TextContent(SchemaModel):
    type: Literal["text"] = "text"

    text: str


class MentionContent(SchemaModel):
    type: Literal["mention"] = "mention"

    # @某个用户
    target_id: int | None = None

    # @全体成员
    is_all: bool = False


class ImageContent(SchemaModel):
    type: Literal["image"] = "image"

    # OneBot / NapCat 的图片标识
    file: str | None = None

    # NapCat 提供的图片 URL
    url: str | None = None

    size: int | None = None

    summary: str | None = None

    sub_type: int | None = None


class ReplyContent(SchemaModel):
    type: Literal["reply"] = "reply"

    # 被引用消息的 message_id
    message_id: int


Content: TypeAlias = Annotated[
    TextContent
    | MentionContent
    | ImageContent
    | ReplyContent,
    Field(discriminator="type"),
]


# ============================================================
# File
# ============================================================


class FileInfo(SchemaModel):
    id: str

    name: str

    size: int | None = None

    busid: int | None = None

    url: str | None = None

    # 文件自身的平台特有信息
    metadata: dict[str, Any] = Field(default_factory=dict)


# ============================================================
# Domain Events
# ============================================================


class MessageReceived(SchemaModel):
    event_type: Literal["message_received"] = "message_received"

    platform: Literal["qq"] = "qq"

    # 当前机器人 QQ
    bot_id: int

    # 我们自己生成的统一会话 ID
    conversation_id: str

    conversation_type: ConversationType

    # 群聊时存在，私聊时为 None
    group_id: int | None = None

    message_id: int

    # Normalizer 会把 Unix timestamp 转成 datetime
    timestamp: datetime

    sender: Sender

    # 消息真正的数据源
    content: list[Content]

    # NapCat / OneBot 特有字段
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def text(self) -> str:
        """
        从所有 TextContent 派生纯文本。

        content 才是真正的数据源，
        text 不单独存储。
        """

        return "".join(
            item.text
            for item in self.content
            if isinstance(item, TextContent)
        ).strip()

    @property
    def mentioned_bot(self) -> bool:
        """
        是否明确 @ 当前机器人。
        """

        return any(
            isinstance(item, MentionContent)
            and item.target_id == self.bot_id
            for item in self.content
        )

    @property
    def reply_to_message_id(self) -> int | None:
        """
        当前消息是否引用了另外一条消息。
        """

        for item in self.content:
            if isinstance(item, ReplyContent):
                return item.message_id

        return None


class MessageRecalled(SchemaModel):
    event_type: Literal["message_recalled"] = "message_recalled"

    platform: Literal["qq"] = "qq"

    bot_id: int

    conversation_id: str
    conversation_type: ConversationType

    group_id: int | None = None

    # 被撤回的消息
    message_id: int

    # 原消息发送者
    sender_id: int

    # 实际执行撤回的人
    # 私聊撤回时可能不存在
    operator_id: int | None = None

    timestamp: datetime

    metadata: dict[str, Any] = Field(default_factory=dict)


class FileUploaded(SchemaModel):
    event_type: Literal["file_uploaded"] = "file_uploaded"

    platform: Literal["qq"] = "qq"

    bot_id: int

    conversation_id: str
    conversation_type: ConversationType

    group_id: int | None = None

    sender_id: int

    timestamp: datetime

    file: FileInfo

    metadata: dict[str, Any] = Field(default_factory=dict)


DomainEvent: TypeAlias = Annotated[
    MessageReceived
    | MessageRecalled
    | FileUploaded,
    Field(discriminator="event_type"),
]