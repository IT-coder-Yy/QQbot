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


class ContextRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class SessionStatus(str, Enum):
    ACTIVE = "active"
    CLOSED = "closed"


class MemoryKind(str, Enum):
    FACT = "fact"
    DECISION = "decision"
    PREFERENCE = "preference"
    PROJECT_STATE = "project_state"


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
    TextContent | MentionContent | ImageContent | ReplyContent,
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

        return "".join(item.text for item in self.content if isinstance(item, TextContent)).strip()

    @property
    def mentioned_bot(self) -> bool:
        """
        是否明确 @ 当前机器人。
        """

        return any(isinstance(item, MentionContent) and item.target_id == self.bot_id for item in self.content)

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
    MessageReceived | MessageRecalled | FileUploaded,
    Field(discriminator="event_type"),
]


# ============================================================
# 上下文管理
# ============================================================


class ContextAttachment(SchemaModel):
    type: Literal["image", "file"]
    id: str | None = None
    name: str | None = None
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextMessage(SchemaModel):
    """Context Store 内部保存的统一消息。"""

    id: str
    role: ContextRole
    content: str
    timestamp: datetime
    # 用户消息才有
    sender_id: int | None = None
    sender_name: str | None = None
    # 对应 QQ / OneBot 的 message_id
    source_message_id: int | None = None
    attachments: list[ContextAttachment] = Field(default_factory=list)
    recalled: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionSummary(SchemaModel):
    narrative: str = ""
    goals: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    important_entities: list[str] = Field(default_factory=list)
    # 已经有多少原始消息被折叠进入 Summary
    compressed_message_count: int = 0
    updated_at: datetime | None = None


class MemoryItem(SchemaModel):
    id: str
    kind: MemoryKind
    content: str
    created_at: datetime
    updated_at: datetime
    source_session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LongTermMemory(SchemaModel):
    stable_facts: list[MemoryItem] = Field(default_factory=list)
    decisions: list[MemoryItem] = Field(default_factory=list)
    preferences: list[MemoryItem] = Field(default_factory=list)
    project_state: list[MemoryItem] = Field(default_factory=list)
    historical_summary: str = ""


class Conversation(SchemaModel):
    conversation_id: str
    conversation_type: ConversationType
    created_at: datetime
    updated_at: datetime
    active_session_id: str | None = None
    long_term_memory: LongTermMemory = Field(default_factory=LongTermMemory)


class Session(SchemaModel):
    session_id: str
    conversation_id: str
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None = None
    summary: SessionSummary = Field(default_factory=SessionSummary)
    recent_messages: list[ContextMessage] = Field(default_factory=list)


class ToolCall(SchemaModel):
    type: Literal["tool_call"] = "tool_call"
    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    timestamp: datetime


class ToolResult(SchemaModel):
    type: Literal["tool_result"] = "tool_result"
    call_id: str
    tool_name: str
    # 真正给模型看的结果
    observation: str
    # 完整原始结果以后可以存在别处
    raw_result_ref: str | None = None
    is_compacted: bool = False
    original_token_count: int | None = None
    current_token_count: int | None = None
    timestamp: datetime


ToolTraceItem: TypeAlias = Annotated[
    ToolCall | ToolResult,
    Field(discriminator="type"),
]


class AgentRun(SchemaModel):
    run_id: str
    conversation_id: str
    session_id: str
    current_turn: ContextMessage
    tool_trace: list[ToolTraceItem] = Field(default_factory=list)
    final_response: str | None = None
    started_at: datetime
    finished_at: datetime | None = None


class ModelContext(SchemaModel):
    # L0：永不压缩
    system_prompt: str
    # L1：极少压缩
    long_term_memory: LongTermMemory
    # L2：允许压缩
    session_summary: SessionSummary
    # L3：滚动窗口
    recent_messages: list[ContextMessage]
    # L4：永远保留
    current_turn: ContextMessage
    # L5：优先压缩
    tool_trace: list[ToolTraceItem] = Field(default_factory=list)


class IntentResult(SchemaModel):
    should_reply: bool

    probability: float = Field(
        ge=0.0,
        le=1.0,
    )

    model: str | None = None
