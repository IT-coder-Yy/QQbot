from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any

from .schema import (
    Content,
    ConversationType,
    DomainEvent,
    FileInfo,
    FileUploaded,
    ImageContent,
    MentionContent,
    MessageReceived,
    MessageRecalled,
    ReplyContent,
    Sender,
    SenderRole,
    TextContent,
)


def normalize_event(event: dict[str, Any]) -> DomainEvent | None:
    """
    OneBot Raw Event 的统一入口。

    返回：
        MessageReceived
        MessageRecalled
        FileUploaded
        None

    None 表示当前事件不需要进入业务层，
    例如 heartbeat。
    """

    post_type = event.get("post_type")

    if post_type == "message":
        return _normalize_message(event)

    if post_type == "notice":
        notice_type = event.get("notice_type")

        if notice_type in {
            "group_recall",
            "friend_recall",
        }:
            return _normalize_recall(event)

        if notice_type == "group_upload":
            return _normalize_file_upload(event)

    return None


# ============================================================
# Message
# ============================================================


def _normalize_message(
    event: dict[str, Any],
) -> MessageReceived | None:

    message_type = event.get("message_type")

    if message_type == "group":
        conversation_type = ConversationType.GROUP

    elif message_type == "private":
        conversation_type = ConversationType.PRIVATE

    else:
        return None

    bot_id = int(event["self_id"])
    user_id = int(event["user_id"])

    group_id: int | None = None

    if conversation_type is ConversationType.GROUP:
        group_id = int(event["group_id"])

    conversation_id = _build_conversation_id(
        bot_id=bot_id,
        conversation_type=conversation_type,
        user_id=user_id,
        group_id=group_id,
    )

    raw_segments = event.get("message", [])

    if not isinstance(raw_segments, list):
        raise ValueError(
            "Normalizer 要求 NapCat 使用 Array 消息格式"
        )

    content, unhandled_segments = _normalize_content(
        raw_segments
    )

    sender_data = event.get("sender") or {}

    sender = Sender(
        id=user_id,
        nickname=sender_data.get("nickname"),
        card=sender_data.get("card"),
        role=_normalize_sender_role(
            sender_data.get("role")
        ),
    )

    metadata = _build_message_metadata(
        event,
        unhandled_segments,
    )

    return MessageReceived(
        bot_id=bot_id,
        conversation_id=conversation_id,
        conversation_type=conversation_type,
        group_id=group_id,
        message_id=int(event["message_id"]),
        timestamp=_normalize_timestamp(event["time"]),
        sender=sender,
        content=content,
        metadata=metadata,
    )


# ============================================================
# Content
# ============================================================


def _normalize_content(
    segments: list[dict[str, Any]],
) -> tuple[list[Content], list[dict[str, Any]]]:

    content: list[Content] = []

    # 暂时不认识的消息段不会丢掉，
    # 放到 metadata 里方便以后继续支持。
    unhandled_segments: list[dict[str, Any]] = []

    for segment in segments:
        segment_type = segment.get("type")
        data = segment.get("data") or {}

        if segment_type == "text":
            content.append(
                TextContent(
                    text=str(data.get("text", ""))
                )
            )

            continue

        if segment_type == "at":
            qq = data.get("qq")

            if qq == "all":
                content.append(
                    MentionContent(
                        target_id=None,
                        is_all=True,
                    )
                )

            elif qq is not None:
                content.append(
                    MentionContent(
                        target_id=int(qq),
                    )
                )

            continue

        if segment_type == "image":
            content.append(
                ImageContent(
                    file=data.get("file"),
                    url=data.get("url"),
                    size=data.get("file_size"),
                    summary=data.get("summary"),
                    sub_type=data.get("sub_type"),
                )
            )

            continue

        if segment_type == "reply":
            reply_id = data.get("id")

            if reply_id is not None:
                content.append(
                    ReplyContent(
                        message_id=int(reply_id)
                    )
                )

            continue

        unhandled_segments.append(segment)

    return content, unhandled_segments


# ============================================================
# Recall
# ============================================================


def _normalize_recall(
    event: dict[str, Any],
) -> MessageRecalled:

    notice_type = event["notice_type"]

    bot_id = int(event["self_id"])
    user_id = int(event["user_id"])

    if notice_type == "group_recall":
        conversation_type = ConversationType.GROUP

        group_id = int(event["group_id"])

        operator_id = int(
            event.get("operator_id", user_id)
        )

    else:
        conversation_type = ConversationType.PRIVATE

        group_id = None
        operator_id = None

    conversation_id = _build_conversation_id(
        bot_id=bot_id,
        conversation_type=conversation_type,
        user_id=user_id,
        group_id=group_id,
    )

    return MessageRecalled(
        bot_id=bot_id,
        conversation_id=conversation_id,
        conversation_type=conversation_type,
        group_id=group_id,
        message_id=int(event["message_id"]),
        sender_id=user_id,
        operator_id=operator_id,
        timestamp=_normalize_timestamp(event["time"]),
        metadata={
            "notice_type": notice_type,
        },
    )


# ============================================================
# File Upload
# ============================================================


def _normalize_file_upload(
    event: dict[str, Any],
) -> FileUploaded:

    bot_id = int(event["self_id"])
    user_id = int(event["user_id"])
    group_id = int(event["group_id"])

    conversation_type = ConversationType.GROUP

    conversation_id = _build_conversation_id(
        bot_id=bot_id,
        conversation_type=conversation_type,
        user_id=user_id,
        group_id=group_id,
    )

    raw_file = event["file"]

    file = FileInfo(
        id=str(raw_file["id"]),
        name=str(raw_file["name"]),
        size=raw_file.get("size"),
        busid=raw_file.get("busid"),
    )

    return FileUploaded(
        bot_id=bot_id,
        conversation_id=conversation_id,
        conversation_type=conversation_type,
        group_id=group_id,
        sender_id=user_id,
        timestamp=_normalize_timestamp(event["time"]),
        file=file,
        metadata={
            "notice_type": "group_upload",
        },
    )


# ============================================================
# Helpers
# ============================================================


def _build_conversation_id(
    *,
    bot_id: int,
    conversation_type: ConversationType,
    user_id: int,
    group_id: int | None,
) -> str:

    if conversation_type is ConversationType.GROUP:

        if group_id is None:
            raise ValueError(
                "群聊事件必须存在 group_id"
            )

        return f"qq:group:{bot_id}:{group_id}"

    return f"qq:private:{bot_id}:{user_id}"


def _normalize_sender_role(
    role: Any,
) -> SenderRole:

    if role == "owner":
        return SenderRole.OWNER

    if role == "admin":
        return SenderRole.ADMIN

    if role == "member":
        return SenderRole.MEMBER

    return SenderRole.UNKNOWN


def _normalize_timestamp(timestamp: Any) -> datetime:
    return datetime.fromtimestamp(
        int(timestamp),
        tz=ZoneInfo("Asia/Shanghai"),
    )

def _build_message_metadata(
    event: dict[str, Any],
    unhandled_segments: list[dict[str, Any]],
) -> dict[str, Any]:

    metadata_keys = (
        "raw_message",
        "sub_type",
        "message_seq",
        "real_id",
        "real_seq",
        "message_format",
        "font",
        "group_name",
    )

    metadata = {
        key: event[key]
        for key in metadata_keys
        if key in event
    }

    if unhandled_segments:
        metadata["unhandled_segments"] = (
            unhandled_segments
        )

    return metadata