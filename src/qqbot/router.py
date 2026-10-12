from enum import Enum

from .schema import (
    ConversationType,
    MessageReceived,
)


class RouteDecision(str, Enum):
    IGNORE = "ignore"

    REPLY = "reply"

    INTENT = "intent"


def decide_route(
    message: MessageReceived,
) -> RouteDecision:
    """
    Router 只处理确定性规则。

    不在这里调用任何模型。
    """

    # 防止机器人处理自己发送的消息。
    if message.sender.id == message.bot_id:
        return RouteDecision.IGNORE

    # 私聊一定交给 Agent。
    if message.conversation_type is ConversationType.PRIVATE:
        return RouteDecision.REPLY

    if message.conversation_type is ConversationType.GROUP:
        # 群聊明确 @Bot：
        # 不需要浪费一次 Jev 调用。
        if message.mentioned_bot:
            return RouteDecision.REPLY

        # 没有 @：
        # 交给 Jev 判断。
        return RouteDecision.INTENT

    return RouteDecision.IGNORE
