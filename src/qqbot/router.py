from .schema import ConversationType, MessageReceived


def should_route_to_agent(
    message: MessageReceived,
) -> bool:
    """
    判断消息是否需要交给 Agent。

    当前规则：
    1. 忽略机器人自己发送的消息
    2. 私聊全部处理
    3. 群聊只有 @机器人 才处理
    """

    if message.sender.id == message.bot_id:
        return False

    if message.conversation_type is ConversationType.PRIVATE:
        return True

    if message.conversation_type is ConversationType.GROUP:
        return message.mentioned_bot

    return False    