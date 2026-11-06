"""主模型的可选引用指令；普通回复保持纯文本。"""

import re

QUOTE_MARKER = "<qqbot-quote/>"
MESSAGE_BREAK = "<qqbot-next/>"
SILENT_MARKER = "<qqbot-silent/>"

CHAT_REPLY_INSTRUCTIONS = f"""需要分开发送时，在两条消息之间单独一行写 {MESSAGE_BREAK}，程序会逐条发送并移除标记。
普通换行仍属于同一条消息；代码、列表、完整步骤不要随意拆散。
历史消息 ID 只是上下文信息，不要在回复正文中输出。"""

GROUP_REPLY_INSTRUCTIONS = f"""【群聊引用回复】
像群友一样自然接话，默认直接回复，不引用。根据聊天内容判断是否需要指明回复对象：
- 消息不多、简单问答、同一人的连续补充、明确的接话关系，通常不需要引用。
- 多人交叉聊天、多个问题混杂，只有直接回复容易让人弄错你在回答哪条消息时，才引用与这句话对应的用户消息。
- 不能仅因为这是群聊、用户 @ 了你、用户用了引用、正文很长或历史消息很多，就使用引用。
逐条判断回复与用户消息的对应关系，不按发送位置决定引用，不强制第一条引用或所有条都引用。
某一条回复需要引用时，在该条开头单独一行写 <qqbot-quote:消息ID/>，再换行写正文；ID 必须来自当前可见的用户消息。
例如第一条只是闲聊，第二条回答消息 123 的问题，可以只在第二条开头写 <qqbot-quote:123/>。多条回复可以各自引用不同用户消息。
不需要引用的条目只写正文。标记由程序移除并转换为 QQ 原生引用，不要另行输出 CQ 码。
这是最终回复的发送指令，工具调用照常进行；不要向群友解释标记，也不要为了引用重复抄写用户原话。"""




def parse_reply(text: str, *, current_message_id: int | None, quote_ids: set[int]) -> list[tuple[str, int | None]]:
    """清除非代码文本行首误生成的消息 ID 并解析引用，正文内的同名文字保持原样。"""
    replies = []
    for part in text.split(MESSAGE_BREAK):
        part = "".join(
            chunk if index % 2 else re.sub(r"(?m)^(<qqbot-quote(?::[^<>]*)?/>\s*)?(?:\[(?:引用)?消息ID:-?\d+\]\s*)+", r"\1", chunk)
            for index, chunk in enumerate(re.split(r"(```.*?```|~~~.*?~~~)", part.strip(), flags=re.S))
        ).strip()
        quote_id = None
        marker = re.match(r"^<qqbot-quote(?::([^<>]*))?/>", part)
        if marker:
            part = part[marker.end():].strip()
            try:
                quote_id = int(marker[1]) if marker[1] is not None else current_message_id
            except ValueError:
                pass
        if part:
            replies.append((part, quote_id if quote_id in quote_ids else None))
    return replies
