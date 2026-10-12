import asyncio
import json

from qqbot.normalizer import normalize_event


def raw_message(message_id=1, user_id=20, group_id=None, mentioned=False):
    data = {
        "post_type": "message",
        "message_type": "group" if group_id else "private",
        "self_id": 10,
        "user_id": user_id,
        "message_id": message_id,
        "time": 1_700_000_000,
        "sender": {"nickname": "测试用户"},
        "message": [{"type": "text", "data": {"text": f"问题 {message_id}"}}],
    }
    if group_id:
        data["group_id"] = group_id
    if mentioned:
        data["message"].append({"type": "at", "data": {"qq": "10"}})
    return data


def message(*args, **kwargs):
    return normalize_event(raw_message(*args, **kwargs))


async def eventually(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0)


class FakeSocket:
    def __init__(self):
        self.sent = []
        self.incoming = asyncio.Queue()

    async def send(self, data):
        self.sent.append(json.loads(data))

    def __aiter__(self):
        return self

    async def __anext__(self):
        data = await self.incoming.get()
        if data is None:
            raise StopAsyncIteration
        return data


def success(action, message_id=101):
    return {"echo": action["echo"], "status": "ok", "retcode": 0, "data": {"message_id": message_id}}
