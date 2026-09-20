import asyncio
import json
import os

import websockets
from dotenv import load_dotenv

from .agent import Agent
from .normalizer import normalize_event
from .router import should_route_to_agent
from .schema import MessageReceived
from .transport import NapCatTransport


load_dotenv()


WS_URL = os.getenv(
    "NAPCAT_WS_URL",
    "ws://127.0.0.1:3001",
)

TOKEN = os.environ["NAPCAT_TOKEN"]


async def main() -> None:
    headers = {
        "Authorization": f"Bearer {TOKEN}",
    }

    agent = Agent()

    async with websockets.connect(
        WS_URL,
        additional_headers=headers,
    ) as ws:
        print("已连接 NapCat")

        transport = NapCatTransport(ws)

        async for raw in ws:
            raw_event = json.loads(raw)

            domain_event = normalize_event(raw_event)

            if not isinstance(
                domain_event,
                MessageReceived,
            ):
                continue

            data = domain_event.model_dump(
                mode="json"
            )

            data["timestamp"] = (
                domain_event.timestamp.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )

            print("=" * 80)
            print(
                json.dumps(
                    data,
                    ensure_ascii=False,
                    indent=2,
                )
            )

            if not should_route_to_agent(
                domain_event
            ):
                continue

            try:
                reply = await agent.run(
                    domain_event
                )

                await transport.reply_text(
                    domain_event,
                    reply,
                    quote=True,
                )

            except Exception as exc:
                print(
                    f"处理消息失败: {exc!r}"
                )


if __name__ == "__main__":
    asyncio.run(main())