import asyncio
import logging
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .router import RouteDecision, decide_route
from .schema import MessageReceived
from .settings import Settings
from .transport import DeliveryError

logger = logging.getLogger(__name__)


@dataclass
class MessageJob:
    message: MessageReceived
    deadline: float
    reply_started: bool = False


class MessageDispatcher:
    """接纳不等待；每会话串行，跨会话共享并发额度。"""

    def __init__(
        self,
        settings: Settings,
        handler: Callable[[MessageJob], Awaitable[None]],
        send_notice: Callable[[MessageReceived, str], Awaitable[None]],
    ) -> None:
        self.settings = settings
        self.handler = handler
        self.send_notice = send_notice
        self._slots = asyncio.Semaphore(settings.concurrency)
        self._queues: dict[str, deque[MessageJob]] = {}
        self._workers: dict[str, asyncio.Task] = {}
        self._notifications: set[asyncio.Task] = set()
        self._seen: OrderedDict[tuple[str, int], float] = OrderedDict()
        self._inflight: set[tuple[str, int]] = set()
        self._last_notice: OrderedDict[str, float] = OrderedDict()
        self._waiting = 0
        self._closed = False

    def submit(self, message: MessageReceived) -> bool:
        if self._closed or decide_route(message) is RouteDecision.IGNORE:
            return False
        now = asyncio.get_running_loop().time()
        key = (message.conversation_id, message.message_id)
        while self._seen and next(iter(self._seen.values())) <= now - self.settings.dedup_ttl:
            self._seen.popitem(last=False)
        if key in self._seen or key in self._inflight:
            return False
        # shortcut: 去重仅在进程内生效；跨重启恢复时改为持久化状态。
        self._seen[key] = now
        if len(self._seen) > self.settings.dedup_capacity:
            self._seen.popitem(last=False)
        queue = self._queues.get(message.conversation_id)
        if self._waiting >= self.settings.queue_limit or (queue is not None and len(queue) >= self.settings.conversation_queue_limit):
            logger.warning("队列已满 conversation=%s message=%s", *key)
            self._notify(message, "现在有点忙，请稍后再试。")
            return False
        if queue is None:
            queue = self._queues[message.conversation_id] = deque()
        queue.append(MessageJob(message, now + self.settings.task_timeout))
        self._waiting += 1
        self._inflight.add(key)
        if message.conversation_id not in self._workers:
            self._workers[message.conversation_id] = asyncio.create_task(self._work(message.conversation_id))
        return True

    async def _work(self, conversation_id: str) -> None:
        queue = self._queues[conversation_id]
        try:
            while queue:
                job = queue[0]
                taken = False
                try:
                    async with asyncio.timeout_at(job.deadline):
                        async with self._slots:
                            if asyncio.get_running_loop().time() >= job.deadline:
                                raise TimeoutError
                            queue.popleft()
                            self._waiting -= 1
                            taken = True
                            await self.handler(job)
                except DeliveryError as exc:
                    logger.warning("发送未确认 conversation=%s message=%s: %s", conversation_id, job.message.message_id, exc)
                except TimeoutError:
                    logger.warning("任务到期 conversation=%s message=%s reply_started=%s", conversation_id, job.message.message_id, job.reply_started)
                    if not job.reply_started:
                        self._notify(job.message, "这次等待超时了，请稍后再试。")
                except Exception:
                    logger.exception("处理失败 conversation=%s message=%s", conversation_id, job.message.message_id)
                    if not job.reply_started:
                        self._notify(job.message, "这次处理失败了，请稍后再试。")
                finally:
                    if not taken:
                        queue.popleft()
                        self._waiting -= 1
                    self._inflight.discard((conversation_id, job.message.message_id))
        finally:
            self._waiting -= len(queue)
            for job in queue:
                self._inflight.discard((conversation_id, job.message.message_id))
            self._queues.pop(conversation_id, None)
            self._workers.pop(conversation_id, None)

    def _notify(self, message: MessageReceived, text: str) -> None:
        if self._closed or decide_route(message) is not RouteDecision.REPLY:
            return
        now = asyncio.get_running_loop().time()
        previous = self._last_notice.get(message.conversation_id)
        if previous is not None and now - previous < self.settings.notice_cooldown:
            return
        if len(self._notifications) >= self.settings.concurrency:
            return
        self._last_notice[message.conversation_id] = now
        self._last_notice.move_to_end(message.conversation_id)
        if len(self._last_notice) > self.settings.dedup_capacity:
            self._last_notice.popitem(last=False)
        task = asyncio.create_task(self._deliver_notice(message, text))
        self._notifications.add(task)
        task.add_done_callback(self._notifications.discard)

    async def _deliver_notice(self, message: MessageReceived, text: str) -> None:
        try:
            async with asyncio.timeout(self.settings.action_timeout):
                await self.send_notice(message, text)
        except Exception:
            logger.warning("提示发送未确认 conversation=%s message=%s", message.conversation_id, message.message_id, exc_info=True)

    async def close(self) -> None:
        self._closed = True
        tasks = [*self._workers.values(), *self._notifications]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        # 刚创建且尚未运行的任务不会进入 finally。
        self._workers.clear()
        self._queues.clear()
        self._notifications.clear()
        self._inflight.clear()
        self._waiting = 0
