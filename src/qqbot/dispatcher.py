import asyncio
import logging
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .router import RouteDecision, decide_route
from .schema import ConversationType, MessageReceived
from .schema import AudioContent, FileContent, ImageContent, TextContent, MentionContent
from .settings import Settings
from .transport import DeliveryError

logger = logging.getLogger(__name__)


@dataclass
class MessageJob:
    message: MessageReceived
    deadline: float
    reply_started: bool = False
    proactive: bool = False
    previous_messages: tuple[MessageReceived, ...] = ()
    superseded: bool = False


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
        self._activity: dict[str, asyncio.Event] = {}
        self._running: dict[str, MessageJob] = {}
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
        if message.conversation_id in self._activity:
            self._activity[message.conversation_id].set()
        # shortcut: 去重仅在进程内生效；跨重启恢复时改为持久化状态。
        self._seen[key] = now
        if len(self._seen) > self.settings.dedup_capacity:
            self._seen.popitem(last=False)
        logger.info(
            "[收到][%s:%s] %s(%s)：%s",
            "群" if message.group_id is not None else "私聊",
            message.group_id if message.group_id is not None else message.sender.id,
            message.sender.display_name,
            message.sender.id,
            message.text or "[非文本消息]",
        )
        queue = self._queues.get(message.conversation_id)
        if self._waiting >= self.settings.queue_limit or (queue is not None and len(queue) >= self.settings.conversation_queue_limit):
            logger.warning("队列已满 conversation=%s message=%s", *key)
            self._notify(message, "现在有点忙，请稍后再试。")
            return False
        if queue is None:
            queue = self._queues[message.conversation_id] = deque()
            self._activity[message.conversation_id] = asyncio.Event()
        media_turn = message.reply_to_message_id is not None or any(isinstance(p, (AudioContent, FileContent, ImageContent)) for p in message.content)
        queue.append(MessageJob(message, now + (self.settings.media_timeout if media_turn else self.settings.task_timeout)))
        self._waiting += 1
        self._inflight.add(key)
        running = self._running.get(message.conversation_id)
        if (self.settings.input_debounce_seconds > 0 and running and not running.proactive and not running.reply_started
                and self._can_merge(running.message, message)
                and all(self._can_merge(running.message, item.message) for item in queue)):
            running.superseded = True
        if message.conversation_id not in self._workers:
            self._workers[message.conversation_id] = asyncio.create_task(self._work(message.conversation_id))
        return True

    @staticmethod
    def _can_merge(first: MessageReceived, second: MessageReceived) -> bool:
        return (isinstance(first, MessageReceived) and isinstance(second, MessageReceived)
                and first.sender.id == second.sender.id and bool(first.text and second.text)
                and all(isinstance(part, (TextContent, MentionContent))
                        and (not isinstance(part, MentionContent) or part.target_id == item.bot_id)
                        for item in (first, second) for part in item.content))


    async def _work(self, conversation_id: str) -> None:
        queue = self._queues[conversation_id]
        activity = self._activity[conversation_id]
        followup = None
        try:
            while queue:
                queued = bool(queue)
                if queue:
                    job = queue[0]
                    followup = None
                taken = False
                source_id = job.message.message_id if isinstance(job.message, MessageReceived) else "proactive"
                try:
                    async with asyncio.timeout_at(job.deadline):
                        if queued and self.settings.input_debounce_seconds > 0 and self._can_merge(job.message, job.message):
                            while all(self._can_merge(job.message, item.message) for item in queue):
                                activity.clear()
                                try:
                                    await asyncio.wait_for(activity.wait(), self.settings.input_debounce_seconds)
                                except TimeoutError:
                                    break
                        async with self._slots:
                            if asyncio.get_running_loop().time() >= job.deadline:
                                raise TimeoutError
                            if queued:
                                queue.popleft()
                                self._waiting -= 1
                                taken = True
                                while (self.settings.input_debounce_seconds > 0 and queue
                                       and self._can_merge(job.message, queue[0].message)):
                                    following = queue.popleft()
                                    job.previous_messages += (job.message,) + following.previous_messages
                                    job.message = following.message
                                    self._waiting -= 1
                                source_id = job.message.message_id if isinstance(job.message, MessageReceived) else "proactive"
                            self._running[conversation_id] = job
                            await self.handler(job)
                except DeliveryError as exc:
                    logger.warning("发送未确认 conversation=%s message=%s: %s", conversation_id, source_id, exc)
                except TimeoutError:
                    logger.warning("任务到期 conversation=%s message=%s reply_started=%s", conversation_id, source_id, job.reply_started)
                    if not job.proactive and not job.reply_started:
                        self._notify(job.message, "这次等待超时了，请稍后再试。")
                except Exception:
                    logger.exception("处理失败 conversation=%s message=%s", conversation_id, source_id)
                    if not job.proactive and not job.reply_started:
                        self._notify(job.message, "这次处理失败了，请稍后再试。")
                finally:
                    self._running.pop(conversation_id, None)
                    if job.superseded and queue and self._can_merge(job.message, queue[0].message):
                        queue[0].previous_messages = job.previous_messages + (job.message,) + queue[0].previous_messages
                    if queued and not taken:
                        queue.popleft()
                        self._waiting -= 1
                    if not job.proactive:
                        for item in (*job.previous_messages, job.message):
                            self._inflight.discard((conversation_id, item.message_id))
        finally:
            self._waiting -= len(queue)
            for job in queue:
                if not job.proactive:
                    self._inflight.discard((conversation_id, job.message.message_id))
            self._queues.pop(conversation_id, None)
            self._workers.pop(conversation_id, None)
            self._activity.pop(conversation_id, None)


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
        self._activity.clear()
        self._running.clear()
        self._notifications.clear()
        self._inflight.clear()
        self._waiting = 0
