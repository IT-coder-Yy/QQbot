"""两表持久化；模型近期消息是档案的投影，压缩不删除档案。"""
from __future__ import annotations

import json
import shutil
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from qqbot.schema import ContextMessage, Conversation, ConversationType, Session, SessionSummary
from .store import InMemoryContextStore


class StorageError(RuntimeError):
    """可以直接告知 QQ 用户的存储错误。"""


TOO_LARGE = "文件太大，无法解析，请拆分后重新发送。"


def dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def size(value) -> int:
    return len(dumps(value).encode("utf-8"))


def milliseconds(value: datetime) -> int:
    return int(value.timestamp() * 1000)


class SQLiteContextStore(InMemoryContextStore):
    persistent = True

    def __init__(self, path: Path, attachments_dir: Path, *, conversation_bytes=50_000_000, attachment_bytes=50_000_000):
        super().__init__()
        self.conversation_limit = conversation_bytes
        self.attachment_limit = attachment_bytes
        if min(conversation_bytes, attachment_bytes) <= 0:
            raise ValueError("存储配额必须大于 0")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.files = attachments_dir.resolve()
        self.files.mkdir(parents=True, exist_ok=True)
        self.protected: set[str] = set()
        self.protected_attachments: set[str] = set()
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 1, 2):
            self.db.close()
            raise RuntimeError(f"不支持的数据库版本: {version}")
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY NOT NULL,
                messages_json TEXT NOT NULL DEFAULT '[]' CHECK(json_valid(messages_json) AND json_type(messages_json)='array'),
                summary_json TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(summary_json) AND json_type(summary_json)='object'),
                summary_through_seq INTEGER NOT NULL DEFAULT 0 CHECK(summary_through_seq >= 0),
                next_message_seq INTEGER NOT NULL DEFAULT 1 CHECK(next_message_seq > summary_through_seq),
                created_at INTEGER NOT NULL,
                last_message_at INTEGER NOT NULL,
                storage_bytes INTEGER NOT NULL DEFAULT 0 CHECK(storage_bytes >= 0)
            );
            CREATE INDEX IF NOT EXISTS conversations_cleanup ON conversations(last_message_at, id);
            CREATE TABLE IF NOT EXISTS attachments (
                id TEXT PRIMARY KEY NOT NULL,
                conversation_id TEXT NOT NULL,
                source_key TEXT NOT NULL,
                source_message_id INTEGER,
                uploader_id INTEGER,
                kind TEXT NOT NULL CHECK(kind IN ('image','audio','document')),
                name TEXT NOT NULL,
                source_json TEXT NOT NULL CHECK(json_valid(source_json)),
                local_path TEXT,
                file_bytes INTEGER NOT NULL DEFAULT 0 CHECK(file_bytes >= 0),
                parsed_content_json TEXT CHECK(parsed_content_json IS NULL OR json_valid(parsed_content_json)),
                status TEXT NOT NULL DEFAULT 'registered' CHECK(status IN ('registered','ready','partial','failed')),
                error TEXT,
                created_at INTEGER NOT NULL,
                last_used_at INTEGER NOT NULL,
                storage_bytes INTEGER NOT NULL DEFAULT 0 CHECK(storage_bytes >= 0),
                UNIQUE(conversation_id, source_key)
            );
            CREATE INDEX IF NOT EXISTS attachments_message ON attachments(conversation_id, source_message_id);
            CREATE INDEX IF NOT EXISTS attachments_conversation ON attachments(conversation_id, created_at);
            CREATE INDEX IF NOT EXISTS attachments_cleanup ON attachments(last_used_at, created_at, id);
        ''')
        if version < 2:
            with self.db:
                self.db.execute("ALTER TABLE conversations ADD COLUMN next_proactive_at INTEGER")
                self.db.execute("PRAGMA user_version=2")
        # tmp 只用于本服务转换；启动时不存在需要恢复的执行任务。
        shutil.rmtree(self.files / "tmp", ignore_errors=True)
        self.cleanup()

    def close(self):
        self.db.close()

    def release(self, conversation_id):
        self.conversations.pop(conversation_id, None)
        self.sessions.pop(conversation_id, None)

    def get_conversation(self, conversation_id):
        if conversation_id not in self.conversations:
            row = self.db.execute("SELECT * FROM conversations WHERE id=?", (conversation_id,)).fetchone()
            if row is None:
                return None
            self.conversations[conversation_id] = Conversation(
                conversation_id=conversation_id,
                conversation_type=ConversationType(conversation_id.split(":")[1]),
                created_at=datetime.fromtimestamp(row["created_at"] / 1000, timezone.utc),
                updated_at=datetime.fromtimestamp(row["last_message_at"] / 1000, timezone.utc),
                active_session_id=conversation_id,
            )
        return self.conversations[conversation_id]

    def save_conversation(self, conversation):
        # 会话创建、消息与摘要的实际写入统一在 save_session 中完成。
        self.conversations[conversation.conversation_id] = conversation

    def get_session(self, session_id):
        if session_id not in self.sessions:
            row = self.db.execute("SELECT * FROM conversations WHERE id=?", (session_id,)).fetchone()
            if row is None:
                return None
            self.sessions[session_id] = Session(
                session_id=session_id, conversation_id=session_id,
                created_at=datetime.fromtimestamp(row["created_at"] / 1000, timezone.utc),
                updated_at=datetime.fromtimestamp(row["last_message_at"] / 1000, timezone.utc),
                summary=SessionSummary.model_validate_json(row["summary_json"]),
                summary_through_seq=row["summary_through_seq"],
                recent_messages=[ContextMessage.model_validate(m) for m in json.loads(row["messages_json"])
                                 if m["seq"] > row["summary_through_seq"]],
            )
        return self.sessions[session_id]

    def history(self, conversation_id):
        row = self.db.execute("SELECT messages_json FROM conversations WHERE id=?", (conversation_id,)).fetchone()
        return json.loads(row[0]) if row else []

    def has_message(self, conversation_id, message_id):
        return any(m.get("source_message_id") == message_id and m["role"] == "user"
                   for m in self.history(conversation_id))

    def next_proactive_time(self, conversation_id):
        row = self.db.execute("SELECT next_proactive_at FROM conversations WHERE id=?", (conversation_id,)).fetchone()
        return datetime.fromtimestamp(row[0] / 1000, timezone.utc) if row and row[0] is not None else None

    def schedule_proactive(self, conversation_id, when):
        old = self.db.execute("SELECT * FROM conversations WHERE id=?", (conversation_id,)).fetchone()
        now = int(time.time() * 1000)
        row = dict(old) if old else dict(id=conversation_id, messages_json="[]", summary_json="{}",
            summary_through_seq=0, next_message_seq=1, created_at=now, last_message_at=now, storage_bytes=0)
        row["next_proactive_at"] = milliseconds(when)
        with self.db:
            self._write_conversation(row)
            self._clean_conversations(current=conversation_id)

    @staticmethod
    def _row_size(row):
        return size({k: v for k, v in dict(row).items() if k != "storage_bytes"})

    def _write_conversation(self, row):
        row["storage_bytes"] = self._row_size(row)
        columns = ",".join(row)
        self.db.execute(f"INSERT OR REPLACE INTO conversations ({columns}) VALUES ({','.join('?' for _ in row)})", tuple(row.values()))

    def save_session(self, session):
        cid = session.conversation_id
        old = self.db.execute("SELECT * FROM conversations WHERE id=?", (cid,)).fetchone()
        row = dict(old) if old else dict(id=cid, messages_json="[]", summary_json="{}", summary_through_seq=0,
                                        next_message_seq=1, created_at=milliseconds(session.created_at),
                                        last_message_at=milliseconds(session.updated_at), storage_bytes=0, next_proactive_at=None)
        archive = {m["id"]: m for m in json.loads(row["messages_json"])}
        for message in session.recent_messages:
            if not message.seq:
                if size(message.model_dump(mode="json")) > self.conversation_limit:
                    raise StorageError("消息太大，无法保存。")
                message.seq = row["next_message_seq"]
                row["next_message_seq"] += 1
            elif message.id not in archive:
                continue  # 已被容量清理的旧消息不能因缓存写回复活。
            archive[message.id] = message.model_dump(mode="json")
        row.update(messages_json=dumps(sorted(archive.values(), key=lambda m: m["seq"])),
                   summary_json=session.summary.model_dump_json(), summary_through_seq=session.summary_through_seq,
                   last_message_at=max(row["last_message_at"], milliseconds(session.updated_at)))
        with self.db:
            self._write_conversation(row)
            self._clean_conversations(current=cid)
        kept = {m["id"] for m in self.history(cid)}
        session.recent_messages[:] = [m for m in session.recent_messages if m.id in kept]
        self.sessions[session.session_id] = session

    def _clean_conversations(self, current=None):
        total = self.db.execute("SELECT coalesce(sum(storage_bytes),0) FROM conversations").fetchone()[0]
        if total <= self.conversation_limit:
            return
        rows = list(self.db.execute("SELECT * FROM conversations ORDER BY last_message_at,id"))
        # 当前会话最后处理；允许丢弃其旧档案，但保留最新消息及摘要。
        rows.sort(key=lambda r: r["id"] == current)
        for value in rows:
            row = dict(value)
            cid = row["id"]
            if cid in self.protected and cid != current:
                continue
            messages = json.loads(row["messages_json"])
            minimum = 1 if cid == current and messages else 0
            removed = 0
            while len(messages) - removed > minimum and total > self.conversation_limit:
                # 外层计量把 JSON 作为字符串；用同一序列化规则计量转义开销。
                reduction = len(dumps(dumps(messages[removed])).encode("utf-8")) - 2 + (1 if len(messages) - removed > 1 else 0)
                total -= reduction
                removed += 1
            row["messages_json"] = dumps(messages[removed:])
            self._write_conversation(row)
            # 重新取得精确差值，避免计数漂移。
            total = self.db.execute("SELECT coalesce(sum(storage_bytes),0) FROM conversations").fetchone()[0]
            if total > self.conversation_limit and cid != current:
                self.db.execute("DELETE FROM conversations WHERE id=?", (cid,))
                total -= row["storage_bytes"]
            if cid != current:
                self.release(cid)
            if total <= self.conversation_limit:
                return
        if total > self.conversation_limit:
            raise StorageError("聊天存储已满，当前内容无法保存，请稍后再试。")

    @contextmanager
    def pin(self, ids):
        added = set(ids) - self.protected_attachments
        self.protected_attachments.update(added)
        try:
            yield
        finally:
            self.protected_attachments.difference_update(added)

    def attachment(self, cid, attachment_id):
        row = self.db.execute("SELECT * FROM attachments WHERE conversation_id=? AND id=?", (cid, attachment_id)).fetchone()
        return dict(row) if row else None

    def attachments(self, cid):
        return [dict(r) for r in self.db.execute("SELECT * FROM attachments WHERE conversation_id=? ORDER BY created_at DESC,id", (cid,))]

    def save_attachment(self, row):
        row = dict(row)
        row["storage_bytes"] = self._row_size(row) + row["file_bytes"]
        if row["storage_bytes"] > self.attachment_limit:
            raise StorageError(TOO_LARGE)
        old = self.attachment(row["conversation_id"], row["id"])
        removed = []
        with self.db:
            columns = ",".join(row)
            self.db.execute(f"INSERT OR REPLACE INTO attachments ({columns}) VALUES ({','.join('?' for _ in row)})", tuple(row.values()))
            removed = self._clean_attachments(current=row["id"])
        if old and old["local_path"] != row["local_path"]:
            removed.append(old["local_path"])
        for name in removed:
            if name:
                self.file_path(name).unlink(missing_ok=True)
        return row

    def _clean_attachments(self, current=None):
        total = self.db.execute("SELECT coalesce(sum(storage_bytes),0) FROM attachments").fetchone()[0]
        removed = []
        for victim in self.db.execute("SELECT * FROM attachments ORDER BY last_used_at,created_at,id").fetchall():
            if total <= self.attachment_limit:
                break
            if victim["id"] == current or victim["id"] in self.protected_attachments:
                continue
            self.db.execute("DELETE FROM attachments WHERE id=?", (victim["id"],))
            total -= victim["storage_bytes"]
            removed.append(victim["local_path"])
        if total > self.attachment_limit:
            raise StorageError("附件空间正在使用，暂时无法保存，请稍后再试。")
        return removed

    def touch_attachment(self, row):
        row["last_used_at"] = int(time.time() * 1000)
        return self.save_attachment(row)

    def file_path(self, name):
        path = (self.files / name).resolve()
        if path.parent != self.files:
            raise StorageError("附件路径无效。")
        return path

    def cleanup(self):
        with self.db:
            self._clean_conversations()
            removed = self._clean_attachments()
        for name in removed:
            if name:
                self.file_path(name).unlink(missing_ok=True)
        self.reconcile()
        # 容量按逻辑数据计量；有大量空闲页时回收 SQLite 物理文件。
        if not self.protected and self.db.execute("PRAGMA freelist_count").fetchone()[0] > 1024:
            self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self.db.execute("VACUUM")

    def reconcile(self):
        for value in self.db.execute("SELECT * FROM attachments").fetchall():
            row = self.attachment(value["conversation_id"], value["id"])
            if row is None:
                continue
            if row["local_path"]:
                path = self.file_path(row["local_path"])
                if path.is_file():
                    if path.stat().st_size != row["file_bytes"]:
                        row["file_bytes"] = path.stat().st_size
                        self.save_attachment(row)
                else:
                    row.update(local_path=None, file_bytes=0, status="failed", error="附件原件已丢失，请重新发送。")
                    self.save_attachment(row)
        referenced = {r[0] for r in self.db.execute("SELECT local_path FROM attachments WHERE local_path IS NOT NULL")}
        for path in self.files.glob("*.blob"):
            if len(path.stem) == 32 and all(c in '0123456789abcdef' for c in path.stem) and path.name not in referenced:
                path.unlink(missing_ok=True)
