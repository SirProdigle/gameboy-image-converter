import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from concurrent.futures import Future
import queue


logger = logging.getLogger(__name__)


class QueueClearedError(Exception):
    """Raised when pending tasks are purged from the queue."""


@dataclass
class MetricsSnapshot:
    timestamp: float
    active_tasks: int
    queued_tasks: int
    total_submitted: int
    total_completed: int
    total_failed: int
    recent_avg_duration: float
    last_task_ended_at: Optional[float]
    last_queue_cleared_at: Optional[float]
    queue_cleared_total: int
    last_queue_clear_count: int


class TaskMetrics:
    def __init__(self, duration_window: int = 25) -> None:
        self._lock = threading.Lock()
        self._active_tasks = 0
        self._queued_tasks = 0
        self._total_submitted = 0
        self._total_completed = 0
        self._total_failed = 0
        self._recent_durations: deque[float] = deque(maxlen=duration_window)
        self._last_task_ended_at: Optional[float] = None
        self._last_queue_cleared_at: Optional[float] = None
        self._queue_cleared_total = 0
        self._last_queue_clear_count = 0

    def task_submitted(self) -> MetricsSnapshot:
        with self._lock:
            self._queued_tasks += 1
            self._total_submitted += 1
            return self._snapshot_locked()

    def retract_submission(self, count: int = 1) -> MetricsSnapshot:
        with self._lock:
            self._queued_tasks = max(self._queued_tasks - count, 0)
            return self._snapshot_locked()

    def task_started(self, task_id: str, task_type: str) -> MetricsSnapshot:
        with self._lock:
            if self._queued_tasks:
                self._queued_tasks -= 1
            self._active_tasks += 1
            return self._snapshot_locked()

    def task_finished(self, task_id: str, task_type: str, duration: float, success: bool = True) -> MetricsSnapshot:
        with self._lock:
            self._active_tasks = max(self._active_tasks - 1, 0)
            if success:
                self._total_completed += 1
            else:
                self._total_failed += 1
            self._recent_durations.append(duration)
            self._last_task_ended_at = time.time()
            return self._snapshot_locked()

    def tasks_cleared(self, count: int) -> MetricsSnapshot:
        if count <= 0:
            return self.get_snapshot()
        with self._lock:
            self._queued_tasks = max(self._queued_tasks - count, 0)
            self._last_queue_cleared_at = time.time()
            self._queue_cleared_total += count
            self._last_queue_clear_count = count
            return self._snapshot_locked()

    def get_snapshot(self) -> MetricsSnapshot:
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> MetricsSnapshot:
        avg_duration = sum(self._recent_durations) / len(self._recent_durations) if self._recent_durations else 0.0
        return MetricsSnapshot(
            timestamp=time.time(),
            active_tasks=self._active_tasks,
            queued_tasks=self._queued_tasks,
            total_submitted=self._total_submitted,
            total_completed=self._total_completed,
            total_failed=self._total_failed,
            recent_avg_duration=avg_duration,
            last_task_ended_at=self._last_task_ended_at,
            last_queue_cleared_at=self._last_queue_cleared_at,
            queue_cleared_total=self._queue_cleared_total,
            last_queue_clear_count=self._last_queue_clear_count,
        )


class TaskExecutor:
    def __init__(self, metrics: TaskMetrics, max_workers: int = 2) -> None:
        self._metrics = metrics
        self._queue: "queue.Queue[tuple[Any, tuple[Any, ...], Dict[str, Any], Future]]" = queue.Queue()
        self._stop_event = threading.Event()
        self._workers: list[threading.Thread] = []
        for index in range(max_workers):
            worker = threading.Thread(target=self._worker, name=f"task-worker-{index}", daemon=True)
            worker.start()
            self._workers.append(worker)

    def submit(self, fn, *args, **kwargs) -> Future:
        future: Future = Future()
        self._queue.put((fn, args, kwargs, future))
        return future

    def clear_pending(self, reason: str) -> int:
        cleared_items: list[tuple[Any, tuple[Any, ...], Dict[str, Any], Future]] = []
        with self._queue.mutex:
            while self._queue.queue:
                cleared_items.append(self._queue.queue.popleft())
            cleared_count = len(cleared_items)
            if cleared_count:
                self._queue.unfinished_tasks = max(self._queue.unfinished_tasks - cleared_count, 0)
        if cleared_items:
            message = f"{reason}. {len(cleared_items)} task(s) dropped from the queue."
        else:
            message = reason
        for _, _, _, future in cleared_items:
            if not future.done():
                future.set_exception(QueueClearedError(message))
        if cleared_items:
            self._metrics.tasks_cleared(len(cleared_items))
        return len(cleared_items)

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                fn, args, kwargs, future = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if fn is None:
                self._queue.task_done()
                break
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)
            finally:
                self._queue.task_done()

    def shutdown(self) -> None:
        self._stop_event.set()
        self._queue.put((None, tuple(), {}, Future()))
        for worker in self._workers:
            worker.join(timeout=1.0)


class DiscordWebhookClient:
    def __init__(self, webhook_url: str, timeout: float = 10.0) -> None:
        self._base_url = webhook_url.rstrip("/")
        self._timeout = timeout

    def send_message(
        self,
        content: str,
        embeds: Optional[list] = None,
        allowed_mentions: Optional[dict] = None,
    ) -> Optional[dict]:
        payload: Dict[str, Any] = {"content": content}
        if embeds:
            payload["embeds"] = embeds
        if allowed_mentions is not None:
            payload["allowed_mentions"] = allowed_mentions
        try:
            response = requests.post(
                self._base_url,
                json=payload,
                params={"wait": "true"},
                timeout=self._timeout,
            )
            response.raise_for_status()
        except requests.RequestException:
            logger.debug("Discord webhook send failed; suppressing error", exc_info=True)
            return None
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError:
            logger.debug("Discord webhook response missing JSON payload; ignoring", exc_info=True)
            return {}

    def edit_message(self, message_id: str, content: str, embeds: Optional[list] = None) -> Optional[dict]:
        url = f"{self._base_url}/messages/{message_id}"
        payload: Dict[str, Any] = {"content": content}
        if embeds:
            payload["embeds"] = embeds
        try:
            response = requests.patch(
                url,
                json=payload,
                params={"wait": "true"},
                timeout=self._timeout,
            )
            response.raise_for_status()
        except requests.RequestException:
            logger.debug("Discord webhook edit failed; suppressing error", exc_info=True)
            return None
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError:
            logger.debug("Discord edit response missing JSON payload; ignoring", exc_info=True)
            return {}

    def delete_message(self, message_id: str) -> bool:
        url = f"{self._base_url}/messages/{message_id}"
        try:
            response = requests.delete(url, timeout=self._timeout)
            response.raise_for_status()
        except requests.RequestException:
            logger.debug("Discord webhook delete failed; suppressing error", exc_info=True)
            return False
        return True


class HeartbeatMonitor:
    def __init__(
        self,
        metrics: TaskMetrics,
        webhook_url: str,
        message_store: Path,
        interval_seconds: int = 60,
        queue_alert_threshold: int = 15,
        queue_alert_duration: int = 60,
        queue_alert_cooldown: int = 60,
    ) -> None:
        self._metrics = metrics
        self._webhook = DiscordWebhookClient(webhook_url)
        self._message_store = message_store
        self._interval = interval_seconds
        self._queue_alert_threshold = queue_alert_threshold
        self._queue_alert_duration = queue_alert_duration
        self._queue_alert_cooldown = queue_alert_cooldown
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._message_id = self._load_message_id()
        self._queue_alert_id: Optional[str] = None
        self._queue_alert_sent_at: float = 0.0
        self._queue_alert_lock = threading.Lock()
        self._last_queue_clear_reason: Optional[str] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="heartbeat-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.post_status()
            except Exception:
                logger.debug("Heartbeat status update failed; suppressing error", exc_info=True)
            self._stop_event.wait(self._interval)

    def post_status(self) -> None:
        snapshot = self._metrics.get_snapshot()
        content = self._format_status(snapshot)
        if not self._message_id and self._message_store.exists():
            self._message_id = self._load_message_id()
        if self._message_id:
            self._webhook.edit_message(self._message_id, content)
            return
        response = self._webhook.send_message(content)
        if response is not None and "id" in response:
            self._message_id = response["id"]
            self._save_message_id(self._message_id)

    def check_queue_threshold(self, queue_size: int) -> None:
        if queue_size <= self._queue_alert_threshold:
            return
        with self._queue_alert_lock:
            now = time.time()
            if now - self._queue_alert_sent_at < self._queue_alert_cooldown:
                return
            self._queue_alert_sent_at = now
        content = (
            f"@Developer Queue length is {queue_size} and exceeded the threshold of "
            f"{self._queue_alert_threshold}."
        )
        response = self._webhook.send_message(content)
        if response is not None and "id" in response:
            message_id = response["id"]
            self._queue_alert_id = message_id
            timer = threading.Timer(self._queue_alert_duration, self._delete_queue_alert, args=(message_id,))
            timer.daemon = True
            timer.start()

    def record_queue_cleared(self, count: int, reason: str) -> None:
        if count <= 0:
            return
        self._last_queue_clear_reason = reason

    def _delete_queue_alert(self, message_id: str) -> None:
        deleted = self._webhook.delete_message(message_id)
        if deleted:
            with self._queue_alert_lock:
                if self._queue_alert_id == message_id:
                    self._queue_alert_id = None

    def _format_status(self, snapshot: MetricsSnapshot) -> str:
        now = int(time.time())
        last_completed = (
            f"<t:{int(snapshot.last_task_ended_at)}:R>" if snapshot.last_task_ended_at else "never"
        )
        last_cleared = (
            f"<t:{int(snapshot.last_queue_cleared_at)}:R>" if snapshot.last_queue_cleared_at else "never"
        )
        queue_note = (
            f"Last clear removed {snapshot.last_queue_clear_count} tasks {last_cleared}."
            if snapshot.last_queue_cleared_at
            else "Queue has not been cleared yet."
        )
        reason_note = f" Reason: {self._last_queue_clear_reason}" if self._last_queue_clear_reason else ""
        avg_duration = f"{snapshot.recent_avg_duration:.2f}s" if snapshot.recent_avg_duration else "n/a"
        content = (
            "**Gameboy Converter — Heartbeat**\n"
            f"• Queue pending: `{snapshot.queued_tasks}` | Active: `{snapshot.active_tasks}`\n"
            f"• Submitted: `{snapshot.total_submitted}` | Completed: `{snapshot.total_completed}` | Failed: `{snapshot.total_failed}`\n"
            f"• Recent avg duration: `{avg_duration}`\n"
            f"• Last task finished: {last_completed}\n"
            f"• {queue_note}{reason_note}\n"
            f"• Last updated: <t:{now}:R>"
        )
        return content

    def _load_message_id(self) -> Optional[str]:
        try:
            if not self._message_store.exists():
                return None
            data = json.loads(self._message_store.read_text())
            return data.get("message_id")
        except Exception:
            logger.debug("Failed to load heartbeat message id; suppressing error", exc_info=True)
            return None

    def _save_message_id(self, message_id: str) -> None:
        try:
            self._message_store.parent.mkdir(parents=True, exist_ok=True)
            self._message_store.write_text(json.dumps({"message_id": message_id}))
        except Exception:
            logger.debug("Failed to persist heartbeat message id; suppressing error", exc_info=True)
