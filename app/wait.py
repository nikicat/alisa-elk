import asyncio
from typing import Any


class PendingTaskRegistry:
    """In-process registry of background LLM tasks, keyed by pending_id.

    Single-process v1 only. For horizontal scaling, swap to a queue.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[Any]] = {}

    def register(self, pending_id: str, task: asyncio.Task[Any]) -> None:
        self._tasks[pending_id] = task

    def get(self, pending_id: str) -> asyncio.Task[Any] | None:
        return self._tasks.get(pending_id)

    def discard(self, pending_id: str) -> None:
        self._tasks.pop(pending_id, None)

    def cancel(self, pending_id: str) -> bool:
        task = self._tasks.pop(pending_id, None)
        if task is None or task.done():
            return False
        task.cancel()
        return True

    def cancel_all(self) -> None:
        for task in list(self._tasks.values()):
            if not task.done():
                task.cancel()
        self._tasks.clear()


_registry = PendingTaskRegistry()


def get_registry() -> PendingTaskRegistry:
    return _registry


def reset_registry_for_tests() -> None:
    global _registry
    _registry.cancel_all()
    _registry = PendingTaskRegistry()


async def wait_for_or_keepalive(task: asyncio.Task[Any], timeout: float) -> Any | None:
    """Wait up to `timeout` seconds. On timeout, return None — leave task running.

    `shield` is required: bare `wait_for` cancels the underlying task on timeout.
    """
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
    except TimeoutError:
        return None
