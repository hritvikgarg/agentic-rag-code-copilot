"""A tiny synchronous event bus and an in-memory audit log."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable, Mapping
from typing import Any

logger = logging.getLogger(__name__)

Handler = Callable[[str, Mapping[str, Any]], None]
WILDCARD = "*"


class EventBus:
    """Publish/subscribe within one process. Handlers run in subscription order, synchronously."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = {}
        self.handler_errors: list[tuple[str, str]] = []

    def subscribe(self, topic: str, handler: Handler) -> None:
        """Subscribe to one topic, or to every topic with ``"*"``."""
        self._handlers.setdefault(topic, []).append(handler)

    def publish(self, topic: str, payload: Mapping[str, Any] | None = None) -> None:
        """Deliver an event. A failing handler is logged and recorded, never re-raised."""
        data = dict(payload or {})
        for handler in [*self._handlers.get(topic, []), *self._handlers.get(WILDCARD, [])]:
            try:
                handler(topic, data)
            except Exception as exc:  # a subscriber must not break the publisher
                logger.exception("handler for %s failed", topic)
                self.handler_errors.append((topic, type(exc).__name__))


class AuditLog:
    """Keeps the most recent events (default 1000); older entries are discarded."""

    def __init__(self, bus: EventBus, *, capacity: int = 1000) -> None:
        self.entries: deque[tuple[str, dict[str, Any]]] = deque(maxlen=capacity)
        bus.subscribe(WILDCARD, self._record)

    def _record(self, topic: str, payload: Mapping[str, Any]) -> None:
        self.entries.append((topic, dict(payload)))
