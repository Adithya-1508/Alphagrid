from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class StreamEvent(BaseModel):
    topic: str = Field(description="Stream topic, e.g. grid.telemetry, market.prices")
    payload: dict[str, Any] = Field(description="Event data payload")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class EventStreamBroker:
    """
    In-memory async Pub/Sub event streaming engine for real-time telemetry & alerts.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable[[StreamEvent], Any]]] = {}
        self._event_history: list[StreamEvent] = []

    def subscribe(self, topic: str, callback: Callable[[StreamEvent], Any]) -> None:
        """Subscribes a listener callback function to a target topic."""
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(callback)

    async def publish(self, topic: str, payload: dict[str, Any]) -> StreamEvent:
        """Publishes an event to a target topic and invokes registered callbacks."""
        event = StreamEvent(topic=topic, payload=payload)
        self._event_history.append(event)

        if topic in self._subscribers:
            for cb in self._subscribers[topic]:
                if asyncio.iscoroutinefunction(cb):
                    await cb(event)
                else:
                    cb(event)

        return event

    def get_history(self, topic: str | None = None) -> list[StreamEvent]:
        """Returns event history, optionally filtered by topic."""
        if topic is None:
            return list(self._event_history)
        return [e for e in self._event_history if e.topic == topic]


_BROKER_INSTANCE: EventStreamBroker | None = None


def get_event_broker() -> EventStreamBroker:
    """Returns singleton EventStreamBroker instance."""
    global _BROKER_INSTANCE
    if _BROKER_INSTANCE is None:
        _BROKER_INSTANCE = EventStreamBroker()
    return _BROKER_INSTANCE
