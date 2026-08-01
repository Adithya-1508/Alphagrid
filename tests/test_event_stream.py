from __future__ import annotations

import pytest

from alphagrid.streaming.event_stream import EventStreamBroker, StreamEvent


@pytest.mark.anyio
async def test_event_stream_pub_sub():
    broker = EventStreamBroker()
    received_events: list[StreamEvent] = []

    def on_telemetry(evt: StreamEvent):
        received_events.append(evt)

    broker.subscribe("grid.telemetry", on_telemetry)

    _ = await broker.publish("grid.telemetry", {"wind_mw": 8500.0, "frequency_hz": 50.01})

    assert len(received_events) == 1
    assert received_events[0].topic == "grid.telemetry"
    assert received_events[0].payload["wind_mw"] == 8500.0
    assert len(broker.get_history()) == 1


@pytest.mark.anyio
async def test_event_stream_topic_filtering():
    broker = EventStreamBroker()
    await broker.publish("market.prices", {"day_ahead": 65.5})
    await broker.publish("model.drift", {"psi": 0.32})

    h_market = broker.get_history("market.prices")
    h_drift = broker.get_history("model.drift")

    assert len(h_market) == 1
    assert len(h_drift) == 1
    assert h_market[0].payload["day_ahead"] == 65.5
