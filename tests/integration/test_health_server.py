"""Integration tests for HealthServer (async HTTP endpoints)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
import pytest_asyncio

from gpt_trader.app.health_server import (
    HealthServer,
    HealthState,
    mark_live,
    mark_ready,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def health_state() -> HealthState:
    """Fresh health state per test."""
    return HealthState()


@pytest_asyncio.fixture
async def server(health_state: HealthState) -> HealthServer:
    server = HealthServer(host="127.0.0.1", port=0, health_state=health_state)
    await server.start()
    yield server
    await server.stop()


class TestHealthServerIntegration:
    """Integration tests for the HealthServer."""

    @pytest.mark.asyncio
    async def test_server_starts_and_stops(self, health_state: HealthState) -> None:
        server = HealthServer(host="127.0.0.1", port=0, health_state=health_state)

        await server.start()
        assert server._running is True
        assert server._server is not None

        await server.stop()
        assert server._running is False
        assert server._server is None

    @pytest.mark.asyncio
    async def test_health_endpoint_healthy(
        self, server: HealthServer, health_state: HealthState
    ) -> None:
        mark_ready(health_state, True)
        response = await _make_request("127.0.0.1", server.port, "/health")
        assert response["status"] in ("healthy", "degraded")
        assert response["live"] is True
        assert response["ready"] is True

    @pytest.mark.asyncio
    async def test_health_endpoint_not_ready(self, server: HealthServer) -> None:
        response = await _make_request("127.0.0.1", server.port, "/health")
        assert response["status"] == "starting"
        assert response["ready"] is False

    @pytest.mark.asyncio
    async def test_live_endpoint(self, server: HealthServer) -> None:
        response = await _make_request("127.0.0.1", server.port, "/live")
        assert response["status"] == "pass"
        assert response["live"] is True

    @pytest.mark.asyncio
    async def test_live_endpoint_not_live(
        self, server: HealthServer, health_state: HealthState
    ) -> None:
        mark_live(health_state, False, "shutting_down")
        response = await _make_request("127.0.0.1", server.port, "/live")
        assert response["status"] == "fail"
        assert response["live"] is False

    @pytest.mark.asyncio
    async def test_ready_endpoint(self, server: HealthServer, health_state: HealthState) -> None:
        mark_ready(health_state, True, "broker_connected")
        response = await _make_request("127.0.0.1", server.port, "/ready")
        assert response["status"] == "pass"
        assert response["ready"] is True
        assert response["reason"] == "broker_connected"

    @pytest.mark.asyncio
    async def test_ready_endpoint_not_ready(self, server: HealthServer) -> None:
        response = await _make_request("127.0.0.1", server.port, "/ready")
        assert response["status"] == "fail"
        assert response["ready"] is False

    @pytest.mark.asyncio
    async def test_not_found_endpoint(self, server: HealthServer) -> None:
        response = await _make_request("127.0.0.1", server.port, "/unknown")
        assert response["error"] == "Not found"


async def _make_request(host: str, port: int, path: str) -> dict[str, Any]:
    """Make HTTP request to health server and return JSON response."""
    reader, writer = await asyncio.open_connection(host, port)

    request = f"GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n"
    writer.write(request.encode())
    await writer.drain()

    response = await asyncio.wait_for(reader.read(), timeout=1.0)
    writer.close()
    await writer.wait_closed()

    # Parse HTTP response - find the JSON body after headers
    response_text = response.decode("utf-8")
    body_start = response_text.find("\r\n\r\n")
    if body_start != -1:
        body = response_text[body_start + 4 :]
        return json.loads(body)
    return {}
