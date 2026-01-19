"""Integration tests for HealthServer (async HTTP endpoints)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from gpt_trader.app.health_server import (
    HealthServer,
    HealthState,
    mark_live,
    mark_ready,
)

pytestmark = pytest.mark.legacy_modernize


class TestHealthServerIntegration:
    """Integration tests for the HealthServer."""

    @pytest.fixture
    def health_state(self) -> HealthState:
        """Fresh health state per test."""
        return HealthState()

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
    async def test_health_endpoint_healthy(self, health_state: HealthState) -> None:
        server = HealthServer(host="127.0.0.1", port=0, health_state=health_state)
        await server.start()
        port = server.port

        try:
            mark_ready(health_state, True)
            response = await _make_request(port, "/health")
            assert response["status"] in ("healthy", "degraded")
            assert response["live"] is True
            assert response["ready"] is True
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_health_endpoint_not_ready(self, health_state: HealthState) -> None:
        server = HealthServer(host="127.0.0.1", port=0, health_state=health_state)
        await server.start()
        port = server.port

        try:
            # Not ready by default
            response = await _make_request(port, "/health")
            assert response["status"] == "starting"
            assert response["ready"] is False
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_live_endpoint(self, health_state: HealthState) -> None:
        server = HealthServer(host="127.0.0.1", port=0, health_state=health_state)
        await server.start()
        port = server.port

        try:
            response = await _make_request(port, "/live")
            assert response["status"] == "pass"
            assert response["live"] is True
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_live_endpoint_not_live(self, health_state: HealthState) -> None:
        server = HealthServer(host="127.0.0.1", port=0, health_state=health_state)
        await server.start()
        port = server.port

        try:
            mark_live(health_state, False, "shutting_down")
            response = await _make_request(port, "/live")
            assert response["status"] == "fail"
            assert response["live"] is False
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_ready_endpoint(self, health_state: HealthState) -> None:
        server = HealthServer(host="127.0.0.1", port=0, health_state=health_state)
        await server.start()
        port = server.port

        try:
            mark_ready(health_state, True, "broker_connected")
            response = await _make_request(port, "/ready")
            assert response["status"] == "pass"
            assert response["ready"] is True
            assert response["reason"] == "broker_connected"
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_ready_endpoint_not_ready(self, health_state: HealthState) -> None:
        server = HealthServer(host="127.0.0.1", port=0, health_state=health_state)
        await server.start()
        port = server.port

        try:
            response = await _make_request(port, "/ready")
            assert response["status"] == "fail"
            assert response["ready"] is False
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_not_found_endpoint(self, health_state: HealthState) -> None:
        server = HealthServer(host="127.0.0.1", port=0, health_state=health_state)
        await server.start()
        port = server.port

        try:
            response = await _make_request(port, "/unknown")
            assert "error" in response
            assert response["error"] == "Not found"
        finally:
            await server.stop()


async def _make_request(port: int, path: str) -> dict[str, Any]:
    """Make HTTP request to health server and return JSON response."""
    import json

    reader, writer = await _open_connection_with_retry("127.0.0.1", port)

    request = f"GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n"
    writer.write(request.encode())
    await writer.drain()

    response = await asyncio.wait_for(reader.read(4096), timeout=5.0)
    writer.close()
    await writer.wait_closed()

    # Parse HTTP response - find the JSON body after headers
    response_text = response.decode("utf-8")
    body_start = response_text.find("\r\n\r\n")
    if body_start != -1:
        body = response_text[body_start + 4 :]
        return json.loads(body)
    return {}


async def _open_connection_with_retry(
    host: str,
    port: int,
    attempts: int = 5,
    delay_seconds: float = 0.05,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    for _ in range(attempts):
        try:
            return await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=1.0,
            )
        except (ConnectionRefusedError, OSError, TimeoutError):
            await asyncio.sleep(delay_seconds)
    return await asyncio.open_connection(host, port)
