"""Tests for the health server module."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from gpt_trader.app.health_server import (
    DEFAULT_HEALTH_PORT,
    HealthServer,
    HealthState,
    add_health_check,
    mark_live,
    mark_ready,
)


class TestHealthState:
    """Tests for HealthState dataclass."""

    def test_initial_state(self) -> None:
        state = HealthState()
        assert state.ready is False
        assert state.live is True
        assert state.reason == "initializing"
        assert state.checks == {}

    def test_set_ready(self) -> None:
        state = HealthState()
        state.set_ready(True, "test_ready")
        assert state.ready is True
        assert state.reason == "test_ready"

    def test_set_live(self) -> None:
        state = HealthState()
        state.set_live(False, "test_shutdown")
        assert state.live is False
        assert state.reason == "test_shutdown"

    def test_add_check_pass(self) -> None:
        state = HealthState()
        state.add_check("broker", True, {"latency_ms": 50})
        assert state.checks["broker"]["status"] == "pass"
        assert state.checks["broker"]["details"]["latency_ms"] == 50

    def test_add_check_fail(self) -> None:
        state = HealthState()
        state.add_check("database", False, {"error": "connection timeout"})
        assert state.checks["database"]["status"] == "fail"
        assert state.checks["database"]["details"]["error"] == "connection timeout"


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


class TestHealthCheckFunctions:
    """Tests for health check helper functions."""

    @pytest.fixture
    def health_state(self) -> HealthState:
        """Fresh health state per test."""
        return HealthState()

    def test_mark_ready(self, health_state: HealthState) -> None:
        mark_ready(health_state, True, "all_systems_go")
        assert health_state.ready is True
        assert health_state.reason == "all_systems_go"

    def test_mark_live(self, health_state: HealthState) -> None:
        mark_live(health_state, False, "graceful_shutdown")
        assert health_state.live is False
        assert health_state.reason == "graceful_shutdown"

    def test_add_health_check_success(self, health_state: HealthState) -> None:
        def check_broker() -> tuple[bool, dict[str, Any]]:
            return True, {"latency_ms": 25}

        add_health_check(health_state, "broker", check_broker)
        assert health_state.checks["broker"]["status"] == "pass"
        assert health_state.checks["broker"]["details"]["latency_ms"] == 25

    def test_add_health_check_failure(self, health_state: HealthState) -> None:
        def check_database() -> tuple[bool, dict[str, Any]]:
            return False, {"error": "connection refused"}

        add_health_check(health_state, "database", check_database)
        assert health_state.checks["database"]["status"] == "fail"

    def test_add_health_check_exception(self, health_state: HealthState) -> None:
        def check_flaky() -> tuple[bool, dict[str, Any]]:
            raise RuntimeError("check failed")

        add_health_check(health_state, "flaky", check_flaky)
        assert health_state.checks["flaky"]["status"] == "fail"
        assert "check failed" in health_state.checks["flaky"]["details"]["error"]


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


def test_default_port_is_8080() -> None:
    assert DEFAULT_HEALTH_PORT == 8080
