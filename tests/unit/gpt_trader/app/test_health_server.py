"""Tests for the health server module."""

from __future__ import annotations

import asyncio
import socket
from typing import Any

import pytest

from gpt_trader.app.health_server import (
    DEFAULT_HEALTH_PORT,
    HealthServer,
    HealthState,
    add_health_check,
    get_health_state,
    mark_live,
    mark_ready,
)


def find_free_port() -> int:
    """Find a free port for testing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return s.getsockname()[1]


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
    def reset_health_state(self) -> None:
        """Reset global health state before each test."""
        state = get_health_state()
        state.ready = False
        state.live = True
        state.reason = "initializing"
        state.checks = {}

    @pytest.mark.asyncio
    async def test_server_starts_and_stops(self, reset_health_state: None) -> None:
        port = find_free_port()
        server = HealthServer(host="127.0.0.1", port=port)

        await server.start()
        assert server._running is True
        assert server._server is not None

        await server.stop()
        assert server._running is False
        assert server._server is None

    @pytest.mark.asyncio
    async def test_health_endpoint_healthy(self, reset_health_state: None) -> None:
        port = find_free_port()
        server = HealthServer(host="127.0.0.1", port=port)
        await server.start()

        try:
            mark_ready(True)
            response = await _make_request(port, "/health")
            assert response["status"] in ("healthy", "degraded")
            assert response["live"] is True
            assert response["ready"] is True
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_health_endpoint_not_ready(self, reset_health_state: None) -> None:
        port = find_free_port()
        server = HealthServer(host="127.0.0.1", port=port)
        await server.start()

        try:
            # Not ready by default
            response = await _make_request(port, "/health")
            assert response["status"] == "starting"
            assert response["ready"] is False
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_live_endpoint(self, reset_health_state: None) -> None:
        port = find_free_port()
        server = HealthServer(host="127.0.0.1", port=port)
        await server.start()

        try:
            response = await _make_request(port, "/live")
            assert response["status"] == "pass"
            assert response["live"] is True
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_live_endpoint_not_live(self, reset_health_state: None) -> None:
        port = find_free_port()
        server = HealthServer(host="127.0.0.1", port=port)
        await server.start()

        try:
            mark_live(False, "shutting_down")
            response = await _make_request(port, "/live")
            assert response["status"] == "fail"
            assert response["live"] is False
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_ready_endpoint(self, reset_health_state: None) -> None:
        port = find_free_port()
        server = HealthServer(host="127.0.0.1", port=port)
        await server.start()

        try:
            mark_ready(True, "broker_connected")
            response = await _make_request(port, "/ready")
            assert response["status"] == "pass"
            assert response["ready"] is True
            assert response["reason"] == "broker_connected"
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_ready_endpoint_not_ready(self, reset_health_state: None) -> None:
        port = find_free_port()
        server = HealthServer(host="127.0.0.1", port=port)
        await server.start()

        try:
            response = await _make_request(port, "/ready")
            assert response["status"] == "fail"
            assert response["ready"] is False
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_not_found_endpoint(self, reset_health_state: None) -> None:
        port = find_free_port()
        server = HealthServer(host="127.0.0.1", port=port)
        await server.start()

        try:
            response = await _make_request(port, "/unknown")
            assert "error" in response
            assert response["error"] == "Not found"
        finally:
            await server.stop()


class TestHealthCheckFunctions:
    """Tests for health check helper functions."""

    @pytest.fixture
    def reset_health_state(self) -> None:
        """Reset global health state before each test."""
        state = get_health_state()
        state.ready = False
        state.live = True
        state.reason = "initializing"
        state.checks = {}

    def test_mark_ready(self, reset_health_state: None) -> None:
        mark_ready(True, "all_systems_go")
        state = get_health_state()
        assert state.ready is True
        assert state.reason == "all_systems_go"

    def test_mark_live(self, reset_health_state: None) -> None:
        mark_live(False, "graceful_shutdown")
        state = get_health_state()
        assert state.live is False
        assert state.reason == "graceful_shutdown"

    def test_add_health_check_success(self, reset_health_state: None) -> None:
        def check_broker() -> tuple[bool, dict[str, Any]]:
            return True, {"latency_ms": 25}

        add_health_check("broker", check_broker)
        state = get_health_state()
        assert state.checks["broker"]["status"] == "pass"
        assert state.checks["broker"]["details"]["latency_ms"] == 25

    def test_add_health_check_failure(self, reset_health_state: None) -> None:
        def check_database() -> tuple[bool, dict[str, Any]]:
            return False, {"error": "connection refused"}

        add_health_check("database", check_database)
        state = get_health_state()
        assert state.checks["database"]["status"] == "fail"

    def test_add_health_check_exception(self, reset_health_state: None) -> None:
        def check_flaky() -> tuple[bool, dict[str, Any]]:
            raise RuntimeError("check failed")

        add_health_check("flaky", check_flaky)
        state = get_health_state()
        assert state.checks["flaky"]["status"] == "fail"
        assert "check failed" in state.checks["flaky"]["details"]["error"]


async def _make_request(port: int, path: str) -> dict[str, Any]:
    """Make HTTP request to health server and return JSON response."""
    import json

    reader, writer = await asyncio.open_connection("127.0.0.1", port)

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


def test_default_port_is_8080() -> None:
    assert DEFAULT_HEALTH_PORT == 8080
