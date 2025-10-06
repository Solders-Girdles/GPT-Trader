# Integration Test Plan - Q4 2025

**Created:** 2025-10-05
**Owner:** QA + Trading Engineering
**Status:** Draft → Review → Approved
**Implementation:** Week 3 (Oct 19-25)

---

## Executive Summary

This plan defines integration test coverage for critical trading system paths identified during the operational audit. Focus areas: broker streaming failover, guardrails integration, WebSocket/REST fallback, and outage handling.

**Target Coverage:** >70% of critical trading paths
**Mock Strategy:** Start with responses library mocks, add live Coinbase sandbox tests if available
**CI Integration:** New tests run via `pytest -m integration` (opt-in, not in default run)

---

## Test Coverage Gaps (Current State)

Based on CODEBASE_HEALTH_ASSESSMENT.md findings:

| Gap | Risk Level | Current Coverage | Target |
|-----|------------|------------------|--------|
| Coinbase streaming failover | 🔴 HIGH | None | Full end-to-end test |
| Guardrails integration | 🟡 MEDIUM | Partial unit tests | Integration test with live orchestrator |
| WebSocket/REST fallback | 🔴 HIGH | None | Streaming service degradation test |
| Broker outage handling | 🟡 MEDIUM | Unit tests only | Recovery orchestrator integration |

---

## Test Organization

### Directory Structure

```
tests/integration/
├── brokerages/
│   ├── test_coinbase_streaming_failover.py       # NEW (Week 3)
│   ├── test_coinbase_rest_api_contracts.py        # Existing
│   └── conftest.py                                # Shared fixtures
├── orchestration/
│   ├── test_guardrails_integration.py             # NEW (Week 3)
│   ├── test_broker_outage_handling.py             # NEW (Week 3)
│   └── conftest.py
├── streaming/
│   ├── test_websocket_rest_fallback.py            # NEW (Week 3)
│   └── conftest.py
├── scenarios/                                      # Existing
│   └── (end-to-end workflow tests)
└── conftest.py                                     # Root fixtures

```

### Pytest Markers (Already Defined)

From pytest.ini analysis:
- `@pytest.mark.integration` - Integration-level tests (skipped by default)
- `@pytest.mark.slow` - Long-running tests (>1s)
- `@pytest.mark.real_api` - Tests that hit real Coinbase APIs (opt-in)
- `@pytest.mark.brokerages` - Brokerage adapter tests
- `@pytest.mark.orchestration` - Orchestration layer tests
- `@pytest.mark.scenario` - Scenario-based integration tests

**New Marker (Week 3):**
- `@pytest.mark.soak` - Extended soak tests (hours/days) - **ADD TO pytest.ini**

---

## Test Specifications

### 1. Coinbase Streaming Failover Test

**File:** `tests/integration/brokerages/test_coinbase_streaming_failover.py`

**Purpose:** Validate that the system gracefully handles WebSocket disconnections and reconnects without data loss.

**Scenario:**
1. Establish WebSocket connection to Coinbase (mock or sandbox)
2. Subscribe to market data feed (e.g., BTC-USD ticker)
3. Simulate WebSocket disconnect (close connection, network error)
4. Verify system detects disconnect and initiates reconnect
5. Verify no duplicate messages after reconnect
6. Verify heartbeat mechanism detects stale connections

**Mocking Strategy:**
- **Phase 1 (Week 3):** Use `responses` library to mock WebSocket handshake, messages
- **Phase 2 (Week 4):** Add live Coinbase sandbox test if API access confirmed

**Test Implementation:**
```python
import pytest
from unittest.mock import AsyncMock, patch
from bot_v2.features.brokerages.coinbase.streaming import CoinbaseStreamingClient

@pytest.mark.integration
@pytest.mark.brokerages
@pytest.mark.asyncio
async def test_coinbase_websocket_reconnect_on_disconnect():
    """
    Test: WebSocket reconnects after unexpected disconnect
    Expected: System reconnects within 5 seconds, no data loss
    """
    client = CoinbaseStreamingClient(...)

    # Mock WebSocket connection
    with patch('websockets.connect', new=AsyncMock()) as mock_ws:
        # Simulate disconnect after 3 messages
        mock_ws.return_value.recv.side_effect = [
            '{"type":"ticker","product_id":"BTC-USD","price":"50000"}',
            '{"type":"ticker","product_id":"BTC-USD","price":"50001"}',
            '{"type":"ticker","product_id":"BTC-USD","price":"50002"}',
            ConnectionResetError("Connection lost"),  # Simulate disconnect
        ]

        await client.connect()
        await client.subscribe(["BTC-USD"])

        messages = []
        async for msg in client.messages():
            messages.append(msg)
            if len(messages) >= 3:
                break

        # Verify reconnect attempted
        assert mock_ws.call_count >= 2  # Initial + reconnect
        assert len(messages) == 3
        assert client.is_connected()

@pytest.mark.integration
@pytest.mark.brokerages
@pytest.mark.slow
@pytest.mark.asyncio
async def test_coinbase_heartbeat_detects_stale_connection():
    """
    Test: Heartbeat mechanism detects stale connection and reconnects
    Expected: Connection reset if no heartbeat received within 30s
    """
    # Implementation with time mocking (freezegun)
    pass
```

**Success Criteria:**
- Test passes with mocked WebSocket
- Reconnect logic validated
- Heartbeat timeout detection works
- No data duplication after reconnect

---

### 2. Guardrails Integration Test

**File:** `tests/integration/orchestration/test_guardrails_integration.py`

**Purpose:** Validate that risk guardrails (position limits, exposure limits, order size limits) are enforced by the orchestration layer.

**Scenario:**
1. Initialize PerpsBot with guardrails enabled
2. Attempt to execute order exceeding position limit
3. Verify order is rejected with appropriate error message
4. Verify rejection is logged to telemetry (MetricsCollector)
5. Validate order policy composition (LiquidityService + RiskGate)

**Test Implementation:**
```python
import pytest
from bot_v2.orchestration.perps_bot_builder import PerpsBotBuilder
from bot_v2.features.live_trade.risk import RiskGateValidator

@pytest.mark.integration
@pytest.mark.orchestration
@pytest.mark.asyncio
async def test_position_limit_guardrail_rejects_oversized_order():
    """
    Test: Position limit guardrail rejects order exceeding max position
    Expected: Order rejected, telemetry incremented, error logged
    """
    # Build bot with position limit = 1.0 BTC
    builder = PerpsBotBuilder()
    bot = (
        builder
        .with_profile("test")
        .with_risk_limits(max_position_btc=1.0)
        .build()
    )

    # Mock broker to return current position = 0.8 BTC
    with patch.object(bot.broker, 'get_position', return_value=0.8):
        # Attempt order for 0.5 BTC (would exceed 1.0 limit)
        result = await bot.execute_order(
            symbol="BTC-USD",
            side="buy",
            quantity=0.5,
        )

        assert result.status == "rejected"
        assert "position_limit" in result.rejection_reason

        # Verify telemetry
        assert bot.metrics.get_counter("orders_rejected_guardrails") == 1

@pytest.mark.integration
@pytest.mark.orchestration
def test_order_policy_composition_layers_risk_gates():
    """
    Test: Order policy correctly composes LiquidityService + RiskGateValidator
    Expected: Order passes through liquidity check → risk gate → execution
    """
    # Test policy composition (unit-integration hybrid)
    pass
```

**Success Criteria:**
- Guardrails reject invalid orders
- Telemetry correctly incremented
- Error messages actionable
- Order policy composition validated

---

### 3. WebSocket/REST Fallback Test

**File:** `tests/integration/streaming/test_websocket_rest_fallback.py`

**Purpose:** Validate that MarketDataService falls back to REST polling when WebSocket streaming fails.

**Scenario:**
1. Initialize MarketDataService with WebSocket streaming
2. Simulate WebSocket failure (repeated disconnects, timeout)
3. Verify system switches to REST polling mode
4. Verify market data updates continue (degraded frequency OK)
5. Verify system returns to WebSocket when available

**Test Implementation:**
```python
import pytest
from bot_v2.features.market_data import MarketDataService

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_market_data_fallback_to_rest_on_websocket_failure():
    """
    Test: MarketDataService falls back to REST when WebSocket fails
    Expected: Market data updates continue via REST polling
    """
    service = MarketDataService(...)

    # Mock WebSocket to fail repeatedly
    with patch('websockets.connect', side_effect=ConnectionError("WebSocket unavailable")):
        await service.start()

        # Verify fallback to REST
        assert service.mode == "rest_polling"

        # Verify market data still updates
        initial_price = service.get_mark_price("BTC-USD")
        await asyncio.sleep(5)  # Wait for REST poll
        updated_price = service.get_mark_price("BTC-USD")

        assert updated_price is not None
        # Price may change or stay same, but should be available

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_market_data_returns_to_websocket_when_available():
    """
    Test: MarketDataService returns to WebSocket when connection restored
    Expected: System switches back to streaming mode for lower latency
    """
    # Implementation with WebSocket recovery simulation
    pass
```

**Success Criteria:**
- Fallback to REST polling works
- Market data remains available
- Return to WebSocket when restored
- Mode transition logged

---

### 4. Broker Outage Handling Test

**File:** `tests/integration/orchestration/test_broker_outage_handling.py`

**Purpose:** Validate that the system gracefully handles broker API outages and initiates recovery procedures.

**Scenario:**
1. Simulate Coinbase API returning 503 (service unavailable)
2. Verify system enters "degraded" mode (no new orders)
3. Verify existing positions are monitored (polling)
4. Verify recovery orchestrator attempts to restore state
5. Verify system resumes normal operation when broker recovers

**Test Implementation:**
```python
import pytest
from bot_v2.orchestration.strategy_orchestrator import StrategyOrchestrator
from bot_v2.state.recovery.workflow import RecoveryWorkflow

@pytest.mark.integration
@pytest.mark.orchestration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_broker_outage_enters_degraded_mode():
    """
    Test: System enters degraded mode on broker outage
    Expected: No new orders, positions monitored, recovery initiated
    """
    orchestrator = StrategyOrchestrator(...)

    # Mock broker to return 503
    with patch.object(orchestrator.broker, 'get_account', side_effect=HTTPError(503)):
        # Trigger health check
        health = await orchestrator.check_broker_health()

        assert health.status == "degraded"
        assert orchestrator.mode == "monitor_only"

        # Verify no new orders accepted
        result = await orchestrator.execute_order(...)
        assert result.status == "rejected"
        assert "broker_unavailable" in result.rejection_reason

@pytest.mark.integration
@pytest.mark.orchestration
@pytest.mark.scenario
@pytest.mark.asyncio
async def test_recovery_workflow_restores_state_after_outage():
    """
    Test: RecoveryWorkflow restores state after broker outage resolved
    Expected: Positions reconciled, state restored, normal mode resumed
    """
    # Implementation with recovery orchestrator
    pass
```

**Success Criteria:**
- Degraded mode activated
- No new orders during outage
- Position monitoring continues
- Recovery workflow executes
- Normal mode resumed after recovery

---

## Mocking Strategy

### Phase 1: responses Library (Week 3)

**For REST API mocking:**
```python
import responses

@responses.activate
def test_coinbase_rest_api_mock():
    responses.add(
        responses.GET,
        "https://api.coinbase.com/api/v3/brokerage/accounts",
        json={"accounts": [{"uuid": "123", "available_balance": {"value": "1000"}}]},
        status=200,
    )
    # Test implementation
```

**For WebSocket mocking:**
```python
from unittest.mock import AsyncMock, patch

@patch('websockets.connect', new_callable=AsyncMock)
async def test_websocket_mock(mock_ws):
    mock_ws.return_value.recv.side_effect = [
        '{"type":"ticker","price":"50000"}',
        '{"type":"heartbeat"}',
    ]
    # Test implementation
```

### Phase 2: Coinbase Sandbox (Week 4, if available)

**Sandbox Environment:**
- API Endpoint: `https://api-public.sandbox.pro.coinbase.com`
- Credentials: Sandbox API key/secret (request from Coinbase)
- Marker: `@pytest.mark.real_api` (opt-in)

**Environment Variables:**
```bash
# .env.sandbox
COINBASE_API_ENDPOINT=https://api-public.sandbox.pro.coinbase.com
COINBASE_API_KEY=sandbox_key_here
COINBASE_API_SECRET=sandbox_secret_here
```

**Sandbox Test Example:**
```python
@pytest.mark.real_api
@pytest.mark.slow
@pytest.mark.asyncio
async def test_coinbase_sandbox_streaming():
    """
    Test: Live Coinbase sandbox WebSocket streaming
    Requires: Sandbox API access configured
    """
    # Use real CoinbaseStreamingClient with sandbox endpoint
    # Verify real WebSocket messages
    pass
```

---

## CI Integration

### pytest.ini Updates (Week 3)

**Add soak marker:**
```ini
[pytest]
markers =
    # ... existing markers ...
    soak: Extended soak tests (hours/days)
```

**Default run excludes integration:**
```ini
addopts =
    -m "not integration and not real_api and not uses_mock_broker and not soak"
```

### CI Pipeline Stages

**1. Fast Unit Tests (Every PR)**
```bash
pytest --maxfail=10  # Default, excludes integration
```

**2. Integration Tests (Daily Nightly)**
```bash
pytest -m "integration and not real_api and not slow"  # Mocked integration
```

**3. Slow Integration Tests (Weekly)**
```bash
pytest -m "integration and slow and not real_api"  # Slower mocked tests
```

**4. Sandbox Tests (Weekly, if available)**
```bash
pytest -m "real_api and integration"  # Live Coinbase sandbox
```

**5. Soak Tests (On-Demand)**
```bash
pytest -m "soak"  # Extended runtime tests
```

---

## Test Fixtures (Shared)

### conftest.py Fixtures

**tests/integration/conftest.py:**
```python
import pytest
from bot_v2.orchestration.perps_bot_builder import PerpsBotBuilder

@pytest.fixture
def mock_broker():
    """Mock broker for integration tests"""
    from unittest.mock import AsyncMock
    broker = AsyncMock()
    broker.get_account.return_value = {"balance": "1000"}
    return broker

@pytest.fixture
async def test_perps_bot(mock_broker):
    """PerpsBot instance with mocked broker"""
    builder = PerpsBotBuilder()
    bot = (
        builder
        .with_profile("test")
        .with_broker(mock_broker)
        .build()
    )
    yield bot
    await bot.shutdown()

@pytest.fixture
def mock_coinbase_rest_responses():
    """responses library activation for Coinbase REST API"""
    import responses
    with responses.RequestsMock() as rsps:
        # Add common Coinbase API responses
        rsps.add(
            responses.GET,
            "https://api.coinbase.com/api/v3/brokerage/accounts",
            json={"accounts": []},
            status=200,
        )
        yield rsps
```

---

## Acceptance Criteria

Before Week 2 → Week 3 transition, this plan must:

- [x] Be reviewed by QA lead
- [x] Be reviewed by trading engineering lead
- [ ] Coinbase sandbox access confirmed (or documented as unavailable)
- [x] Mock strategy approved (responses library for Week 3)
- [x] CI integration plan validated
- [x] Pytest markers defined

**Review Gate:** End of Week 2 (Oct 18)

---

## Rollout Plan

### Week 3 (Oct 19-25)

**Monday-Tuesday:** Create test files with mocked implementations
- `test_coinbase_streaming_failover.py`
- `test_guardrails_integration.py`
- `test_websocket_rest_fallback.py`
- `test_broker_outage_handling.py`

**Wednesday-Thursday:** Implement test fixtures, validate CI integration

**Friday:** Code review, merge to main

### Week 4 (Oct 26 - Nov 2)

**Monday:** Add live Coinbase sandbox tests (if access confirmed)

**Tuesday-Wednesday:** Execute soak tests, validate monitoring

**Thursday:** Final review, update docs

---

## Success Metrics

| Metric | Baseline | Target (Week 3) | Actual |
|--------|----------|-----------------|--------|
| Integration test count | ~300 | +4 critical tests | TBD |
| Integration coverage | Unknown | >70% critical paths | TBD |
| CI test execution time | ~2min (unit) | <5min (unit+integration) | TBD |
| Flaky test rate | <1% | <2% | TBD |

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Coinbase sandbox unavailable | 🟡 MEDIUM | 🟡 MEDIUM | Use mocks for Week 3, defer live tests |
| Integration tests flaky | 🟡 MEDIUM | 🔴 HIGH | Retry logic, longer timeouts, mock stability |
| CI pipeline too slow | 🟢 LOW | 🟡 MEDIUM | Parallel test execution (pytest-xdist) |
| Mocks don't match reality | 🟡 MEDIUM | 🔴 HIGH | Validate mocks against Coinbase API docs, add sandbox tests later |

---

## Related Documents

- [CLEANUP_CHECKLIST.md](../ops/CLEANUP_CHECKLIST.md) - Week 3 task tracking
- [CODEBASE_HEALTH_ASSESSMENT.md](../ops/CODEBASE_HEALTH_ASSESSMENT.md) - Coverage gaps identified
- [pytest.ini](/pytest.ini) - Test configuration
- [REFACTORING_2025_RUNBOOK.md](../architecture/REFACTORING_2025_RUNBOOK.md) - Architecture context

---

**Approval Signatures:**
- QA Lead: ___________________________ Date: __________
- Trading Engineering Lead: ___________________________ Date: __________
- Architecture Lead: ___________________________ Date: __________
