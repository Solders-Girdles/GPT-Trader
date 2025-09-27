## test: Improve test coverage and stability

### Summary
Comprehensive test suite improvements focusing on stability, coverage, and quality. Adds critical orchestration coverage, de‑flakes streaming tests, reduces mocking, and introduces a CI coverage gate.

### Changes

#### CI/CD
- Added coverage-check job with warning-only gate at 70% (`--cov-fail-under=70`).
- Uploads HTML coverage report as an artifact for review.

#### New Test Coverage
- `broker_factory.py`: Environment-driven broker selection (sandbox exchange, prod advanced JWT, unsupported broker).
- `mock_broker.py`: Quotes, market order → position updates, cancel SUBMITTED limit order.
- `adapters.py`: Interface compliance for data, analyze, backtest, optimize, ML, regime, sizing, monitor, trading, plus `AdapterFactory` registry wiring.
- E2E smoke test: Runs CLI with `--profile dev --dry-run --dev-fast` using `MockBroker` (no WS threads), verifies `health.json`.

#### Un-skipped Tests (9 total)
- WebSocket auth gap detection.
- WebSocket auth provider injection (via `ws_auth_provider`).
- Sandbox mode detection (broker_factory).
- HTTP request layer path composition (updated to `/api/v3/brokerage/market/products`).
- Improvements connection validation (uses public `client.get_accounts`).
- MarkCache TTL tests (2): set/get and expiry without datetime monkeypatching.
- Event persistence tests (2): positions and funding with proper product catalog mocking.

#### Quality Improvements
- Reduced mocking in preflight validation: 4 tests switched to real `Product` dataclass.
- Removed problematic sleep statements; streaming tests now use short, bounded polling.
- Fixed fragile datetime handling in cache tests by directly aging timestamps.

### Metrics
- Tests: 270 → 383 (+42%).
- Skips: 101 → 78 (−23%).
- Runtime: ~5s (maintained fast execution).
- Coverage: Baseline added in CI; ready to bump gate to 75% in the next PR.

### Validation
- Lint/format: `poetry run ruff check .` and `poetry run black --check .`.
- Types: `poetry run mypy src/bot_v2`.
- Tests: `poetry run pytest -q` and unit-only coverage `poetry run pytest --cov=bot_v2 --cov-report=term tests/unit/bot_v2`.

### Risks & Mitigations
- CI gate is warning-only to avoid blocking while we continue cleanup.
- Un-skipped tests were stabilized (3x runs where applicable) and use deterministic patterns.

### Follow-ups (planned next PRs)
- Drop skips below 75 by re-enabling `TestFundingCalculator` (align with accrual behavior) and selective endpoint tests.
- Bump CI coverage gate to 75% once stable.
- Consider splitting `test_product_catalog.py` into mapping/cache/funding modules for clarity.

