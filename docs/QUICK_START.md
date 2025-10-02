# Quick Start

For the complete setup guide with detailed instructions, see:

**[Complete Setup Guide](guides/complete_setup_guide.md)**

## TL;DR for Experienced Users

```bash
# Install dependencies
poetry install

# Run dev profile (uses mock broker, no real trades)
poetry run perps-bot --profile dev --dev-fast

# Run tests
poetry run pytest -q

# Verify Phase 0 coverage gates stay green
pytest --cov=src/bot_v2 --cov-report=term --cov-report=json -q
python scripts/check_package_coverage.py phase0
```

For environment configuration, API setup, and production deployment, see the [Complete Setup Guide](guides/complete_setup_guide.md).

## Contributor Test Suites

- **Monitoring evaluations**: `pytest -m monitoring` (frozen-time flows, alert escalations)
- **Brokerage adapters**: `pytest -m brokerages` (REST + websocket contracts)
- **High impact CI slice**: `pytest -m high_impact` (fast regression guardrail)
- **Coverage overview**: `python scripts/check_package_coverage.py --show-all`

**Raised fail-under thresholds (Phase 0):**
- `security` ≥ 88%
- `config` ≥ 86%
- `cli` ≥ 93%

The coverage check script uses the JSON artifact produced by pytest-cov; re-run the two commands shown in the TL;DR after writing tests to refresh the numbers.

## When to Read Next

**New Contributors:**
- **[Testing Guide](guides/testing.md)** - Understand the test suite structure, run tests, and write new tests
- **[Complete Setup Guide](guides/complete_setup_guide.md)** - Detailed environment setup, API configuration, and deployment
- **[Production Guide](guides/production.md)** - Production deployment, monitoring, and operational procedures

**Experienced Developers:**
- Jump directly to [ARCHITECTURE.md](ARCHITECTURE.md) for system design and component details
- See [docs/README.md](README.md) for the complete documentation index
