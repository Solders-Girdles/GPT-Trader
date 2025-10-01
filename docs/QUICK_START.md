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
```

For environment configuration, API setup, and production deployment, see the [Complete Setup Guide](guides/complete_setup_guide.md).

## When to Read Next

**New Contributors:**
- **[Testing Guide](guides/testing.md)** - Understand the test suite structure, run tests, and write new tests
- **[Complete Setup Guide](guides/complete_setup_guide.md)** - Detailed environment setup, API configuration, and deployment
- **[Production Guide](guides/production.md)** - Production deployment, monitoring, and operational procedures

**Experienced Developers:**
- Jump directly to [ARCHITECTURE.md](ARCHITECTURE.md) for system design and component details
- See [docs/README.md](README.md) for the complete documentation index
