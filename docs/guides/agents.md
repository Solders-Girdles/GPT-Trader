# AI Agent Development Guide

This guide points AI agents to the canonical resources for GPT-Trader V2. The bot operates **spot-first** with dormant perps logic that activates only when Coinbase grants INTX access.

## Primary Resources

**Start here:**
- **[`docs/agents/Agents.md`](../agents/Agents.md)** – Comprehensive playbook for all AI assistants (commands, workflows, testing, directory structure)
- **[`docs/agents/CLAUDE.md`](../agents/CLAUDE.md)** – Claude-specific patterns and tips
- **[`docs/agents/Gemini.md`](../agents/Gemini.md)** – Gemini-specific patterns and tips

**Supporting documentation:**
- [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md) – System architecture and design
- [`docs/guides/complete_setup_guide.md`](complete_setup_guide.md) – Environment and credential setup
- [`docs/guides/testing.md`](testing.md) – Test metrics and commands
- [`README.md`](../../README.md) – Quick start and daily operations

## Quick Reference

### Essential Commands
```bash
poetry install                                   # Install dependencies
poetry run perps-bot --profile dev --dev-fast    # Dev cycle (mock broker)
poetry run pytest -q                             # Full test suite
```

### Key Facts
- **Default mode:** Spot trading (BTC-USD, ETH-USD, etc.)
- **Perps mode:** Gated behind `COINBASE_ENABLE_DERIVATIVES=1` + INTX access
- **Entry point:** `poetry run perps-bot` (see CLI_REFERENCE.md for all options)
- **Experimental features:** Archived on 2025-09-29; avoid unless explicitly requested

For detailed workflows, directory navigation, operational tooling, and testing patterns, refer to [`docs/agents/Agents.md`](../agents/Agents.md).
