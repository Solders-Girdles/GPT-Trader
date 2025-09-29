# Changelog

## Unreleased

### Naming Alignment: `qty` → `quantity`
- Core brokerage interfaces now expose `quantity` exclusively; legacy `qty` aliases have been removed across serializers and dataclasses.
- Coinbase adapter, live execution, deterministic broker stub, and strategy paths emit `quantity` only in logs and telemetry to keep downstream metrics consistent.
- CLI order tooling only accepts `--order-quantity`; the legacy `--order-qty` alias has been removed.
- Historical rollout details live in repository history (Wave 1 status notes).

### Module Cleanup & Broker Modernization
- **BREAKING**: Removed deprecated modules (`execution_v3`, `week2_filters`, `perps_baseline_v2`, legacy Coinbase helper modules).
- **BREAKING**: Migrated from `MockBroker` to `DeterministicBroker` for development/testing workflows.
- **FIX**: Resolved CLI import failures and enforced module removal consistency via `test_removed_aliases.py`.
- **DOCS**: Updated architecture, development guides, and paper trading documentation to reflect current patterns.
- **TESTS**: Consolidated test structure, removed stale files, and eliminated legacy testing patterns.
- All active test suites pass with the modernized broker implementation and clean module structure.
