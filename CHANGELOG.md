# Changelog

All notable changes to GPT-Trader will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2025-08-30

### Added
- Connection keep-alive for Coinbase client (20-40ms latency reduction)
- Deterministic backoff jitter to prevent thundering herd
- Enhanced rate limiting with sliding window and auto-throttling
- CI/CD guards against type regression
- Comprehensive performance tests
- Debugging documentation for proxy/firewall issues

### Changed
- **BREAKING**: Consolidated all broker types to use `brokerages.core.interfaces`
- **BREAKING**: Order fields renamed (order_id → id, quantity → qty, etc.)
- ExecutionEngine now uses core Order fields exclusively
- All tests updated to import from core interfaces
- Improved Coinbase README with performance tuning guide

### Deprecated
- Imports from `live_trade.types` (use `brokerages.core.interfaces` instead)

### Fixed
- ExecutionEngine using incorrect Order field names
- Test failures from type mismatches
- Facade imports and field access patterns

### Performance
- 15-30% throughput increase in high-volume scenarios
- 20-40ms latency reduction per API request
- Better retry distribution under load

## [2.0.0] - Previous Release

### Added
- Bot V2 architecture with vertical slices
- Coinbase Advanced Trade API integration
- Paper trading engine
- ML strategy integration
- Comprehensive backtesting framework

### Changed
- Complete rewrite of broker abstraction layer
- New modular architecture
- Improved error handling

### Fixed
- Various stability improvements
- Memory leak in backtesting engine
- Race conditions in order execution

---

For detailed release notes, see [RELEASE_NOTES_v2.1.0.md](RELEASE_NOTES_v2.1.0.md)