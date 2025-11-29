# GPT-Trader Project Roadmap

**Last Updated**: 2025-11-28
**Status**: Active

---

## 🎯 Strategic Vision

Transition GPT-Trader from a "functional prototype" to an **institutional-grade trading system**.
Focus on **operational excellence**, **observability**, and **durable state management** to support unattended production operation.

---

## 📍 Current Status (November 2025)

### ✅ Strengths
- **Resilience**: Crash-proof architecture with SQLite persistence and state recovery.
- **Architecture**: Decoupled vertical slices (`core/`, `live_trade`, `optimize`) with clean composition.
- **Quality**: 3,400+ tests covering unit, contract, and crash-recovery scenarios.
- **Core Functionality**: Verified Spot trading on Coinbase Advanced Trade.

### ⚠️ Areas for Improvement
- **Observability**: System is "silent" - lacks proactive alerting (Slack/Email) for critical errors.
- **Monitoring**: No unified health dashboard or exposed health check endpoints.
- **Runbooks**: Lack of operational documentation for handling alerts.

---

## 📅 Phase 2: Production Hardening (Immediate Priority)

### Priority 1: Observability & Alerting
**Goal**: Transform the system from "silent" to "proactive".
- [x] **Notification Service**: Implement a multi-channel `NotificationService` (Console, File, Webhook/Slack). (Completed: `gpt_trader.monitoring.notifications`)
- [x] **Critical Alerts**: Hook into `EventStore` to trigger alerts on `error` events or `circuit_breaker` trips. (Completed: Wired into `TradingEngine` and `bot.py`)
- [x] **Heartbeat**: Implement a "dead man's switch" heartbeat log to confirm the loop is alive. (Completed: `HeartbeatService` implemented and wired)

### Priority 2: Operational Monitoring
**Goal**: Provide real-time insight into bot health without tailing logs.
- [x] **Health Endpoints**: Expose a lightweight HTTP server (or file-based status) for external monitoring (e.g., UptimeRobot). (Completed: `StatusReporter` implemented)
- [ ] **Metrics Dashboard**: Refine the Prometheus exporter to include queue depths and API latency.

### Priority 3: Documentation & Runbooks
**Goal**: Ensure any engineer can operate the bot.
- [x] **Alert Runbooks**: "If you see Alert X, do Y." (Completed: `docs/operations/RUNBOOKS.md`)
- [ ] **Deployment Guide**: Standardized systemd/Docker deployment steps.

---

## 🔮 Phase 3: Feature Expansion (Q1 2026)

### 🚀 Perps Activation
- [x] **Paper Trading**: Rigorous testing of `BaselinePerpsStrategy` using the new `EventStore` for state. (Completed: `paper_trade_stress_test.py` validated 30-day simulation)
- [x] **Funding Rate**: Implement funding rate accrual logic in the simulator. (Completed: `FundingProcessor` integrated)

### 🧠 Advanced Optimization
- [x] **Walk-Forward Analysis**: Implement WFA to validate strategy robustness. (Completed: `WalkForwardOptimizer` implemented)

---

## 📝 Change Log

| Date | Change | Author |
|------|--------|--------|
| 2025-11-28 | **Phase 2 Launch**: Shifted focus to Production Hardening (Alerting & Monitoring). Completed Q4 Architecture goals. | Gemini Agent |
| 2025-11-28 | **Roadmap Reset**: Created new roadmap based on code audit. Focus shifts to Durability & Architecture. | Gemini Agent |
