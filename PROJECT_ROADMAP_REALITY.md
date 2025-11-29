# GPT-Trader Project Roadmap: The "Reality Check" Edition

**Status**: Active / De-Hyped
**Date**: 2025-11-28

## 🧐 The "Senior Dev" Assessment

The codebase is a classic example of "Enterprise Python" - robust, typed, and slightly over-architected for its current task, but solid.
We have verified that the "ML-driven" claims were marketing fluff (now removed). The system is a reliable Technical Analysis bot with a crash-proof SQLite persistence layer.

**Verdict**: The architecture (`ApplicationContainer`, `ServiceRegistry`, `EventStore`) is actually sound. It allows for easy testing and modularity. We will **keep** it rather than pruning it.

---

## 🛣️ Phase 1: Operation "Truth & Cleanup" (Completed)

**Goal**: Align documentation with reality and remove "fake" complexity.

- [x] **Readme Reality Check**: "ML-driven" claims removed. "About the name" section added.
- [x] **Architecture Pruning**:
    -   *Decision*: Architecture verified as robust and pragmatic. No pruning needed.
    -   `adaptive_portfolio` ghost feature removed from docs.
- [x] **Test Audit**:
    -   `tests/unit/gpt_trader/features/live_trade/test_state_recovery.py` **PASSED** (Crash recovery confirmed).
    -   `tests/unit/gpt_trader/features/brokerages/coinbase/test_websocket_mixin.py` **PASSED** (Streaming confirmed).

## 🛠️ Phase 2: Strategy Development (Completed)

**Goal**: Make the bot actually trade intelligently.

- [x] **Strategy Abstraction Layer** (Completed 2025-11-28):
    -   Created `TradingStrategy` protocol in `interfaces.py`
    -   Created `create_strategy()` factory in `factory.py`
    -   Refactored `TradingEngine` to use factory (strategy injection)
    -   Added `strategy_type` config field: `"baseline"` | `"mean_reversion"`

- [x] **MeanReversionStrategy** (Completed 2025-11-28):
    -   Z-Score based entries: Long when Z < -2, Short when Z > +2
    -   Volatility-targeted position sizing
    -   16 unit tests passing
    -   File: `src/gpt_trader/features/live_trade/strategies/mean_reversion/strategy.py`

- [ ] **Actual ML Integration** (Optional but promised):
    -   If we keep the name "GPT-Trader", we need... GPT.
    -   *Idea*: Use an LLM (via API) to analyze *news sentiment* and feed it into the `EventStore` as a signal.
    -   *Idea*: Train a small `scikit-learn` model on the collected data and inference it in `decide()`.

## 📊 Phase 3: Operational Sanity (Completed)

**Goal**: Run it without babysitting.

- [x] **Simple Monitoring** (Completed 2025-11-28):
    -   StatusReporter writes to `var/data/status.json` every 60s (configurable via `STATUS_FILE`, `STATUS_INTERVAL`)
    -   Health assessment includes: engine running, recent errors, price staleness, heartbeat health
    -   Webhook notifications via `WEBHOOK_URL` env var (Slack/Discord compatible)
    -   NotificationService with rate limiting, deduplication, and multi-backend support
- [x] **Cost Control** (Completed 2025-11-28):
    -   `EventStore.prune(max_rows)` deletes oldest events, keeping configurable limit
    -   `DatabaseEngine.prune_by_count()` handles SQLite pruning
    -   Automatic hourly pruning task in TradingEngine (default: 1M events max)
    -   4 unit tests for pruning functionality

## 📝 Decision Record

1.  **Keep EventStore**: Confirmed useful for debugging and state recovery.
2.  **Architecture**: Keep `ApplicationContainer` and `ServiceRegistry` for testability.
3.  **Name**: Keep "GPT-Trader" but redefine it as "AI-Assisted Development".

---

*Author: The Cynical Senior Dev (via Gemini)*
