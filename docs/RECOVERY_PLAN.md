# GPT-Trader: State of the Union & Recovery Plan

**Date**: November 19, 2025
**Status**: Assessment Complete

## 1. Executive Summary

The project is in a **much better state than feared**. Contrary to the concern of "lack of cohesion," the codebase is structured, modern, and well-tested. The "Operational but not trading" status is accurate and stems from a specific, solvable configuration/logic issue, not a systemic failure.

The documentation (`PROJECT_ROADMAP.md`) is up-to-date and accurately reflects reality. The immediate blocker is that the `BaselinePerpsStrategy` is too conservative or lacks sufficient data to trigger signals, resulting in a permanent "HOLD" state.

## 2. True State of the Codebase

| Aspect | Status | Assessment |
| :--- | :--- | :--- |
| **Architecture** | 🟢 Healthy | Clean "Vertical Slice" architecture (`src/bot_v2/features`). Dependencies are well-managed with Poetry. |
| **Testing** | 🟢 Strong | **3,698 tests** collected. Refactoring is in progress but the suite is comprehensive. |
| **Documentation** | 🟢 Excellent | `PROJECT_ROADMAP.md` and `README.md` are current (Nov 18). |
| **Functionality** | 🟡 Partial | Bot starts and runs cycles. **Critical Gap**: Strategy logic returns "HOLD" 100% of the time. |
| **Legacy Code** | 🟢 Minimal | No obvious "dead" V1 code cluttering the main `src/bot_v2` directory. |

### Key Findings
*   **Strategy Logic**: The `BaselinePerpsStrategy` relies on a simple Moving Average (MA) crossover (5/20 periods). If the bot doesn't fetch enough historical data on startup, or if the market is ranging, it will never trade. This is the "Hold" bug.
*   **Test Suite**: The test suite is massive (~3.7k tests). The "Test Refactoring" effort is justified to keep this manageable.

## 3. Recovery Plan

We do not need a "restart". We need a **kickstart**.

### Phase 0: Cleanup (Immediate)
**Goal**: Reduce cognitive load and remove deprecated content.
1.  **Purge Legacy Artifacts**: Remove unused directories (e.g., `htmlcov_baseline`, `backtesting` if empty/unused) and old config files.
2.  **Archive Documentation**: Move outdated planning docs to `docs/archive`.
3.  **Codebase Sweep**: Scan for and remove any remaining V1 code or unused "dead" files.

### Phase 1: Ignition (Immediate - Next 24 Hours)
**Goal**: Force the bot to make a trade in Dev Mode.
1.  **Debug Strategy**:
    *   Verify `recent_marks` contains enough data points (>20) for the MA calculation.
    *   Temporarily create a `ForceBuyStrategy` or lower the MA periods (e.g., 2/5) to trigger signals easier for testing.
2.  **Verify Execution**:
    *   Confirm the "Buy" signal actually results in a placed order in the mock broker.

### Phase 2: Stabilization (Next 3 Days)
**Goal**: Clear technical debt to allow smooth feature development.
1.  **Complete Test Refactoring**: Finish the "Phase 3" split of `test_execution_coordinator.py`. This is already in progress and should be finished to unblock CI.
2.  **Config Validation**: Ensure the strategy config is correctly loaded.

### Phase 3: Evolution (Next 2 Weeks)
**Goal**: Make the bot smart.
1.  **Better Signals**: The MA crossover is too basic. Implement a slightly more robust signal (e.g., RSI + Bollinger Bands) to get meaningful trading activity.
2.  **Live Testing**: Move to `canary` profile with real (small) funds once signals are reliable.

## 4. Immediate Next Steps

I propose we proceed with **Phase 1: Ignition**.

1.  I will create a reproduction script to simulate the strategy decision process and see exactly why it returns HOLD.
2.  I will fix the data feed or config to ensure signals are generated.
3.  We will verify a trade happens.

**Shall I proceed with debugging the Strategy Logic?**
