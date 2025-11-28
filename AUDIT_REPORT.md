# GPT-Trader Audit Report: Current State Analysis

**Date:** November 2025
**Auditor:** Jules (AI Software Engineer)

## 1. Executive Summary

The GPT-Trader codebase represents a "Sophisticated Prototype". It features high-quality testing patterns (Property-based testing), a clean Vertical Slice architecture, and strict typing standards. However, it suffers from a **critical operational disconnection**: the sophisticated `RiskManager` is fully implemented but **completely bypassed** by the active `TradingEngine`.

**Status:** ⚠️ **Operational but Unsafe**
- **Spot Trading:** Can technically execute trades.
- **Risk Controls:** **INACTIVE** (Bypassed).
- **Perps:** Dormant/Standby as intended.

---

## 2. Critical Findings (The "Must Fix")

### 🚨 Risk Management Bypass
The `TradingEngine` (`src/gpt_trader/features/live_trade/engines/strategy.py`) executes trades directly against the broker:
```python
# current implementation
await asyncio.to_thread(self.context.broker.place_order, ...)
```
It **never checks** with `LiveRiskManager`. Furthermore, `LiveRiskManager.check_order` is currently a stub that returns `True` unconditionally. The complex validation logic (`pre_trade_validate`, daily loss limits, leverage caps) sits unused.

### 🔌 Protocol Mismatches
The `ServiceRegistry` implementation causes Mypy errors because it defines a Protocol (`ServiceRegistryProtocol`) for internal wiring but returns the concrete `ServiceRegistry` class in factory methods. This is "Enterprise Creep" that adds friction without value.

### 🧪 Test Suite Fragility
While the suite reports 3100+ tests, the `tests/contract` module is broken due to import errors (referencing non-existent classes). The heavy reliance on `hypothesis` is a strength, but the environment setup (`poetry install`) required manual intervention for dev dependencies, suggesting a drift in `pyproject.toml` or lockfiles.

---

## 3. Dimensional Analysis

### 📐 Code Hygiene & Standards
- **Mypy:** 13 errors found (down from "94" in history, but not "0").
  - *Cause:* Missing `types-requests`, `types-PyYAML` stubs, and Protocol mismatches.
- **Linting:** Excellent. Ruff reports zero violations.
- **Naming:** 95% compliance. 31 accepted exceptions (`# naming: allow`) is a healthy sign of pragmatic enforcement.

### 🏗️ Architecture
- **Vertical Slices:** The directory structure is sound. Features are well-colocated.
- **Orchestration:** The `TradingBot` -> `TradingEngine` delegation is clean.
- **Container:** `ApplicationContainer` contains some hardcoded "Mock Mode" logic. While technically "coupling", it is far simpler than a dynamic plugin system and fits the "Avoid Enterprise Creep" mantra.

### 📉 Perps vs. Spot
- **Isolation:** The "Perps" logic is effectively dormant behind `COINBASE_ENABLE_DERIVATIVES`.
- **Leakage:** The active engine is named `TradingEngine` but hard-wires `BaselinePerpsStrategy`. If this strategy is intended for Spot trading, the naming is confusing. If it's *only* for Perps, then the Spot bot is effectively running a Perps strategy on Spot symbols, which may lead to logic errors (e.g., expecting Funding Rates).

---

## 4. Recommendations & Next Phase Plan

### Phase 1: Safety First (Immediate)
1.  **Wire the Risk Manager:** Modify `TradingEngine` to explicitly call `RiskManager.pre_trade_validate` before placing any order.
2.  **Un-stub Risk Checks:** Remove the `return True` stub in `check_order` and connect it to the actual validation logic.
3.  **Fix Mypy:** Install missing stubs and relax the `ServiceRegistryProtocol` strictness to clear the 13 errors.

### Phase 2: Operational Hardening
1.  **Spot Strategy Separation:** Rename `BaselinePerpsStrategy` or create a distinct `SpotStrategy` to clarify intent and avoid "Perps logic on Spot assets".
2.  **Fix Contract Tests:** Repair the imports in `tests/contract/test_coinbase_api_contract.py` so the contract tests actually run in CI.

### Phase 3: Documentation Cleanup
1.  **Archive Roadmap:** Move `PROJECT_ROADMAP.md` to `docs/archive/`.
2.  **Restore AGENTS.md:** Create a valid `AGENTS.md` (or link `docs/agents/CLAUDE.md` to it) so future agents have clear instructions.

---

**Auditor's Note:** The project is in better shape than most "AI-generated" codebases due to the strong testing foundation. The primary gap is simply *connecting* the built components (Risk -> Engine). Once wired, this system will be quite robust.
