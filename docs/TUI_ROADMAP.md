# TUI Roadmap

This document tracks the development of the Textual-based Terminal User Interface (TUI) for GPT-Trader.

## Phase 1: Foundation (Completed & Enhanced)

The goal of Phase 1 was to establish a robust, testable, and well-structured foundation for the TUI.

- [x] **Robust State Management**
    - [x] Implement `TuiState` using `textual.reactive` for efficient UI updates.
    - [x] Create typed data structures in `types.py`.
    - [x] Implement `update_from_bot_status` to safely handle dictionary updates.
- [x] **Testing Infrastructure**
    - [x] Create shared fixtures (`mock_bot`, `mock_app`).
    - [x] Implement helper functions (`wait_for_widget`, `assert_widget_visible`).
- [x] **Layout and Design**
    - [x] Implement Nord theme colors in `styles.tcss`.
    - [x] Create responsive grid layout in `screens.py`.
- [ ] **Data Layer Upgrade (Retroactive)**
    - [ ] **Rich Data Support**: Update `StatusReporter` to support nested JSON structures (e.g., signal breakdowns) instead of stringifying everything.
    - [ ] **State Deserialization**: Update `TuiState` to correctly parse complex nested data from the bot.

## Phase 2: Foundation Hardening (In Progress)

**Goal:** Ensure the application is stable, crash-resistant, and type-safe before adding complexity.

- [ ] **Error Boundaries**
    - [ ] Wrap all widget update methods in try/except blocks.
    - [ ] Implement a global error handler.
- [ ] **Event Loop Stability**
    - [ ] Verify that heavy TUI rendering does not block the trading bot's asyncio loop.
    - [ ] Implement "circuit breakers" for UI updates.
- [ ] **Type Safety Audit**
    - [ ] Run strict `mypy` checks on `gpt_trader.tui`.
    - [ ] **Signal Typing**: Define explicit `TypedDict` or `dataclass` structures for Ensemble Signal data to ensure safe parsing in the TUI.

## Phase 3: Data Fidelity & Basic Observability

**Goal:** Ensure the dashboard is a reliable monitor ("Trust what you see").

- [ ] **Data Fidelity Verification**
    - [ ] Audit `StatusReporter` integration to ensure all widgets receive accurate, real-time data.
    - [ ] Verify update frequency and latency.
- [ ] **Enhanced Observability**
    - [ ] **Log Filtering**: Add ability to filter logs by level (INFO, WARN, ERROR) in `LogWidget`.
    - [ ] **Connection Status**: Robustly display API connection health and latency.
    - [ ] **System Health**: Accurate CPU/Memory usage tracking.

## Phase 4: Critical Operational Control

**Goal:** Safe control of the bot's lifecycle ("Safe Control").

- [ ] **Panic Button**
    - [ ] Implement "Flatten & Stop" functionality for emergency shutdowns.
    - [ ] Add confirmation modal.
- [ ] **Bot Lifecycle**
    - [ ] Robust Start/Stop control with status feedback.
    - [ ] Graceful shutdown handling.

## Phase 5: Operational Control & Inspection

**Goal:** Deep visibility and runtime control of the bot's internal state.

- [ ] **Ensemble Inspection**
    - [ ] **Signal Inspector Widget**: Create a widget to view detailed breakdown of ensemble decisions (signal weights, individual votes).
    - [ ] **Strategy State Visualization**: Visualize internal state of strategies (e.g., indicator values) using the rich data from Phase 1.
- [ ] **Advanced Configuration**
    - [ ] **Runtime Config Editor**: Expand `ConfigModal` to allow editing strategy parameters and risk limits.
    - [ ] **Dynamic Strategy Tuning**: Enable/Disable specific signals or adjust weights at runtime.

## Phase 6: Active Management (Future)

**Goal:** Active trading intervention ("Intervention").

- [ ] **Manual Order Entry**
    - [ ] Create a form widget for submitting manual orders.
- [ ] **Position Management**
    - [ ] Add ability to close specific positions individually.

## Phase 7: Advanced Visualization

**Goal:** Deep market insight ("Analysis").

- [ ] **Market Depth**
    - [ ] `OrderBookWidget`: Visualize depth of market.
- [ ] **Charting**
    - [ ] `PriceChartWidget`: ASCII/Braille based price history chart.

## Phase 8: Production Readiness

- [ ] **User Configuration**
    - [ ] Keybinding configuration.
    - [ ] Theme customization.
- [ ] **Distribution**
    - [ ] Standalone executable packaging.
