# Strategy System Evolution: Signal Ensemble Architecture

## Vision
To transition from a monolithic strategy class to a modular **Signal Ensemble** system. This architecture treats trading signals as independent, pluggable components ("Alphas") that are aggregated by a sophisticated "Combiner" ("Brain"). This allows for infinite scalability in trading intelligence—we can add 100 signals without changing the execution engine.

## Core Concepts

1.  **Signal Generator**: A stateless or stateful component that analyzes specific market data and outputs a normalized `Signal`.
2.  **Signal Output**: A standardized object containing:
    -   `signal`: Direction and strength (-1.0 to +1.0).
    -   `confidence`: Probability of success (0.0 to 1.0).
    -   `metadata`: Context for debugging (e.g., "RSI=85").
3.  **Combiner**: The logic that aggregates multiple signals into a single `NetSignal`. It is context-aware (e.g., "Ignore trend signals in ranging markets").
4.  **Ensemble Strategy**: The execution shell that runs the pipeline: `Data -> Signals -> Combiner -> Execution`.

---

## Detailed Implementation Steps

### Phase 1: Foundation (Interfaces & Types)
**Objective**: Define the rigid contracts that allow components to interact seamlessly.

#### 1.1 Define Data Structures
**File**: `src/gpt_trader/features/live_trade/signals/types.py`
-   `SignalType` (Enum): `TREND`, `MEAN_REVERSION`, `VOLATILITY`, `SENTIMENT`.
-   `SignalOutput` (Dataclass):
    ```python
    @dataclass
    class SignalOutput:
        name: str
        type: SignalType
        strength: float  # -1.0 (Strong Sell) to +1.0 (Strong Buy)
        confidence: float # 0.0 to 1.0
        metadata: dict[str, Any]
    ```

#### 1.2 Define Protocols
**File**: `src/gpt_trader/features/live_trade/signals/protocol.py`
-   `SignalGenerator` (Protocol):
    ```python
    def generate(self, context: StrategyContext) -> SignalOutput: ...
    ```
-   `SignalCombiner` (Protocol):
    ```python
    def combine(self, signals: list[SignalOutput], context: StrategyContext) -> SignalOutput: ...
    ```

### Phase 2: Core Signal Library
**Objective**: Port existing logic into granular, reusable signal components.

#### 2.1 Trend Signal (MA Crossover)
**File**: `src/gpt_trader/features/live_trade/signals/trend.py`
-   **Config**: `fast_period`, `slow_period`.
-   **Logic**:
    -   Calculate Fast/Slow MAs.
    -   **Bullish**: Fast > Slow. Strength scales with divergence.
    -   **Bearish**: Fast < Slow.
    -   **Output**: `SignalOutput(type=TREND, strength=...)`

#### 2.2 Mean Reversion Signal (Z-Score)
**File**: `src/gpt_trader/features/live_trade/signals/mean_reversion.py`
-   **Config**: `window`, `z_threshold`.
-   **Logic**:
    -   Calculate Z-Score of price.
    -   **Buy**: Z < -Threshold.
    -   **Sell**: Z > +Threshold.
    -   **Output**: `SignalOutput(type=MEAN_REVERSION, strength=...)`

#### 2.3 RSI Signal (Momentum)
**File**: `src/gpt_trader/features/live_trade/signals/momentum.py`
-   **Config**: `period`, `overbought`, `oversold`.
-   **Logic**:
    -   **Buy**: RSI < Oversold.
    -   **Sell**: RSI > Overbought.
    -   **Output**: `SignalOutput(type=MEAN_REVERSION, strength=...)`

### Phase 3: The Brain (Regime-Aware Combiner)
**Objective**: Implement the logic that intelligently weighs signals based on market context.

#### 3.1 ADX Indicator
**File**: `src/gpt_trader/features/live_trade/indicators.py`
-   Implement `average_directional_index(high, low, close, period)`.

#### 3.2 Regime Detection
**File**: `src/gpt_trader/features/live_trade/combiners/regime.py`
-   **Logic**:
    -   Calculate ADX(14).
    -   **Trending**: ADX > 25.
    -   **Ranging**: ADX < 20.
    -   **Transition**: 20 <= ADX <= 25.

#### 3.3 Weighted Combination
**File**: `src/gpt_trader/features/live_trade/combiners/regime.py`
-   **Config**:
    -   `trend_weights`: { "trending": 1.0, "ranging": 0.0 }
    -   `mean_rev_weights`: { "trending": 0.0, "ranging": 1.0 }
-   **Algorithm**:
    1.  Determine Regime.
    2.  For each signal:
        -   Get base weight from config.
        -   Apply regime multiplier (e.g., if Ranging, Trend Signal * 0.0).
    3.  Sum weighted signals.
    4.  Normalize result (-1.0 to 1.0).

### Phase 4: Execution (Ensemble Strategy)
**Objective**: Create the strategy class that ties it all together.

#### 4.1 Ensemble Strategy Class
**File**: `src/gpt_trader/features/live_trade/strategies/ensemble.py`
-   **Init**: Loads Signals and Combiner from config.
-   **Decide Loop**:
    1.  `context = build_context(...)`
    2.  `signals = [s.generate(context) for s in self.signals]`
    3.  `net_signal = self.combiner.combine(signals, context)`
    4.  **Execution Logic**:
        -   If `net_signal.strength > threshold`: BUY
        -   If `net_signal.strength < -threshold`: SELL
        -   If `net_signal.strength` crosses zero: CLOSE
    5.  **Risk Overlay**: Check Stop Loss / Take Profit (overrides signal).

### Phase 5: Configuration & Integration
**Objective**: Expose this power to the user via YAML.

#### 5.1 Configuration Schema
**File**: `src/gpt_trader/orchestration/configuration/bot_config/bot_config.py`
-   Update `StrategyConfig` to support:
    ```yaml
    strategy:
      type: "ensemble"
      signals:
        - name: "trend_follower"
          type: "trend_ma"
          params: { fast: 10, slow: 50 }
        - name: "mean_reverter"
          type: "z_score"
          params: { window: 20 }
      combiner:
        type: "regime_aware"
        params: { adx_period: 14 }
    ```

---

## Verification Plan

### Unit Tests
1.  **Signals**: Test each signal in isolation with synthetic data (e.g., perfect sine wave for mean reversion, linear trend for trend signal).
2.  **Combiner**: Test that weights shift correctly when ADX crosses thresholds.
3.  **Ensemble**: Test end-to-end flow.

### Integration Test
-   **Paper Trading**: Run with `ensemble` profile.
-   **Logs**: Verify logs show:
    -   "Regime: TRENDING (ADX=35)"
    -   "Signal[Trend]: +0.8 (Weight: 1.0)"
    -   "Signal[MeanRev]: -0.5 (Weight: 0.0)"
    -   "Net Signal: +0.8 -> BUY"
