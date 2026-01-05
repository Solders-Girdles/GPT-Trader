# TUI Style Guide & Component Rules

This document defines the visual standards and component patterns for the GPT-Trader TUI dashboard. Following these guidelines ensures consistency across all screens and widgets.

## Table of Contents

- [Design Principles](#design-principles)
- [Color System](#color-system)
- [Typography](#typography)
- [Spacing](#spacing)
- [Component Rules](#component-rules)
  - [Tile Headers](#tile-headers)
  - [Inline Banners](#inline-banners)
  - [Data Tables](#data-tables)
  - [Empty States](#empty-states)
  - [Loading States](#loading-states)
  - [Value Changes](#value-changes)
  - [Badges & Status Indicators](#badges--status-indicators)
  - [Unified Status Classes](#unified-status-classes)
  - [Unified Threshold System](#unified-threshold-system)
  - [Staleness & Freshness Indicators](#staleness--freshness-indicators)
  - [Strategy Decision Display](#strategy-decision-display)
  - [Buttons](#buttons)
  - [Focus & Keyboard Navigation](#focus--keyboard-navigation)
- [Layout Rules](#layout-rules)
- [Performance Budgets](#performance-budgets)
- [Accessibility Requirements](#accessibility-requirements)

---

## Design Principles

1. **Data Density Over Decoration**: Maximize information per screen space. Trading dashboards must convey critical data at a glance.

2. **Keyboard-First Navigation**: All actions accessible via keyboard. Mouse is secondary.

3. **Visual Hierarchy Through Contrast**: Use color and weight, not borders, to establish hierarchy.

4. **Consistent Interaction Patterns**: Same visual feedback across all interactive elements.

5. **Graceful Degradation**: Show meaningful states during loading, errors, and empty data.

---

## Color System

### Background Layers

| Token | Hex | Usage |
|-------|-----|-------|
| `$bg-primary` | `#101010` | Main background, deepest layer |
| `$bg-secondary` | `#1a1a1a` | Panels, containers |
| `$bg-elevated` | `#252525` | Headers, cards, elevated content |
| `$bg-highlight` | `#303030` | Hover states, active inputs |

### Text Colors

| Token | Hex | Contrast | Usage |
|-------|-----|----------|-------|
| `$text-primary` | `#FFFFFF` | 15.5:1 | Primary data, values, headers |
| `$text-secondary` | `#B0B0B0` | 7.5:1 | Labels, secondary info |
| `$text-muted` | `#606060` | 4.1:1 | Hints, disabled text |

### Semantic Colors

| Token | Hex | Usage |
|-------|-----|-------|
| `$success` | `#00FF41` | Positive P&L, successful operations |
| `$warning` | `#FFD700` | Warnings, cautions, pending states |
| `$error` | `#FF0033` | Negative P&L, errors, critical states |
| `$info` | `#007AFF` | Information, neutral highlights |
| `$accent` | `#007AFF` | Focus rings, primary actions |

### Muted Variants

Use `*-muted` variants (10% opacity) for backgrounds behind semantic text:
- `.success-muted` for positive value backgrounds
- `.error-muted` for negative value backgrounds
- `.warning-muted` for warning backgrounds

---

## Typography

### Text Styles

```tcss
/* Headers */
.widget-header {
    text-style: bold;
    color: $text-primary;
}

/* Labels */
.label {
    color: $text-secondary;
}

/* Values */
.value {
    color: $text-primary;
    text-style: bold;
}

/* Muted/Hint text */
.muted {
    color: $text-muted;
}
```

### Do's and Don'ts

✅ **DO**: Use `text-style: bold` for emphasis
❌ **DON'T**: Use ALL CAPS for emphasis (reserve for headers only)

✅ **DO**: Use `$text-secondary` for labels
❌ **DON'T**: Use `$text-muted` for interactive element labels

---

## Spacing

### Spacing Scale

| Token | Value | Usage |
|-------|-------|-------|
| `$spacing-xs` | 1 | Tight spacing, inline elements |
| `$spacing-sm` | 2 | Standard element padding |
| `$spacing-md` | 3 | Section padding |
| `$spacing-lg` | 4 | Major section gaps |

### Semantic Tokens

| Token | Value | Usage |
|-------|-------|-------|
| `$gutter-tight` | 1 | Compact layouts |
| `$gutter-standard` | 2 | Standard layouts |
| `$tile-padding` | 1 2 | Tile internal padding |
| `$widget-content-padding` | 1 2 | Widget content area |

---

## Component Rules

### Tile Headers

Tiles use refined headers with accent indicators instead of heavy backgrounds.

```tcss
/* Standard tile header */
.widget-header {
    color: $text-primary;
    text-style: bold;
    background: transparent;
    padding: 0 1;
    border-left: wide $accent;
    height: 2;
    margin-bottom: 1;
}

/* Subtle variant for secondary sections */
.header-subtle {
    color: $text-secondary;
    background: transparent;
    border-bottom: solid $border-hairline;
    height: 1;
}
```

**Rules:**
- Headers use `border-left: wide $accent` as visual indicator
- No heavy background fills - keep background transparent
- Text always `$text-primary` with `bold` style
- Height of 2 for standard headers, 1 for subtle

### Inline Banners

Toast-like notifications within tiles for contextual feedback.

```tcss
.inline-banner {
    height: auto;
    width: 100%;
    padding: 0 1;
    background: $bg-elevated;
    border-left: wide $accent-muted;
    color: $text-secondary;
}

/* Severity variants */
.inline-banner.info    { border-left: wide $info;    color: $info; }
.inline-banner.success { border-left: wide $success; color: $success; }
.inline-banner.warning { border-left: wide $warning; color: $warning; }
.inline-banner.error   { border-left: wide $error;   color: $error; }
```

**Rules:**
- Use `border-left` color to indicate severity, not background
- Keep backgrounds subtle (use `*-muted` variants if needed)
- Maximum 2 lines of text

### Data Tables

Tables use minimal styling with clear row separation.

```tcss
DataTable {
    padding: 0;
    border: none;
    background: $bg-primary;
}

DataTable > .datatable--header {
    background: $bg-elevated;
    border-bottom: solid $border-light;
}

DataTable > .datatable--cursor {
    background: $accent-subtle;
    border-left: wide $accent-muted;
}

DataTable:focus > .datatable--cursor {
    background: $overlay-focus;
    border-left: wide $accent;
}
```

**Rules:**
- No vertical cell separators (`border-right: none`)
- Header row gets `$bg-elevated` background
- Cursor row uses `border-left` indicator, not full background
- Sort indicators: `▲` ascending, `▼` descending

### Empty States

Use `TileEmptyState` for all empty/error states. This provides consistent structure with icon, title, subtitle, and action hints.

```python
from gpt_trader.tui.widgets.tile_states import TileEmptyState

# Basic empty state
yield TileEmptyState(
    title="No Positions",
    subtitle="Positions appear when trades are opened",
    icon="◇",
    actions=["[S] Start Bot", "[R] Refresh"],
    id="positions-empty",
)

# Error state with retry affordance
yield TileEmptyState(
    title="Connection Failed",
    subtitle="Check API credentials and network",
    icon="⚠",
    actions=["[R] Reconnect", "[C] Config"],
    id="connection-error",
)
```

**Standard Icons:**
| Icon | Usage |
|------|-------|
| `◇` | No data (positions, trades) |
| `◌` | Waiting/pending (orders, watchlist) |
| `○` | Empty history (trades, alerts) |
| `⚠` | Error/failure states |

**Rules:**
- Always use `TileEmptyState` instead of raw `.empty-state` Labels
- Include relevant action hints for keyboard shortcuts that exist
- Use `⚠` icon for all error/failure states
- Update dynamically with `empty_state.update_state(subtitle="New message")`
- For compact spaces (AlertInbox), hide icon via CSS:
  ```tcss
  #empty-inbox .empty-icon { display: none; height: 0; }
  ```

**Error State Pattern:**
```python
# On render/connection failure
tile_empty_state(
    "Connection Failed",
    "Cannot retrieve data",
    icon="⚠",
    actions=["[R] Reconnect", "[L] Logs"],
)
```

### Loading States

Use skeleton placeholders during data loading.

```tcss
.skeleton {
    background: $bg-elevated;
    color: transparent;
}

.skeleton-text {
    height: 1;
    width: 60%;
}

.skeleton-value {
    height: 1;
    width: 40%;
}
```

**Rules:**
- Match skeleton dimensions to expected content
- Use `color: transparent` to hide any placeholder text
- Never show "Loading..." text alone - use skeletons

### Value Changes

Highlight value changes with brief visual feedback.

```tcss
.value-changed-up {
    color: $success;
    text-style: bold;
    background: $success-muted;
}

.value-changed-down {
    color: $error;
    text-style: bold;
    background: $error-muted;
}

.value-flash {
    background: $accent-subtle;
}
```

**Rules:**
- Apply class for 1-2 seconds, then remove
- Use `$success` for increases, `$error` for decreases
- Background tint helps values stand out briefly

### Badges & Status Indicators

Compact labels for status and categorization.

```tcss
.badge {
    padding: 0 1;
    height: 1;
    background: $bg-elevated;
    color: $text-primary;
    text-style: bold;
}

.badge-success { background: $success-muted; color: $success; }
.badge-warning { background: $warning-muted; color: $warning; }
.badge-error   { background: $error-muted;   color: $error; }
.badge-info    { background: $info-muted;    color: $info; }
```

**Rules:**
- Always include `text-style: bold`
- Use semantic variants for status
- Keep text short (1-2 words)

### Unified Status Classes

The TUI uses a centralized threshold system for consistent status indicators across all widgets. Status is always one of three levels: **OK**, **WARNING**, or **CRITICAL**.

```tcss
/* Unified status classes - use these instead of widget-specific variants */
.status-ok       { color: $success; }
.status-warning  { color: $warning; text-style: italic; }
.status-critical { color: $error;   text-style: bold underline; }
```

**Accessibility:** Status classes include text-style differentiation (not just color):
- OK: Normal weight (green)
- WARNING: Italic (yellow)
- CRITICAL: Bold + underline (red)

**Usage in Python:**
```python
from gpt_trader.tui.thresholds import (
    StatusLevel,
    get_status_class,
    get_status_color,
    get_status_icon,
)

# Get appropriate class for CSS
status = StatusLevel.WARNING
css_class = get_status_class(status)  # "status-warning"
color = get_status_color(status)       # "yellow"
icon = get_status_icon(status)         # "⚠"
```

### Unified Threshold System

All metrics use centralized thresholds defined in `gpt_trader.tui.thresholds`. This ensures consistent status boundaries across all widgets.

#### Performance Thresholds

```python
from gpt_trader.tui.thresholds import (
    get_latency_status,
    get_cpu_status,
    get_memory_status,
    get_fps_status,
)

# Latency: OK < 50ms, WARNING < 150ms, CRITICAL >= 150ms
status = get_latency_status(latency_ms=75.0)  # StatusLevel.WARNING

# CPU: OK < 50%, WARNING < 80%, CRITICAL >= 80%
status = get_cpu_status(cpu_percent=65.0)  # StatusLevel.WARNING
```

| Metric | OK | WARNING | CRITICAL |
|--------|-----|---------|----------|
| Latency | < 50ms | 50-150ms | ≥ 150ms |
| CPU | < 50% | 50-80% | ≥ 80% |
| Memory | < 60% | 60-80% | ≥ 80% |
| FPS | ≥ 0.5 | 0.2-0.5 | < 0.2 |

#### Risk Thresholds

```python
from gpt_trader.tui.thresholds import (
    get_loss_ratio_status,
    get_risk_score_status,
    get_risk_status_label,
)

# Loss ratio uses abs() to handle negative losses correctly
status = get_loss_ratio_status(
    current_loss_pct=-0.05,  # -5% loss
    limit_pct=0.10,          # 10% limit
)  # StatusLevel.WARNING (50% of limit used)

# Risk score maps to LOW/MEDIUM/HIGH
label = get_risk_status_label(status)  # "MEDIUM"
```

| Metric | OK (LOW) | WARNING (MEDIUM) | CRITICAL (HIGH) |
|--------|----------|------------------|-----------------|
| Loss Ratio | < 50% of limit | 50-75% of limit | ≥ 75% of limit |
| Risk Score | < 2 | 2-5 | ≥ 5 |

**Important:** `get_loss_ratio_status()` uses `abs()` internally to correctly handle negative loss values.

#### Confidence Thresholds

Confidence is inverted - higher is better:

```python
from gpt_trader.tui.thresholds import (
    get_confidence_status,
    get_confidence_label,
    format_confidence_with_badge,
)

# HIGH confidence = OK (green), LOW confidence = CRITICAL (red)
status = get_confidence_status(0.75)  # StatusLevel.OK
label = get_confidence_label(status)   # "HIGH"

# Format with badge for display
text, css_class = format_confidence_with_badge(0.75)
# text = "0.75 HIGH", css_class = "status-ok"
```

| Confidence | Status | Label | Color |
|------------|--------|-------|-------|
| ≥ 0.70 | OK | HIGH | Green |
| 0.40-0.70 | WARNING | MED | Yellow |
| < 0.40 | CRITICAL | LOW | Red |

### Staleness & Freshness Indicators

Use `TileBanner` and staleness helpers to show data freshness state.

#### TileBanner Component

```python
from gpt_trader.tui.widgets.tile_states import TileBanner

# In compose()
yield TileBanner(id="my-banner", classes="tile-banner hidden")

# Update based on state
banner = self.query_one(TileBanner)
banner.update_banner("Data is 30s stale", severity="warning")
# Or hide when fresh:
banner.hide()
```

#### Staleness Helpers

```python
from gpt_trader.tui.staleness_helpers import (
    get_staleness_banner,
    get_freshness_display,
    get_empty_state_config,
)

# Get banner text and severity based on state
result = get_staleness_banner(state)
if result:
    text, severity = result  # ("Data 30s stale", "warning")
    banner.update_banner(text, severity=severity)
else:
    banner.hide()  # Data is fresh

# Get freshness indicator for header
freshness = get_freshness_display(state)
if freshness:
    text, css_class = freshness  # ("● 2s", "fresh")
```

**Staleness Thresholds:**
| Age | State | Display |
|-----|-------|---------|
| < 10s | Fresh | Green dot `● 2s` |
| 10-30s | Stale | Yellow banner "Data Xs stale" |
| > 30s | Critical | Red banner "Data Xs stale - may be outdated" |

### Strategy Decision Display

Strategy decisions use confidence badges and regime indicators for clarity.

#### Confidence with Badge

Display confidence values with semantic badges:

```python
# In strategy widget table
from gpt_trader.tui.thresholds import (
    get_confidence_status,
    get_confidence_label,
    get_status_color,
)

conf_status = get_confidence_status(decision.confidence)
conf_label = get_confidence_label(conf_status)
conf_color = get_status_color(conf_status)

# Outputs: "[green]0.85 HIGH[/green]"
confidence_cell = f"[{conf_color}]{decision.confidence:.2f} {conf_label}[/{conf_color}]"
```

#### Regime Badge (Ensemble Strategies)

Display market regime for ensemble strategies:

```python
# Check for regime in indicators
regime = decision.indicators.get("regime")
if regime:
    regime_icon = "📈" if regime == "trending" else "📊" if regime == "ranging" else "⚪"
    regime_text = f"{regime_icon} {regime.upper()}"

    # Optionally include ADX
    adx = decision.indicators.get("adx")
    if adx is not None:
        regime_text += f" (ADX: {adx:.1f})"
```

**Regime Icons:**
| Regime | Icon | Meaning |
|--------|------|---------|
| Trending | 📈 | Trend-following signals weighted higher |
| Ranging | 📊 | Mean-reversion signals weighted higher |
| Neutral | ⚪ | Balanced signal weights |

#### Indicator Grouping

Group strategy indicators by semantic category:

```python
INDICATOR_CATEGORIES = {
    "trend": ["trend", "crossover_signal", "short_ma", "long_ma", "adx"],
    "momentum": ["rsi", "rsi_signal", "momentum"],
    "regime": ["regime"],
    "order_flow": ["aggressor_ratio", "trade_count", "volume", "vwap"],
    "microstructure": ["spread_bps", "spread", "quality"],
}
```

### Buttons

Consistent button styling across all variants.

```tcss
Button {
    border: solid $border-subtle;
    background: $bg-elevated;
    color: $text-primary;
}

Button:hover {
    background: $bg-highlight;
    border: solid $accent-muted;
}

Button:focus {
    border: solid $accent;
    background: $accent-subtle;
}

Button.-primary {
    background: $accent;
    color: $bg-primary;
}

Button.-success {
    background: $success-dim;
    color: $text-primary;
}

Button.-error {
    background: $error-dim;
    color: $text-primary;
}
```

**Rules:**
- Default buttons use `$bg-elevated` background
- Hover lightens background to `$bg-highlight`
- Focus adds `$accent` border
- Primary/Success/Error variants use semantic colors

### Focus & Keyboard Navigation

Clear focus indicators for accessibility.

```tcss
*:focus {
    border: solid $accent;
}

Button:focus,
Input:focus,
Select:focus {
    border: thick $accent;
}

.tile-focused {
    border: solid $accent;
}
```

**Rules:**
- All focusable elements must show visible focus ring
- Use `thick` border for primary interactive elements
- Tile focus uses class-based application

---

## Layout Rules

### Bento Grid

The main dashboard uses a 4-column grid:

```tcss
.bento-grid {
    layout: grid;
    grid-size: 4;
    grid-gutter: 1;
}

/* Tile spans */
.tile-hero    { column-span: 2; row-span: 2; }
.tile-account { column-span: 2; row-span: 1; }
.tile-market  { column-span: 2; row-span: 2; }
.tile-system  { column-span: 2; row-span: 1; }
.tile-logs    { column-span: 4; row-span: 1; }
```

### Tile Structure

Every tile follows this structure:

```
┌─────────────────────────────────┐
│ HEADER ← border-left accent     │
├─────────────────────────────────┤
│                                 │
│         CONTENT AREA            │
│                                 │
├─────────────────────────────────┤
│ [Actions hint when focused]     │
└─────────────────────────────────┘
```

### Responsive Breakpoints

| Terminal Width | Layout |
|----------------|--------|
| < 80 cols | Compact mode: reduced padding, condensed widgets |
| 80-120 cols | Standard mode |
| > 120 cols | Expanded mode: additional columns visible |

---

## Performance Budgets

The TUI enforces these performance targets:

### Frame Timing

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| **FPS** | ≥ 0.5 | 0.2-0.5 | < 0.2 |
| **Avg Frame Time** | < 50ms | 50-100ms | > 100ms |
| **P95 Frame Time** | < 100ms | 100-200ms | > 200ms |
| **Max Frame Time** | < 200ms | 200-500ms | > 500ms |

### Memory

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| **Memory %** | < 50% | 50-80% | > 80% |
| **RSS Growth** | < 10MB/hour | 10-50MB/hour | > 50MB/hour |

### Update Cadence

| Data Type | Target Interval | Max Staleness |
|-----------|-----------------|---------------|
| Price data | 1s | 5s |
| Account balance | 5s | 30s |
| System metrics | 1s | 10s |
| Logs | Real-time | 1s |

### Re-render Costs

| Operation | Budget |
|-----------|--------|
| Full state update | < 50ms |
| Single widget update | < 10ms |
| Table row update | < 5ms |

The performance overlay (accessible via command palette) surfaces these metrics in real-time.

---

## Accessibility Requirements

### Contrast Ratios (WCAG AA)

- **Primary text**: ≥ 7:1 (achieved: 15.5:1)
- **Secondary text**: ≥ 4.5:1 (achieved: 7.5:1)
- **Muted text**: ≥ 3:1 (achieved: 4.1:1)

### Keyboard Navigation

- All actions accessible via keyboard
- Clear focus indicators on all interactive elements
- Logical tab order following visual layout
- Arrow key navigation for tile grid

### High Contrast Mode

Accessible via theme toggle (Dark → Light → High Contrast):

```tcss
/* High contrast overrides */
$text-muted: #999999;      /* Boosted from #606060 */
$border-subtle: #555555;   /* Boosted from #333333 */
$accent: #00AAFF;          /* Brighter blue */
```

### State Indicators

Use classes instead of ARIA attributes (Textual limitation):

```tcss
.state-selected { /* selected item */ }
.state-disabled { /* disabled element */ }
.state-invalid  { /* validation error */ }
```

---

## Quick Reference Card

```
HEADERS
├── .widget-header     → Bold, accent border-left
└── .header-subtle     → Secondary, hairline border-bottom

BANNERS
├── .inline-banner       → Neutral
├── .inline-banner.info    → Blue
├── .inline-banner.success → Green
├── .inline-banner.warning → Gold
└── .inline-banner.error   → Red

BADGES
├── .badge           → Neutral
├── .badge-success   → Green
├── .badge-warning   → Gold
├── .badge-error     → Red
└── .badge-info      → Blue

UNIFIED STATUS (use these for metrics)
├── .status-ok       → Green (OK, HIGH confidence, LOW risk)
├── .status-warning  → Yellow + italic (WARNING, MED confidence, MEDIUM risk)
└── .status-critical → Red + bold underline (CRITICAL, LOW confidence, HIGH risk)

THRESHOLDS (from gpt_trader.tui.thresholds)
├── Performance
│   ├── get_latency_status()  → < 50ms OK, < 150ms WARN
│   ├── get_cpu_status()      → < 50% OK, < 80% WARN
│   └── get_memory_status()   → < 60% OK, < 80% WARN
├── Risk
│   ├── get_loss_ratio_status() → < 50% OK, < 75% WARN (uses abs()!)
│   └── get_risk_score_status() → < 2 LOW, < 5 MEDIUM
└── Confidence (inverted - higher = better)
    └── get_confidence_status() → ≥ 0.7 HIGH, ≥ 0.4 MED

STALENESS
├── TileBanner       → Contextual banner in tiles
├── get_staleness_banner()   → Returns (text, severity) or None
└── get_freshness_display()  → Returns (text, css_class) for headers

STRATEGY DISPLAY
├── Confidence badge → "[green]0.85 HIGH[/green]"
├── Regime icons    → 📈 TRENDING | 📊 RANGING | ⚪ NEUTRAL
└── Indicator groups → trend | momentum | regime | order_flow | microstructure

STATES
├── TileEmptyState   → Empty/error states (replaces .empty-state)
│   ├── icon ◇      → No data
│   ├── icon ◌      → Waiting/pending
│   ├── icon ○      → Empty history
│   └── icon ⚠      → Error/failure
├── TileLoadingState → Loading with spinner
├── .skeleton        → Loading placeholder
├── .value-changed-up   → Green flash
├── .value-changed-down → Red flash
└── .state-disabled     → 40% opacity

FOCUS
├── *:focus              → Accent border
├── .tile-focused        → Tile focus ring
└── .keyboard-focused    → Enhanced focus
```
