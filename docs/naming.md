# GPT-Trader Naming Standards

This document defines the naming conventions for code, configuration, CLI surfaces, and documentation across the GPT-Trader project. Consistent naming improves clarity, reduces rename churn, and makes the codebase more maintainable for both humans and AI agents.

## 1. Scope & Goals

- **Increase clarity**: Use descriptive, unambiguous names across all surfaces
- **Reduce churn**: Establish clear casing, terminology, and abbreviation rules
- **Enable automation**: Support automated naming checks via `scripts/agents/naming_inventory.py`
- **Preserve compatibility**: Document sanctioned exceptions for external API integration

## 2. Naming Categories & Rules

### 2.1 Modules & Packages

**Format:** `snake_case` for Python modules and packages.

**Rules:**
- Use descriptive nouns or noun phrases (`risk_limits`, not `rl`)
- Avoid double abbreviations or cryptic shorthand
- Prefer specificity over brevity (`execution_engine` over `exec`)

**Banned abbreviations:**
- ❌ `cfg` → ✅ `config`
- ❌ `svc` → ✅ `service`
- ❌ `mgr` → ✅ `manager`
- ❌ `util` → ✅ Use domain-specific names (e.g., `asyncio_helpers`, `time_helpers`)

**Examples:**
```python
# Good
from bot_v2.features.live_trade import risk_manager
from bot_v2.orchestration import runtime_engine

# Bad
from bot_v2.features.live_trade import risk_mgr  # naming: allow - example only
from bot_v2.orchestration import rt_eng  # naming: allow - example only
```

### 2.2 Classes & Data Structures

**Format:** `PascalCase`.

**Rules:**
- Classes should be nouns (e.g., `Portfolio`, `RiskLimit`, `ExecutionEngine`)
- Mixins and protocols must carry suffixes (`LoggingMixin`, `ExecutableProtocol`)
- Avoid generic terms like `Helper` unless the class truly provides cross-cutting utilities

**Banned abbreviations:**
- ❌ `Mgr` → ✅ `Manager`
- ❌ `Cfg` → ✅ `Config`
- ❌ `Svc` → ✅ `Service`
- ❌ `Impl` → Use descriptive names (`CoinbaseExecutor` not `ExecutorImpl`)

**Examples:**
```python
# Good
class RiskManager:
    pass

class TelemetryEngine:
    pass

# Bad
class RiskMgr:  # naming: allow - example only
    pass

class TelemetryImpl:  # naming: allow - example only
    pass
```

### 2.3 Functions & Methods

**Format:** `snake_case`.

**Rules:**
- Start with verbs for actions (`fetch_account_snapshot`, `validate_order`)
- Use noun phrases for pure accessors (`risk_limits`, `portfolio_balance`)
- Avoid redundant `async` suffix; Python's `async def` keyword is sufficient

**Banned abbreviations:**
- ❌ `calc` → ✅ `calculate`
- ❌ `upd` → ✅ `update`
- ❌ `cfg` → ✅ `config`
- ❌ `mgr` → ✅ `manager`

**Examples:**
```python
# Good
async def fetch_portfolio_balance(account_id: str) -> Decimal:
    pass

def calculate_position_size(capital: Decimal, risk_pct: Decimal) -> Decimal:
    pass

# Bad
async def get_balance_async(account_id: str) -> Decimal:  # naming: allow - example only
    pass

def calc_pos_size(capital: Decimal, risk_pct: Decimal) -> Decimal:  # naming: allow - example only
    pass
```

### 2.4 Variables & Attributes

**Format:** `snake_case` for mutable state; `UPPER_SNAKE_CASE` for constants.

**Rules:**
- Prefer full words over abbreviations
- Use domain terms that match documentation (`portfolio`, `position`, `exposure`)
- **Quantity terminology**: Always spell out `quantity` instead of `qty`
- **Amount terminology**: Use `amount` instead of `amt`

**Banned abbreviations:**
- ❌ `qty` → ✅ `quantity`
- ❌ `amt` → ✅ `amount`
- ❌ `cfg` → ✅ `config`
- ❌ `svc` → ✅ `service`

**Constants:**
```python
# Good
MAX_POSITION_SIZE = Decimal("10000")
DEFAULT_RISK_PERCENTAGE = Decimal("0.02")

# Bad
MAX_POS_SIZE = Decimal("10000")  # naming: allow - example only
DEFAULT_RISK_PCT = Decimal("0.02")  # naming: allow - example only
```

**Variables:**
```python
# Good
order_quantity = calculate_order_quantity(capital, risk_pct)
filled_quantity = execution_report.filled_quantity

# Bad
order_qty = calculate_order_qty(capital, risk_pct)  # naming: allow - example only
filled_qty = execution_report.filled_qty  # naming: allow - example only
```

### 2.5 Configuration Keys & Environment Variables

**Format:**
- Config files (JSON/YAML/TOML): `snake_case` keys
- Environment variables: `UPPER_SNAKE_CASE`

**Rules:**
- Prefix environment variables with subsystem name (`COINBASE_`, `RISK_`, `TELEMETRY_`)
- Document every new key in README + config templates
- Avoid redundant suffixes like `_CFG` or `_CONF`

**Examples:**
```yaml
# Good - config/risk/coinbase.yaml
risk_limits:
  max_position_size: 10000
  max_daily_loss: 500

# Bad
risk_limits:  # naming: allow - example only
  max_pos_size: 10000  # naming: allow - example only
  max_daily_loss_amt: 500  # naming: allow - example only
```

```bash
# Good - environment variables
export COINBASE_API_KEY="..."
export RISK_MAX_POSITION_SIZE="10000"

# Bad
export COINBASE_KEY="..."  # naming: allow - example only
export RISK_MAX_POS_SIZE="10000"  # naming: allow - example only
```

### 2.6 CLI Flags & Commands

**Format:** Long options use `--kebab-case`; short options only when already established.

**Rules:**
- Align CLI terminology with configuration keys
- Provide backward-compatible aliases for at least one sprint when renaming
- Document aliases in `--help` output

**Examples:**
```bash
# Good
coinbase-trader run --profile dev --max-position-size 10000

# Bad
coinbase-trader run --profile dev --max-pos-size 10000  # naming: allow - example only
```

### 2.7 External API Exceptions

When interfacing with third-party APIs (Coinbase, Prometheus), **keep their canonical field names** to avoid mapping errors and confusion.

**Rules:**
- Preserve external field names in API client code
- Document each exception with an inline comment
- Convert to internal naming conventions at the boundary layer

**Examples:**
```python
# Good - preserving Coinbase API field names
@dataclass
class CoinbaseOrderResponse:
    order_id: str
    product_id: str
    size: str  # Coinbase API uses 'size', not 'quantity'
    filled_size: str  # Coinbase API field name

    def to_internal_order(self) -> Order:
        """Convert to internal domain model with standardized naming."""
        return Order(
            order_id=self.order_id,
            product_id=self.product_id,
            quantity=Decimal(self.size),  # Convert to 'quantity' internally
            filled_quantity=Decimal(self.filled_size),
        )
```

## 3. Abbreviation Policy

### Approved Abbreviations

The following abbreviations are allowed without restriction:

- **API** - Application Programming Interface
- **CLI** - Command Line Interface
- **PnL** - Profit and Loss
- **ID** - Identifier
- **UUID** - Universally Unique Identifier
- **URL** - Uniform Resource Locator
- **JSON** - JavaScript Object Notation
- **YAML** - YAML Ain't Markup Language
- **UTC** - Coordinated Universal Time

### Requesting New Abbreviations

New abbreviations require maintainer approval:

1. Open a discussion in GitHub Issues or PR comments
2. Provide justification (industry standard, reduces verbosity significantly)
3. If approved, add to this list and to `scripts/agents/naming_inventory.py` exceptions

## 4. File Naming Conventions

### Source Files

- Python modules: `snake_case.py` (e.g., `risk_manager.py`, `execution_engine.py`)
- Avoid `*_utils.py` or `*_helpers.py` - use domain-specific names:
  - ❌ `async_utils.py` → ✅ `asyncio_helpers.py`
  - ❌ `time_utils.py` → ✅ `time_helpers.py`
  - ❌ `logging_utils.py` → ✅ `logging.py` (when in a dedicated directory)

### Configuration Files

- Use descriptive names: `coinbase_spot.prod.yaml`, `risk_limits.dev.yaml`
- Avoid abbreviations in filenames: `coinbase_perps.yaml` → `coinbase_derivatives.yaml` (when appropriate)

### Documentation Files

- Use lowercase with underscores: `naming_standards.md`, `api_reference.md`
- Avoid redundant prefixes: `doc_naming_standards.md` → `naming_standards.md`

## 5. Enforcement & Tooling

### Automated Checks

The `scripts/agents/naming_inventory.py` script scans for banned patterns:

```bash
# Run the naming inventory
python scripts/agents/naming_inventory.py \
    --summary docs/agents/naming_inventory.md \
    --json var/agents/naming_inventory.json

# View current findings
cat docs/agents/naming_inventory.md
```

### Suppressing False Positives

Use `# naming: allow` inline comments to suppress warnings for:

- External API field names
- Temporarily grandfathered code during migration
- Legitimate domain-specific abbreviations

**Examples:**
```python
# External API field name - keep as-is
coinbase_response = {"order_id": "123", "size": "1.5"}  # naming: allow

# Temporary during migration wave
def legacy_calc_qty(amount: Decimal) -> Decimal:  # naming: allow - Wave 1 migration
    return calculate_quantity(amount)
```

### Pre-commit Hook (Planned)

A pre-commit hook will be added to run naming checks automatically:

```yaml
# .pre-commit-config.yaml (planned)
- repo: local
  hooks:
    - id: naming-inventory
      name: Check naming standards
      entry: python scripts/agents/naming_inventory.py
      language: system
      types: [python]
      pass_filenames: false
```

### CI Integration (Planned)

The naming inventory will run in CI and fail the build if new violations are introduced:

```yaml
# .github/workflows/ci.yml (planned)
- name: Check naming standards
  run: |
    python scripts/agents/naming_inventory.py --json naming_report.json
    # Fail if new violations exceed baseline
```

## 6. Migration Strategy

When renaming existing code, follow these guidelines:

### Small-Scope Renames (single module)

1. Rename identifiers within the module
2. Update all references in the same PR
3. Run full test suite to catch missed references

### Large-Scope Renames (cross-module)

1. **Add new names** alongside old names (e.g., property aliases)
2. **Update callers** progressively across multiple PRs
3. **Deprecate old names** with warnings after one sprint
4. **Remove old names** after two sprints or when usage drops to zero

**Example - Parameter alias during migration:**
```python
def create_order(
    quantity: Decimal,
    qty: Decimal | None = None,  # Deprecated alias
) -> Order:
    """Create an order.

    Args:
        quantity: Order quantity (preferred)
        qty: DEPRECATED - Use 'quantity' instead
    """
    if qty is not None:
        warnings.warn("Parameter 'qty' is deprecated, use 'quantity'", DeprecationWarning)
        quantity = qty
    return Order(quantity=quantity)
```

### File Renames

For file or module renames:

1. Create new file with updated name
2. Add `# Deprecated - import from new location` stub in old file
3. Update all imports in separate commits
4. Remove old file after confirming zero usage

## 7. Review Checklist

Before submitting a PR, verify:

- [ ] No banned abbreviations introduced (run `scripts/agents/naming_inventory.py`)
- [ ] New names follow category-specific rules (modules, classes, functions, variables)
- [ ] External API exceptions are documented with inline comments
- [ ] Configuration keys and environment variables match conventions
- [ ] CLI flags use `--kebab-case` and align with config keys
- [ ] File names are descriptive and avoid `*_utils.py` pattern
- [ ] Tests updated to reflect renamed identifiers

## 8. Examples & Anti-Patterns

### Example: Risk Manager Module

```python
# Good naming throughout
# File: src/bot_v2/features/live_trade/risk/manager.py

from decimal import Decimal
from dataclasses import dataclass

@dataclass
class RiskLimit:
    """Risk limit configuration."""
    max_position_size: Decimal
    max_daily_loss: Decimal

class RiskManager:
    """Manages risk limits and validation."""

    def __init__(self, config: RiskLimit):
        self.config = config

    def validate_order_quantity(self, quantity: Decimal) -> bool:
        """Validate order quantity against position size limit."""
        return quantity <= self.config.max_position_size

    def calculate_maximum_quantity(self, price: Decimal, capital: Decimal) -> Decimal:
        """Calculate maximum order quantity given price and available capital."""
        max_value = min(
            self.config.max_position_size * price,
            capital,
        )
        return max_value / price
```

### Anti-Pattern: Abbreviated Names

```python
# Bad - banned abbreviations throughout
# File: src/bot_v2/features/live_trade/risk/mgr.py  # naming: allow - example only

from decimal import Decimal
from dataclasses import dataclass

@dataclass
class RiskCfg:  # naming: allow - example only
    """Risk limit cfg."""  # naming: allow - example only
    max_pos_size: Decimal  # naming: allow - example only
    max_daily_loss_amt: Decimal  # naming: allow - example only

class RiskMgr:  # naming: allow - example only
    """Manages risk limits and validation."""

    def __init__(self, cfg: RiskCfg):  # naming: allow - example only
        self.cfg = cfg  # naming: allow - example only

    def validate_order_qty(self, qty: Decimal) -> bool:  # naming: allow - example only
        """Validate order qty against position size limit."""  # naming: allow - example only
        return qty <= self.cfg.max_pos_size  # naming: allow - example only

    def calc_max_qty(self, price: Decimal, capital: Decimal) -> Decimal:  # naming: allow - example only
        """Calc maximum order qty given price and available capital."""  # naming: allow - example only
        max_val = min(  # naming: allow - example only
            self.cfg.max_pos_size * price,  # naming: allow - example only
            capital,
        )
        return max_val / price  # naming: allow - example only
```

## 9. Additional Resources

- **Naming Inventory**: `docs/agents/naming_inventory.md` - Current violations and cleanup plan
- **Inventory Script**: `scripts/agents/naming_inventory.py` - Automated scanner
- **Contributing Guide**: `CONTRIBUTING.md` - Development workflow and quality standards
- **Python Style Guide**: [PEP 8](https://pep8.org/) - General Python conventions

## 10. Questions & Feedback

For questions or suggestions about these naming standards:

1. Open a GitHub Discussion in the repository
2. Reference this document in PR reviews when suggesting naming improvements
3. Propose additions to the approved abbreviations list with justification

---

**Effective Date**: 2025-11-20

**Status**: Active - enforced via code review and automated tooling
