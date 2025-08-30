# 🔒 The Isolation Principle

## **Core Rule: One Slice = One Context = Complete Feature**

### ❌ **What We Must Avoid**
```
features/backtest/
├── backtest.py
└── signals.py  → imports from shared/strategies  ❌ BREAKS ISOLATION!
```

### ✅ **What We Must Do**
```
features/backtest/
├── backtest.py
├── signals.py
└── strategies.py  → Everything needed is LOCAL ✅
```

## 🎯 **Why Isolation Matters for AI Agents**

1. **Single Context Loading**
   - Agent loads ONE directory
   - Everything is there
   - No navigation needed

2. **Zero Dependencies**
   - No shared/
   - No utils/
   - No cross-slice imports

3. **Complete Understanding**
   - Read one slice, understand everything
   - No hidden dependencies
   - No surprise imports

## 📊 **The Cost of Shared Code**

```python
# BAD: Using shared code
from shared.strategies import MomentumStrategy  # +200 tokens
from shared.utils import validate_data          # +100 tokens
from shared.types import MarketData            # +150 tokens
# Total: 450 extra tokens just for "shared" code!

# GOOD: Everything local
from .strategies import MomentumStrategy  # 0 extra tokens (already in context)
from .validation import validate_data     # 0 extra tokens (already in context)
from .types import MarketData            # 0 extra tokens (already in context)
# Total: 0 extra tokens!
```

## 🔧 **Implementation Rules**

### Rule 1: **Duplicate, Don't Share**
```python
# Each slice gets its own copy of what it needs
features/backtest/strategies.py     # Backtest's copy
features/paper_trade/strategies.py  # Paper trade's copy
features/optimize/strategies.py     # Optimize's copy
```

### Rule 2: **Local Types Only**
```python
# Bad
from shared.types import Order  ❌

# Good
from .types import Order  ✅
```

### Rule 3: **No Cross-Slice Imports**
```python
# Bad
from features.backtest import BacktestResult  ❌

# Good
from .types import BacktestResult  ✅
```

### Rule 4: **Configuration Over Code Sharing**
```python
# Instead of importing shared strategies, pass as config
def run_backtest(strategy_config: dict, ...):
    strategy = create_local_strategy(strategy_config)
```

## 📁 **Correct Structure**

```
features/
├── backtest/
│   ├── __init__.py
│   ├── backtest.py         # Entry point
│   ├── data.py             # Data fetching
│   ├── signals.py          # Signal generation
│   ├── execution.py        # Trade simulation
│   ├── metrics.py          # Metrics calculation
│   ├── strategies.py       # LOCAL strategy implementations
│   ├── validation.py       # LOCAL validation
│   └── types.py            # LOCAL types
│
├── paper_trade/
│   ├── __init__.py
│   ├── paper_trade.py      # Entry point
│   ├── strategies.py       # DUPLICATE of strategies (intentional!)
│   ├── validation.py       # DUPLICATE of validation (intentional!)
│   └── types.py            # DUPLICATE of types (intentional!)
│
└── NO shared/ or utils/ directories!
```

## 🚫 **Anti-Patterns to Avoid**

1. **"DRY" Obsession**
   - DRY (Don't Repeat Yourself) is LESS important than isolation
   - Duplication is GOOD when it maintains isolation

2. **Shared Utils**
   - No `shared/utils.py`
   - No `common/helpers.py`
   - Each slice has its own utilities

3. **Base Classes in Shared**
   - No `shared/base_strategy.py`
   - Each slice defines what it needs

## ✅ **Benefits of Complete Isolation**

1. **Token Efficiency**
   - Load one directory, get everything
   - No chasing imports
   - No loading shared code

2. **Change Safety**
   - Modify one slice without fear
   - Can't break other slices
   - True independence

3. **AI Agent Paradise**
   - One context to rule them all
   - Complete understanding from one directory
   - No mental model of dependencies needed

## 📝 **Duplication Guidelines**

**What to Duplicate:**
- Small utilities (< 50 lines)
- Type definitions
- Validation functions
- Strategy implementations
- Constants and configurations

**What NOT to Duplicate:**
- External API clients (use a service)
- Database connections (use a service)
- Large algorithms (> 200 lines)

## 🎯 **The Ultimate Test**

Can an AI agent understand and modify a feature by loading ONLY that feature's directory?

- ✅ YES = Perfect isolation
- ❌ NO = Fix the dependencies

## 💡 **Remember**

> "A little duplication is better than a lot of dependencies"
> 
> For AI agents, isolation > DRY

---

**The Isolation Principle is the key to token-efficient, AI-friendly code!**