# Paper Trading Implementation Plan (Accurate)

## Current State (Verified)
- **Scripts**: 30 files (need to reduce to <10)
- **Coinbase Tests**: 29/29 passing ✅
- **Paper Engine**: Basic functionality only
- **Unified Entry**: `scripts/paper_trade.py` working ✅

## Implementation Roadmap

### Phase 1: Script Consolidation (Day 1-2)
```bash
# Archive redundant paper scripts
mv scripts/paper_trade_coinbase.py archived/paper_scripts_2025/
mv scripts/paper_trade_live.py archived/paper_scripts_2025/
mv scripts/paper_trade_strategies_*.py archived/paper_scripts_2025/
mv scripts/parallel_paper_trading.py archived/paper_scripts_2025/

# Move utilities
mkdir -p scripts/utils
mv scripts/check_*.py scripts/utils/
mv scripts/validate_*.py scripts/utils/
mv scripts/monitor_*.py scripts/utils/

# Result: ~8 scripts in scripts/
```

### Phase 2: PaperExecutionEngine Enhancement (Day 3-4)

#### Add to `src/bot_v2/orchestration/execution.py`:
```python
class PaperExecutionEngine:
    def __init__(self, commission=0.006, slippage=0.001, 
                 initial_capital=10000, config=None):
        # Add config support
        self.config = config or {}
        self.max_position_pct = self.config.get('max_position_pct', 0.25)
        self.max_exposure = self.config.get('max_exposure', 0.90)
        self.min_cash_reserve = self.config.get('min_cash', 1000)
        
    def validate_order(self, symbol: str, amount_usd: float) -> bool:
        """Enforce portfolio constraints."""
        # Position size limit
        if amount_usd > self.cash * self.max_position_pct:
            return False
        
        # Exposure limit
        total_exposure = sum(p.quantity * p.current_price 
                           for p in self.positions.values())
        portfolio_value = self.cash + total_exposure
        if (total_exposure + amount_usd) > portfolio_value * self.max_exposure:
            return False
            
        # Minimum cash
        if self.cash - amount_usd < self.min_cash_reserve:
            return False
            
        return True
    
    def enforce_product_rules(self, symbol: str, quantity: float, price: float):
        """Apply exchange rules (min size, step size, notional)."""
        from .utils import ProductCatalog, quantize_to_increment
        
        catalog = ProductCatalog()
        product = catalog.get(self._broker._client, symbol)
        
        # Round to increments
        quantity = quantize_to_increment(quantity, product.step_size)
        price = quantize_to_increment(price, product.price_increment)
        
        # Check minimums
        if quantity < product.min_size:
            raise ValueError(f"Below min size {product.min_size}")
            
        notional = quantity * price
        if notional < product.min_notional:
            raise ValueError(f"Below min notional {product.min_notional}")
            
        return quantity, price
    
    def calculate_equity(self) -> float:
        """Calculate total portfolio value."""
        position_value = sum(
            pos.quantity * pos.current_price 
            for pos in self.positions.values()
        )
        return self.cash + position_value
```

### Phase 3: Persistence Layer (Day 5-6)

#### Create `src/bot_v2/features/paper_trade/event_store.py`:
```python
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional

class EventStore:
    def __init__(self, db_path: str = "paper_trades.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_tables()
    
    def _init_tables(self):
        """Create tables for trades and snapshots."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                commission REAL,
                pnl REAL,
                reason TEXT,
                strategy TEXT
            );
            
            CREATE TABLE IF NOT EXISTS equity_snapshots (
                timestamp TEXT PRIMARY KEY,
                cash REAL NOT NULL,
                positions_value REAL NOT NULL,
                total_equity REAL NOT NULL,
                positions_json TEXT
            );
            
            CREATE INDEX IF NOT EXISTS idx_trades_symbol 
                ON trades(symbol);
            CREATE INDEX IF NOT EXISTS idx_trades_timestamp 
                ON trades(timestamp);
        """)
        self.conn.commit()
    
    def record_trade(self, trade: Dict):
        """Record a trade to the database."""
        self.conn.execute(
            """INSERT INTO trades 
               (timestamp, symbol, side, quantity, price, commission, pnl, reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (trade['timestamp'], trade['symbol'], trade['side'],
             trade['quantity'], trade['price'], trade['commission'],
             trade.get('pnl'), trade.get('reason'))
        )
        self.conn.commit()
    
    def snapshot_equity(self, cash: float, positions: Dict):
        """Save portfolio snapshot."""
        positions_value = sum(
            p['quantity'] * p['current_price'] 
            for p in positions.values()
        )
        
        self.conn.execute(
            """INSERT OR REPLACE INTO equity_snapshots 
               VALUES (?, ?, ?, ?, ?)""",
            (datetime.now().isoformat(), cash, positions_value,
             cash + positions_value, json.dumps(positions))
        )
        self.conn.commit()
    
    def get_metrics(self) -> Dict:
        """Calculate performance metrics."""
        cursor = self.conn.execute("""
            SELECT 
                COUNT(*) as total_trades,
                COUNT(CASE WHEN pnl > 0 THEN 1 END) as winners,
                COUNT(CASE WHEN pnl < 0 THEN 1 END) as losers,
                SUM(pnl) as total_pnl,
                AVG(pnl) as avg_pnl,
                MAX(pnl) as best_trade,
                MIN(pnl) as worst_trade
            FROM trades WHERE pnl IS NOT NULL
        """)
        
        row = cursor.fetchone()
        if row[0] == 0:  # No trades
            return {}
            
        win_rate = row[1] / row[0] if row[0] > 0 else 0
        
        return {
            'total_trades': row[0],
            'win_rate': win_rate,
            'total_pnl': row[3] or 0,
            'avg_pnl': row[4] or 0,
            'best_trade': row[5] or 0,
            'worst_trade': row[6] or 0
        }
```

### Phase 4: Monitoring (Day 7)

#### Simple Console Dashboard:
```python
# Add to scripts/paper_trade.py
def show_dashboard(engine, event_store):
    """Display console dashboard."""
    import os
    os.system('clear' if os.name == 'posix' else 'cls')
    
    print("="*50)
    print("         PAPER TRADING DASHBOARD")
    print("="*50)
    
    equity = engine.calculate_equity()
    returns = ((equity - engine.initial_capital) / engine.initial_capital) * 100
    
    print(f"\n📊 Portfolio Summary:")
    print(f"  Total Equity: ${equity:,.2f}")
    print(f"  Cash: ${engine.cash:,.2f}")
    print(f"  Returns: {returns:+.2f}%")
    
    if engine.positions:
        print(f"\n📈 Open Positions:")
        for symbol, pos in engine.positions.items():
            pnl = (pos.current_price - pos.entry_price) * pos.quantity
            pnl_pct = ((pos.current_price / pos.entry_price) - 1) * 100
            print(f"  {symbol}: {pos.quantity:.6f} @ ${pos.entry_price:.2f}")
            print(f"    Current: ${pos.current_price:.2f} | P&L: ${pnl:+,.2f} ({pnl_pct:+.1f}%)")
    
    metrics = event_store.get_metrics()
    if metrics:
        print(f"\n📈 Performance:")
        print(f"  Total Trades: {metrics['total_trades']}")
        print(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
        print(f"  Total P&L: ${metrics['total_pnl']:+,.2f}")
```

### Phase 5: Testing & Polish (Day 8-10)

#### Key Tests to Add:
```bash
# Paper engine tests
tests/unit/bot_v2/features/paper_trade/
  ├── test_paper_engine.py (enhance existing)
  ├── test_portfolio_constraints.py (new)
  ├── test_product_rules.py (new)
  └── test_event_store.py (new)

# Integration tests  
tests/integration/paper_trade/
  ├── test_offline_trading.py (using fixtures)
  └── test_sandbox_trading.py (if creds available)
```

## Security Fix (Immediate)

```bash
# Remove .env.production from tracking
git rm --cached .env.production
echo ".env.production" >> .gitignore
git commit -m "chore: remove tracked env file for security"

# Document in README
echo "Copy .env.template to .env.production and add your keys" >> README.md
```

## Success Metrics
- [ ] Scripts reduced to <10 in scripts/
- [ ] Paper engine with constraints + product rules
- [ ] Trade persistence with SQLite
- [ ] Console dashboard showing P&L
- [ ] 80%+ test coverage on paper trading
- [ ] Single command execution working
- [ ] No sensitive files in git

## Final Commands

```bash
# Setup
cp .env.template .env.production
# Add your CDP keys to .env.production

# Run paper trading
PYTHONPATH=src python scripts/paper_trade.py \
    --symbols BTC-USD,ETH-USD \
    --capital 10000 \
    --cycles 10 \
    --interval 30

# Run with dashboard (future)
PYTHONPATH=src python scripts/paper_trade.py \
    --symbols BTC-USD \
    --capital 10000 \
    --dashboard
```

---
*Timeline: 7-10 days*
*Current Progress: ~40% (foundation laid)*
*Next Critical Step: Complete script consolidation*