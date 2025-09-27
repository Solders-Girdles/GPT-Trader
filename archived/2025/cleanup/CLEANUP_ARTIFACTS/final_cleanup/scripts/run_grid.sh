#!/usr/bin/env bash
set -euo pipefail

# Common args (edit if you want)
SYMS="--symbol-list data/universe/sp100.csv"
DATES="--start 2019-01-01 --end 2020-12-31"
COMMON="--strategy trend_breakout --regime on --cost-bps 5 --exit-mode stop"

# Parameter grid (tweak as needed)
ATR_KS=(2.3 2.6 2.8)
RISK_PCTS=(0.35 0.5)
MAX_POS=(8 10)
ENTRY_CONFIRMS=(1 2)
REBAL_PCTS=(0 0.002)

# Output table
STAMP=$(date +"%Y%m%d-%H%M%S")
OUT="data/backtests/grid_results_${STAMP}.csv"
mkdir -p scripts data/backtests
echo "atr_k,risk_pct,max_positions,entry_confirm,min_rebalance_pct,total_return,cagr,sharpe,max_drawdown,total_costs,summary_path" > "$OUT"

for atrk in "${ATR_KS[@]}"; do
  for risk in "${RISK_PCTS[@]}"; do
    for mp in "${MAX_POS[@]}"; do
      for ec in "${ENTRY_CONFIRMS[@]}"; do
        for rp in "${REBAL_PCTS[@]}"; do
          echo ">>> Run: atr_k=${atrk}, risk=${risk}, max_pos=${mp}, entry_confirm=${ec}, min_rebal=${rp}"
          poetry run gpt-trader backtest \
            $COMMON $SYMS $DATES \
            --atr-k "$atrk" --risk-pct "$risk" --max-positions "$mp" \
            --entry-confirm "$ec" --min-rebalance-pct "$rp" >/dev/null

          # latest summary file
          SUM=$(ls -t data/backtests/*_summary.csv | head -n1)

          # extract metrics and append a CSV row
          python - <<PY >> "$OUT"
import csv, sys
from pathlib import Path

sum_path = Path("$SUM")
d = {}
with open(sum_path) as f:
    for row in csv.reader(f):
        if not row: continue
        k, v = row[0], row[1]
        d[k] = v

# print one CSV line with params + metrics
print(",".join(map(str, [
    "$atrk",
    "$risk",
    "$mp",
    "$ec",
    "$rp",
    d.get("total_return",""),
    d.get("cagr",""),
    d.get("sharpe",""),
    d.get("max_drawdown",""),
    d.get("total_costs",""),
    sum_path.as_posix(),
])))
PY
        done
      done
    done
  done
done

echo "Grid complete -> $OUT"
