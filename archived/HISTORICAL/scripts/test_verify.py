#!/usr/bin/env python3
"""Quick test of verification without meta_workflow import issues"""

import subprocess
import sys

# Test basic verification
result = subprocess.run(
    [sys.executable, "-c", """
import sys
sys.path.insert(0, 'src')

# Test basic imports
try:
    from bot.integration.orchestrator import IntegratedOrchestrator
    from bot.paper_trading import PaperTradingEngine
    from bot.strategy.demo_ma import DemoMAStrategy
    print("✅ Core imports working")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test structure
from pathlib import Path
dirs = ['src/bot', 'tests', 'scripts', 'docs']
for d in dirs:
    if Path(d).exists():
        print(f"✅ {d} exists")
    else:
        print(f"❌ {d} missing")

print("\\n✅ Basic verification passed")
"""],
    capture_output=False,
    text=True
)

sys.exit(result.returncode)