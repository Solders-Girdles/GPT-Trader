#!/usr/bin/env python3
"""
Quick diagnostic check without logging noise.
Run this for instant component health check.
"""
import os
import sys

# Suppress all logging
os.environ['SUPPRESS_LOGS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress stderr temporarily
import contextlib
import io

def quiet_import(module_path):
    """Import a module quietly and return success status."""
    with contextlib.redirect_stderr(io.StringIO()):
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                if '.' in module_path:
                    parts = module_path.split('.')
                    module = __import__(module_path)
                    for part in parts[1:]:
                        module = getattr(module, part)
                else:
                    __import__(module_path)
                return True, None
            except Exception as e:
                return False, str(e)

def main():
    """Run quick component checks."""
    components = {
        'Config': 'bot.config',
        'Pipeline': 'bot.dataflow.pipeline',
        'Strategies': 'bot.strategy',
        'Risk': 'bot.risk.config',
        'Portfolio': 'bot.portfolio.allocator',
        'Backtest': 'bot.backtest.engine',
        'ML': 'bot.ml.models',
    }
    
    print("=== QUICK COMPONENT CHECK ===")
    working = []
    broken = []
    
    for name, module in components.items():
        success, error = quiet_import(module)
        if success:
            print(f"✅ {name}")
            working.append(name)
        else:
            # Extract just the error type
            error_type = error.split(':')[0] if ':' in error else error
            print(f"❌ {name}: {error_type}")
            broken.append(name)
    
    print(f"\n📊 Summary: {len(working)}/{len(components)} working")
    
    if broken:
        print(f"🔧 Fix these first: {', '.join(broken)}")
        
        # Give specific guidance
        if 'Pipeline' in broken:
            print("   → Check: tests/unit/dataflow/")
        if 'Risk' in broken:
            print("   → Check: missing __init__.py or imports")
        if 'ML' in broken:
            print("   → Check: scikit-learn installed")
    
    return 0 if len(broken) == 0 else 1

if __name__ == "__main__":
    sys.exit(main())