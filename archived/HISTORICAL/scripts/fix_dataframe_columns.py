#!/usr/bin/env python3
"""
Fix DataFrame Column Case Sensitivity
======================================
Converts all Title Case column references to lowercase.
"""

import os
import re
from pathlib import Path


def fix_column_references(content: str) -> tuple[str, int]:
    """Fix column references in file content."""
    original = content
    changes = 0
    
    # Pattern to match DataFrame column access with Title Case
    patterns = [
        (r'(\[[\'"]\s*)Close(\s*[\'\"]\])', r'\1close\2'),
        (r'(\[[\'"]\s*)Open(\s*[\'\"]\])', r'\1open\2'),
        (r'(\[[\'"]\s*)High(\s*[\'\"]\])', r'\1high\2'),
        (r'(\[[\'"]\s*)Low(\s*[\'\"]\])', r'\1low\2'),
        (r'(\[[\'"]\s*)Volume(\s*[\'\"]\])', r'\1volume\2'),
    ]
    
    for pattern, replacement in patterns:
        content, n = re.subn(pattern, replacement, content)
        changes += n
    
    # Also fix in docstrings and comments
    content = content.replace("['Open', 'High', 'Low', 'Close', 'Volume']", 
                             "['open', 'high', 'low', 'close', 'volume']")
    if "['Open', 'High', 'Low', 'Close', 'Volume']" in original:
        changes += 1
    
    return content, changes


def fix_file(filepath: Path) -> bool:
    """Fix a single file."""
    try:
        content = filepath.read_text()
        fixed_content, changes = fix_column_references(content)
        
        if changes > 0:
            # Create backup
            backup_path = filepath.with_suffix(filepath.suffix + '.bak')
            backup_path.write_text(content)
            
            # Write fixed content
            filepath.write_text(fixed_content)
            print(f"  ✅ Fixed {filepath.name}: {changes} changes")
            return True
        else:
            print(f"  ⏭️  {filepath.name}: No changes needed")
            return False
    except Exception as e:
        print(f"  ❌ Error fixing {filepath}: {e}")
        return False


def main():
    print("="*60)
    print("🔧 FIXING DATAFRAME COLUMN CASE SENSITIVITY")
    print("="*60)
    print()
    
    # Strategy files to fix
    strategy_dir = Path("src/bot/strategy")
    strategy_files = [
        "demo_ma.py",
        "trend_breakout.py",
        "mean_reversion.py",
        "momentum.py",
        "volatility.py",
        "components.py",
        "validation_pipeline.py",
        "enhanced_trend_breakout.py",
        "optimized_ma.py",
        "talib_optimized_ma.py",
    ]
    
    print("📁 Fixing strategy files...")
    fixed_count = 0
    for filename in strategy_files:
        filepath = strategy_dir / filename
        if filepath.exists():
            if fix_file(filepath):
                fixed_count += 1
        else:
            print(f"  ⚠️  {filename}: File not found")
    
    print(f"\n✅ Fixed {fixed_count} files")
    
    # Also check other directories that might have column references
    other_dirs = [
        ("src/bot/indicators", "*.py"),
        ("src/bot/backtest", "*.py"),
        ("src/bot/portfolio", "*.py"),
        ("tests", "**/*.py"),
    ]
    
    print("\n📁 Checking other directories...")
    for dir_path, pattern in other_dirs:
        base_path = Path(dir_path)
        if base_path.exists():
            files = list(base_path.glob(pattern))
            if files:
                print(f"\n  Checking {dir_path}...")
                for filepath in files:
                    # Skip __pycache__ and backup files
                    if '__pycache__' in str(filepath) or filepath.suffix == '.bak':
                        continue
                    
                    content = filepath.read_text()
                    if any(col in content for col in ["['Close']", '["Close"]', 
                                                       "['Open']", '["Open"]',
                                                       "['High']", '["High"]',
                                                       "['Low']", '["Low"]',
                                                       "['Volume']", '["Volume"]']):
                        if fix_file(filepath):
                            fixed_count += 1
    
    print("\n" + "="*60)
    print(f"🎯 TOTAL FILES FIXED: {fixed_count}")
    print("="*60)
    
    # Verify the fix
    print("\n🔍 Verifying fix...")
    verify_script = Path("scripts/verify_dataframe_fix.py")
    if verify_script.exists():
        import subprocess
        result = subprocess.run(["python", str(verify_script)], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Verification passed!")
        else:
            print("❌ Verification failed. Check the output:")
            print(result.stdout)
            print(result.stderr)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())