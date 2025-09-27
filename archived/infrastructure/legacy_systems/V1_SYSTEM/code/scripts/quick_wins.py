# \!/usr/bin/env python3
"""Quick wins implementation"""

import sys
from pathlib import Path

sys.path.insert(0, "src")


def fix_backtest_parameters():
    """Fix parameter mismatch"""
    print("Fixing backtest parameters...")
    commands_file = Path("src/bot/cli/commands.py")
    if commands_file.exists():
        content = commands_file.read_text()
        content = content.replace('"start_date":', '"start":')
        content = content.replace('"end_date":', '"end":')
        commands_file.write_text(content)
        print("✅ Fixed")
        return True
    return False


def fix_test_imports():
    """Fix test imports"""
    print("Fixing test imports...")
    conftest = Path("tests/conftest.py")
    if conftest.exists():
        content = conftest.read_text()
        if "sys.path.insert" not in content:
            fix = "import sys\nimport os\nsys.path.insert(0, os.path.abspath('src'))\n\n"
            conftest.write_text(fix + content)
            print("✅ Fixed")
        else:
            print("Already fixed")
        return True
    return False


def main():
    print("Running quick wins...")
    fix_backtest_parameters()
    fix_test_imports()
    print(r"Done\!")


if __name__ == "__main__":
    main()
