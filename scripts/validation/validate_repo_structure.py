#!/usr/bin/env python3
"""
Repository Structure Validation Script

This script validates that the repository structure follows the expected conventions:
- src/ layout with proper package structure
- No 'src.' imports
- Proper pytest configuration
- Clean root directory
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict

class Colors:
    """Terminal colors for output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


class RepoValidator:
    """Validates repository structure and conventions"""
    
    def __init__(self, repo_root: Path = Path.cwd()):
        self.repo_root = repo_root
        self.issues = defaultdict(list)
        self.warnings = defaultdict(list)
        self.successes = []
    
    def validate_all(self) -> bool:
        """Run all validations"""
        print(f"{Colors.BLUE}🔍 Repository Structure Validation{Colors.NC}")
        print("=" * 50)
        
        # Run all checks
        self.check_pytest_config()
        self.check_package_structure()
        self.check_imports()
        self.check_root_hygiene()
        self.check_test_discovery()
        self.check_runner_functionality()
        
        # Report results
        self.print_report()
        
        return len(self.issues) == 0
    
    def check_pytest_config(self):
        """Check pytest.ini configuration"""
        pytest_ini = self.repo_root / "pytest.ini"
        
        if not pytest_ini.exists():
            self.issues["pytest"]["Missing pytest.ini in root"]
            return
        
        content = pytest_ini.read_text()
        
        # Check for pythonpath
        if "pythonpath = src" not in content:
            self.issues["pytest"].append("Missing 'pythonpath = src' in pytest.ini")
        else:
            self.successes.append("✓ pytest.ini has correct pythonpath")
        
        # Check for testpaths
        if "testpaths = tests" not in content:
            self.warnings["pytest"].append("Missing 'testpaths = tests' in pytest.ini")
        else:
            self.successes.append("✓ pytest.ini has correct testpaths")
    
    def check_package_structure(self):
        """Check package structure and __init__.py files"""
        src_dir = self.repo_root / "src"
        bot_v2_dir = src_dir / "bot_v2"
        
        if not src_dir.exists():
            self.issues["package"].append("Missing src/ directory")
            return
        
        if not bot_v2_dir.exists():
            self.issues["package"].append("Missing src/bot_v2/ directory")
            return
        
        # Check for __init__.py
        init_file = bot_v2_dir / "__init__.py"
        if not init_file.exists():
            self.issues["package"].append("Missing src/bot_v2/__init__.py")
        else:
            self.successes.append("✓ src/bot_v2/__init__.py exists")
        
        # Check pyproject.toml
        pyproject = self.repo_root / "pyproject.toml"
        if pyproject.exists():
            content = pyproject.read_text()
            if 'packages = [{ include = "bot_v2", from = "src" }]' in content:
                self.successes.append("✓ pyproject.toml has correct package configuration")
            else:
                self.issues["package"].append("Incorrect package configuration in pyproject.toml")
    
    def check_imports(self):
        """Check for incorrect import patterns"""
        # Find all Python files
        py_files = []
        for pattern in ["scripts/**/*.py", "src/**/*.py", "tests/**/*.py"]:
            py_files.extend(self.repo_root.glob(pattern))
        
        src_import_files = []
        sys_path_files = []
        
        for py_file in py_files:
            # Skip archived files
            if "archived" in str(py_file):
                continue
            
            try:
                content = py_file.read_text()
                
                # Check for src. imports
                if re.search(r'from src\.bot_v2', content) or re.search(r'import src\.bot_v2', content):
                    src_import_files.append(str(py_file.relative_to(self.repo_root)))
                
                # Check for sys.path manipulation
                if re.search(r'sys\.path\.(insert|append)', content):
                    sys_path_files.append(str(py_file.relative_to(self.repo_root)))
            except:
                pass
        
        if src_import_files:
            self.issues["imports"].append(f"Found {len(src_import_files)} files with 'src.' imports:")
            for f in src_import_files[:10]:  # Show first 10
                self.issues["imports"].append(f"  - {f}")
            if len(src_import_files) > 10:
                self.issues["imports"].append(f"  ... and {len(src_import_files) - 10} more")
        else:
            self.successes.append("✓ No 'src.' imports found")
        
        if sys_path_files:
            self.warnings["imports"].append(f"Found {len(sys_path_files)} files with sys.path manipulation:")
            for f in sys_path_files[:5]:
                self.warnings["imports"].append(f"  - {f}")
        else:
            self.successes.append("✓ No sys.path manipulations found")
    
    def check_root_hygiene(self):
        """Check for clutter in root directory"""
        root_files = list(self.repo_root.glob("*.py"))
        root_json = list(self.repo_root.glob("*.json"))
        root_sh = list(self.repo_root.glob("*.sh"))
        
        # Expected files in root
        expected_py = {"setup.py", "conftest.py"}
        expected_json = {"package.json", "tsconfig.json"}
        expected_sh = {"setup.sh", "install.sh"}
        
        # Find unexpected files
        unexpected_py = [f.name for f in root_files if f.name not in expected_py]
        unexpected_json = [f.name for f in root_json if f.name not in expected_json]
        unexpected_sh = [f.name for f in root_sh if f.name not in expected_sh]
        
        if unexpected_py:
            self.warnings["root"].append(f"Found {len(unexpected_py)} Python files in root:")
            for f in unexpected_py[:5]:
                self.warnings["root"].append(f"  - {f}")
        
        if len(unexpected_json) > 5:  # Allow a few JSON files
            self.warnings["root"].append(f"Found {len(unexpected_json)} JSON files in root (consider moving to results/)")
        
        if unexpected_sh:
            self.warnings["root"].append(f"Found {len(unexpected_sh)} shell scripts in root:")
            for f in unexpected_sh[:5]:
                self.warnings["root"].append(f"  - {f}")
        
        if not (unexpected_py or len(unexpected_json) > 5 or unexpected_sh):
            self.successes.append("✓ Root directory is clean")
    
    def check_test_discovery(self):
        """Check if pytest can discover tests"""
        try:
            result = subprocess.run(
                ["pytest", "--collect-only", "-q"],
                capture_output=True,
                text=True,
                cwd=self.repo_root,
                timeout=10
            )
            
            if result.returncode == 0:
                # Extract test count
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if "passed" in line or "selected" in line:
                        self.successes.append(f"✓ Pytest can discover tests: {line}")
                        return
                
                if result.stdout:
                    self.successes.append("✓ Pytest discovery works")
            else:
                self.issues["tests"].append(f"Pytest discovery failed: {result.stderr[:200]}")
        except Exception as e:
            self.warnings["tests"].append(f"Could not run pytest: {e}")
    
    def check_runner_functionality(self):
        """Check if main runner scripts have correct imports"""
        runner = self.repo_root / "scripts" / "run_perps_bot.py"
        
        if not runner.exists():
            self.warnings["runner"].append("Main runner script not found at scripts/run_perps_bot.py")
            return
        
        content = runner.read_text()
        
        # Check for src. imports
        if "from bot_v2" in content:
            self.issues["runner"].append("Main runner still uses 'src.' imports")
        else:
            self.successes.append("✓ Main runner uses correct imports")
    
    def print_report(self):
        """Print validation report"""
        print()
        
        # Print successes
        if self.successes:
            print(f"{Colors.GREEN}✅ Passed Checks:{Colors.NC}")
            for success in self.successes:
                print(f"  {success}")
            print()
        
        # Print warnings
        if self.warnings:
            print(f"{Colors.YELLOW}⚠️  Warnings:{Colors.NC}")
            for category, items in self.warnings.items():
                print(f"\n  [{category.upper()}]")
                for item in items:
                    print(f"    {item}")
            print()
        
        # Print issues
        if self.issues:
            print(f"{Colors.RED}❌ Issues Found:{Colors.NC}")
            for category, items in self.issues.items():
                print(f"\n  [{category.upper()}]")
                for item in items:
                    print(f"    {item}")
            print()
        
        # Summary
        print("=" * 50)
        if not self.issues:
            print(f"{Colors.GREEN}✅ All critical checks passed!{Colors.NC}")
            return True
        else:
            total_issues = sum(len(items) for items in self.issues.values())
            print(f"{Colors.RED}❌ Found {total_issues} critical issues that need fixing{Colors.NC}")
            print(f"\nRun the fix script with: {Colors.YELLOW}bash scripts/fix_repo_structure.sh{Colors.NC}")
            return False


def main():
    """Main entry point"""
    validator = RepoValidator()
    success = validator.validate_all()
    
    # Additional quick checks
    print("\n" + "=" * 50)
    print(f"{Colors.BLUE}Quick Import Test:{Colors.NC}")
    
    # Try to import the package
    try:
        sys.path.insert(0, str(Path.cwd() / "src"))
        import bot_v2.features.live_trade
        print(f"{Colors.GREEN}✓ Package import successful{Colors.NC}")
    except ImportError as e:
        print(f"{Colors.RED}✗ Package import failed: {e}{Colors.NC}")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()