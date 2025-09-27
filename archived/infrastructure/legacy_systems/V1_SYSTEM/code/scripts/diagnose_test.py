#!/usr/bin/env python3
"""
Diagnose test failures with helpful context.
Usage: poetry run python scripts/diagnose_test.py tests/unit/module/test_file.py::TestClass::test_method
"""
import sys
import subprocess
import re
from pathlib import Path

def diagnose_test_failure(test_path):
    """Diagnose why a test is failing."""
    print(f"🔍 Diagnosing: {test_path}\n")
    
    # Parse test path
    if '::' in test_path:
        file_path = test_path.split('::')[0]
    else:
        file_path = test_path
    
    if not Path(file_path).exists():
        print(f"❌ Test file not found: {file_path}")
        return
    
    # Step 1: Check if test file imports work
    print("1. Checking test imports...")
    module_path = file_path.replace('/', '.').replace('.py', '')
    result = subprocess.run(
        f'poetry run python -c "import {module_path}"',
        shell=True,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("   ❌ Import error in test file:")
        error_lines = result.stderr.split('\n')
        for line in error_lines:
            if 'ModuleNotFoundError' in line or 'ImportError' in line:
                print(f"      {line}")
                
                # Extract module name
                match = re.search(r"'([^']+)'", line)
                if match:
                    missing_module = match.group(1)
                    print(f"\n   💡 Fix: Check if '{missing_module}' exists or has __init__.py")
        return
    else:
        print("   ✅ Test file imports successfully")
    
    # Step 2: Check for patch decorators
    print("\n2. Checking @patch decorators...")
    with open(file_path) as f:
        content = f.read()
    
    patches = re.findall(r'@patch\(["\']([^"\']+)["\']\)', content)
    if patches:
        print(f"   Found {len(patches)} @patch decorators:")
        for patch_path in patches:
            # Check if it starts with src.bot
            if patch_path.startswith('src.bot'):
                print(f"   ❌ {patch_path}")
                print(f"      Fix: Change to '{patch_path.replace('src.bot', 'bot')}'")
            else:
                # Try to import the patched module
                module_check = patch_path.rsplit('.', 1)[0]
                check_result = subprocess.run(
                    f'poetry run python -c "import {module_check}"',
                    shell=True,
                    capture_output=True,
                    text=True
                )
                if check_result.returncode == 0:
                    print(f"   ✅ {patch_path}")
                else:
                    print(f"   ❌ {patch_path} (module not found)")
    
    # Step 3: Check fixtures
    print("\n3. Checking test fixtures...")
    fixtures = re.findall(r'def test_\w+\([^)]+\)', content)
    for fixture_line in fixtures:
        params = fixture_line.split('(')[1].split(')')[0]
        if params and params != 'self':
            fixture_names = [p.strip() for p in params.split(',') if p.strip() != 'self']
            print(f"   Test uses fixtures: {', '.join(fixture_names)}")
            
            # Check if fixtures exist
            for fixture_name in fixture_names:
                fixture_check = subprocess.run(
                    f'poetry run pytest --fixtures -q 2>/dev/null | grep "^{fixture_name}"',
                    shell=True,
                    capture_output=True,
                    text=True
                )
                if fixture_check.stdout:
                    print(f"      ✅ {fixture_name} found")
                else:
                    print(f"      ❌ {fixture_name} not found")
                    print(f"         Check: tests/fixtures/factories.py or conftest.py")
    
    # Step 4: Run test with verbose output
    print("\n4. Running test with detailed output...")
    test_result = subprocess.run(
        f'poetry run pytest {test_path} -xvs 2>&1 | tail -30',
        shell=True,
        capture_output=True,
        text=True
    )
    
    if 'PASSED' in test_result.stdout:
        print("   ✅ Test is now passing!")
    elif 'FAILED' in test_result.stdout:
        print("   ❌ Test still failing. Key error:")
        
        # Extract the actual error
        lines = test_result.stdout.split('\n')
        for i, line in enumerate(lines):
            if 'AssertionError' in line or 'Error' in line:
                print(f"      {line}")
                # Print next few lines for context
                for j in range(i+1, min(i+3, len(lines))):
                    if lines[j].strip():
                        print(f"      {lines[j]}")
                break
    
    # Step 5: Suggest solutions
    print("\n5. Suggested fixes based on diagnosis:")
    
    if 'src.bot' in content:
        print("   → Remove 'src.' from all @patch decorators")
    
    if 'AssertionError: Expected' in test_result.stdout and 'to have been called' in test_result.stdout:
        print("   → Mock not being called - check patch path matches actual import")
        print("   → Verify the mocked function is actually called in the code under test")
    
    if 'fixture' in test_result.stdout and 'not found' in test_result.stdout:
        print("   → Missing fixture - check tests/fixtures/factories.py")
        print("   → Or add fixture to conftest.py")
    
    if 'ModuleNotFoundError' in test_result.stdout:
        print("   → Missing module - check imports and __init__.py files")
    
    print("\n📚 Check KNOWN_FAILURES.md for more solutions")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: poetry run python scripts/diagnose_test.py <test_path>")
        print("Example: poetry run python scripts/diagnose_test.py tests/unit/dataflow/test_pipeline.py::TestDataPipeline::test_fetch_symbol_data_success")
        sys.exit(1)
    
    diagnose_test_failure(sys.argv[1])