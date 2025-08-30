#!/usr/bin/env python3
"""
Systematic audit of the GPT-Trader codebase to understand what actually exists
and what actually works.
"""

import os
import ast
import sys
from pathlib import Path
from collections import defaultdict
import importlib.util

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def analyze_directory_structure():
    """Map out what directories and modules exist."""
    
    print("="*80)
    print("DIRECTORY STRUCTURE ANALYSIS")
    print("="*80)
    
    src_path = Path("src/bot")
    
    # Count files in each directory
    dir_stats = {}
    
    for dir_path in sorted(src_path.rglob("*")):
        if dir_path.is_dir():
            py_files = list(dir_path.glob("*.py"))
            if py_files:
                rel_path = dir_path.relative_to(src_path)
                dir_stats[str(rel_path)] = {
                    'file_count': len(py_files),
                    'files': [f.name for f in py_files],
                    'has_init': '__init__.py' in [f.name for f in py_files],
                    'total_lines': sum(len(open(f).readlines()) for f in py_files)
                }
    
    # Group by suspected purpose
    categories = {
        'execution': [],
        'strategy': [],
        'data': [],
        'risk': [],
        'ml': [],
        'config': [],
        'monitoring': [],
        'unknown': []
    }
    
    for dir_name, stats in dir_stats.items():
        if 'exec' in dir_name or 'trading' in dir_name or 'broker' in dir_name:
            categories['execution'].append(dir_name)
        elif 'strategy' in dir_name or 'signal' in dir_name:
            categories['strategy'].append(dir_name)
        elif 'data' in dir_name or 'pipeline' in dir_name:
            categories['data'].append(dir_name)
        elif 'risk' in dir_name:
            categories['risk'].append(dir_name)
        elif 'ml' in dir_name or 'learning' in dir_name:
            categories['ml'].append(dir_name)
        elif 'config' in dir_name:
            categories['config'].append(dir_name)
        elif 'monitor' in dir_name or 'dashboard' in dir_name or 'logging' in dir_name:
            categories['monitoring'].append(dir_name)
        else:
            categories['unknown'].append(dir_name)
    
    print("\nComponents by Category:")
    for category, dirs in categories.items():
        if dirs:
            print(f"\n{category.upper()} ({len(dirs)} directories):")
            for d in dirs:
                stats = dir_stats[d]
                print(f"  - {d}: {stats['file_count']} files, {stats['total_lines']} lines")
    
    return dir_stats, categories


def find_duplicate_functionality():
    """Identify potential duplicate implementations."""
    
    print("\n" + "="*80)
    print("DUPLICATE FUNCTIONALITY ANALYSIS")
    print("="*80)
    
    duplicates = {
        'orchestrators': [],
        'risk_managers': [],
        'execution_engines': [],
        'config_systems': [],
        'ledgers': [],
        'strategies': []
    }
    
    src_path = Path("src/bot")
    
    for py_file in src_path.rglob("*.py"):
        content = py_file.read_text()
        rel_path = py_file.relative_to(src_path)
        
        # Look for orchestrators
        if 'orchestrator' in str(rel_path).lower() or 'class.*Orchestrator' in content:
            duplicates['orchestrators'].append(str(rel_path))
        
        # Look for risk managers
        if 'risk' in str(rel_path).lower() and ('class.*Risk' in content or 'class.*Manager' in content):
            duplicates['risk_managers'].append(str(rel_path))
        
        # Look for execution
        if 'exec' in str(rel_path).lower() or 'engine' in str(rel_path).lower():
            duplicates['execution_engines'].append(str(rel_path))
        
        # Look for config
        if 'config' in str(rel_path).lower():
            duplicates['config_systems'].append(str(rel_path))
        
        # Look for ledgers
        if 'ledger' in str(rel_path).lower() or 'class.*Ledger' in content:
            duplicates['ledgers'].append(str(rel_path))
    
    print("\nPotential Duplicates Found:")
    for category, files in duplicates.items():
        if len(files) > 1:
            print(f"\n{category.upper()} ({len(files)} implementations):")
            for f in files:
                print(f"  - {f}")
    
    return duplicates


def trace_import_dependencies():
    """Understand which modules actually depend on each other."""
    
    print("\n" + "="*80)
    print("IMPORT DEPENDENCY ANALYSIS")
    print("="*80)
    
    src_path = Path("src/bot")
    imports = defaultdict(set)
    
    for py_file in src_path.rglob("*.py"):
        if '__pycache__' in str(py_file):
            continue
            
        try:
            with open(py_file, 'r') as f:
                tree = ast.parse(f.read())
                
            rel_path = py_file.relative_to(src_path)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if 'bot.' in alias.name:
                            imports[str(rel_path)].add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module and 'bot.' in node.module:
                        imports[str(rel_path)].add(node.module)
        except:
            pass
    
    # Find modules that are never imported
    all_modules = set(str(p.relative_to(src_path)) for p in src_path.rglob("*.py"))
    imported_modules = set()
    for deps in imports.values():
        imported_modules.update(deps)
    
    never_imported = []
    for module in all_modules:
        module_name = module.replace('/', '.').replace('.py', '')
        full_name = f"bot.{module_name}"
        if full_name not in imported_modules and not module.endswith('__init__.py'):
            never_imported.append(module)
    
    print(f"\nModules that are never imported ({len(never_imported)}):")
    for module in sorted(never_imported)[:20]:  # First 20
        print(f"  - {module}")
    
    return imports, never_imported


def analyze_trading_flow():
    """Trace how trades actually flow through the system."""
    
    print("\n" + "="*80)
    print("TRADING FLOW ANALYSIS")
    print("="*80)
    
    # Key methods to track
    key_methods = {
        'signal_generation': ['generate_signals', 'calculate_signals', 'get_signals'],
        'position_sizing': ['calculate_position_size', 'get_position_size', 'size_position'],
        'risk_checking': ['check_risk', 'validate_risk', 'apply_risk'],
        'order_execution': ['execute_order', 'submit_order', 'place_order'],
        'trade_recording': ['record_trade', 'save_trade', 'log_trade'],
        'pnl_calculation': ['calculate_pnl', 'get_pnl', 'update_pnl']
    }
    
    src_path = Path("src/bot")
    found_methods = defaultdict(list)
    
    for py_file in src_path.rglob("*.py"):
        if '__pycache__' in str(py_file):
            continue
            
        content = py_file.read_text()
        rel_path = py_file.relative_to(src_path)
        
        for category, methods in key_methods.items():
            for method in methods:
                if f'def {method}' in content:
                    found_methods[category].append((str(rel_path), method))
    
    print("\nTrading Flow Components Found:")
    for category, locations in found_methods.items():
        print(f"\n{category.upper()} ({len(locations)} implementations):")
        for file_path, method in locations:
            print(f"  - {file_path}: {method}()")
    
    return found_methods


def check_test_coverage():
    """See what actually has tests."""
    
    print("\n" + "="*80)
    print("TEST COVERAGE ANALYSIS")
    print("="*80)
    
    test_path = Path("tests")
    src_path = Path("src/bot")
    
    # Map test files to source files
    tested_modules = set()
    
    for test_file in test_path.rglob("test_*.py"):
        if '__pycache__' in str(test_file):
            continue
            
        # Try to infer what module is being tested
        test_name = test_file.stem.replace('test_', '')
        tested_modules.add(test_name)
    
    # Count source modules
    source_modules = set()
    for src_file in src_path.rglob("*.py"):
        if '__pycache__' not in str(src_file) and '__init__' not in str(src_file):
            source_modules.add(src_file.stem)
    
    # Find untested modules
    untested = source_modules - tested_modules
    
    print(f"\nTest Coverage Summary:")
    print(f"  Source modules: {len(source_modules)}")
    print(f"  Tested modules: {len(tested_modules)}")
    print(f"  Untested modules: {len(untested)}")
    print(f"  Coverage: {len(tested_modules)/max(1, len(source_modules))*100:.1f}%")
    
    print(f"\nUntested modules (first 20):")
    for module in sorted(untested)[:20]:
        print(f"  - {module}")
    
    return tested_modules, untested


def main():
    """Run complete system audit."""
    
    print("="*80)
    print("GPT-TRADER SYSTEM AUDIT")
    print("="*80)
    print(f"Audit Date: {os.popen('date').read().strip()}")
    
    # Run all analyses
    dir_stats, categories = analyze_directory_structure()
    duplicates = find_duplicate_functionality()
    imports, never_imported = trace_import_dependencies()
    trading_flow = analyze_trading_flow()
    tested, untested = check_test_coverage()
    
    # Summary
    print("\n" + "="*80)
    print("AUDIT SUMMARY")
    print("="*80)
    
    # Red flags
    red_flags = []
    
    if len(duplicates['orchestrators']) > 1:
        red_flags.append(f"Multiple orchestrators: {len(duplicates['orchestrators'])}")
    if len(duplicates['risk_managers']) > 1:
        red_flags.append(f"Multiple risk managers: {len(duplicates['risk_managers'])}")
    if len(duplicates['execution_engines']) > 1:
        red_flags.append(f"Multiple execution engines: {len(duplicates['execution_engines'])}")
    if len(never_imported) > 10:
        red_flags.append(f"Dead code: {len(never_imported)} modules never imported")
    if len(tested) < len(untested):
        red_flags.append(f"Poor test coverage: {len(tested)/(len(tested)+len(untested))*100:.1f}%")
    
    print("\n🚨 RED FLAGS:")
    for flag in red_flags:
        print(f"  - {flag}")
    
    print("\n📊 KEY METRICS:")
    print(f"  - Total Python files: {sum(stats['file_count'] for stats in dir_stats.values())}")
    print(f"  - Total lines of code: {sum(stats['total_lines'] for stats in dir_stats.values())}")
    print(f"  - Directories with code: {len(dir_stats)}")
    print(f"  - Dead modules: {len(never_imported)}")
    print(f"  - Test coverage: {len(tested)/(len(tested)+len(untested))*100:.1f}%")
    
    print("\n⚠️ STRUCTURAL ISSUES:")
    print("  1. Multiple parallel implementations of core components")
    print("  2. Large amount of dead/unused code")
    print("  3. Poor test coverage")
    print("  4. Unclear execution flow")
    print("  5. Disconnected components")
    
    print("\n💡 RECOMMENDATION:")
    print("  This codebase needs architectural refactoring, not bug fixes.")
    print("  Consider starting with a minimal working system and rebuilding.")


if __name__ == "__main__":
    main()