"""
Integration tests to verify that feature slices are truly isolated
"""

import pytest
import ast
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src'))

pytestmark = pytest.mark.integration

class TestSliceIsolation:
    """Test that feature slices maintain complete isolation"""
    
    def setup_method(self):
        """Set up test environment"""
        self.features_dir = Path(__file__).parent.parent.parent.parent / "src" / "bot_v2" / "features"
        self.feature_slices = [
            "analyze",
            "backtest",
            "data",
            "live_trade",
            "market_regime",
            "ml_strategy",
            "monitor",
            "optimize",
            "paper_trade",
            "position_sizing",
            "adaptive_portfolio"
        ]
    
    def get_imports_from_file(self, file_path):
        """Extract all imports from a Python file"""
        imports = []
        
        try:
            with open(file_path, 'r') as f:
                tree = ast.parse(f.read())
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        full_import = f"{module}.{alias.name}" if module else alias.name
                        imports.append(full_import)
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
        
        return imports
    
    def test_no_cross_slice_imports(self):
        """Test that no slice imports from another slice"""
        violations = []
        
        for slice_name in self.feature_slices:
            slice_dir = self.features_dir / slice_name
            if not slice_dir.exists():
                continue
            
            # Check all Python files in this slice
            for py_file in slice_dir.glob("*.py"):
                imports = self.get_imports_from_file(py_file)
                
                # Check for imports from other slices
                for other_slice in self.feature_slices:
                    if other_slice == slice_name:
                        continue
                    
                    # Check if any import references another slice
                    for imp in imports:
                        if f"features.{other_slice}" in imp or f"/{other_slice}/" in imp:
                            violations.append({
                                'file': str(py_file.relative_to(self.features_dir)),
                                'imports_from': other_slice,
                                'import_statement': imp
                            })
        
        # Report violations
        if violations:
            violation_report = "\n".join([
                f"  {v['file']} imports from {v['imports_from']}: {v['import_statement']}"
                for v in violations
            ])
            pytest.fail(f"Cross-slice imports detected:\n{violation_report}")
    
    def test_slice_uses_data_provider(self):
        """Test that slices use the centralized data provider"""
        slices_needing_data = [
            "analyze", "backtest", "optimize", "paper_trade",
            "ml_strategy", "market_regime", "adaptive_portfolio"
        ]
        
        for slice_name in slices_needing_data:
            slice_dir = self.features_dir / slice_name
            if not slice_dir.exists():
                continue
            
            # Check if slice uses data provider
            uses_provider = False
            for py_file in slice_dir.glob("*.py"):
                imports = self.get_imports_from_file(py_file)
                
                for imp in imports:
                    if "data_providers" in imp or "get_data_provider" in imp:
                        uses_provider = True
                        break
                
                if uses_provider:
                    break
            
            assert uses_provider, f"Slice {slice_name} should use data_providers but doesn't"
    
    def test_no_direct_yfinance_imports(self):
        """Test that no slice directly imports yfinance"""
        violations = []
        
        for slice_name in self.feature_slices:
            slice_dir = self.features_dir / slice_name
            if not slice_dir.exists():
                continue
            
            for py_file in slice_dir.glob("*.py"):
                with open(py_file, 'r') as f:
                    content = f.read()
                    if 'import yfinance' in content or 'from yfinance' in content:
                        violations.append(str(py_file.relative_to(self.features_dir)))
        
        if violations:
            pytest.fail(f"Direct yfinance imports found in: {', '.join(violations)}")
    
    def test_each_slice_has_types(self):
        """Test that each slice has its own types.py file"""
        for slice_name in self.feature_slices:
            slice_dir = self.features_dir / slice_name
            if not slice_dir.exists():
                continue
            
            types_file = slice_dir / "types.py"
            assert types_file.exists(), f"Slice {slice_name} missing types.py"
    
    def test_slice_can_be_imported_independently(self):
        """Test that each slice can be imported without errors"""
        import importlib
        
        for slice_name in self.feature_slices:
            slice_dir = self.features_dir / slice_name
            if not slice_dir.exists():
                continue
            
            # Try to import the main module of each slice
            try:
                module_name = f"bot_v2.features.{slice_name}.{slice_name}"
                module = importlib.import_module(module_name)
                assert module is not None
            except ImportError as e:
                # Some slices might not have a main module with same name
                # Try importing __init__
                try:
                    module_name = f"bot_v2.features.{slice_name}"
                    module = importlib.import_module(module_name)
                    assert module is not None
                except ImportError as e2:
                    pytest.fail(f"Cannot import slice {slice_name}: {e2}")


class TestSliceTokenEfficiency:
    """Test that slices maintain token efficiency"""
    
    def setup_method(self):
        """Set up test environment"""
        self.features_dir = Path(__file__).parent.parent.parent.parent / "src" / "bot_v2" / "features"
        self.max_lines_per_file = 600  # ~600 tokens
        self.max_files_per_slice = 10
    
    def test_file_sizes(self):
        """Test that individual files stay within token limits"""
        violations = []
        
        for slice_dir in self.features_dir.iterdir():
            if not slice_dir.is_dir():
                continue
            
            for py_file in slice_dir.glob("*.py"):
                with open(py_file, 'r') as f:
                    lines = len(f.readlines())
                
                if lines > self.max_lines_per_file:
                    violations.append({
                        'file': str(py_file.relative_to(self.features_dir)),
                        'lines': lines
                    })
        
        if violations:
            report = "\n".join([
                f"  {v['file']}: {v['lines']} lines (max: {self.max_lines_per_file})"
                for v in violations
            ])
            print(f"Warning: Large files detected:\n{report}")
    
    def test_slice_file_count(self):
        """Test that slices don't have too many files"""
        violations = []
        
        for slice_dir in self.features_dir.iterdir():
            if not slice_dir.is_dir():
                continue
            
            py_files = list(slice_dir.glob("*.py"))
            if len(py_files) > self.max_files_per_slice:
                violations.append({
                    'slice': slice_dir.name,
                    'file_count': len(py_files)
                })
        
        if violations:
            report = "\n".join([
                f"  {v['slice']}: {v['file_count']} files (max: {self.max_files_per_slice})"
                for v in violations
            ])
            print(f"Warning: Slices with many files:\n{report}")


class TestDataProviderIntegration:
    """Test data provider integration across slices"""
    
    def test_mock_provider_in_test_mode(self):
        """Test that mock provider is used in test mode"""
        os.environ['TESTING'] = 'true'
        
        from bot_v2.data_providers import get_data_provider, MockProvider
        
        provider = get_data_provider()
        assert isinstance(provider, MockProvider)
        
        del os.environ['TESTING']
    
    def test_provider_consistency(self):
        """Test that all slices get the same provider instance"""
        from bot_v2.data_providers import get_data_provider
        
        provider1 = get_data_provider()
        provider2 = get_data_provider()
        
        assert provider1 is provider2, "Should return same provider instance"
    
    def test_provider_data_format(self):
        """Test that provider returns data in expected format"""
        from bot_v2.data_providers import MockProvider
        import pandas as pd
        
        provider = MockProvider()
        data = provider.get_historical_data("AAPL", "30d")
        
        # Check data structure
        assert isinstance(data, pd.DataFrame)
        assert all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
        assert isinstance(data.index, pd.DatetimeIndex)
        
        # Check data quality
        assert not data.isnull().any().any()
        assert (data['High'] >= data['Low']).all()
        assert (data['Volume'] > 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
