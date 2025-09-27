"""
Implementation Gate - Validates code implementation against requirements and quality standards.

This gate ensures that:
1. Code meets functional requirements
2. Quality standards are maintained
3. Architecture patterns are followed
4. Security and performance requirements are met
"""

import ast
import os
import subprocess
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re


class ImplementationStatus(Enum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"


@dataclass
class QualityMetric:
    """Represents a quality metric measurement."""
    name: str
    value: float
    threshold: float
    status: ImplementationStatus
    description: str


@dataclass
class ImplementationResult:
    """Results of implementation validation."""
    component_id: str
    domain: str
    status: ImplementationStatus
    quality_metrics: List[QualityMetric]
    security_issues: List[str]
    performance_issues: List[str]
    architecture_violations: List[str]
    coverage_percentage: float
    test_results: Dict[str, Any]


class ImplementationGate:
    """Validates implementation quality and adherence to standards."""
    
    def __init__(self):
        self.quality_thresholds = {
            "cyclomatic_complexity": 10,
            "function_length": 50,
            "class_length": 300,
            "test_coverage": 80.0,
            "maintainability_index": 70.0
        }
        
        self.security_patterns = [
            r"eval\s*\(",  # Dangerous eval usage
            r"exec\s*\(",  # Dangerous exec usage
            r"__import__\s*\(",  # Dynamic imports
            r"pickle\.loads?\s*\(",  # Pickle security
            r"subprocess\..*shell\s*=\s*True",  # Shell injection
        ]
        
        self.architecture_rules = {
            "domain_isolation": "No cross-domain imports outside interfaces",
            "dependency_injection": "Use dependency injection for external services",
            "error_handling": "Proper error handling and logging",
            "configuration": "Configuration externalized from code"
        }
    
    def analyze_code_quality(self, file_path: str) -> List[QualityMetric]:
        """Analyze code quality metrics for a Python file."""
        metrics = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Analyze cyclomatic complexity
            complexity = self._calculate_complexity(tree)
            metrics.append(QualityMetric(
                name="cyclomatic_complexity",
                value=complexity,
                threshold=self.quality_thresholds["cyclomatic_complexity"],
                status=ImplementationStatus.PASS if complexity <= self.quality_thresholds["cyclomatic_complexity"] else ImplementationStatus.FAIL,
                description=f"Average cyclomatic complexity: {complexity}"
            ))
            
            # Analyze function lengths
            avg_function_length = self._calculate_avg_function_length(tree)
            metrics.append(QualityMetric(
                name="function_length",
                value=avg_function_length,
                threshold=self.quality_thresholds["function_length"],
                status=ImplementationStatus.PASS if avg_function_length <= self.quality_thresholds["function_length"] else ImplementationStatus.WARNING,
                description=f"Average function length: {avg_function_length} lines"
            ))
            
            # Analyze class lengths
            avg_class_length = self._calculate_avg_class_length(tree)
            if avg_class_length > 0:
                metrics.append(QualityMetric(
                    name="class_length",
                    value=avg_class_length,
                    threshold=self.quality_thresholds["class_length"],
                    status=ImplementationStatus.PASS if avg_class_length <= self.quality_thresholds["class_length"] else ImplementationStatus.WARNING,
                    description=f"Average class length: {avg_class_length} lines"
                ))
            
        except Exception as e:
            metrics.append(QualityMetric(
                name="parse_error",
                value=0,
                threshold=0,
                status=ImplementationStatus.FAIL,
                description=f"Failed to parse file: {str(e)}"
            ))
        
        return metrics
    
    def check_security_issues(self, file_path: str) -> List[str]:
        """Check for common security issues in code."""
        issues = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            for pattern in self.security_patterns:
                matches = re.finditer(pattern, content, re.MULTILINE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    issues.append(f"Line {line_num}: Potential security issue - {pattern}")
            
            # Check for hardcoded secrets
            secret_patterns = [
                r"password\s*=\s*['\"][^'\"]+['\"]",
                r"api_key\s*=\s*['\"][^'\"]+['\"]",
                r"secret\s*=\s*['\"][^'\"]+['\"]",
                r"token\s*=\s*['\"][^'\"]+['\"]"
            ]
            
            for pattern in secret_patterns:
                matches = re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    issues.append(f"Line {line_num}: Potential hardcoded secret")
            
        except Exception as e:
            issues.append(f"Failed to analyze security: {str(e)}")
        
        return issues
    
    def check_architecture_compliance(self, file_path: str, domain: str) -> List[str]:
        """Check compliance with architecture rules."""
        violations = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Check for cross-slice imports (bot_v2 architecture)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module and "src/bot_v2/features/" in str(node.module):
                        imported_slice = self._extract_slice_from_import(node.module)
                        current_slice = self._extract_slice_from_path(file_path)
                        if imported_slice and imported_slice != current_slice and not self._is_interface_import(node.module):
                            violations.append(f"Cross-slice import violation: importing from {imported_slice} slice")
            
            # Check for proper error handling
            function_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            for func in function_nodes:
                if not self._has_error_handling(func):
                    violations.append(f"Function '{func.name}' lacks proper error handling")
            
            # Check for configuration externalization
            if self._has_hardcoded_config(content):
                violations.append("Hardcoded configuration detected - should be externalized")
            
        except Exception as e:
            violations.append(f"Failed to analyze architecture compliance: {str(e)}")
        
        return violations
    
    def run_tests(self, component_path: str) -> Dict[str, Any]:
        """Run tests for a component and return results."""
        test_results = {
            "status": "unknown",
            "coverage": 0.0,
            "passed": 0,
            "failed": 0,
            "errors": []
        }
        
        try:
            # Look for test files
            test_files = []
            for root, dirs, files in os.walk(component_path):
                for file in files:
                    if file.startswith("test_") and file.endswith(".py"):
                        test_files.append(os.path.join(root, file))
            
            if not test_files:
                test_results["errors"].append("No test files found")
                return test_results
            
            # Run pytest with coverage
            cmd = ["python", "-m", "pytest", "--cov=" + component_path, "--cov-report=json"] + test_files
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode == 0:
                test_results["status"] = "passed"
            else:
                test_results["status"] = "failed"
                test_results["errors"].append(result.stderr)
            
            # Parse coverage report
            coverage_file = "coverage.json"
            if os.path.exists(coverage_file):
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                    test_results["coverage"] = coverage_data.get("totals", {}).get("percent_covered", 0.0)
                os.remove(coverage_file)  # Cleanup
            
        except Exception as e:
            test_results["errors"].append(f"Test execution failed: {str(e)}")
        
        return test_results
    
    def validate_component(self, component_path: str, component_id: str, domain: str) -> ImplementationResult:
        """Validate a complete component implementation."""
        quality_metrics = []
        security_issues = []
        architecture_violations = []
        test_results = {"status": "not_run"}
        coverage = 0.0
        
        # Analyze all Python files in component
        for root, dirs, files in os.walk(component_path):
            for file in files:
                if file.endswith('.py') and not file.startswith('test_'):
                    file_path = os.path.join(root, file)
                    
                    # Quality analysis
                    file_metrics = self.analyze_code_quality(file_path)
                    quality_metrics.extend(file_metrics)
                    
                    # Security analysis
                    file_security = self.check_security_issues(file_path)
                    security_issues.extend(file_security)
                    
                    # Architecture analysis
                    file_violations = self.check_architecture_compliance(file_path, domain)
                    architecture_violations.extend(file_violations)
        
        # Run tests
        test_results = self.run_tests(component_path)
        coverage = test_results.get("coverage", 0.0)
        
        # Determine overall status
        has_failures = any(m.status == ImplementationStatus.FAIL for m in quality_metrics)
        has_security_issues = len(security_issues) > 0
        has_architecture_violations = len(architecture_violations) > 0
        has_test_failures = test_results.get("status") != "passed"
        low_coverage = coverage < self.quality_thresholds["test_coverage"]
        
        if has_failures or has_security_issues or has_test_failures:
            status = ImplementationStatus.FAIL
        elif has_architecture_violations or low_coverage:
            status = ImplementationStatus.WARNING
        else:
            status = ImplementationStatus.PASS
        
        return ImplementationResult(
            component_id=component_id,
            domain=domain,
            status=status,
            quality_metrics=quality_metrics,
            security_issues=security_issues,
            performance_issues=[],  # Could be extended
            architecture_violations=architecture_violations,
            coverage_percentage=coverage,
            test_results=test_results
        )
    
    def generate_implementation_report(self, results: List[ImplementationResult]) -> str:
        """Generate comprehensive implementation validation report."""
        report = []
        report.append("# Implementation Gate Validation Report")
        
        # Summary
        total = len(results)
        passed = sum(1 for r in results if r.status == ImplementationStatus.PASS)
        warned = sum(1 for r in results if r.status == ImplementationStatus.WARNING)
        failed = total - passed - warned
        
        report.append(f"\n## Summary")
        report.append(f"- Total Components: {total}")
        report.append(f"- Passed: {passed}")
        report.append(f"- Warnings: {warned}")
        report.append(f"- Failed: {failed}")
        
        # Domain breakdown
        domain_stats = {}
        for result in results:
            domain = result.domain
            if domain not in domain_stats:
                domain_stats[domain] = {"total": 0, "passed": 0, "warned": 0, "failed": 0}
            
            domain_stats[domain]["total"] += 1
            if result.status == ImplementationStatus.PASS:
                domain_stats[domain]["passed"] += 1
            elif result.status == ImplementationStatus.WARNING:
                domain_stats[domain]["warned"] += 1
            else:
                domain_stats[domain]["failed"] += 1
        
        report.append("\n## Domain Breakdown")
        for domain, stats in domain_stats.items():
            report.append(f"### {domain}")
            report.append(f"- Total: {stats['total']}")
            report.append(f"- Passed: {stats['passed']}")
            report.append(f"- Warnings: {stats['warned']}")
            report.append(f"- Failed: {stats['failed']}")
        
        # Individual results
        report.append("\n## Component Details")
        for result in results:
            report.append(f"\n### {result.component_id} ({result.domain})")
            report.append(f"Status: {result.status.value.upper()}")
            report.append(f"Test Coverage: {result.coverage_percentage:.1f}%")
            
            if result.quality_metrics:
                report.append("\nQuality Metrics:")
                for metric in result.quality_metrics:
                    status_icon = "✅" if metric.status == ImplementationStatus.PASS else "❌" if metric.status == ImplementationStatus.FAIL else "⚠️"
                    report.append(f"- {status_icon} {metric.name}: {metric.value} (threshold: {metric.threshold})")
            
            if result.security_issues:
                report.append("\nSecurity Issues:")
                for issue in result.security_issues:
                    report.append(f"- ❌ {issue}")
            
            if result.architecture_violations:
                report.append("\nArchitecture Violations:")
                for violation in result.architecture_violations:
                    report.append(f"- ⚠️ {violation}")
        
        return "\n".join(report)
    
    def _calculate_complexity(self, tree: ast.AST) -> float:
        """Calculate average cyclomatic complexity."""
        complexities = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = self._function_complexity(node)
                complexities.append(complexity)
        
        return sum(complexities) / len(complexities) if complexities else 1.0
    
    def _function_complexity(self, func_node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.With):
                complexity += 1
        
        return complexity
    
    def _calculate_avg_function_length(self, tree: ast.AST) -> float:
        """Calculate average function length in lines."""
        lengths = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                length = node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 1
                lengths.append(length)
        
        return sum(lengths) / len(lengths) if lengths else 0
    
    def _calculate_avg_class_length(self, tree: ast.AST) -> float:
        """Calculate average class length in lines."""
        lengths = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                length = node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 1
                lengths.append(length)
        
        return sum(lengths) / len(lengths) if lengths else 0
    
    def _extract_slice_from_import(self, module: str) -> Optional[str]:
        """Extract slice name from import path."""
        if "src/bot_v2/features/" in module:
            parts = module.split("/")
            features_index = parts.index("features") + 1
            if features_index < len(parts):
                return parts[features_index]
        return None
    
    def _extract_slice_from_path(self, file_path: str) -> Optional[str]:
        """Extract slice name from file path."""
        if "src/bot_v2/features/" in file_path:
            parts = file_path.split("/")
            features_index = parts.index("features") + 1
            if features_index < len(parts):
                return parts[features_index]
        return None
    
    def _is_interface_import(self, module: str) -> bool:
        """Check if import is from an interface module."""
        return "interfaces" in module or "base" in module
    
    def _has_error_handling(self, func_node: ast.FunctionDef) -> bool:
        """Check if function has proper error handling."""
        for node in ast.walk(func_node):
            if isinstance(node, ast.Try):
                return True
        return False
    
    def _has_hardcoded_config(self, content: str) -> bool:
        """Check for hardcoded configuration values."""
        # Simple heuristic - look for common config patterns
        config_patterns = [
            r"host\s*=\s*['\"]localhost['\"]",
            r"port\s*=\s*\d{4,5}",
            r"url\s*=\s*['\"]https?://[^'\"]+['\"]"
        ]
        
        for pattern in config_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        return False


# Example usage
if __name__ == "__main__":
    gate = ImplementationGate()
    
    # Example validation (would normally validate actual component paths)
    print("Implementation Gate initialized")
    print(f"Quality thresholds: {gate.quality_thresholds}")
    print(f"Architecture rules: {gate.architecture_rules}")