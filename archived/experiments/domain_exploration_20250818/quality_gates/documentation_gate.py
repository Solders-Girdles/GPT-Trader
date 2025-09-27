"""
Documentation Gate - Validates documentation quality and completeness.

This gate ensures that:
1. All components have proper documentation
2. Documentation is up-to-date with implementation
3. API documentation is complete and accurate
4. Knowledge transfer documentation exists
"""

import os
import re
import ast
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class DocumentationType(Enum):
    README = "readme"
    API = "api"
    ARCHITECTURAL = "architectural"
    USER_GUIDE = "user_guide"
    TROUBLESHOOTING = "troubleshooting"
    CHANGELOG = "changelog"


class DocumentationStatus(Enum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    MISSING = "missing"
    OUTDATED = "outdated"


@dataclass
class DocumentationItem:
    """Represents a documentation item."""
    name: str
    type: DocumentationType
    path: str
    status: DocumentationStatus
    last_modified: datetime
    word_count: int
    coverage_score: float
    issues: List[str]


@dataclass
class APIDocumentation:
    """Represents API documentation completeness."""
    module_path: str
    total_functions: int
    documented_functions: int
    total_classes: int
    documented_classes: int
    missing_docstrings: List[str]
    incomplete_docstrings: List[str]


class DocumentationGate:
    """Validates documentation quality and completeness."""
    
    def __init__(self):
        self.required_docs = {
            DocumentationType.README: "README.md",
            DocumentationType.API: "API.md",
            DocumentationType.ARCHITECTURAL: "ARCHITECTURE.md"
        }
        
        self.quality_thresholds = {
            "min_word_count": 100,
            "api_coverage": 80.0,
            "docstring_coverage": 90.0,
            "freshness_days": 30
        }
        
        self.documentation_patterns = {
            "function_signature": r"def\s+(\w+)\s*\(",
            "class_definition": r"class\s+(\w+)\s*[\(:]",
            "docstring": r'"""[\s\S]*?"""',
            "api_endpoint": r"@app\.route\s*\(\s*['\"]([^'\"]+)['\"]",
            "parameter_doc": r":param\s+(\w+):",
            "return_doc": r":return[s]?:",
            "example_code": r"```python[\s\S]*?```"
        }
    
    def analyze_api_documentation(self, source_path: str) -> APIDocumentation:
        """Analyze API documentation completeness for Python modules."""
        total_functions = 0
        documented_functions = 0
        total_classes = 0
        documented_classes = 0
        missing_docstrings = []
        incomplete_docstrings = []
        
        for root, dirs, files in os.walk(source_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        tree = ast.parse(content)
                        
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef):
                                total_functions += 1
                                if self._has_docstring(node):
                                    if self._is_complete_docstring(node, content):
                                        documented_functions += 1
                                    else:
                                        incomplete_docstrings.append(f"{file}:{node.name}")
                                else:
                                    missing_docstrings.append(f"{file}:{node.name}")
                            
                            elif isinstance(node, ast.ClassDef):
                                total_classes += 1
                                if self._has_docstring(node):
                                    if self._is_complete_docstring(node, content):
                                        documented_classes += 1
                                    else:
                                        incomplete_docstrings.append(f"{file}:{node.name}")
                                else:
                                    missing_docstrings.append(f"{file}:{node.name}")
                    
                    except Exception as e:
                        missing_docstrings.append(f"{file}: Parse error - {str(e)}")
        
        return APIDocumentation(
            module_path=source_path,
            total_functions=total_functions,
            documented_functions=documented_functions,
            total_classes=total_classes,
            documented_classes=documented_classes,
            missing_docstrings=missing_docstrings,
            incomplete_docstrings=incomplete_docstrings
        )
    
    def validate_documentation_structure(self, component_path: str) -> Dict[str, Any]:
        """Validate that required documentation files exist."""
        issues = []
        warnings = []
        found_docs = {}
        
        # Check for required documentation files
        for doc_type, filename in self.required_docs.items():
            file_path = os.path.join(component_path, filename)
            if os.path.exists(file_path):
                found_docs[doc_type] = file_path
            else:
                issues.append(f"Missing required documentation: {filename}")
        
        # Check for additional documentation
        additional_docs = []
        for root, dirs, files in os.walk(component_path):
            for file in files:
                if file.endswith(('.md', '.rst', '.txt')) and file not in self.required_docs.values():
                    additional_docs.append(os.path.join(root, file))
        
        # Check documentation organization
        docs_dir = os.path.join(component_path, "docs")
        if additional_docs and not os.path.exists(docs_dir):
            warnings.append("Consider organizing documentation in a 'docs' directory")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "found_docs": found_docs,
            "additional_docs": additional_docs
        }
    
    def analyze_documentation_quality(self, doc_path: str) -> DocumentationItem:
        """Analyze the quality of a documentation file."""
        issues = []
        
        if not os.path.exists(doc_path):
            return DocumentationItem(
                name=os.path.basename(doc_path),
                type=self._infer_doc_type(doc_path),
                path=doc_path,
                status=DocumentationStatus.MISSING,
                last_modified=datetime.min,
                word_count=0,
                coverage_score=0.0,
                issues=["File does not exist"]
            )
        
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic metrics
            word_count = len(content.split())
            last_modified = datetime.fromtimestamp(os.path.getmtime(doc_path))
            
            # Quality checks
            if word_count < self.quality_thresholds["min_word_count"]:
                issues.append(f"Documentation too brief: {word_count} words < {self.quality_thresholds['min_word_count']}")
            
            # Check for common documentation elements
            coverage_score = self._calculate_coverage_score(content, doc_path)
            
            # Check freshness
            days_old = (datetime.now() - last_modified).days
            if days_old > self.quality_thresholds["freshness_days"]:
                issues.append(f"Documentation may be outdated: {days_old} days old")
            
            # Check for broken links
            broken_links = self._check_broken_links(content, doc_path)
            if broken_links:
                issues.extend([f"Broken link: {link}" for link in broken_links])
            
            # Determine status
            if issues:
                if word_count == 0:
                    status = DocumentationStatus.MISSING
                elif coverage_score < 0.5:
                    status = DocumentationStatus.INCOMPLETE
                else:
                    status = DocumentationStatus.OUTDATED
            else:
                status = DocumentationStatus.COMPLETE
            
            return DocumentationItem(
                name=os.path.basename(doc_path),
                type=self._infer_doc_type(doc_path),
                path=doc_path,
                status=status,
                last_modified=last_modified,
                word_count=word_count,
                coverage_score=coverage_score,
                issues=issues
            )
        
        except Exception as e:
            return DocumentationItem(
                name=os.path.basename(doc_path),
                type=self._infer_doc_type(doc_path),
                path=doc_path,
                status=DocumentationStatus.INCOMPLETE,
                last_modified=datetime.min,
                word_count=0,
                coverage_score=0.0,
                issues=[f"Error reading file: {str(e)}"]
            )
    
    def validate_component_documentation(self, component_path: str, component_id: str) -> Dict[str, Any]:
        """Validate all documentation for a component."""
        structure_validation = self.validate_documentation_structure(component_path)
        api_documentation = self.analyze_api_documentation(component_path)
        
        # Analyze individual documentation files
        doc_analyses = []
        for doc_type, doc_path in structure_validation["found_docs"].items():
            analysis = self.analyze_documentation_quality(doc_path)
            doc_analyses.append(analysis)
        
        # Calculate overall scores
        api_coverage = 0.0
        if api_documentation.total_functions + api_documentation.total_classes > 0:
            total_items = api_documentation.total_functions + api_documentation.total_classes
            documented_items = api_documentation.documented_functions + api_documentation.documented_classes
            api_coverage = (documented_items / total_items) * 100
        
        # Determine overall status
        structure_valid = structure_validation["valid"]
        api_sufficient = api_coverage >= self.quality_thresholds["api_coverage"]
        docs_complete = all(doc.status == DocumentationStatus.COMPLETE for doc in doc_analyses)
        
        overall_status = "complete" if structure_valid and api_sufficient and docs_complete else "incomplete"
        
        return {
            "component_id": component_id,
            "status": overall_status,
            "structure_validation": structure_validation,
            "api_documentation": api_documentation,
            "api_coverage": api_coverage,
            "documentation_files": doc_analyses,
            "overall_score": self._calculate_overall_score(doc_analyses, api_coverage)
        }
    
    def generate_documentation_report(self, validations: List[Dict[str, Any]]) -> str:
        """Generate comprehensive documentation validation report."""
        report = []
        report.append("# Documentation Gate Validation Report")
        
        # Summary
        total_components = len(validations)
        complete_components = sum(1 for v in validations if v["status"] == "complete")
        
        report.append(f"\n## Summary")
        report.append(f"- Total Components: {total_components}")
        report.append(f"- Complete Documentation: {complete_components}")
        report.append(f"- Completion Rate: {(complete_components/total_components)*100:.1f}%")
        
        # API Coverage Summary
        total_api_items = sum(
            v["api_documentation"].total_functions + v["api_documentation"].total_classes 
            for v in validations
        )
        total_documented = sum(
            v["api_documentation"].documented_functions + v["api_documentation"].documented_classes 
            for v in validations
        )
        overall_api_coverage = (total_documented / total_api_items * 100) if total_api_items > 0 else 0
        
        report.append(f"\n## API Documentation Coverage")
        report.append(f"- Overall Coverage: {overall_api_coverage:.1f}%")
        report.append(f"- Total API Items: {total_api_items}")
        report.append(f"- Documented Items: {total_documented}")
        
        # Component Details
        report.append("\n## Component Documentation Status")
        
        for validation in validations:
            component_id = validation["component_id"]
            status = validation["status"]
            api_coverage = validation["api_coverage"]
            overall_score = validation["overall_score"]
            
            status_icon = "✅" if status == "complete" else "❌"
            report.append(f"\n### {component_id} {status_icon}")
            report.append(f"- Status: {status}")
            report.append(f"- API Coverage: {api_coverage:.1f}%")
            report.append(f"- Overall Score: {overall_score:.1f}/100")
            
            # Structure issues
            structure = validation["structure_validation"]
            if structure["issues"]:
                report.append("- Structure Issues:")
                for issue in structure["issues"]:
                    report.append(f"  - ❌ {issue}")
            
            # Documentation files
            for doc in validation["documentation_files"]:
                doc_status_icon = {
                    DocumentationStatus.COMPLETE: "✅",
                    DocumentationStatus.INCOMPLETE: "⚠️",
                    DocumentationStatus.MISSING: "❌",
                    DocumentationStatus.OUTDATED: "🕐"
                }.get(doc.status, "❓")
                
                report.append(f"- {doc.name} {doc_status_icon}")
                report.append(f"  - Words: {doc.word_count}")
                report.append(f"  - Coverage: {doc.coverage_score:.1f}")
                
                if doc.issues:
                    for issue in doc.issues[:3]:  # Show first 3 issues
                        report.append(f"  - Issue: {issue}")
        
        # Recommendations
        report.append("\n## Recommendations")
        
        incomplete_components = [v for v in validations if v["status"] != "complete"]
        if incomplete_components:
            report.append("### Priority Actions")
            for validation in incomplete_components[:5]:  # Top 5 issues
                component_id = validation["component_id"]
                
                # Find most critical issues
                critical_issues = []
                if not validation["structure_validation"]["valid"]:
                    critical_issues.extend(validation["structure_validation"]["issues"])
                
                if validation["api_coverage"] < self.quality_thresholds["api_coverage"]:
                    critical_issues.append(f"Low API coverage: {validation['api_coverage']:.1f}%")
                
                report.append(f"- **{component_id}**: {critical_issues[0] if critical_issues else 'General improvements needed'}")
        
        # Best Practices
        report.append("\n### Best Practices")
        report.append("- Maintain API coverage above 80%")
        report.append("- Update documentation with code changes")
        report.append("- Include examples in API documentation")
        report.append("- Organize additional docs in 'docs' directory")
        report.append("- Use consistent documentation templates")
        
        return "\n".join(report)
    
    def suggest_documentation_improvements(self, validation: Dict[str, Any]) -> List[str]:
        """Suggest specific improvements for component documentation."""
        suggestions = []
        
        # Structure improvements
        structure = validation["structure_validation"]
        if structure["issues"]:
            suggestions.append("Add missing required documentation files")
        
        # API improvements
        api_coverage = validation["api_coverage"]
        if api_coverage < self.quality_thresholds["api_coverage"]:
            suggestions.append(f"Improve API documentation coverage (currently {api_coverage:.1f}%)")
        
        api_doc = validation["api_documentation"]
        if api_doc.missing_docstrings:
            suggestions.append(f"Add docstrings to {len(api_doc.missing_docstrings)} functions/classes")
        
        if api_doc.incomplete_docstrings:
            suggestions.append(f"Complete docstrings for {len(api_doc.incomplete_docstrings)} functions/classes")
        
        # Documentation quality improvements
        for doc in validation["documentation_files"]:
            if doc.status != DocumentationStatus.COMPLETE:
                if doc.word_count < self.quality_thresholds["min_word_count"]:
                    suggestions.append(f"Expand {doc.name} documentation")
                
                if doc.coverage_score < 0.7:
                    suggestions.append(f"Improve coverage in {doc.name}")
        
        return suggestions
    
    def _infer_doc_type(self, file_path: str) -> DocumentationType:
        """Infer documentation type from file path."""
        filename = os.path.basename(file_path).lower()
        
        if "readme" in filename:
            return DocumentationType.README
        elif "api" in filename:
            return DocumentationType.API
        elif "architecture" in filename or "arch" in filename:
            return DocumentationType.ARCHITECTURAL
        elif "user" in filename or "guide" in filename:
            return DocumentationType.USER_GUIDE
        elif "troubleshoot" in filename or "faq" in filename:
            return DocumentationType.TROUBLESHOOTING
        elif "changelog" in filename or "history" in filename:
            return DocumentationType.CHANGELOG
        else:
            return DocumentationType.README
    
    def _has_docstring(self, node: ast.AST) -> bool:
        """Check if a function or class has a docstring."""
        if (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and 
            node.body and 
            isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Constant) and 
            isinstance(node.body[0].value.value, str)):
            return True
        return False
    
    def _is_complete_docstring(self, node: ast.AST, content: str) -> bool:
        """Check if docstring is complete (has params and return info for functions)."""
        if not self._has_docstring(node):
            return False
        
        docstring = node.body[0].value.value
        
        # For functions, check for parameter and return documentation
        if isinstance(node, ast.FunctionDef):
            has_params = len(node.args.args) <= 1  # Only 'self' parameter
            has_return = node.returns is None
            
            if not has_params:
                # Should document parameters
                if ":param" not in docstring and "Args:" not in docstring:
                    return False
            
            if not has_return:
                # Should document return value
                if ":return" not in docstring and "Returns:" not in docstring:
                    return False
        
        # Check for minimum content (more than just a one-liner)
        return len(docstring.strip()) > 20
    
    def _calculate_coverage_score(self, content: str, file_path: str) -> float:
        """Calculate documentation coverage score based on expected elements."""
        score = 0.0
        max_score = 100.0
        
        # Expected elements for different document types
        doc_type = self._infer_doc_type(file_path)
        
        if doc_type == DocumentationType.README:
            expected_sections = ["description", "installation", "usage", "example"]
            section_score = 25.0
        elif doc_type == DocumentationType.API:
            expected_sections = ["endpoint", "parameter", "response", "example"]
            section_score = 25.0
        else:
            expected_sections = ["overview", "detail", "example"]
            section_score = 33.3
        
        # Check for expected sections
        content_lower = content.lower()
        for section in expected_sections:
            if section in content_lower:
                score += section_score
        
        return min(score, max_score)
    
    def _check_broken_links(self, content: str, doc_path: str) -> List[str]:
        """Check for broken internal links in documentation."""
        broken_links = []
        
        # Find markdown links
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        matches = re.findall(link_pattern, content)
        
        base_dir = os.path.dirname(doc_path)
        
        for link_text, link_url in matches:
            # Only check relative file links
            if not link_url.startswith(('http://', 'https://', 'mailto:')):
                if link_url.startswith('#'):
                    # Internal anchor - skip for now
                    continue
                
                # Resolve relative path
                full_path = os.path.join(base_dir, link_url)
                if not os.path.exists(full_path):
                    broken_links.append(link_url)
        
        return broken_links
    
    def _calculate_overall_score(self, doc_analyses: List[DocumentationItem], api_coverage: float) -> float:
        """Calculate overall documentation score."""
        if not doc_analyses:
            return api_coverage
        
        # Average documentation scores
        doc_scores = [doc.coverage_score for doc in doc_analyses]
        avg_doc_score = sum(doc_scores) / len(doc_scores)
        
        # Weighted combination: 60% doc quality, 40% API coverage
        overall_score = (avg_doc_score * 0.6) + (api_coverage * 0.4)
        
        return overall_score


# Example usage
if __name__ == "__main__":
    gate = DocumentationGate()
    
    print("Documentation Gate initialized")
    print(f"Quality thresholds: {gate.quality_thresholds}")
    print(f"Required documentation: {gate.required_docs}")
    
    # Example validation (would normally validate actual component paths)
    # validation = gate.validate_component_documentation("/path/to/component", "COMP-001")
    # print("Component validation complete")