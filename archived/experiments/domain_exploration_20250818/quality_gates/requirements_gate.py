"""
Requirements Gate - Validates domain requirements before implementation.

This gate ensures that:
1. Requirements are well-defined and measurable
2. Domain boundaries are respected
3. Dependencies are explicitly documented
4. Quality criteria are established
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json


class RequirementStatus(Enum):
    DRAFT = "draft"
    APPROVED = "approved"
    REJECTED = "rejected"
    UNDER_REVIEW = "under_review"


class Priority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class DomainRequirement:
    """Represents a requirement within a specific domain."""
    id: str
    domain: str
    title: str
    description: str
    acceptance_criteria: List[str]
    dependencies: List[str]
    priority: Priority
    status: RequirementStatus
    estimated_effort: int  # Story points
    quality_gates: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "domain": self.domain,
            "title": self.title,
            "description": self.description,
            "acceptance_criteria": self.acceptance_criteria,
            "dependencies": self.dependencies,
            "priority": self.priority.value,
            "status": self.status.value,
            "estimated_effort": self.estimated_effort,
            "quality_gates": self.quality_gates
        }


class RequirementsGate:
    """Validates requirements against domain architecture and quality standards."""
    
    def __init__(self):
        self.valid_domains = {
            "ml_intelligence", "trading_execution", "risk_management", 
            "data_pipeline", "infrastructure"
        }
        self.required_quality_gates = [
            "unit_tests", "integration_tests", "documentation", 
            "security_review", "performance_validation"
        ]
    
    def validate_requirement(self, requirement: DomainRequirement) -> Dict[str, Any]:
        """Validate a single requirement against quality standards."""
        errors = []
        warnings = []
        
        # Validate domain
        if requirement.domain not in self.valid_domains:
            errors.append(f"Invalid domain: {requirement.domain}")
        
        # Validate acceptance criteria
        if not requirement.acceptance_criteria:
            errors.append("Acceptance criteria cannot be empty")
        elif len(requirement.acceptance_criteria) < 2:
            warnings.append("Consider adding more detailed acceptance criteria")
        
        # Validate quality gates
        missing_gates = set(self.required_quality_gates) - set(requirement.quality_gates)
        if missing_gates:
            warnings.append(f"Missing recommended quality gates: {list(missing_gates)}")
        
        # Validate effort estimation
        if requirement.estimated_effort <= 0:
            errors.append("Effort estimation must be positive")
        elif requirement.estimated_effort > 13:
            warnings.append("Large effort estimation - consider breaking down the requirement")
        
        # Validate description quality
        if len(requirement.description) < 50:
            warnings.append("Description seems too brief - consider adding more detail")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "requirement_id": requirement.id
        }
    
    def validate_requirements_set(self, requirements: List[DomainRequirement]) -> Dict[str, Any]:
        """Validate a set of requirements for consistency and completeness."""
        results = []
        domain_distribution = {}
        priority_distribution = {}
        
        for req in requirements:
            # Validate individual requirement
            validation = self.validate_requirement(req)
            results.append(validation)
            
            # Track distribution
            domain_distribution[req.domain] = domain_distribution.get(req.domain, 0) + 1
            priority_distribution[req.priority.value] = priority_distribution.get(req.priority.value, 0) + 1
        
        # Check for dependency cycles
        dependency_issues = self._check_dependency_cycles(requirements)
        
        # Check for balanced distribution
        balance_warnings = []
        if len(domain_distribution) == 1:
            balance_warnings.append("All requirements in single domain - consider cross-domain integration")
        
        critical_count = priority_distribution.get("critical", 0)
        total_count = len(requirements)
        if critical_count / total_count > 0.5:
            balance_warnings.append("High proportion of critical requirements - consider prioritization review")
        
        return {
            "individual_results": results,
            "domain_distribution": domain_distribution,
            "priority_distribution": priority_distribution,
            "dependency_issues": dependency_issues,
            "balance_warnings": balance_warnings,
            "overall_valid": all(r["valid"] for r in results) and not dependency_issues
        }
    
    def _check_dependency_cycles(self, requirements: List[DomainRequirement]) -> List[str]:
        """Check for circular dependencies in requirements."""
        # Build dependency graph
        graph = {}
        for req in requirements:
            graph[req.id] = req.dependencies
        
        # Simple cycle detection
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node, path):
            if node in rec_stack:
                cycle_start = path.index(node)
                cycles.append(" -> ".join(path[cycle_start:] + [node]))
                return
            
            if node in visited or node not in graph:
                return
            
            visited.add(node)
            rec_stack.add(node)
            
            for dependency in graph[node]:
                dfs(dependency, path + [node])
            
            rec_stack.remove(node)
        
        for req_id in graph:
            if req_id not in visited:
                dfs(req_id, [])
        
        return cycles
    
    def generate_requirements_report(self, requirements: List[DomainRequirement]) -> str:
        """Generate a comprehensive requirements validation report."""
        validation = self.validate_requirements_set(requirements)
        
        report = []
        report.append("# Requirements Gate Validation Report")
        report.append(f"\nTotal Requirements: {len(requirements)}")
        report.append(f"Overall Status: {'PASS' if validation['overall_valid'] else 'FAIL'}")
        
        # Domain distribution
        report.append("\n## Domain Distribution")
        for domain, count in validation["domain_distribution"].items():
            report.append(f"- {domain}: {count}")
        
        # Priority distribution
        report.append("\n## Priority Distribution")
        for priority, count in validation["priority_distribution"].items():
            report.append(f"- {priority}: {count}")
        
        # Issues
        if validation["dependency_issues"]:
            report.append("\n## Dependency Issues")
            for issue in validation["dependency_issues"]:
                report.append(f"- Cycle detected: {issue}")
        
        if validation["balance_warnings"]:
            report.append("\n## Balance Warnings")
            for warning in validation["balance_warnings"]:
                report.append(f"- {warning}")
        
        # Individual results
        report.append("\n## Individual Requirement Validation")
        for result in validation["individual_results"]:
            req_id = result["requirement_id"]
            status = "PASS" if result["valid"] else "FAIL"
            report.append(f"\n### {req_id}: {status}")
            
            if result["errors"]:
                report.append("Errors:")
                for error in result["errors"]:
                    report.append(f"- {error}")
            
            if result["warnings"]:
                report.append("Warnings:")
                for warning in result["warnings"]:
                    report.append(f"- {warning}")
        
        return "\n".join(report)
    
    def save_requirements(self, requirements: List[DomainRequirement], filepath: str):
        """Save requirements to JSON file."""
        data = {
            "requirements": [req.to_dict() for req in requirements],
            "validation_timestamp": str(datetime.now()),
            "validation_version": "1.0"
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# Example usage and testing
if __name__ == "__main__":
    import datetime
    
    # Example requirements
    example_requirements = [
        DomainRequirement(
            id="ML-001",
            domain="ml_intelligence",
            title="Strategy Selection Model",
            description="Implement ML model for dynamic strategy selection based on market conditions and historical performance",
            acceptance_criteria=[
                "Model achieves >70% accuracy on historical data",
                "Response time < 100ms for real-time predictions",
                "Confidence scores provided for all predictions"
            ],
            dependencies=[],
            priority=Priority.CRITICAL,
            status=RequirementStatus.APPROVED,
            estimated_effort=8,
            quality_gates=["unit_tests", "integration_tests", "documentation", "performance_validation"]
        ),
        DomainRequirement(
            id="TRADE-001",
            domain="trading_execution",
            title="Order Management System",
            description="Core order management system with support for multiple order types and risk checks",
            acceptance_criteria=[
                "Support for market, limit, and stop orders",
                "Pre-trade risk validation",
                "Order state tracking and audit trail"
            ],
            dependencies=["ML-001"],
            priority=Priority.CRITICAL,
            status=RequirementStatus.APPROVED,
            estimated_effort=13,
            quality_gates=["unit_tests", "integration_tests", "documentation", "security_review"]
        )
    ]
    
    gate = RequirementsGate()
    validation = gate.validate_requirements_set(example_requirements)
    
    print("Validation Results:")
    print(f"Overall Valid: {validation['overall_valid']}")
    print(f"Domain Distribution: {validation['domain_distribution']}")
    print(f"Priority Distribution: {validation['priority_distribution']}")
    
    report = gate.generate_requirements_report(example_requirements)
    print("\n" + report)