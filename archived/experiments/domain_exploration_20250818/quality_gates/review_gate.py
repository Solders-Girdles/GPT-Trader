"""
Review Gate - Validates code review process and quality standards.

This gate ensures that:
1. Code changes are properly reviewed
2. Review feedback is addressed
3. Knowledge transfer occurs during reviews
4. Review quality standards are maintained
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import re


class ReviewStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    APPROVED = "approved"
    REJECTED = "rejected"
    REQUIRES_CHANGES = "requires_changes"


class ReviewerRole(Enum):
    DOMAIN_EXPERT = "domain_expert"
    SECURITY_REVIEWER = "security_reviewer"
    ARCHITECTURE_REVIEWER = "architecture_reviewer"
    GENERAL_REVIEWER = "general_reviewer"


@dataclass
class ReviewComment:
    """Represents a review comment."""
    id: str
    reviewer: str
    content: str
    line_number: Optional[int]
    severity: str  # info, warning, error, blocker
    addressed: bool = False
    resolution: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CodeReview:
    """Represents a code review session."""
    id: str
    component_id: str
    domain: str
    author: str
    reviewers: List[str]
    status: ReviewStatus
    comments: List[ReviewComment]
    files_changed: List[str]
    lines_added: int
    lines_removed: int
    created_at: datetime
    completed_at: Optional[datetime] = None
    approval_threshold: int = 2  # Minimum approvals needed


class ReviewGate:
    """Validates code review process and enforces quality standards."""
    
    def __init__(self):
        self.review_requirements = {
            "min_reviewers": 2,
            "require_domain_expert": True,
            "max_review_time_hours": 48,
            "require_security_review_for_domains": {
                "trading_execution", "risk_management", "infrastructure"
            },
            "max_files_per_review": 10,
            "max_lines_per_review": 500
        }
        
        self.severity_thresholds = {
            "blocker": 0,  # No blockers allowed
            "error": 2,    # Max 2 errors
            "warning": 5   # Max 5 warnings
        }
        
        self.domain_experts = {
            "ml_intelligence": ["ml-strategy-director", "feature-engineer", "model-trainer"],
            "trading_execution": ["trading-ops-lead", "paper-trade-manager", "live-trade-operator"],
            "risk_management": ["risk-analyst", "compliance-officer"],
            "data_pipeline": ["data-pipeline-engineer", "market-data-specialist"],
            "infrastructure": ["devops-lead", "deployment-engineer", "monitoring-specialist"]
        }
    
    def validate_review_setup(self, review: CodeReview) -> Dict[str, Any]:
        """Validate that a review is properly set up."""
        issues = []
        warnings = []
        
        # Check minimum reviewers
        if len(review.reviewers) < self.review_requirements["min_reviewers"]:
            issues.append(f"Insufficient reviewers: {len(review.reviewers)} < {self.review_requirements['min_reviewers']}")
        
        # Check for domain expert
        if self.review_requirements["require_domain_expert"]:
            domain_experts = self.domain_experts.get(review.domain, [])
            has_domain_expert = any(reviewer in domain_experts for reviewer in review.reviewers)
            if not has_domain_expert:
                issues.append(f"No domain expert assigned for {review.domain}")
        
        # Check for security reviewer if required
        if review.domain in self.review_requirements["require_security_review_for_domains"]:
            has_security_reviewer = any("security" in reviewer.lower() for reviewer in review.reviewers)
            if not has_security_reviewer:
                warnings.append(f"Security review recommended for {review.domain}")
        
        # Check change size
        if len(review.files_changed) > self.review_requirements["max_files_per_review"]:
            warnings.append(f"Large change: {len(review.files_changed)} files (consider splitting)")
        
        total_lines = review.lines_added + review.lines_removed
        if total_lines > self.review_requirements["max_lines_per_review"]:
            warnings.append(f"Large change: {total_lines} lines (consider splitting)")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings
        }
    
    def validate_review_quality(self, review: CodeReview) -> Dict[str, Any]:
        """Validate the quality of review comments and feedback."""
        issues = []
        warnings = []
        metrics = {}
        
        # Count comments by severity
        severity_counts = {"blocker": 0, "error": 0, "warning": 0, "info": 0}
        unaddressed_comments = []
        
        for comment in review.comments:
            severity_counts[comment.severity] += 1
            if not comment.addressed and comment.severity in ["blocker", "error"]:
                unaddressed_comments.append(comment)
        
        # Check severity thresholds
        for severity, count in severity_counts.items():
            threshold = self.severity_thresholds.get(severity)
            if threshold is not None and count > threshold:
                if severity == "blocker":
                    issues.append(f"Blocking issues must be resolved: {count} blockers")
                else:
                    issues.append(f"Too many {severity}s: {count} > {threshold}")
        
        # Check unaddressed critical comments
        if unaddressed_comments:
            issues.append(f"{len(unaddressed_comments)} critical comments unaddressed")
        
        # Check review timing
        if review.completed_at:
            review_duration = (review.completed_at - review.created_at).total_seconds() / 3600
            max_hours = self.review_requirements["max_review_time_hours"]
            if review_duration > max_hours:
                warnings.append(f"Long review cycle: {review_duration:.1f} hours > {max_hours}")
        
        # Calculate review coverage
        total_comments = len(review.comments)
        substantive_comments = sum(1 for c in review.comments if len(c.content) > 20)
        coverage_ratio = substantive_comments / max(total_comments, 1)
        
        metrics.update({
            "total_comments": total_comments,
            "substantive_comments": substantive_comments,
            "coverage_ratio": coverage_ratio,
            "severity_distribution": severity_counts,
            "unaddressed_critical": len(unaddressed_comments)
        })
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "metrics": metrics
        }
    
    def validate_review_completion(self, review: CodeReview) -> Dict[str, Any]:
        """Validate that review is properly completed."""
        issues = []
        warnings = []
        
        if review.status != ReviewStatus.APPROVED:
            if review.status == ReviewStatus.PENDING:
                issues.append("Review not started")
            elif review.status == ReviewStatus.IN_PROGRESS:
                issues.append("Review not completed")
            elif review.status == ReviewStatus.REJECTED:
                issues.append("Review rejected - changes required")
            elif review.status == ReviewStatus.REQUIRES_CHANGES:
                issues.append("Review requires changes")
        
        # Check that all blockers and errors are addressed
        unresolved_critical = [
            c for c in review.comments 
            if c.severity in ["blocker", "error"] and not c.addressed
        ]
        
        if unresolved_critical:
            issues.append(f"{len(unresolved_critical)} critical issues unresolved")
        
        # Check approval count
        approvals = self._count_approvals(review)
        if approvals < review.approval_threshold:
            issues.append(f"Insufficient approvals: {approvals} < {review.approval_threshold}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "approvals": approvals
        }
    
    def analyze_review_patterns(self, reviews: List[CodeReview]) -> Dict[str, Any]:
        """Analyze patterns across multiple reviews."""
        if not reviews:
            return {"error": "No reviews to analyze"}
        
        # Time analysis
        review_times = []
        for review in reviews:
            if review.completed_at:
                duration = (review.completed_at - review.created_at).total_seconds() / 3600
                review_times.append(duration)
        
        # Comment analysis
        total_comments = sum(len(r.comments) for r in reviews)
        avg_comments_per_review = total_comments / len(reviews)
        
        # Domain distribution
        domain_counts = {}
        for review in reviews:
            domain_counts[review.domain] = domain_counts.get(review.domain, 0) + 1
        
        # Reviewer participation
        reviewer_counts = {}
        for review in reviews:
            for reviewer in review.reviewers:
                reviewer_counts[reviewer] = reviewer_counts.get(reviewer, 0) + 1
        
        # Quality trends
        approval_rate = sum(1 for r in reviews if r.status == ReviewStatus.APPROVED) / len(reviews)
        
        return {
            "total_reviews": len(reviews),
            "avg_review_time_hours": sum(review_times) / len(review_times) if review_times else 0,
            "avg_comments_per_review": avg_comments_per_review,
            "approval_rate": approval_rate,
            "domain_distribution": domain_counts,
            "top_reviewers": sorted(reviewer_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            "review_time_trend": review_times[-10:] if len(review_times) >= 10 else review_times
        }
    
    def suggest_reviewers(self, component_id: str, domain: str, files_changed: List[str]) -> List[str]:
        """Suggest appropriate reviewers for a change."""
        suggested = []
        
        # Always include domain experts
        domain_experts = self.domain_experts.get(domain, [])
        suggested.extend(domain_experts[:2])  # Max 2 domain experts
        
        # Add security reviewer for sensitive domains
        if domain in self.review_requirements["require_security_review_for_domains"]:
            suggested.append("security-reviewer")
        
        # Add architecture reviewer for large changes
        if len(files_changed) > 5:
            suggested.append("tech-lead-orchestrator")
        
        # Remove duplicates while preserving order
        seen = set()
        result = []
        for reviewer in suggested:
            if reviewer not in seen:
                seen.add(reviewer)
                result.append(reviewer)
        
        return result
    
    def generate_review_report(self, review: CodeReview) -> str:
        """Generate comprehensive review report."""
        setup_validation = self.validate_review_setup(review)
        quality_validation = self.validate_review_quality(review)
        completion_validation = self.validate_review_completion(review)
        
        report = []
        report.append(f"# Code Review Report: {review.id}")
        report.append(f"Component: {review.component_id} ({review.domain})")
        report.append(f"Author: {review.author}")
        report.append(f"Status: {review.status.value}")
        
        # Summary
        report.append("\n## Summary")
        report.append(f"- Files changed: {len(review.files_changed)}")
        report.append(f"- Lines added: {review.lines_added}")
        report.append(f"- Lines removed: {review.lines_removed}")
        report.append(f"- Total comments: {len(review.comments)}")
        report.append(f"- Reviewers: {', '.join(review.reviewers)}")
        
        # Validation results
        validations = {
            "Setup": setup_validation,
            "Quality": quality_validation,
            "Completion": completion_validation
        }
        
        for section_name, validation in validations.items():
            report.append(f"\n## {section_name} Validation")
            status = "✅ PASS" if validation["valid"] else "❌ FAIL"
            report.append(f"Status: {status}")
            
            if validation.get("issues"):
                report.append("\nIssues:")
                for issue in validation["issues"]:
                    report.append(f"- ❌ {issue}")
            
            if validation.get("warnings"):
                report.append("\nWarnings:")
                for warning in validation["warnings"]:
                    report.append(f"- ⚠️ {warning}")
        
        # Comment analysis
        if review.comments:
            report.append("\n## Comment Analysis")
            severity_counts = {"blocker": 0, "error": 0, "warning": 0, "info": 0}
            for comment in review.comments:
                severity_counts[comment.severity] += 1
            
            for severity, count in severity_counts.items():
                if count > 0:
                    report.append(f"- {severity.capitalize()}: {count}")
            
            unaddressed = [c for c in review.comments if not c.addressed and c.severity in ["blocker", "error"]]
            if unaddressed:
                report.append(f"\nUnaddressed Critical Comments: {len(unaddressed)}")
                for comment in unaddressed[:5]:  # Show first 5
                    report.append(f"- {comment.severity.upper()}: {comment.content[:100]}...")
        
        return "\n".join(report)
    
    def _count_approvals(self, review: CodeReview) -> int:
        """Count explicit approvals in review."""
        # In a real implementation, this would check for explicit approval comments
        # For now, we'll use a simple heuristic
        if review.status == ReviewStatus.APPROVED:
            return len(review.reviewers)
        return 0


# Example usage and testing
if __name__ == "__main__":
    # Example review
    review = CodeReview(
        id="REV-001",
        component_id="ML-001",
        domain="ml_intelligence",
        author="backend-developer",
        reviewers=["ml-strategy-director", "feature-engineer"],
        status=ReviewStatus.APPROVED,
        comments=[
            ReviewComment(
                id="C1",
                reviewer="ml-strategy-director",
                content="Consider adding input validation for edge cases",
                line_number=45,
                severity="warning",
                addressed=True,
                resolution="Added validation checks"
            ),
            ReviewComment(
                id="C2",
                reviewer="feature-engineer",
                content="Good implementation of feature engineering pipeline",
                line_number=None,
                severity="info",
                addressed=True
            )
        ],
        files_changed=["strategy_selection.py", "model.py"],
        lines_added=150,
        lines_removed=20,
        created_at=datetime.now() - timedelta(hours=12),
        completed_at=datetime.now()
    )
    
    gate = ReviewGate()
    
    print("Review Setup Validation:")
    setup_result = gate.validate_review_setup(review)
    print(f"Valid: {setup_result['valid']}")
    
    print("\nReview Quality Validation:")
    quality_result = gate.validate_review_quality(review)
    print(f"Valid: {quality_result['valid']}")
    
    print("\nReview Completion Validation:")
    completion_result = gate.validate_review_completion(review)
    print(f"Valid: {completion_result['valid']}")
    
    print("\nSuggested Reviewers for new change:")
    suggestions = gate.suggest_reviewers("TRADE-001", "trading_execution", ["order_management.py"])
    print(f"Suggestions: {suggestions}")
    
    print("\nFull Review Report:")
    report = gate.generate_review_report(review)
    print(report)