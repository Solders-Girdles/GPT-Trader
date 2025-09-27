#!/usr/bin/env python3
"""
Validate and auto-update the knowledge layer to prevent staleness.
Run this AFTER any changes to ensure knowledge stays current.
"""
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
import re

class KnowledgeValidator:
    def __init__(self):
        self.issues = []
        self.updates_made = []
        
    def validate_all(self):
        """Run all validation checks."""
        print("🔍 Validating Knowledge Layer...")
        
        self.check_project_state_freshness()
        self.check_for_new_report_files()
        self.check_test_map_accuracy()
        self.check_dependencies_consistency()
        self.check_known_failures_relevance()
        self.auto_update_state()
        
        return self.report_results()
    
    def check_project_state_freshness(self):
        """Ensure PROJECT_STATE.json isn't stale."""
        with open("PROJECT_STATE.json") as f:
            state = json.load(f)
        
        # Check if verification is older than 24 hours
        last_run = state["system_status"].get("last_test_run")
        if last_run:
            last_time = datetime.fromisoformat(last_run)
            if datetime.now() - last_time > timedelta(hours=24):
                self.issues.append("PROJECT_STATE.json verification is >24 hours old")
                print("  ⚠️  State verification is stale (>24 hours)")
        
        # Check for unverified components
        unverified = [k for k, v in state["components"].items() 
                     if not v.get("verified")]
        if unverified:
            self.issues.append(f"Unverified components: {unverified}")
            print(f"  ⚠️  Unverified components: {unverified}")
    
    def check_for_new_report_files(self):
        """Detect and flag new report files."""
        forbidden_patterns = [
            "*_REPORT.md",
            "*_COMPLETE.md",
            "*_STATUS.md",
            "*_IMPLEMENTATION.md",
            "*_SUMMARY.md",
            "*_PROGRESS.md"
        ]
        
        found_reports = []
        for pattern in forbidden_patterns:
            files = list(Path(".").glob(pattern))
            found_reports.extend(files)
        
        if found_reports:
            self.issues.append(f"Found {len(found_reports)} forbidden report files")
            print(f"  ❌ Found forbidden reports: {[str(f) for f in found_reports]}")
            print("     These should be deleted and info moved to PROJECT_STATE.json")
    
    def check_test_map_accuracy(self):
        """Verify TEST_MAP.json points to real test files."""
        with open("TEST_MAP.json") as f:
            test_map = json.load(f)
        
        missing_tests = []
        for component, info in test_map["test_mapping"].items():
            for test_file in info.get("unit_tests", []):
                if not Path(test_file).exists():
                    missing_tests.append(test_file)
        
        if missing_tests:
            self.issues.append(f"TEST_MAP references {len(missing_tests)} missing tests")
            print(f"  ⚠️  Missing tests in TEST_MAP: {missing_tests[:3]}...")
    
    def check_dependencies_consistency(self):
        """Ensure DEPENDENCIES.json matches PROJECT_STATE.json."""
        with open("DEPENDENCIES.json") as f:
            deps = json.load(f)
        with open("PROJECT_STATE.json") as f:
            state = json.load(f)
        
        # Check all components in dependencies exist in state
        for comp in deps["component_dependencies"]:
            if comp not in state["components"] and comp != "portfolio":
                self.issues.append(f"Component '{comp}' in DEPENDENCIES but not PROJECT_STATE")
        
        # Check dependency info matches
        for comp, info in state["components"].items():
            if comp in deps["component_dependencies"]:
                dep_info = deps["component_dependencies"][comp]
                if info.get("depends_on") != dep_info.get("depends_on", []):
                    self.issues.append(f"Dependency mismatch for {comp}")
    
    def check_known_failures_relevance(self):
        """Check if solutions in KNOWN_FAILURES still work."""
        # This is harder to automate, but we can check structure
        known_failures = Path("KNOWN_FAILURES.md")
        if known_failures.exists():
            content = known_failures.read_text()
            
            # Check for TODO markers that indicate incomplete solutions
            if "TODO" in content or "FIXME" in content:
                self.issues.append("KNOWN_FAILURES.md contains TODO/FIXME markers")
            
            # Check last modified date
            mtime = datetime.fromtimestamp(known_failures.stat().st_mtime)
            if datetime.now() - mtime > timedelta(days=30):
                print("  ℹ️  KNOWN_FAILURES.md hasn't been updated in 30+ days")
    
    def auto_update_state(self):
        """Auto-update PROJECT_STATE.json with current info."""
        with open("PROJECT_STATE.json", "r") as f:
            state = json.load(f)
        
        # Update last_updated
        old_date = state["last_updated"]
        state["last_updated"] = datetime.now().strftime("%Y-%m-%d")
        
        if old_date != state["last_updated"]:
            self.updates_made.append("Updated last_updated timestamp")
        
        # Check for components with very old verification
        for comp, info in state["components"].items():
            last_verified = info.get("last_verified")
            if last_verified:
                verified_time = datetime.fromisoformat(last_verified)
                if datetime.now() - verified_time > timedelta(days=7):
                    info["verified"] = False
                    self.updates_made.append(f"Marked {comp} as unverified (>7 days old)")
        
        # Save updates
        if self.updates_made:
            with open("PROJECT_STATE.json", "w") as f:
                json.dump(state, f, indent=2)
            print(f"  ✅ Auto-updated PROJECT_STATE.json ({len(self.updates_made)} changes)")
    
    def report_results(self):
        """Report validation results."""
        print("\n" + "="*50)
        print("KNOWLEDGE LAYER VALIDATION RESULTS")
        print("="*50)
        
        if not self.issues:
            print("✅ Knowledge layer is current and consistent!")
            return True
        else:
            print(f"⚠️  Found {len(self.issues)} issues:\n")
            for i, issue in enumerate(self.issues, 1):
                print(f"  {i}. {issue}")
            
            print("\n📋 Required Actions:")
            print("  1. Run: poetry run python scripts/verify_capabilities.py")
            print("  2. Update PROJECT_STATE.json with results")
            print("  3. Delete any report files found")
            print("  4. Update TEST_MAP.json if tests moved")
            
            return False

def update_after_change(component: str, status: str, error: str = None):
    """
    Helper to update knowledge after fixing a component.
    Call this from agents after making changes.
    """
    with open("PROJECT_STATE.json", "r") as f:
        state = json.load(f)
    
    if component in state["components"]:
        old_status = state["components"][component]["status"]
        state["components"][component]["status"] = status
        state["components"][component]["verified"] = True
        state["components"][component]["last_verified"] = datetime.now().isoformat()
        
        if error and error not in state["components"][component].get("known_issues", []):
            state["components"][component].setdefault("known_issues", []).append(error)
        
        with open("PROJECT_STATE.json", "w") as f:
            json.dump(state, f, indent=2)
        
        print(f"✅ Updated {component}: {old_status} → {status}")
        
        # Add to KNOWN_FAILURES if new error
        if error:
            add_to_known_failures(component, error, f"Fix: {status}")

def add_to_known_failures(component: str, error: str, solution: str):
    """Add a new error/solution to KNOWN_FAILURES.md."""
    with open("KNOWN_FAILURES.md", "a") as f:
        f.write(f"\n### Error: `{error}` in {component}\n")
        f.write(f"**Added**: {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"**Solution:**\n```\n{solution}\n```\n")
    print(f"📝 Added new solution to KNOWN_FAILURES.md")

if __name__ == "__main__":
    validator = KnowledgeValidator()
    success = validator.validate_all()
    
    # Check for report files to delete
    import glob
    reports = glob.glob("*_REPORT.md") + glob.glob("*_COMPLETE.md")
    if reports:
        print(f"\n🗑️  Delete these report files:")
        for r in reports:
            print(f"  rm {r}")
    
    sys.exit(0 if success else 1)