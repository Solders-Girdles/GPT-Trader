#!/usr/bin/env python3
"""
CLAUDE.md Update Script
Helps maintain CLAUDE.md as the single source of truth for project status.
"""

import datetime
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional
import argparse

class ClaudeMDUpdater:
    """Manages updates to CLAUDE.md file"""
    
    def __init__(self, claude_md_path: str = "CLAUDE.md"):
        self.claude_md_path = Path(claude_md_path)
        self.content = self._read_claude_md()
        self.today = datetime.date.today().isoformat()
        
    def _read_claude_md(self) -> str:
        """Read current CLAUDE.md content"""
        if not self.claude_md_path.exists():
            raise FileNotFoundError(f"CLAUDE.md not found at {self.claude_md_path}")
        return self.claude_md_path.read_text()
    
    def _write_claude_md(self) -> None:
        """Write updated content back to CLAUDE.md"""
        self.claude_md_path.write_text(self.content)
        print(f"✅ Updated CLAUDE.md successfully")
    
    def update_date(self) -> None:
        """Update the date in Current Sprint Status"""
        self.content = re.sub(
            r'\*\*Date\*\*: \d{4}-\d{2}-\d{2}',
            f'**Date**: {self.today}',
            self.content
        )
        print(f"📅 Updated date to {self.today}")
    
    def mark_task_complete(self, task_id: str) -> None:
        """Mark a specific task as complete"""
        # Try to find and mark the task
        pattern = rf'- \[ \] (.*{re.escape(task_id)}.*)'
        if re.search(pattern, self.content):
            self.content = re.sub(pattern, r'- [x] \1', self.content)
            print(f"✅ Marked {task_id} as complete")
        else:
            print(f"⚠️  Task {task_id} not found or already complete")
    
    def add_issue(self, severity: str, description: str, impact: str) -> None:
        """Add a new issue to the Active Issues section"""
        issue_template = f"""
### Issue #{self._get_next_issue_number()}: {description}
**Discovered:** {self.today}
**Severity:** {severity}
**Impact:** {impact}

#### Problem
[To be detailed]

#### Proposed Solution
[To be determined]

#### Resolution Status
- [ ] Solution designed
- [ ] Tasks added to roadmap
- [ ] Implementation started
- [ ] Testing complete
- [ ] Issue resolved
"""
        
        # Find the right section based on severity
        section = "High Priority Issues" if severity in ["Critical", "High"] else "Medium Priority Issues"
        
        # Insert the issue
        insertion_point = f"### {section}\n"
        if insertion_point in self.content:
            insert_pos = self.content.find(insertion_point) + len(insertion_point)
            # Find the next line after the comment
            next_line = self.content.find('\n', insert_pos) + 1
            self.content = self.content[:next_line] + issue_template + self.content[next_line:]
            print(f"🚨 Added new {severity} issue: {description}")
        else:
            print(f"⚠️  Could not find {section} section")
    
    def _get_next_issue_number(self) -> int:
        """Get the next available issue number"""
        issues = re.findall(r'Issue #(\d+):', self.content)
        if issues:
            return max(int(i) for i in issues) + 1
        return 1
    
    def update_current_focus(self, focus: str, next_step: str) -> None:
        """Update today's focus and next step"""
        # Update Today's Focus
        self.content = re.sub(
            r"### Today's Focus:.*?\n",
            f"### Today's Focus: {focus}\n",
            self.content
        )
        
        # Update Next Step
        self.content = re.sub(
            r'\*\*Next Step\*\*:.*?\n',
            f'**Next Step**: {next_step}\n',
            self.content
        )
        
        print(f"🎯 Updated focus: {focus}")
        print(f"➡️  Next step: {next_step}")
    
    def add_decision(self, decision: str, rationale: str, alternatives: str = "") -> None:
        """Add a new decision to the Key Decisions Log"""
        decision_template = f"""
### {self.today}: {decision}
**Decision**: {decision}
**Rationale**: {rationale}
**Alternatives**: {alternatives if alternatives else "N/A"}
"""
        
        # Find the Key Decisions Log section
        section_marker = "## 🎯 Key Decisions Log"
        if section_marker in self.content:
            # Insert after the section header
            insert_pos = self.content.find(section_marker)
            next_section = self.content.find("\n## ", insert_pos + 1)
            if next_section == -1:
                next_section = len(self.content)
            
            # Find the first decision entry or end of section
            first_decision = self.content.find("\n### ", insert_pos, next_section)
            if first_decision != -1:
                # Insert before the first decision (newest first)
                self.content = self.content[:first_decision] + decision_template + self.content[first_decision:]
            else:
                # No decisions yet, add after section header
                insert_after_header = self.content.find("\n", insert_pos) + 1
                self.content = self.content[:insert_after_header] + decision_template + self.content[insert_after_header:]
            
            print(f"📝 Added decision: {decision}")
    
    def update_metrics(self, metrics: Dict[str, str]) -> None:
        """Update success metrics"""
        for metric, value in metrics.items():
            pattern = rf'(- \[ \] {re.escape(metric)})[^\n]*'
            if value.lower() == "complete":
                replacement = rf'- [x] {metric}'
            else:
                replacement = rf'\1: {value}'
            
            if re.search(pattern, self.content):
                self.content = re.sub(pattern, replacement, self.content)
                print(f"📊 Updated metric: {metric} = {value}")
    
    def daily_summary(self, completed_tasks: List[str], 
                     tomorrow_priorities: List[str],
                     blockers: Optional[List[str]] = None) -> None:
        """Generate and update daily summary"""
        
        # Update completed tasks
        for task in completed_tasks:
            self.mark_task_complete(task)
        
        # Update tomorrow's priorities section
        priorities_text = "\n".join(f"{i+1}. {p}" for i, p in enumerate(tomorrow_priorities))
        
        self.content = re.sub(
            r"### Tomorrow's Priorities\n(?:.*?\n)*?(?=\n#|\Z)",
            f"### Tomorrow's Priorities\n{priorities_text}\n\n",
            self.content,
            flags=re.DOTALL
        )
        
        print(f"📝 Updated tomorrow's priorities ({len(tomorrow_priorities)} items)")
        
        # Add blockers if any
        if blockers:
            blockers_text = "\n".join(f"- {b}" for b in blockers)
            self.content = re.sub(
                r"### Blockers\n(?:.*?\n)*?(?=\n#|\Z)",
                f"### Blockers\n{blockers_text}\n\n",
                self.content,
                flags=re.DOTALL
            )
            print(f"⚠️  Added {len(blockers)} blockers")

def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description="Update CLAUDE.md with project status")
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Task completion
    task_parser = subparsers.add_parser('task', help='Mark task as complete')
    task_parser.add_argument('task_id', help='Task ID (e.g., MON-001)')
    
    # Add issue
    issue_parser = subparsers.add_parser('issue', help='Add new issue')
    issue_parser.add_argument('severity', choices=['Critical', 'High', 'Medium', 'Low'])
    issue_parser.add_argument('description', help='Brief issue description')
    issue_parser.add_argument('--impact', default='TBD', help='Impact description')
    
    # Update focus
    focus_parser = subparsers.add_parser('focus', help='Update current focus')
    focus_parser.add_argument('focus', help='Current focus description')
    focus_parser.add_argument('next_step', help='Next step description')
    
    # Add decision
    decision_parser = subparsers.add_parser('decision', help='Log a decision')
    decision_parser.add_argument('decision', help='Decision made')
    decision_parser.add_argument('rationale', help='Why this decision')
    decision_parser.add_argument('--alternatives', default='', help='Alternatives considered')
    
    # Daily update
    daily_parser = subparsers.add_parser('daily', help='Daily summary update')
    daily_parser.add_argument('--completed', nargs='+', default=[], help='Completed task IDs')
    daily_parser.add_argument('--tomorrow', nargs='+', default=[], help='Tomorrow priorities')
    daily_parser.add_argument('--blockers', nargs='+', default=[], help='Current blockers')
    
    # Update date
    subparsers.add_parser('date', help='Update date to today')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize updater
    updater = ClaudeMDUpdater()
    
    # Execute command
    if args.command == 'task':
        updater.mark_task_complete(args.task_id)
    elif args.command == 'issue':
        updater.add_issue(args.severity, args.description, args.impact)
    elif args.command == 'focus':
        updater.update_current_focus(args.focus, args.next_step)
    elif args.command == 'decision':
        updater.add_decision(args.decision, args.rationale, args.alternatives)
    elif args.command == 'daily':
        updater.daily_summary(args.completed, args.tomorrow, args.blockers)
    elif args.command == 'date':
        updater.update_date()
    
    # Save changes
    updater._write_claude_md()

if __name__ == "__main__":
    main()