# CLAUDE.md Maintenance Workflow

**Purpose:** Establish systematic processes for keeping CLAUDE.md as the single source of truth throughout GPT-Trader's development lifecycle.  
**Last Updated:** 2025-08-14  
**Status:** Active Process Document  

## 🎯 Why CLAUDE.md Matters

CLAUDE.md serves as:
1. **Context Provider**: Instant orientation for every Claude Code session
2. **Progress Tracker**: Current phase, tasks, and metrics
3. **Issue Logger**: Active problems and their solutions
4. **Workflow Guide**: Standard procedures and templates
5. **Decision Record**: Key architectural and strategic choices

## 📋 Update Triggers

### Daily Updates (End of Day)
```bash
# Daily update checklist
- [ ] Tasks completed today (mark in CLAUDE.md)
- [ ] New issues discovered
- [ ] Tomorrow's priority tasks
- [ ] Any blockers encountered
```

### Weekly Updates (Friday)
```bash
# Weekly review checklist
- [ ] Week's accomplishments
- [ ] Next week's focus
- [ ] Metrics update
- [ ] Issue resolution status
- [ ] Roadmap adjustments
```

### Event-Driven Updates
- After completing a major milestone
- When discovering critical issues
- After making architectural decisions
- When priorities change
- After team meetings/reviews

## 🔄 Standard Update Workflows

### Workflow 1: Issue Discovery & Planning

```markdown
## 🚨 Active Issues

### Issue #[NUMBER]: [Brief Description]
**Discovered:** [Date]
**Severity:** Critical | High | Medium | Low
**Impact:** [What's affected]

#### Problem
[Detailed description of the issue]

#### Proposed Solution
[Planned approach to fix]

#### Tasks Required
- [ ] [Task ID]: [Description] (Est: Xh)
- [ ] [Task ID]: [Description] (Est: Xh)

#### Integration Plan
- Fits into Week [X] of Phase [Y]
- Dependencies: [List any]
- Testing approach: [Brief description]

#### Resolution Status
- [ ] Solution designed
- [ ] Tasks added to roadmap
- [ ] Implementation started
- [ ] Testing complete
- [ ] Issue resolved
```

### Workflow 2: Task Completion

```markdown
## ✅ Recent Completions

### [Date]: [Component Name]
**Task IDs:** MON-001, MON-002
**Time Taken:** Estimated 4h, Actual 3.5h
**Key Changes:**
- Implemented [feature]
- Fixed [issue]
- Improved [metric]

**Lessons Learned:**
- [What worked well]
- [What could improve]

**Next Steps:**
- Continue with [Task ID]
- Test [component]
```

### Workflow 3: Metric Tracking

```markdown
## 📊 Current Metrics

### Week [X] Performance
| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Autonomy Level | 70% | 45% | 🟡 In Progress |
| Human Tasks/Day | 5 | 12 | 🔴 Needs Work |
| System Uptime | 99.95% | 99.97% | 🟢 Achieved |
| Model Accuracy | 60% | 58% | 🟡 Close |

### Trends
- Autonomy improving +5% per week
- Alert volume decreasing as expected
- Performance stable
```

## 📝 CLAUDE.md Section Templates

### 1. Current Sprint Status
```markdown
## 🏃 Current Sprint: Phase 3, Week 2

### This Week's Focus: A/B Testing Framework
- [x] MON-009: Traffic splitting mechanism
- [x] MON-010: Performance metric collection
- [ ] MON-011: Statistical significance testing <- CURRENT
- [ ] MON-012: Auto-promotion logic

### Today's Priority
Working on statistical significance testing for A/B framework.
Need to implement t-test and chi-square test with proper p-value calculation.

### Blockers
- None currently

### Tomorrow's Plan
- Complete MON-011
- Start MON-012
- Review test coverage
```

### 2. Issue Tracking Section
```markdown
## 🐛 Active Issues & Solutions

### High Priority
1. **Memory leak in online learning** (Issue #045)
   - Solution: Implement circular buffer
   - Tasks: ADAPT-045, ADAPT-046
   - ETA: 2 days

### Medium Priority
1. **Dashboard performance with >1000 metrics** (Issue #046)
   - Solution: Add pagination and caching
   - Tasks: OPS-047, OPS-048
   - ETA: Next sprint

### Resolved This Week
- ✅ Database connection pool exhaustion (Issue #044)
```

### 3. Decision Log
```markdown
## 🎯 Key Decisions

### 2025-08-14: Statistical Test Choice
**Decision:** Use Welch's t-test instead of Student's t-test
**Reason:** Handles unequal variances better
**Impact:** More robust A/B testing
**Alternatives Considered:** Mann-Whitney U, Permutation tests

### 2025-08-13: Alert Bundling Window
**Decision:** 60-second bundling window
**Reason:** Balance between responsiveness and noise reduction
**Impact:** 50% reduction in alert volume
```

## 🔧 Automation Scripts

### Daily Update Script
```python
#!/usr/bin/env python3
# update_claude_md.py

import datetime
import re

def update_claude_md():
    """Update CLAUDE.md with daily progress"""
    
    # Read current CLAUDE.md
    with open('CLAUDE.md', 'r') as f:
        content = f.read()
    
    # Update date
    today = datetime.date.today().isoformat()
    content = re.sub(
        r'Last Updated: \d{4}-\d{2}-\d{2}',
        f'Last Updated: {today}',
        content
    )
    
    # Get completed tasks from git commits
    completed_tasks = get_completed_tasks_from_git()
    
    # Update task checklist
    for task in completed_tasks:
        content = content.replace(f'- [ ] {task}', f'- [x] {task}')
    
    # Add today's summary
    summary = generate_daily_summary()
    content = add_section(content, "Daily Progress", summary)
    
    # Write updated content
    with open('CLAUDE.md', 'w') as f:
        f.write(content)
    
    print(f"✅ CLAUDE.md updated for {today}")

def get_completed_tasks_from_git():
    """Extract task IDs from today's commits"""
    # Implementation here
    pass

def generate_daily_summary():
    """Generate summary of today's work"""
    # Implementation here
    pass

if __name__ == "__main__":
    update_claude_md()
```

### Issue Integration Script
```python
#!/usr/bin/env python3
# integrate_issue.py

def add_issue_to_roadmap(issue_id, description, tasks, priority="medium"):
    """Add new issue and its tasks to the roadmap"""
    
    # Read current roadmap
    with open('docs/PHASE_3_TASK_BREAKDOWN.md', 'r') as f:
        roadmap = f.read()
    
    # Find appropriate week based on priority
    week = determine_week_for_priority(priority)
    
    # Generate new task IDs
    new_tasks = generate_task_ids(tasks, week)
    
    # Insert tasks into roadmap
    roadmap = insert_tasks_into_week(roadmap, week, new_tasks)
    
    # Update CLAUDE.md with issue
    update_claude_md_with_issue(issue_id, description, new_tasks)
    
    print(f"✅ Issue #{issue_id} integrated into Week {week}")
    print(f"📝 New tasks: {', '.join(new_tasks)}")

def determine_week_for_priority(priority):
    """Determine which week to add tasks based on priority"""
    if priority == "critical":
        return "current"
    elif priority == "high":
        return "next"
    else:
        return "backlog"
```

## 📊 Workflow Decision Tree

```
New Information Arrives
├── Is it an issue?
│   ├── Yes → Use Issue Discovery Workflow
│   │   ├── Log in Active Issues section
│   │   ├── Create solution plan
│   │   ├── Generate tasks
│   │   └── Integrate into roadmap
│   └── No → Continue
│
├── Is it a completion?
│   ├── Yes → Use Task Completion Workflow
│   │   ├── Mark tasks complete
│   │   ├── Log lessons learned
│   │   └── Update metrics
│   └── No → Continue
│
├── Is it a decision?
│   ├── Yes → Use Decision Log
│   │   ├── Document decision
│   │   ├── Record rationale
│   │   └── Note alternatives
│   └── No → Continue
│
└── Is it a metric update?
    ├── Yes → Update Metrics Section
    └── No → Add to Notes
```

## 🎯 Best Practices

### DO's
1. **Update immediately** when discovering issues
2. **Be specific** with task IDs and estimates
3. **Link related items** (issues → tasks → completions)
4. **Include context** for future Claude sessions
5. **Track metrics** consistently
6. **Document decisions** with rationale
7. **Review weekly** for accuracy

### DON'Ts
1. **Don't delay updates** - stale info hurts productivity
2. **Don't be vague** - "various fixes" isn't helpful
3. **Don't delete history** - archive instead
4. **Don't skip metrics** - trends matter
5. **Don't forget dependencies** - note what blocks what

## 📝 CLAUDE.md Structure Template

```markdown
# Claude Code Assistant Guide for GPT-Trader

## 🎯 Current Focus
[Immediate priority and context]

## 📊 Current Status
[Phase, week, and progress]

## 🏃 Active Sprint
[Current tasks and today's work]

## 🚨 Active Issues
[Problems being worked on]

## ✅ Recent Completions
[Last 5 completed items]

## 📈 Metrics Dashboard
[Key performance indicators]

## 🔧 Quick Commands
[Frequently used commands]

## 📚 Key Files
[Important files to review]

## 🎯 Decisions Log
[Recent architectural decisions]

## 📝 Notes
[Miscellaneous important info]
```

## 🔄 Integration with Development Cycle

### Morning Routine
1. Read CLAUDE.md for context
2. Check Active Issues section
3. Review today's tasks
4. Note any overnight alerts/issues

### During Development
1. Update task status as you work
2. Log issues immediately when found
3. Document decisions as they're made
4. Track time for estimates

### End of Day
1. Run daily update script
2. Mark completed tasks
3. Add tomorrow's priorities
4. Update metrics if changed

### Weekly Review
1. Analyze metrics trends
2. Adjust roadmap if needed
3. Archive completed issues
4. Plan next week's focus

## 🚀 Advanced Workflows

### Workflow 4: Performance Regression
```markdown
## ⚠️ Performance Regression Detected

### Regression: Model accuracy dropped 5%
**Detected:** 2025-08-14, 14:30 UTC
**Baseline:** 62% accuracy
**Current:** 57% accuracy

#### Investigation
- [ ] Check recent feature changes
- [ ] Review data quality
- [ ] Analyze prediction distribution
- [ ] Check for concept drift

#### Root Cause
[To be determined]

#### Remediation Plan
1. Rollback to previous model (immediate)
2. Investigate root cause (1 day)
3. Fix and retrain (2 days)
4. Validate fix (1 day)

#### Prevention
- Add regression tests
- Enhance monitoring
- Implement gradual rollout
```

### Workflow 5: Roadmap Adjustment
```markdown
## 📅 Roadmap Adjustment

### Change Request: Prioritize Risk Monitoring
**Reason:** Regulatory requirement
**Impact:** Delay adaptive learning by 1 week

#### Original Plan
- Week 3-4: Risk Monitoring
- Week 5-6: Adaptive Learning

#### Adjusted Plan
- Week 2-3: Risk Monitoring (moved up)
- Week 4: Integration & Testing
- Week 5-6: Adaptive Learning

#### Tasks Affected
- RISK-* tasks move to Week 2
- ADAPT-* tasks move to Week 5
- New integration tasks added

#### Stakeholder Approval
- [ ] Product Owner
- [ ] Tech Lead
- [ ] Risk Team
```

## 📊 Metrics for Workflow Effectiveness

Track these to ensure the workflow is working:

| Metric | Target | Measurement |
|--------|--------|-------------|
| Update Frequency | Daily | Git commits to CLAUDE.md |
| Issue Resolution Time | <3 days | Issue created → resolved |
| Task Estimation Accuracy | ±20% | Estimated vs actual time |
| Context Switching | <2/day | How often priorities change |
| Documentation Completeness | 100% | Decisions with rationale |

## 🎉 Success Indicators

You know the workflow is working when:
1. **Every Claude session starts productively** - no time wasted on context
2. **Issues are caught early** - before they become critical
3. **Progress is measurable** - clear metrics and trends
4. **Team is aligned** - everyone knows current state
5. **Decisions are traceable** - can explain why things were done

## 🔧 Tooling Integration

### Git Hooks
```bash
# .git/hooks/pre-commit
#!/bin/bash
# Remind to update CLAUDE.md

echo "Did you update CLAUDE.md with:"
echo "  - Completed tasks?"
echo "  - New issues discovered?"
echo "  - Important decisions?"
echo ""
echo "Press Enter to continue or Ctrl+C to cancel and update"
read
```

### VS Code Snippets
```json
{
  "CLAUDE Issue": {
    "prefix": "claude-issue",
    "body": [
      "### Issue #${1:number}: ${2:description}",
      "**Discovered:** ${CURRENT_YEAR}-${CURRENT_MONTH}-${CURRENT_DATE}",
      "**Severity:** ${3|Critical,High,Medium,Low|}",
      "**Impact:** ${4:what's affected}",
      "",
      "#### Problem",
      "${5:detailed description}",
      "",
      "#### Proposed Solution",
      "${6:planned approach}",
      "",
      "#### Tasks Required",
      "- [ ] ${7:task_id}: ${8:description} (Est: ${9:X}h)",
      "",
      "#### Resolution Status",
      "- [ ] Solution designed",
      "- [ ] Tasks added to roadmap",
      "- [ ] Implementation started",
      "- [ ] Testing complete",
      "- [ ] Issue resolved"
    ]
  }
}
```

## 📝 Conclusion

This workflow ensures CLAUDE.md remains the living heart of the project, providing:
- **Immediate context** for every session
- **Clear progress tracking** 
- **Issue management** integration
- **Decision documentation**
- **Metric visibility**

By following these workflows, the development team maintains focus, tracks progress effectively, and ensures every Claude Code session starts with perfect context.

---

**Document Status:** Active  
**Review Frequency:** Weekly  
**Owner:** Development Team  
**Last Review:** 2025-08-14