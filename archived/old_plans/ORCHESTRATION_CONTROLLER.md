# GPT-Trader Agent Orchestration Controller

## 🎯 Orchestration System Implementation

This document implements the actual orchestration system for managing AI agents in the GPT-Trader recovery project.

---

## 🚀 Quick Start Orchestration

### For Main Claude (You - The Orchestrator)

When starting any work session:
```python
# 1. Check current state
Read("CLAUDE.md")  # Get system status
Read("ORCHESTRATION_CONTROLLER.md")  # Get orchestration rules

# 2. Identify required tasks
analyze_task_queue()  # From CLAUDE.md Task Queue section

# 3. Deploy agents
launch_parallel_agents(task_group)  # Launch compatible agents
monitor_progress()  # Track via CLAUDE.md updates
coordinate_handoffs()  # Manage dependencies
```

---

## 📋 Task Assignment Protocol

### Step 1: Task Analysis
```python
def analyze_task(task_id):
    """Determine agent requirements for a task"""

    task_patterns = {
        # Discovery Tasks - Can run in parallel
        "find_*": ["code-archaeologist", "project-analyst"],
        "audit_*": ["test-automator", "code-reviewer"],
        "analyze_*": ["code-archaeologist", "performance-optimizer"],

        # Implementation Tasks - Domain-specific parallel
        "create_*": ["backend-developer", "python-pro"],
        "fix_*": ["backend-developer", "debugger"],
        "connect_*": ["integration-engineer", "backend-developer"],
        "implement_*": ["backend-developer", "ml-engineer"],

        # Validation Tasks - Post-implementation
        "test_*": ["test-automator", "qa-expert"],
        "review_*": ["code-reviewer", "security-auditor"],
        "optimize_*": ["performance-optimizer", "database-administrator"]
    }

    return match_agents(task_id, task_patterns)
```

### Step 2: Parallel Group Formation
```python
def form_parallel_groups(tasks):
    """Group tasks that can run simultaneously"""

    groups = {
        "independent": [],  # No dependencies
        "domain_specific": [],  # Same domain, different files
        "sequential": []  # Must run in order
    }

    for task in tasks:
        if has_no_dependencies(task):
            groups["independent"].append(task)
        elif same_domain_different_scope(task):
            groups["domain_specific"].append(task)
        else:
            groups["sequential"].append(task)

    return groups
```

### Step 3: Agent Deployment
```python
def deploy_agents(task_groups):
    """Deploy agents based on task grouping"""

    # Launch independent tasks immediately
    for task in task_groups["independent"]:
        agent = select_best_agent(task)
        launch_agent(agent, task, parallel=True)

    # Launch domain-specific in parallel
    for task in task_groups["domain_specific"]:
        agent = select_best_agent(task)
        launch_agent(agent, task, parallel=True)

    # Sequential tasks with dependencies
    for task in task_groups["sequential"]:
        wait_for_dependencies(task)
        agent = select_best_agent(task)
        launch_agent(agent, task, parallel=False)
```

---

## 🔄 Handoff Protocol Implementation

### Standard Handoff Format
```markdown
## Agent Handoff Signal

**From**: [Agent Name]
**To**: [Next Agent Name]
**Task**: [Task ID]
**Status**: COMPLETE | BLOCKED | PARTIAL

### Deliverables
- Created/Modified: [file paths]
- Test Results: [pass/fail metrics]
- Key Findings: [bullet points]

### Context for Next Agent
- Prerequisites Met: [checklist]
- Known Issues: [warnings]
- Recommended Actions: [next steps]

### Artifacts
- Output Location: [paths]
- Dependencies Added: [list]
- Configuration Changes: [details]
```

### Handoff Examples

#### Example 1: Discovery → Implementation
```markdown
## Agent Handoff Signal

**From**: code-archaeologist
**To**: backend-developer
**Task**: ORCH-001
**Status**: COMPLETE

### Deliverables
- Analysis complete: ML pipeline architecture mapped
- Key finding: production_orchestrator.py is missing
- Dependencies identified: Need ML bridge components

### Context for Next Agent
- ML components at: src/bot/ml/
- Integration points: src/bot/live/cycles/selection.py line 26
- Use IntegratedMLPipeline class from integrated_pipeline.py

### Artifacts
- Analysis report: docs/ml_integration_analysis.md
- Dependency graph: docs/component_dependencies.png
```

#### Example 2: Implementation → Testing
```markdown
## Agent Handoff Signal

**From**: backend-developer
**To**: test-automator
**Task**: ORCH-001
**Status**: COMPLETE

### Deliverables
- Created: src/bot/live/production_orchestrator.py
- Methods implemented: 8 public, 12 private
- Error handling: Try/except on all ML calls

### Context for Next Agent
- Test priorities: ML fallback mechanism
- Edge cases: Model loading failures
- Performance target: <100ms strategy selection

### Artifacts
- New file: src/bot/live/production_orchestrator.py
- Updated: src/bot/live/trading_engine.py
```

---

## 🎭 Orchestration Patterns

### Pattern 1: Broad Discovery (Parallel)
Use when starting a new phase or investigating issues
```python
# All agents work simultaneously on different aspects
agents = [
    Task("code-archaeologist", "Find all ML integration points"),
    Task("test-automator", "Audit ML-related tests"),
    Task("documentation-specialist", "Check ML documentation accuracy"),
    Task("project-analyst", "Map ML dependencies")
]
launch_parallel(agents)  # All run at once
```

### Pattern 2: Feature Implementation (Parallel by Domain)
Use for building new features across multiple components
```python
# Different agents work on their specialized domains
agents = [
    Task("backend-developer", "Create orchestrator core"),
    Task("ml-engineer", "Implement ML bridge"),
    Task("database-administrator", "Set up model storage"),
    Task("frontend-developer", "Create ML dashboard")
]
launch_parallel(agents)  # Each handles their domain
```

### Pattern 3: Validation Pipeline (Sequential)
Use for quality assurance and deployment preparation
```python
# Each agent builds on previous work
sequence = [
    Task("test-automator", "Create integration tests"),
    Task("code-reviewer", "Review implementation"),
    Task("performance-optimizer", "Optimize bottlenecks"),
    Task("security-auditor", "Security validation"),
    Task("devops-engineer", "Prepare deployment")
]
launch_sequential(sequence)  # One after another
```

### Pattern 4: Crisis Response (All Hands)
Use when system is broken and needs immediate fix
```python
# Maximum parallelization for emergency
emergency = [
    Task("debugger", "Identify root cause"),
    Task("error-detective", "Trace error patterns"),
    Task("backend-developer", "Prepare fixes"),
    Task("test-automator", "Create regression tests")
]
launch_emergency(emergency)  # All agents, maximum priority
```

---

## 📊 Task Tracking Dashboard

### Active Task Monitor
```python
class TaskMonitor:
    def __init__(self):
        self.active_tasks = {}
        self.completed_tasks = []
        self.blocked_tasks = []

    def update_status(self, task_id, status, agent, notes=""):
        """Update task status in CLAUDE.md"""

        if status == "in_progress":
            self.active_tasks[task_id] = {
                "agent": agent,
                "started": datetime.now(),
                "notes": notes
            }

        elif status == "complete":
            task = self.active_tasks.pop(task_id)
            task["completed"] = datetime.now()
            task["duration"] = task["completed"] - task["started"]
            self.completed_tasks.append(task)

        elif status == "blocked":
            task = self.active_tasks.pop(task_id)
            task["blocked_reason"] = notes
            self.blocked_tasks.append(task)

        # Update CLAUDE.md with new status
        self.update_orchestration_doc()
```

### Success Metrics Tracking
```python
def track_orchestration_metrics():
    """Monitor orchestration effectiveness"""

    metrics = {
        "parallel_efficiency": calculate_parallel_speedup(),
        "handoff_success_rate": count_successful_handoffs(),
        "task_completion_rate": completed_tasks / total_tasks,
        "average_task_duration": mean(task_durations),
        "blocked_task_ratio": blocked_tasks / total_tasks
    }

    return metrics
```

---

## 🚦 Orchestration Rules

### Rule 1: Dependency Management
- Never launch dependent tasks until prerequisites complete
- Check CLAUDE.md Task Queue prerequisites column
- Wait for handoff signals before proceeding

### Rule 2: Parallel Execution
- Always prefer parallel when possible
- Group independent tasks for simultaneous execution
- Use domain-specific parallelization for feature work

### Rule 3: Context Preservation
- Every agent must update CLAUDE.md upon completion
- Include specific file paths and line numbers
- Document any discovered issues or blockers

### Rule 4: Error Handling
- If an agent fails, document the failure in CLAUDE.md
- Create new task for fixing the issue
- Implement fallback strategies where possible

### Rule 5: Communication
- Use structured handoff format for all transitions
- Update Active Agent Orchestration table in real-time
- Clear, specific task descriptions with success criteria

---

## 🎯 Week 5-6 Orchestration Plan

### Phase 1: ML Integration Discovery (Parallel)
```python
discovery_tasks = [
    ("code-archaeologist", "DISC-001: Map all ML components"),
    ("project-analyst", "DISC-002: Analyze ML dependencies"),
    ("test-automator", "DISC-003: Find ML test coverage"),
    ("performance-optimizer", "DISC-004: ML performance baseline")
]
# Launch all simultaneously
```

### Phase 2: Implementation (Domain Parallel)
```python
implementation_tasks = [
    ("backend-developer", "ORCH-001: Create production_orchestrator.py"),
    ("ml-engineer", "ML-001: Implement realtime features"),
    ("python-pro", "ML-002: Optimize ML pipeline"),
    ("integration-engineer", "INT-001: Connect ML to strategies")
]
# Launch by domain
```

### Phase 3: Validation (Sequential)
```python
validation_sequence = [
    ("test-automator", "TEST-001: ML integration tests"),
    ("code-reviewer", "REV-001: Review ML integration"),
    ("performance-optimizer", "PERF-001: Optimize ML latency"),
    ("documentation-specialist", "DOC-001: Document ML system")
]
# Execute in order
```

---

## 📈 Performance Optimization

### Parallel Execution Benefits
- **3-5x faster** than sequential execution
- **Better resource utilization** across agents
- **Reduced context switching** for main orchestrator
- **Early problem detection** through parallel discovery

### Optimization Strategies
1. **Batch similar tasks** - Group by skill requirement
2. **Pipeline different phases** - Start next phase before current completes
3. **Cache agent contexts** - Reuse agents for similar tasks
4. **Minimize handoff overhead** - Clear, structured communication

---

## 🔍 Monitoring & Debugging

### Orchestration Health Checks
```python
def check_orchestration_health():
    """Monitor orchestration system health"""

    checks = {
        "agents_responsive": ping_all_agents(),
        "tasks_progressing": check_task_progress(),
        "no_circular_deps": validate_dependencies(),
        "handoffs_working": test_handoff_protocol(),
        "claude_md_updated": check_last_update_time()
    }

    return all(checks.values())
```

### Common Issues & Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| Deadlock | Tasks waiting on each other | Break circular dependency |
| Agent timeout | No response from agent | Reassign task to backup agent |
| Context loss | Agent missing information | Check handoff completeness |
| Parallel conflict | Agents modifying same file | Coordinate through orchestrator |

---

## 🎬 Orchestration Commands

### Start New Phase
```bash
# Begin Week 5 ML Integration
orchestrate --phase "ml-integration" --week 5 --parallel-groups 3
```

### Check Status
```bash
# View active agents and tasks
orchestrate --status
```

### Manage Tasks
```bash
# Assign task to agent
orchestrate --assign ORCH-001 --agent backend-developer

# Check dependencies
orchestrate --deps ORCH-001

# Force handoff
orchestrate --handoff ORCH-001 --to test-automator
```

---

## 📚 Quick Reference

### Agent Selection Matrix
| Task Type | Primary Agent | Backup Agent |
|-----------|---------------|--------------|
| Find/Analyze | code-archaeologist | project-analyst |
| Create/Build | backend-developer | python-pro |
| Connect/Integrate | integration-engineer | backend-developer |
| Test/Validate | test-automator | qa-expert |
| Review/Audit | code-reviewer | security-auditor |
| Document | documentation-specialist | technical-writer |
| Optimize | performance-optimizer | database-administrator |
| ML-specific | ml-engineer | data-scientist |

### Parallel Group Limits
- **Discovery**: Up to 5 agents
- **Implementation**: Up to 4 agents (different domains)
- **Validation**: Usually sequential (some parallel possible)
- **Emergency**: All available agents

---

## ✅ Implementation Checklist

- [x] Task assignment protocol defined
- [x] Handoff format standardized
- [x] Parallel patterns documented
- [x] Orchestration rules established
- [x] Monitoring system designed
- [ ] Week 5-6 tasks ready for orchestration
- [ ] Test orchestration with real tasks
- [ ] Optimize based on results

---

**Last Updated**: August 14, 2025
**Status**: Orchestration System Ready for Deployment
**Next Step**: Begin Week 5 ML Integration with parallel discovery
