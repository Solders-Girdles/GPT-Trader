#!/usr/bin/env python3
"""
Example of how to properly structure tasks for Claude Code agents.

Remember: Agents are stateless and isolated. They need complete context.
"""

# ❌ BAD: How NOT to delegate to agents
bad_examples = [
    "Fix the broken tests",  # What tests? What's broken?
    "Continue from the last agent",  # Agents have no memory
    "Coordinate with the test agent",  # Agents can't communicate
    "Check the state files",  # Which files? What to check?
    "Make sure everything works",  # Too vague
]

# ✅ GOOD: How to properly delegate
good_examples = [
    """
    Task: Fix the import error in test_demo_ma.py
    1. Read /Users/rj/PycharmProjects/GPT-Trader/tests/unit/strategy/test_demo_ma.py
    2. Error: "cannot import name 'DemoMAStrategy' from 'bot.strategy'"
    3. Fix: Change line 3 to: from bot.strategy.demo_ma import DemoMAStrategy
    4. Verify: Run 'poetry run pytest tests/unit/strategy/test_demo_ma.py::test_import'
    5. Return: "FIXED" if test passes, or the full error message
    """,
    
    """
    Analyze performance bottlenecks in the backtest engine:
    1. Read src/bot/backtest/engine.py 
    2. Look for: nested loops, DataFrame.iterrows(), redundant calculations
    3. Return a list with format: [line_number, issue, suggested_fix]
    4. Focus on the run_backtest() method
    """,
    
    """
    Validate the risk management configuration:
    1. Read src/bot/risk/config.py
    2. Check that all RiskConfig fields have valid defaults
    3. Run: python -c "from bot.risk.config import RiskConfig; RiskConfig()"
    4. If it fails, fix the validation errors
    5. Return the working configuration as JSON
    """,
]

# How the main agent should process responses
def handle_agent_response(task_type: str, response: str) -> str:
    """
    Main agent processes sub-agent responses and decides next steps.
    
    Sub-agents return text. Main agent must:
    1. Parse the response
    2. Decide if task succeeded
    3. Update PROJECT_STATE.json if needed
    4. Delegate follow-up tasks if needed
    """
    if task_type == "test_fix":
        if "FIXED" in response:
            # Update PROJECT_STATE.json
            return "Update component status to 'working'"
        else:
            # Parse error and delegate more specific fix
            return f"Delegate new task with error: {response}"
    
    elif task_type == "analysis":
        # Parse bottlenecks and create fix tasks
        issues = eval(response)  # Agent returns list
        for line, issue, fix in issues:
            # Create specific fix task for each issue
            pass
    
    return "Process response and decide next action"

# The workflow that actually works
WORKING_WORKFLOW = """
1. Main agent reads PROJECT_STATE.json
2. Main agent identifies what needs fixing
3. Main agent creates COMPLETE task description
4. Main agent delegates to ONE agent with ALL context
5. Agent returns text response
6. Main agent interprets response
7. Main agent updates files/state
8. Main agent verifies with scripts/verify_capabilities.py
"""

if __name__ == "__main__":
    print("This is documentation, not executable code.")
    print("Use these patterns when delegating to Claude Code agents.")
    print("\nRemember: Agents are stateless. Include ALL context.")