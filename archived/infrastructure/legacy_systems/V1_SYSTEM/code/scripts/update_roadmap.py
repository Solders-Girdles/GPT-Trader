#!/usr/bin/env python3
"""
Update ROADMAP.json when milestones complete or tasks change.
Keeps strategic plan in sync with actual progress.
"""
import json
from datetime import datetime
from pathlib import Path

def update_milestone_status(milestone_id: str, new_status: str):
    """Update a milestone's status when completed."""
    with open("ROADMAP.json", "r") as f:
        roadmap = json.load(f)
    
    updated = False
    for phase_name, phase in roadmap["phases"].items():
        for milestone in phase.get("milestones", []):
            if milestone["id"] == milestone_id:
                old_status = milestone.get("status", "pending")
                milestone["status"] = new_status
                milestone["completed_date"] = datetime.now().isoformat()
                updated = True
                print(f"✅ Milestone {milestone_id}: {old_status} → {new_status}")
                
                # Update current focus if this was current
                if roadmap["current_focus"]["milestone_id"] == milestone_id:
                    # Move to next milestone
                    next_milestone = find_next_milestone(roadmap)
                    if next_milestone:
                        roadmap["current_focus"] = {
                            "milestone_id": next_milestone["id"],
                            "milestone_title": next_milestone["title"],
                            "next_tasks": next_milestone.get("tasks", [])[:3],
                            "blocking_issues": []
                        }
                        print(f"📍 Current focus moved to: {next_milestone['title']}")
                break
    
    if updated:
        roadmap["last_updated"] = datetime.now().strftime("%Y-%m-%d")
        with open("ROADMAP.json", "w") as f:
            json.dump(roadmap, f, indent=2)
        
        # Check if phase complete
        check_phase_completion(roadmap)
    
    return updated

def find_next_milestone(roadmap):
    """Find the next pending milestone in priority order."""
    current_phase = roadmap.get("current_phase")
    
    # First check current phase
    if current_phase in roadmap["phases"]:
        phase = roadmap["phases"][current_phase]
        for milestone in phase.get("milestones", []):
            if milestone.get("status", "pending") == "pending":
                return milestone
    
    # Then check next phases
    for phase_name, phase in roadmap["phases"].items():
        if phase.get("status") not in ["completed", "blocked"]:
            for milestone in phase.get("milestones", []):
                if milestone.get("status", "pending") == "pending":
                    # Update current phase
                    roadmap["current_phase"] = phase_name
                    return milestone
    
    return None

def check_phase_completion(roadmap):
    """Check if current phase is complete and update accordingly."""
    current_phase = roadmap.get("current_phase")
    if current_phase in roadmap["phases"]:
        phase = roadmap["phases"][current_phase]
        milestones = phase.get("milestones", [])
        
        if all(m.get("status") == "completed" for m in milestones):
            phase["status"] = "completed"
            phase["completed_date"] = datetime.now().isoformat()
            print(f"🎉 Phase '{current_phase}' COMPLETED!")
            
            # Find next phase
            for phase_name, next_phase in roadmap["phases"].items():
                if next_phase.get("status") == "pending":
                    roadmap["current_phase"] = phase_name
                    print(f"📈 Moving to phase: {phase_name}")
                    break

def add_blocker(blocker_description: str):
    """Add a blocker to current milestone."""
    with open("ROADMAP.json", "r") as f:
        roadmap = json.load(f)
    
    current_focus = roadmap.get("current_focus", {})
    blockers = current_focus.setdefault("blocking_issues", [])
    
    if blocker_description not in blockers:
        blockers.append(blocker_description)
        roadmap["last_updated"] = datetime.now().strftime("%Y-%m-%d")
        
        with open("ROADMAP.json", "w") as f:
            json.dump(roadmap, f, indent=2)
        
        print(f"🚧 Added blocker: {blocker_description}")
        return True
    return False

def get_current_tasks():
    """Get current tasks from roadmap for todo list."""
    with open("ROADMAP.json", "r") as f:
        roadmap = json.load(f)
    
    current = roadmap.get("current_focus", {})
    tasks = current.get("next_tasks", [])
    
    return {
        "milestone": current.get("milestone_title", "Unknown"),
        "milestone_id": current.get("milestone_id"),
        "tasks": tasks,
        "blockers": current.get("blocking_issues", [])
    }

def sync_with_project_state():
    """Sync roadmap with PROJECT_STATE.json progress."""
    with open("ROADMAP.json", "r") as f:
        roadmap = json.load(f)
    with open("PROJECT_STATE.json", "r") as f:
        state = json.load(f)
    
    updates = []
    
    # Check each milestone against component status
    for phase_name, phase in roadmap["phases"].items():
        for milestone in phase.get("milestones", []):
            component = milestone.get("component")
            if component and component in state["components"]:
                comp_status = state["components"][component]["status"]
                
                # Update milestone based on component status
                if comp_status == "working" and milestone.get("status") != "completed":
                    milestone["status"] = "completed"
                    updates.append(f"{milestone['id']}: completed (component working)")
                elif comp_status == "partial" and milestone.get("status") == "pending":
                    milestone["status"] = "in_progress"
                    updates.append(f"{milestone['id']}: in_progress (component partial)")
    
    if updates:
        roadmap["last_updated"] = datetime.now().strftime("%Y-%m-%d")
        with open("ROADMAP.json", "w") as f:
            json.dump(roadmap, f, indent=2)
        print(f"📊 Synced with PROJECT_STATE.json:")
        for update in updates:
            print(f"   {update}")
    
    return len(updates)

def generate_status_report():
    """Generate a status report from roadmap."""
    with open("ROADMAP.json", "r") as f:
        roadmap = json.load(f)
    
    print("="*50)
    print("STRATEGIC ROADMAP STATUS")
    print("="*50)
    print(f"Current Phase: {roadmap['current_phase']}")
    print(f"Overall Goal: {roadmap['overall_goal']}")
    print()
    
    # Phase status
    for phase_name, phase in roadmap["phases"].items():
        status = phase.get("status", "pending")
        symbol = "✅" if status == "completed" else "🔄" if status == "in_progress" else "⏳"
        print(f"{symbol} {phase_name}: {status}")
        
        if phase_name == roadmap["current_phase"]:
            # Show milestones for current phase
            for m in phase.get("milestones", []):
                m_status = m.get("status", "pending")
                m_symbol = "✅" if m_status == "completed" else "🔄" if m_status == "in_progress" else "⏳"
                print(f"   {m_symbol} {m['id']}: {m['title']}")
    
    print()
    print(f"Current Focus: {roadmap['current_focus']['milestone_title']}")
    print("Next Tasks:")
    for task in roadmap['current_focus'].get('next_tasks', []):
        print(f"  - {task}")
    
    if roadmap['current_focus'].get('blocking_issues'):
        print("\n⚠️ Blockers:")
        for blocker in roadmap['current_focus']['blocking_issues']:
            print(f"  - {blocker}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "complete" and len(sys.argv) > 2:
            milestone_id = sys.argv[2]
            update_milestone_status(milestone_id, "completed")
        
        elif command == "block" and len(sys.argv) > 2:
            blocker = " ".join(sys.argv[2:])
            add_blocker(blocker)
        
        elif command == "sync":
            sync_with_project_state()
        
        elif command == "status":
            generate_status_report()
        
        elif command == "tasks":
            tasks = get_current_tasks()
            print(f"Current milestone: {tasks['milestone']}")
            for task in tasks['tasks']:
                print(f"  - {task}")
    else:
        # Default: show status
        generate_status_report()