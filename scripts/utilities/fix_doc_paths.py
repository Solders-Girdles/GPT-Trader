#!/usr/bin/env python3
"""Fix outdated paths in documentation files."""

import os
import re
from pathlib import Path

def update_file_references(file_path):
    """Update references to moved knowledge files."""
    
    replacements = {
        # Core knowledge files moved to .knowledge/
        r"(?<!')PROJECT_STATE\.json": ".knowledge/PROJECT_STATE.json",
        r"(?<!')KNOWN_FAILURES\.md": ".knowledge/KNOWN_FAILURES.md",
        r"(?<!')SYSTEM_REALITY\.md": ".knowledge/SYSTEM_REALITY.md",
        r"(?<!')ROADMAP\.json": ".knowledge/ROADMAP.json",
        
        # Files moved to docs/knowledge/
        r"(?<!')WORKFLOW\.md(?!')": "docs/knowledge/WORKFLOW.md",
        r"(?<!')DIAGNOSTICS\.md(?!')": "docs/knowledge/DIAGNOSTICS.md",
        r"(?<!')IMPORTS\.md(?!')": "docs/knowledge/IMPORTS.md",
        r"(?<!')TEST_MAP\.json": "docs/knowledge/TEST_MAP.json",
        r"(?<!')DEPENDENCIES\.json": "docs/knowledge/DEPENDENCIES.json",
        r"(?<!')AGENT_WORKFLOW\.md": "docs/knowledge/AGENT_WORKFLOW.md",
        r"(?<!')AGENT_DELEGATION_GUIDE\.md": "docs/knowledge/AGENT_DELEGATION_GUIDE.md",
        r"(?<!')TASK_TEMPLATES\.md": "docs/knowledge/TASK_TEMPLATES.md",
    }
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        original = content
        changes = []
        
        for pattern, replacement in replacements.items():
            # Skip if already has the correct path
            if replacement in pattern:
                continue
                
            # Count replacements
            matches = re.findall(pattern, content)
            if matches:
                content = re.sub(pattern, replacement, content)
                changes.append(f"  - {pattern.replace('(?<!', '').replace(r'\.', '.').replace(')', '')} → {replacement} ({len(matches)} occurrences)")
        
        if content != original:
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"✅ Updated {file_path}")
            for change in changes:
                print(change)
            return True
        else:
            print(f"✓ No updates needed for {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False

def main():
    """Update all documentation files."""
    
    docs_dir = Path("docs/knowledge")
    root_files = ["CLAUDE.md", "README.md"]
    
    updated_count = 0
    
    # Update files in docs/knowledge/
    for file in docs_dir.glob("*.md"):
        if update_file_references(file):
            updated_count += 1
    
    # Update root files
    for file_name in root_files:
        file_path = Path(file_name)
        if file_path.exists():
            if update_file_references(file_path):
                updated_count += 1
    
    print(f"\n📊 Summary: Updated {updated_count} files")

if __name__ == "__main__":
    main()