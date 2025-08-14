#!/bin/bash
# Simple CLAUDE.md daily update script
# More reliable than complex Python parsing

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get current date
DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H:%M")

# Function to mark task complete
mark_complete() {
    TASK_ID=$1
    echo -e "${GREEN}✅ Marking $TASK_ID as complete${NC}"
    
    # Simple sed replacement - works with multiple formats
    sed -i.bak "s/\[ \] $TASK_ID/[x] $TASK_ID/g" CLAUDE.md
    sed -i.bak "s/🟡 $TASK_ID/✅ $TASK_ID/g" CLAUDE.md
    sed -i.bak "s/📅 $TASK_ID/✅ $TASK_ID/g" CLAUDE.md
}

# Function to add completed task with time
add_completed() {
    TASK_ID=$1
    DESC=$2
    TIME_SPENT=$3
    
    echo -e "${GREEN}✅ Adding completed: $TASK_ID${NC}"
    
    # Find the completed section and append
    sed -i.bak "/### ✅ Completed/a\\
- ✅ $TASK_ID: $DESC (${TIME_SPENT}m)" CLAUDE.md
}

# Function to update progress bar
update_progress() {
    COMPLETED=$1
    TOTAL=$2
    PERCENT=$((COMPLETED * 100 / TOTAL))
    
    # Create progress bar
    FILLED=$((PERCENT / 5))
    EMPTY=$((20 - FILLED))
    BAR=""
    
    for i in $(seq 1 $FILLED); do BAR="${BAR}█"; done
    for i in $(seq 1 $EMPTY); do BAR="${BAR}░"; done
    
    echo -e "${YELLOW}📊 Progress: $BAR $PERCENT% ($COMPLETED/$TOTAL)${NC}"
    
    # Update in file
    sed -i.bak "s/Progress:.*tasks)/Progress: $BAR $PERCENT% ($COMPLETED\/$TOTAL tasks)/g" CLAUDE.md
}

# Function to add issue
add_issue() {
    SEVERITY=$1
    DESC=$2
    
    echo -e "${RED}🐛 Adding issue: $DESC${NC}"
    
    # Add to issues section
    sed -i.bak "/### 🔴 Blocked\/Issues/a\\
- 🐛 [$SEVERITY] $DESC (Added $TIME)" CLAUDE.md
}

# Function to add learning
add_learning() {
    LEARNING=$1
    
    echo -e "${GREEN}💡 Adding learning: $LEARNING${NC}"
    
    # Add to learnings section
    sed -i.bak "/## 💡 Key Learnings/a\\
- **$DATE**: $LEARNING" CLAUDE.md
}

# Function to update current focus
update_focus() {
    CURRENT=$1
    NEXT=$2
    
    echo -e "${YELLOW}🎯 Updating focus${NC}"
    
    sed -i.bak "s/\*\*Now Working On\*\*:.*/\*\*Now Working On\*\*: $CURRENT/g" CLAUDE.md
    sed -i.bak "s/\*\*Next Up\*\*:.*/\*\*Next Up\*\*: $NEXT/g" CLAUDE.md
}

# Function to show usage
usage() {
    echo "Usage: $0 [command] [args]"
    echo ""
    echo "Commands:"
    echo "  complete TASK_ID              - Mark task as complete"
    echo "  add-done TASK_ID DESC TIME    - Add completed task with time"
    echo "  progress COMPLETED TOTAL       - Update progress bar"
    echo "  issue SEVERITY DESC            - Add an issue"
    echo "  learning TEXT                  - Add a learning"
    echo "  focus CURRENT NEXT            - Update current focus"
    echo "  daily                         - Run daily summary"
    echo ""
    echo "Examples:"
    echo "  $0 complete MON-006"
    echo "  $0 add-done MON-007 'Created dashboard' 45"
    echo "  $0 progress 7 15"
    echo "  $0 issue HIGH 'Import error in integration'"
    echo "  $0 learning 'CUSUM needs smaller k value'"
    echo "  $0 focus 'MON-007' 'MON-009'"
}

# Main command processing
case "$1" in
    complete)
        mark_complete "$2"
        ;;
    add-done)
        add_completed "$2" "$3" "$4"
        ;;
    progress)
        update_progress "$2" "$3"
        ;;
    issue)
        add_issue "$2" "$3"
        ;;
    learning)
        add_learning "$2"
        ;;
    focus)
        update_focus "$2" "$3"
        ;;
    daily)
        echo -e "${GREEN}Running daily summary...${NC}"
        # Add date header
        echo "" >> CLAUDE.md
        echo "---" >> CLAUDE.md
        echo "## Daily Summary: $DATE" >> CLAUDE.md
        echo "**Generated:** $TIME" >> CLAUDE.md
        
        # Count tasks
        COMPLETED=$(grep -c "✅" CLAUDE.md)
        echo "**Tasks Completed Today:** $COMPLETED" >> CLAUDE.md
        
        echo -e "${GREEN}✅ Daily summary added${NC}"
        ;;
    *)
        usage
        exit 1
        ;;
esac

echo -e "${GREEN}✅ CLAUDE.md updated successfully${NC}"

# Create backup
cp CLAUDE.md "backups/CLAUDE_${DATE}_${TIME}.md" 2>/dev/null || true