# V2 Demos

## Purpose
V2-native demonstrations showcasing vertical slice functionality.

## V2 Demo Principles
- Each demo targets a single slice
- Complete isolation - no cross-slice imports
- Self-contained examples (~500 tokens to understand)
- Clear documentation of what slice is being demonstrated

## Available V2 Demos
(To be created as needed)

## Demo Template
```python
#!/usr/bin/env python
"""V2 Demo: [Slice Name] - [What it demonstrates]"""

from bot_v2.features.[slice] import *

def demo_[slice]_functionality():
    """Demonstrate [slice] capabilities."""
    # Demo code here
    pass

if __name__ == "__main__":
    print("V2 Demo: [Slice Name]")
    demo_[slice]_functionality()
```