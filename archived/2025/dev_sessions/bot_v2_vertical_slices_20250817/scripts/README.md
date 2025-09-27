# V2 Scripts

## Purpose
V2-native utility scripts that work with the vertical slice architecture.

## V2 Architecture Requirements
- All scripts must use V2 slice imports: `from src.bot_v2.features.[slice] import *`
- No cross-slice dependencies allowed
- Each script should target a specific slice
- Scripts must respect slice isolation principles

## Available V2 Scripts
(To be created as needed)

## Script Template
```python
#!/usr/bin/env python
"""V2 script for [purpose]"""

from src.bot_v2.features.[slice] import *

def main():
    # Script logic here
    pass

if __name__ == "__main__":
    main()
```