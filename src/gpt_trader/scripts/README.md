# V2 Scripts

## Purpose
V2-native utility scripts that work with the vertical slice architecture.

## V2 Architecture Requirements
- Prefer explicit imports: `from gpt_trader.features.[slice] import module`
- Avoid wildcard imports; they complicate static analysis and tooling
- No cross-slice dependencies allowed
- Each script should target a specific slice
- Scripts must respect slice isolation principles

## Available V2 Scripts
(To be created as needed)

## Script Template
```python
#!/usr/bin/env python
"""V2 script for [purpose]"""

from gpt_trader.features.[slice] import module_a, module_b


def main() -> None:
    # Script logic here
    module_a.do_work()


if __name__ == "__main__":
    main()
```
