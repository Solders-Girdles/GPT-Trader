# Composition Root Migration Guide

This guide explains how to migrate to the new composition root pattern introduced in Issue #91.

## Overview

The composition root pattern provides a centralized way to manage application dependencies through a dependency injection container. This improves testability, maintainability, and makes the application architecture more explicit.

## What Changed

### New Components

1. **`src/app/container.py`** - Main dependency injection container
2. **`src/app/__init__.py`** - Package exports for the app module
3. **Updated PerpsBot** - Added `from_container` class method and container support
4. **Updated PerpsBotBuilder** - Added container-based construction option
5. **Updated bootstrap** - Added container-enabled bootstrap functions

### Backward Compatibility

All existing code continues to work without changes. The container approach is opt-in, meaning:

- Existing CLI commands work unchanged
- Existing scripts work unchanged
- Existing tests work unchanged
- Existing bot creation code works unchanged

## Migration Paths

### 1. Direct Container Usage (Recommended for New Code)

```python
# Old approach
from bot_v2.orchestration.perps_bot_builder import create_perps_bot
bot = create_perps_bot(config)

# New approach (recommended)
from app.container import create_application_container
container = create_application_container(config)
bot = container.create_perps_bot()
```

### 2. PerpsBot.from_container() Method

```python
# New approach using class method
from app.container import create_application_container
from bot_v2.orchestration.perps_bot import PerpsBot

container = create_application_container(config)
bot = PerpsBot.from_container(container)
```

### 3. Builder with Container Flag

```python
# Existing builder with container enabled
from bot_v2.orchestration.perps_bot_builder import PerpsBotBuilder

bot = PerpsBotBuilder(use_container=True).with_config(config).build()
```

### 4. Bootstrap with Container

```python
# Bootstrap with container
from bot_v2.orchestration.perps_bootstrap import prepare_perps_bot_with_container

result = prepare_perps_bot_with_container(config)
container = result.registry.extras["container"]
bot = PerpsBot.from_container(container)
```

## Benefits of Container Approach

1. **Dependency Management** - All dependencies are managed in one place
2. **Lazy Initialization** - Services are created only when needed
3. **Singleton Behavior** - Services are shared by default
4. **Testability** - Easy to mock dependencies for testing
5. **Configuration** - Centralized configuration management
6. **Lifecycle Management** - Clear service lifecycle

## When to Use Each Approach

### Use Container Approach For:
- New application code
- Complex dependency graphs
- Test suites that need dependency injection
- Applications requiring service lifecycle management

### Use Legacy Approach For:
- Existing production code (no immediate need to change)
- Simple bot creation without complex dependencies
- Backward compatibility requirements

## Testing with Container

```python
# Unit testing with container
def test_bot_with_mocked_dependencies():
    from app.container import ApplicationContainer
    from unittest.mock import Mock

    container = ApplicationContainer(config)

    # Mock dependencies
    container._broker = Mock()
    container._event_store = Mock()

    bot = container.create_perps_bot()
    # Test bot with mocked dependencies
```

## Migration Checklist

- [ ] Identify bot creation points in your code
- [ ] Decide on migration strategy (gradual or complete)
- [ ] Update tests to use container where beneficial
- [ ] Update documentation for new code
- [ ] Consider adding container-specific configuration options
- [ ] Update CI/CD if needed (should work without changes)

## Troubleshooting

### Import Errors
Make sure to import from the correct module:
```python
# Correct
from app.container import ApplicationContainer

# Incorrect
from src.app.container import ApplicationContainer
```

### Container Not Found
If `bot.container` is None, the bot was created using the legacy approach:
```python
if bot.container is None:
    print("Bot created using legacy approach")
else:
    print("Bot created using container approach")
```

### Service Creation Issues
Services are created lazily. Access them to trigger creation:
```python
container = ApplicationContainer(config)
broker = container.broker  # This triggers broker creation
```

## Examples

### Complete Example with Container

```python
from app.container import create_application_container
from bot_v2.orchestration.configuration import BotConfig, Profile

# Create configuration
config = BotConfig.from_profile(Profile.DEV, symbols=["BTC-USD"])

# Create container
container = create_application_container(config)

# Create bot
bot = container.create_perps_bot()

# Run bot
import asyncio
asyncio.run(bot.run())
```

### Example with Custom Settings

```python
from app.container import create_application_container
from bot_v2.orchestration.runtime_settings import RuntimeSettings

# Custom settings
settings = RuntimeSettings()
settings.coinbase_default_quote = "USDT"
settings.data_dir = Path("/custom/data/path")

# Create container with custom settings
container = create_application_container(config, settings)
bot = container.create_perps_bot()
```

## Future Roadmap

1. **Phase 1**: Container available as opt-in (current state)
2. **Phase 2**: Gradual migration of existing code to container
3. **Phase 3**: Container becomes default approach
4. **Phase 4**: Legacy approach deprecated (future)

## Support

For questions about the composition root migration:
- Check the test files in `tests/unit/app/` and `tests/integration/test_composition_root.py`
- Review the container implementation in `src/app/container.py`
- Consult the existing bot creation code in `src/bot_v2/orchestration/`
