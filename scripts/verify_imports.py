
import sys
sys.path.append("src")

try:
    from bot_v2.orchestration.managers.unified_state import UnifiedStateManager, ReduceOnlyModeSource
    print("UnifiedStateManager imported successfully")
except ImportError as e:
    print(f"Failed to import UnifiedStateManager: {e}")

try:
    from bot_v2.orchestration.perps_bot.bot import PerpsBot
    print("PerpsBot imported successfully")
except ImportError as e:
    print(f"Failed to import PerpsBot: {e}")
except Exception as e:
    print(f"PerpsBot import raised other error: {e}")
