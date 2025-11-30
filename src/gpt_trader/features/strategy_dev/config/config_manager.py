"""Configuration manager with YAML support and hot-reload.

Provides:
- ConfigManager: Unified configuration management with YAML loading
"""

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from gpt_trader.features.strategy_dev.config.registry import StrategyRegistry
from gpt_trader.features.strategy_dev.config.strategy_profile import StrategyProfile

logger = logging.getLogger(__name__)


@dataclass
class ConfigManager:
    """Unified configuration manager.

    Features:
    - Load configurations from YAML files
    - Environment variable overrides
    - Hot-reload support
    - Strategy registry integration
    """

    config_path: Path | None = None
    registry: StrategyRegistry = field(default_factory=StrategyRegistry)
    enable_hot_reload: bool = False
    hot_reload_interval: float = 5.0

    _config_cache: dict[str, Any] = field(default_factory=dict)
    _file_mtimes: dict[str, float] = field(default_factory=dict)
    _callbacks: dict[str, list[Callable]] = field(default_factory=dict)
    _reload_thread: threading.Thread | None = None
    _stop_reload: bool = False

    def __post_init__(self) -> None:
        """Initialize manager."""
        if self.config_path:
            self.config_path = Path(self.config_path)
            self._load_all_configs()

        if self.enable_hot_reload:
            self._start_hot_reload()

    def _load_all_configs(self) -> None:
        """Load all configuration files from config path."""
        if not self.config_path or not self.config_path.exists():
            return

        # Load YAML files
        for yaml_file in self.config_path.glob("*.yaml"):
            self._load_yaml_file(yaml_file)

        for yml_file in self.config_path.glob("*.yml"):
            self._load_yaml_file(yml_file)

        # Load JSON files
        for json_file in self.config_path.glob("*.json"):
            self._load_json_file(json_file)

        logger.info(f"Loaded {len(self._config_cache)} configuration files")

    def _load_yaml_file(self, file_path: Path) -> None:
        """Load a YAML configuration file."""
        if not YAML_AVAILABLE:
            logger.warning(f"YAML not available, skipping {file_path}")
            return

        try:
            with open(file_path) as f:
                content = yaml.safe_load(f)

            # Track file modification time
            self._file_mtimes[str(file_path)] = file_path.stat().st_mtime

            # Store in cache
            name = file_path.stem
            self._config_cache[name] = {
                "source": str(file_path),
                "type": "yaml",
                "loaded_at": datetime.now().isoformat(),
                "content": content,
            }

            # Check if this is a strategy profile
            if self._is_strategy_profile(content):
                profile = StrategyProfile.from_dict(content)
                self.registry.register(profile, file_path=file_path)

            logger.debug(f"Loaded YAML config: {name}")

        except Exception as e:
            logger.error(f"Error loading YAML file {file_path}: {e}")

    def _load_json_file(self, file_path: Path) -> None:
        """Load a JSON configuration file."""
        try:
            with open(file_path) as f:
                content = json.load(f)

            # Track file modification time
            self._file_mtimes[str(file_path)] = file_path.stat().st_mtime

            # Store in cache
            name = file_path.stem
            self._config_cache[name] = {
                "source": str(file_path),
                "type": "json",
                "loaded_at": datetime.now().isoformat(),
                "content": content,
            }

            # Check if this is a strategy profile
            if self._is_strategy_profile(content):
                profile = StrategyProfile.from_dict(content)
                self.registry.register(profile, file_path=file_path)

            logger.debug(f"Loaded JSON config: {name}")

        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")

    def _is_strategy_profile(self, content: dict) -> bool:
        """Check if content looks like a strategy profile."""
        required_keys = {"name"}
        strategy_keys = {"signals", "risk", "regime", "execution"}

        has_required = required_keys.issubset(content.keys())
        has_strategy = bool(strategy_keys.intersection(content.keys()))

        return has_required and has_strategy

    def _start_hot_reload(self) -> None:
        """Start hot reload monitoring thread."""
        self._stop_reload = False
        self._reload_thread = threading.Thread(
            target=self._hot_reload_loop,
            daemon=True,
        )
        self._reload_thread.start()
        logger.info("Hot reload monitoring started")

    def _hot_reload_loop(self) -> None:
        """Monitor configuration files for changes."""
        while not self._stop_reload:
            try:
                for file_path_str, old_mtime in list(self._file_mtimes.items()):
                    file_path = Path(file_path_str)
                    if file_path.exists():
                        current_mtime = file_path.stat().st_mtime
                        if current_mtime > old_mtime:
                            logger.info(f"Config file changed: {file_path.name}")
                            self._reload_file(file_path)
            except Exception as e:
                logger.error(f"Error in hot reload loop: {e}")

            time.sleep(self.hot_reload_interval)

    def _reload_file(self, file_path: Path) -> None:
        """Reload a single configuration file."""
        if file_path.suffix in (".yaml", ".yml"):
            self._load_yaml_file(file_path)
        elif file_path.suffix == ".json":
            self._load_json_file(file_path)

        self._emit("config_reloaded", str(file_path))

    def stop_hot_reload(self) -> None:
        """Stop hot reload monitoring."""
        self._stop_reload = True
        if self._reload_thread:
            self._reload_thread.join(timeout=2.0)
        logger.info("Hot reload monitoring stopped")

    def on(self, event: str, callback: Callable) -> None:
        """Register callback for config events.

        Events:
        - config_reloaded: Config file reloaded
        """
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    def _emit(self, event: str, data: Any) -> None:
        """Emit event to callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in config callback: {e}")

    def get(self, name: str, default: Any = None) -> Any:
        """Get configuration by name.

        Args:
            name: Configuration name (file stem)
            default: Default value if not found

        Returns:
            Configuration content
        """
        cached = self._config_cache.get(name)
        if cached:
            return self._apply_env_overrides(cached["content"], name)
        return default

    def get_value(self, name: str, key: str, default: Any = None) -> Any:
        """Get a specific value from a configuration.

        Args:
            name: Configuration name
            key: Dot-separated key path (e.g., "risk.max_position_size")
            default: Default value

        Returns:
            Configuration value
        """
        config = self.get(name)
        if not config:
            return default

        # Navigate nested keys
        keys = key.split(".")
        value = config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def _apply_env_overrides(self, config: dict, prefix: str) -> dict:
        """Apply environment variable overrides to config.

        Environment variables should be named:
        {PREFIX}_{KEY} for top-level keys
        {PREFIX}_{SECTION}_{KEY} for nested keys

        Args:
            config: Configuration dictionary
            prefix: Environment variable prefix

        Returns:
            Configuration with overrides applied
        """
        result = config.copy()
        env_prefix = prefix.upper().replace("-", "_")

        for key, value in os.environ.items():
            if not key.startswith(env_prefix + "_"):
                continue

            # Parse the environment variable name
            parts = key[len(env_prefix) + 1 :].lower().split("_")

            # Apply to config
            target = result
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]

            # Set value with type conversion
            final_key = parts[-1]
            target[final_key] = self._convert_env_value(value)

        return result

    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Boolean
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False

        # Number
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # JSON array/object
        if value.startswith(("[", "{")):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass

        return value

    def load_profile(self, file_path: str | Path) -> StrategyProfile:
        """Load a strategy profile from a file.

        Args:
            file_path: Path to profile file

        Returns:
            Loaded strategy profile
        """
        file_path = Path(file_path)

        if file_path.suffix in (".yaml", ".yml"):
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML required for YAML files")
            with open(file_path) as f:
                data = yaml.safe_load(f)
        else:
            with open(file_path) as f:
                data = json.load(f)

        profile = StrategyProfile.from_dict(data)
        self.registry.register(profile, file_path=file_path)

        return profile

    def save_profile(
        self,
        profile: StrategyProfile,
        file_path: str | Path | None = None,
        format: str = "yaml",
    ) -> Path:
        """Save a strategy profile to file.

        Args:
            profile: Strategy profile to save
            file_path: Output path (auto-generated if None)
            format: File format (yaml, json)

        Returns:
            Path to saved file
        """
        if file_path is None:
            if not self.config_path:
                raise ValueError("No config_path set and no file_path provided")
            ext = ".yaml" if format == "yaml" else ".json"
            file_path = self.config_path / f"{profile.name}{ext}"
        else:
            file_path = Path(file_path)

        data = profile.to_dict()

        if format == "yaml":
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML required for YAML output")
            with open(file_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        else:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved profile to {file_path}")
        return file_path

    def create_template_profile(self, name: str = "template") -> StrategyProfile:
        """Create a template strategy profile.

        Args:
            name: Profile name

        Returns:
            Template profile
        """
        from gpt_trader.features.strategy_dev.config.strategy_profile import (
            ExecutionConfig,
            RegimeConfig,
            RiskConfig,
            SignalConfig,
        )

        profile = StrategyProfile(
            name=name,
            description="Template strategy profile - customize for your needs",
            author="System",
            signals=[
                SignalConfig(
                    name="momentum",
                    weight=0.4,
                    parameters={"period": 20, "threshold": 0.6},
                ),
                SignalConfig(
                    name="mean_reversion",
                    weight=0.3,
                    parameters={"period": 14, "oversold": 30, "overbought": 70},
                ),
                SignalConfig(
                    name="trend",
                    weight=0.3,
                    parameters={"fast_period": 12, "slow_period": 26},
                ),
            ],
            risk=RiskConfig(),
            regime=RegimeConfig(),
            execution=ExecutionConfig(),
            tags=["template"],
        )

        return profile

    def list_configs(self) -> list[dict[str, Any]]:
        """List all loaded configurations.

        Returns:
            List of configuration metadata
        """
        return [
            {
                "name": name,
                "source": data["source"],
                "type": data["type"],
                "loaded_at": data["loaded_at"],
            }
            for name, data in self._config_cache.items()
        ]

    def reload_all(self) -> int:
        """Reload all configuration files.

        Returns:
            Number of files reloaded
        """
        self._config_cache.clear()
        self._file_mtimes.clear()
        self._load_all_configs()
        return len(self._config_cache)

    def summary(self) -> dict[str, Any]:
        """Get configuration manager summary.

        Returns:
            Summary data
        """
        return {
            "config_path": str(self.config_path) if self.config_path else None,
            "hot_reload_enabled": self.enable_hot_reload,
            "configs_loaded": len(self._config_cache),
            "registry": self.registry.summary(),
            "configs": self.list_configs(),
        }

    def __del__(self) -> None:
        """Cleanup on destruction."""
        if self.enable_hot_reload:
            self.stop_hot_reload()
