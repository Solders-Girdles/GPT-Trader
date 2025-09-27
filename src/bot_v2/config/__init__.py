"""
Comprehensive configuration system for GPT-Trader V2

Provides centralized configuration management with JSON/YAML support,
environment variable overrides, validation, and hot-reload capabilities.
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from datetime import datetime

from ..errors import ConfigurationError
from ..validation import validate_config, Validator

logger = logging.getLogger(__name__)


@dataclass
class ConfigMetadata:
    """Metadata about a configuration"""
    version: str
    last_updated: datetime
    description: str
    source: str  # file path or "defaults"/"environment"


class ConfigLoader:
    """Centralized configuration loader with caching and validation"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize config loader
        
        Args:
            config_dir: Directory containing config files. 
                       Defaults to project_root/config
        """
        if config_dir is None:
            # Default to config directory in project root
            project_root = Path(__file__).parent.parent.parent.parent
            config_dir = project_root / "config"
        
        self.config_dir = Path(config_dir)
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._metadata: Dict[str, ConfigMetadata] = {}
        self._validators: Dict[str, Dict[str, Validator]] = {}
        self._file_mtimes: Dict[str, float] = {}
        
        # Environment variable prefix
        self.env_prefix = "BOT_V2_"
        
        # Initialize default configurations
        self._init_defaults()
    
    def _init_defaults(self):
        """Initialize default configurations"""
        self._configs['system'] = {
            'log_level': 'INFO',
            'data_provider': 'yfinance',
            'timezone': 'US/Eastern',
            'cache_enabled': True,
            'cache_ttl_seconds': 300,
            'max_retries': 3,
            'retry_delay': 1.0
        }
        
        self._configs['backtest'] = {
            'initial_capital': 10000.0,
            'commission': 0.001,
            'slippage': 0.0005,
            'min_data_points': 30,
            'enable_shorting': False,
            'position_size': 0.1
        }
        
        self._configs['paper_trade'] = {
            'initial_capital': 10000.0,
            'commission': 0.0,
            'slippage': 0.001,
            'update_frequency': 60,  # seconds
            'max_positions': 10,
            'enable_shorting': False
        }
        
        self._configs['optimize'] = {
            'population_size': 50,
            'generations': 20,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'elitism_rate': 0.1,
            'parallel_workers': 4
        }
        
        self._configs['monitor'] = {
            'metrics_interval': 60,  # seconds
            'alert_cooldown': 300,  # seconds
            'error_threshold': 10,
            'performance_window': 3600,  # seconds
            'enable_alerts': True
        }
        
        self._configs['risk'] = {
            'max_position_size': 0.25,
            'max_daily_loss': 0.02,
            'max_drawdown': 0.10,
            'stop_loss': 0.05,
            'take_profit': 0.10,
            'max_correlation': 0.7
        }
    
    def get_config(
        self,
        slice_name: str,
        force_reload: bool = False
    ) -> Dict[str, Any]:
        """Get configuration for a slice
        
        Args:
            slice_name: Name of the slice (e.g., 'backtest', 'paper_trade')
            force_reload: Force reload from file
            
        Returns:
            Configuration dictionary
        """
        # Check if reload needed
        if not force_reload and slice_name in self._configs:
            if self._should_reload(slice_name):
                force_reload = True
        
        if force_reload or slice_name not in self._configs:
            self._load_config(slice_name)
        
        # Apply environment overrides
        config = self._apply_env_overrides(slice_name, self._configs[slice_name].copy())
        
        # Validate if validator exists
        if slice_name in self._validators:
            config = validate_config(config, self._validators[slice_name])
        
        return config
    
    def _should_reload(self, slice_name: str) -> bool:
        """Check if config file has been modified"""
        config_path = self._get_config_path(slice_name)
        
        if not config_path or not config_path.exists():
            return False
        
        current_mtime = config_path.stat().st_mtime
        last_mtime = self._file_mtimes.get(str(config_path), 0)
        
        return current_mtime > last_mtime
    
    def _get_config_path(self, slice_name: str) -> Optional[Path]:
        """Get path to config file"""
        # Try different extensions
        for ext in ['.json', '.yaml', '.yml']:
            path = self.config_dir / f"{slice_name}_config{ext}"
            if path.exists():
                return path
        
        return None
    
    def _load_config(self, slice_name: str):
        """Load configuration from file"""
        config_path = self._get_config_path(slice_name)
        
        if not config_path:
            # Use defaults if no file exists
            if slice_name not in self._configs:
                logger.warning(f"No config file found for {slice_name}, using defaults")
                self._configs[slice_name] = {}
            return
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix == '.json':
                    config = json.load(f)
                else:
                    config = yaml.safe_load(f)
            
            # Merge with defaults
            if slice_name in self._configs:
                defaults = self._configs[slice_name].copy()
                defaults.update(config)
                config = defaults
            
            self._configs[slice_name] = config
            self._file_mtimes[str(config_path)] = config_path.stat().st_mtime
            
            # Extract metadata if present
            if 'version' in config:
                self._metadata[slice_name] = ConfigMetadata(
                    version=config.get('version', '1.0'),
                    last_updated=datetime.fromisoformat(config.get('last_updated', datetime.now().isoformat())),
                    description=config.get('description', ''),
                    source=str(config_path)
                )
            
            logger.info(f"Loaded config for {slice_name} from {config_path}")
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load config for {slice_name}",
                config_key=slice_name,
                context={'path': str(config_path), 'error': str(e)}
            )
    
    def _apply_env_overrides(self, slice_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides
        
        Environment variables should be named:
        BOT_V2_<SLICE>_<KEY>
        
        For example:
        BOT_V2_BACKTEST_INITIAL_CAPITAL=20000
        """
        prefix = f"{self.env_prefix}{slice_name.upper()}_"
        
        for env_key, env_value in os.environ.items():
            if env_key.startswith(prefix):
                config_key = env_key[len(prefix):].lower()
                
                # Convert value to appropriate type
                try:
                    # Try to parse as JSON first (handles arrays, objects)
                    config[config_key] = json.loads(env_value)
                except json.JSONDecodeError:
                    # Try to convert to number
                    try:
                        if '.' in env_value:
                            config[config_key] = float(env_value)
                        else:
                            config[config_key] = int(env_value)
                    except ValueError:
                        # Keep as string
                        if env_value.lower() in ('true', 'false'):
                            config[config_key] = env_value.lower() == 'true'
                        else:
                            config[config_key] = env_value
                
                logger.debug(f"Override {slice_name}.{config_key} from environment")
        
        return config
    
    def set_validator(self, slice_name: str, validators: Dict[str, Validator]):
        """Set validators for a slice configuration"""
        self._validators[slice_name] = validators
    
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all loaded configurations"""
        result = {}
        for slice_name in self._configs:
            result[slice_name] = self.get_config(slice_name)
        return result
    
    def save_config(self, slice_name: str, config: Dict[str, Any]):
        """Save configuration to file"""
        config_path = self.config_dir / f"{slice_name}_config.json"
        
        # Add metadata
        config['version'] = config.get('version', '1.0')
        config['last_updated'] = datetime.now().isoformat()
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            
            logger.info(f"Saved config for {slice_name} to {config_path}")
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to save config for {slice_name}",
                config_key=slice_name,
                context={'path': str(config_path), 'error': str(e)}
            )
    
    def reload_all(self):
        """Reload all configurations from files"""
        for slice_name in list(self._configs.keys()):
            self.get_config(slice_name, force_reload=True)


# Global config loader instance
_config_loader = None


def get_config_loader() -> ConfigLoader:
    """Get global config loader instance"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


def get_config(slice_name: str, force_reload: bool = False) -> Dict[str, Any]:
    """Convenience function to get config for a slice"""
    return get_config_loader().get_config(slice_name, force_reload)


def set_config_loader(loader: ConfigLoader):
    """Set custom config loader"""
    global _config_loader
    _config_loader = loader


# Configuration decorator
def with_config(slice_name: str):
    """Decorator to inject configuration into function"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            config = get_config(slice_name)
            return func(*args, config=config, **kwargs)
        return wrapper
    return decorator


# Export main components
__all__ = [
    'ConfigLoader',
    'ConfigMetadata',
    'get_config_loader',
    'get_config',
    'set_config_loader',
    'with_config'
]