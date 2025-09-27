"""
Configuration management for adaptive portfolio.

Handles loading, validation, and hot-reloading of configuration files.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

from .types import (
    PortfolioConfig, TierConfig, RiskProfile, PositionConstraints,
    TradingRules, CostStructure, MarketConstraints, ValidationResult,
    PortfolioTier
)


class ConfigManager:
    """Manages adaptive portfolio configuration with validation and hot-reload."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize with default or custom config path."""
        if config_path is None:
            # Default to config in project root
            project_root = Path(__file__).parent.parent.parent.parent.parent
            config_path = project_root / "config" / "adaptive_portfolio_config.json"
        
        self.config_path = Path(config_path)
        self._config: Optional[PortfolioConfig] = None
        self._last_modified: Optional[float] = None
    
    def load_config(self, force_reload: bool = False) -> PortfolioConfig:
        """Load configuration from file with optional hot-reload."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        # Check if reload needed
        current_modified = self.config_path.stat().st_mtime
        if not force_reload and self._config and current_modified == self._last_modified:
            return self._config
        
        # Load and parse config
        with open(self.config_path, 'r') as f:
            raw_config = json.load(f)
        
        # Convert to typed config
        config = self._parse_config(raw_config)
        
        # Validate configuration
        validation = self.validate_config(config)
        if not validation.is_valid:
            raise ValueError(f"Invalid configuration: {validation.errors}")
        
        # Cache and return
        self._config = config
        self._last_modified = current_modified
        return config
    
    def _parse_config(self, raw: Dict[str, Any]) -> PortfolioConfig:
        """Parse raw JSON into typed configuration objects."""
        
        # Parse tiers
        tiers = {}
        for tier_name, tier_data in raw["tiers"].items():
            positions = PositionConstraints(
                min_positions=tier_data["positions"]["min"],
                max_positions=tier_data["positions"]["max"],
                target_positions=tier_data["positions"]["target"],
                min_position_size=tier_data["min_position_size"]
            )
            
            risk = RiskProfile(
                daily_limit_pct=tier_data["risk"]["daily_limit_pct"],
                quarterly_limit_pct=tier_data["risk"]["quarterly_limit_pct"],
                position_stop_loss_pct=tier_data["risk"]["position_stop_loss_pct"],
                max_sector_exposure_pct=tier_data["risk"]["max_sector_exposure_pct"]
            )
            
            trading = TradingRules(
                max_trades_per_week=tier_data["trading"]["max_trades_per_week"],
                account_type=tier_data["trading"]["account_type"],
                settlement_days=tier_data["trading"]["settlement_days"],
                pdt_compliant=tier_data["trading"]["pdt_compliant"]
            )
            
            tiers[tier_name] = TierConfig(
                name=tier_data["name"],
                range=(tier_data["range"][0], tier_data["range"][1]),
                positions=positions,
                min_position_size=tier_data["min_position_size"],
                strategies=tier_data["strategies"],
                risk=risk,
                trading=trading
            )
        
        # Parse costs
        costs = CostStructure(
            commission_per_trade=raw["costs"]["commission_per_trade"],
            spread_estimate_pct=raw["costs"]["spread_estimate_pct"],
            slippage_pct=raw["costs"]["slippage_pct"],
            financing_rate_annual_pct=raw["costs"]["financing_rate_annual_pct"]
        )
        
        # Parse market constraints
        market_constraints = MarketConstraints(
            min_share_price=raw["market_constraints"]["min_share_price"],
            max_share_price=raw["market_constraints"]["max_share_price"],
            min_daily_volume=raw["market_constraints"]["min_daily_volume"],
            excluded_sectors=raw["market_constraints"]["excluded_sectors"],
            excluded_symbols=raw["market_constraints"]["excluded_symbols"],
            market_hours_only=raw["market_constraints"]["market_hours_only"]
        )
        
        return PortfolioConfig(
            version=raw["version"],
            last_updated=raw["last_updated"],
            description=raw["description"],
            tiers=tiers,
            costs=costs,
            market_constraints=market_constraints,
            validation=raw["validation"],
            rebalancing=raw["rebalancing"]
        )
    
    def validate_config(self, config: PortfolioConfig) -> ValidationResult:
        """Validate configuration for logical consistency."""
        errors = []
        warnings = []
        suggestions = []
        
        # Validate each tier
        for tier_name, tier in config.tiers.items():
            # Check position sizing math
            min_capital, max_capital = tier.range
            max_positions = tier.positions.max_positions
            min_position = tier.min_position_size
            
            if min_capital / max_positions < min_position:
                errors.append(
                    f"{tier_name}: Minimum capital ({min_capital}) / max positions "
                    f"({max_positions}) = {min_capital/max_positions:.0f} is less than "
                    f"minimum position size ({min_position})"
                )
            
            # Check risk limits
            if tier.risk.daily_limit_pct > 10:
                errors.append(f"{tier_name}: Daily risk limit too high ({tier.risk.daily_limit_pct}%)")
            
            if tier.risk.quarterly_limit_pct > 50:
                errors.append(f"{tier_name}: Quarterly risk limit too high ({tier.risk.quarterly_limit_pct}%)")
            
            # Check PDT compliance
            if min_capital < 25000 and not tier.trading.pdt_compliant:
                warnings.append(
                    f"{tier_name}: Portfolio under $25K should be PDT compliant"
                )
            
            # Suggestions for optimization
            if tier.positions.max_positions > 10 and min_capital < 10000:
                suggestions.append(
                    f"{tier_name}: Consider fewer positions for small portfolio"
                )
        
        # Validate tier ranges don't overlap
        ranges = [(name, tier.range) for name, tier in config.tiers.items()]
        ranges.sort(key=lambda x: x[1][0])  # Sort by min value
        
        for i in range(len(ranges) - 1):
            current_name, (_, current_max) = ranges[i]
            next_name, (next_min, _) = ranges[i + 1]
            
            if current_max > next_min:
                errors.append(
                    f"Tier ranges overlap: {current_name} max ({current_max}) > "
                    f"{next_name} min ({next_min})"
                )
        
        # Validate cost assumptions
        if config.costs.spread_estimate_pct > 1.0:
            warnings.append("Spread estimate seems high (>1%)")
        
        if config.costs.slippage_pct > 0.5:
            warnings.append("Slippage estimate seems high (>0.5%)")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def get_tier_for_capital(self, capital: float) -> str:
        """Determine appropriate tier for given capital amount."""
        config = self.load_config()
        
        for tier_name, tier in config.tiers.items():
            min_val, max_val = tier.range
            if min_val <= capital < max_val:
                return tier_name
        
        # Default to largest tier if above all ranges
        return "large"
    
    def save_config(self, config: PortfolioConfig, backup: bool = True) -> None:
        """Save configuration to file with optional backup."""
        if backup and self.config_path.exists():
            backup_path = self.config_path.with_suffix('.json.backup')
            backup_path.write_bytes(self.config_path.read_bytes())
        
        # Convert back to JSON format
        raw_config = self._config_to_dict(config)
        
        with open(self.config_path, 'w') as f:
            json.dump(raw_config, f, indent=2)
        
        # Update cache
        self._config = config
        self._last_modified = self.config_path.stat().st_mtime
    
    def _config_to_dict(self, config: PortfolioConfig) -> Dict[str, Any]:
        """Convert typed config back to dictionary for JSON serialization."""
        # This is a simplified version - would need full implementation
        # for production use
        raise NotImplementedError("Config serialization not yet implemented")


# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """Get global config manager instance."""
    global _config_manager
    if _config_manager is None or config_path is not None:
        _config_manager = ConfigManager(config_path)
    return _config_manager


def load_portfolio_config(config_path: Optional[str] = None) -> PortfolioConfig:
    """Load portfolio configuration (convenience function)."""
    return get_config_manager(config_path).load_config()


def validate_portfolio_config(config_path: Optional[str] = None) -> ValidationResult:
    """Validate portfolio configuration (convenience function)."""
    config = load_portfolio_config(config_path)
    return get_config_manager().validate_config(config)


def get_current_tier(capital: float, config_path: Optional[str] = None) -> str:
    """Get appropriate tier for capital amount (convenience function)."""
    return get_config_manager(config_path).get_tier_for_capital(capital)