"""
Data Quality Framework for GPT-Trader Strategy Training

Provides comprehensive data validation, cleaning, and quality scoring for historical datasets.
Works with the Historical Data Manager to ensure clean, validated data for strategy training.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class DataQualityIssueType(Enum):
    """Types of data quality issues"""

    MISSING_DATA = "missing_data"
    DUPLICATE_TIMESTAMPS = "duplicate_timestamps"
    OUTLIERS = "outliers"
    INCONSISTENT_OHLC = "inconsistent_ohlc"
    NEGATIVE_PRICES = "negative_prices"
    NEGATIVE_VOLUME = "negative_volume"
    GAPS_IN_DATA = "gaps_in_data"
    STALE_DATA = "stale_data"
    EXTREME_VOLATILITY = "extreme_volatility"
    CORPORATE_ACTION_ANOMALY = "corporate_action_anomaly"


class DataQualitySeverity(Enum):
    """Severity levels for data quality issues"""

    CRITICAL = "critical"  # Data unusable
    HIGH = "high"  # Significant impact on analysis
    MEDIUM = "medium"  # Moderate impact
    LOW = "low"  # Minor impact
    INFO = "info"  # Informational only


@dataclass
class DataQualityIssue:
    """Represents a single data quality issue"""

    issue_type: DataQualityIssueType
    severity: DataQualitySeverity
    description: str
    affected_rows: int
    percentage: float
    symbol: str | None = None
    timestamp_range: tuple[datetime, datetime] | None = None
    suggested_action: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataQualityReport:
    """Comprehensive data quality assessment report"""

    symbol: str
    total_records: int
    date_range: tuple[datetime, datetime]
    quality_score: float  # 0-100 scale
    issues: list[DataQualityIssue]
    cleaning_applied: list[str]
    recommendations: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def critical_issues(self) -> list[DataQualityIssue]:
        """Get critical issues that make data unusable"""
        return [issue for issue in self.issues if issue.severity == DataQualitySeverity.CRITICAL]

    @property
    def is_usable(self) -> bool:
        """Check if data is usable for strategy training"""
        return len(self.critical_issues) == 0 and self.quality_score >= 70.0


@dataclass
class DataCleaningConfig:
    """Configuration for data cleaning operations"""

    # Missing data handling
    fill_missing_method: str = "forward"  # forward, backward, interpolate, drop
    max_consecutive_missing: int = 5  # Maximum consecutive missing values to fill

    # Outlier handling
    outlier_method: str = "iqr"  # iqr, zscore, isolation_forest
    outlier_action: str = "cap"  # cap, remove, flag
    outlier_threshold: float = 3.0

    # Volume handling
    zero_volume_action: str = "keep"  # keep, remove, flag
    negative_volume_action: str = "remove"  # remove, zero, flag

    # OHLC consistency
    fix_ohlc_inconsistencies: bool = True

    # Corporate actions
    handle_splits: bool = True
    handle_dividends: bool = True
    split_threshold: float = 0.8  # Detect splits if price drops more than 20%

    # Data continuity
    max_gap_days: int = 7  # Maximum gap in trading days
    weekend_handling: str = "ignore"  # ignore, flag

    # Quality thresholds
    min_quality_score: float = 70.0
    min_completeness: float = 0.95


class BaseDataValidator(ABC):
    """Base class for data validators"""

    def __init__(self, config: DataCleaningConfig) -> None:
        self.config = config

    @abstractmethod
    def validate(self, data: pd.DataFrame, symbol: str) -> list[DataQualityIssue]:
        """Validate data and return list of issues"""
        pass


class MissingDataValidator(BaseDataValidator):
    """Validates missing data patterns"""

    def validate(self, data: pd.DataFrame, symbol: str) -> list[DataQualityIssue]:
        issues = []

        # Check overall missing data
        missing_counts = data.isnull().sum()
        len(data) * len(data.columns)

        for column in data.columns:
            if column in ["Symbol", "Source"]:  # Skip metadata columns
                continue

            missing_count = missing_counts[column]
            if missing_count > 0:
                percentage = (missing_count / len(data)) * 100

                severity = DataQualitySeverity.INFO
                if percentage > 10:
                    severity = DataQualitySeverity.CRITICAL
                elif percentage > 5:
                    severity = DataQualitySeverity.HIGH
                elif percentage > 2:
                    severity = DataQualitySeverity.MEDIUM

                # Check for consecutive missing values
                is_missing = data[column].isnull()
                consecutive_missing = []
                current_streak = 0
                max_streak = 0

                for missing in is_missing:
                    if missing:
                        current_streak += 1
                        max_streak = max(max_streak, current_streak)
                    else:
                        if current_streak > 0:
                            consecutive_missing.append(current_streak)
                        current_streak = 0

                suggested_action = (
                    "forward_fill"
                    if max_streak <= self.config.max_consecutive_missing
                    else "interpolate"
                )

                issues.append(
                    DataQualityIssue(
                        issue_type=DataQualityIssueType.MISSING_DATA,
                        severity=severity,
                        description=f"Missing data in {column}: {missing_count} values ({percentage:.1f}%)",
                        affected_rows=missing_count,
                        percentage=percentage,
                        symbol=symbol,
                        suggested_action=suggested_action,
                        metadata={
                            "column": column,
                            "max_consecutive": max_streak,
                            "consecutive_streaks": consecutive_missing,
                        },
                    )
                )

        return issues


class OutlierValidator(BaseDataValidator):
    """Validates and detects outliers in price/volume data"""

    def validate(self, data: pd.DataFrame, symbol: str) -> list[DataQualityIssue]:
        issues = []

        price_columns = ["Open", "High", "Low", "Close"]
        price_columns = [col for col in price_columns if col in data.columns]

        for column in price_columns + (["Volume"] if "Volume" in data.columns else []):
            if data[column].isnull().all():
                continue

            outliers = self._detect_outliers(data[column])
            outlier_count = outliers.sum()

            if outlier_count > 0:
                percentage = (outlier_count / len(data)) * 100

                severity = DataQualitySeverity.LOW
                if percentage > 5:
                    severity = DataQualitySeverity.HIGH
                elif percentage > 2:
                    severity = DataQualitySeverity.MEDIUM

                issues.append(
                    DataQualityIssue(
                        issue_type=DataQualityIssueType.OUTLIERS,
                        severity=severity,
                        description=f"Outliers detected in {column}: {outlier_count} values ({percentage:.1f}%)",
                        affected_rows=outlier_count,
                        percentage=percentage,
                        symbol=symbol,
                        suggested_action="cap_outliers",
                        metadata={
                            "column": column,
                            "method": self.config.outlier_method,
                            "threshold": self.config.outlier_threshold,
                        },
                    )
                )

        return issues

    def _detect_outliers(self, series: pd.Series) -> pd.Series:
        """Detect outliers using configured method"""
        if self.config.outlier_method == "iqr":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (series < lower_bound) | (series > upper_bound)

        elif self.config.outlier_method == "zscore":
            z_scores = np.abs(stats.zscore(series, nan_policy="omit"))
            return z_scores > self.config.outlier_threshold

        elif self.config.outlier_method == "isolation_forest":
            try:
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(series.values.reshape(-1, 1))

                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_predictions = iso_forest.fit_predict(data_scaled)
                return outlier_predictions == -1
            except Exception:
                # Fallback to IQR method
                return self._detect_outliers_iqr(series)

        return pd.Series([False] * len(series), index=series.index)


class ConsistencyValidator(BaseDataValidator):
    """Validates OHLC consistency and other data relationships"""

    def validate(self, data: pd.DataFrame, symbol: str) -> list[DataQualityIssue]:
        issues = []

        # Check OHLC relationships
        if all(col in data.columns for col in ["Open", "High", "Low", "Close"]):
            issues.extend(self._validate_ohlc_consistency(data, symbol))

        # Check for negative prices
        price_columns = ["Open", "High", "Low", "Close"]
        for col in price_columns:
            if col in data.columns:
                negative_count = (data[col] <= 0).sum()
                if negative_count > 0:
                    percentage = (negative_count / len(data)) * 100
                    issues.append(
                        DataQualityIssue(
                            issue_type=DataQualityIssueType.NEGATIVE_PRICES,
                            severity=DataQualitySeverity.CRITICAL,
                            description=f"Negative or zero prices in {col}: {negative_count} values ({percentage:.1f}%)",
                            affected_rows=negative_count,
                            percentage=percentage,
                            symbol=symbol,
                            suggested_action="remove_rows",
                            metadata={"column": col},
                        )
                    )

        # Check for negative volume
        if "Volume" in data.columns:
            negative_volume = (data["Volume"] < 0).sum()
            if negative_volume > 0:
                percentage = (negative_volume / len(data)) * 100
                issues.append(
                    DataQualityIssue(
                        issue_type=DataQualityIssueType.NEGATIVE_VOLUME,
                        severity=DataQualitySeverity.HIGH,
                        description=f"Negative volume: {negative_volume} values ({percentage:.1f}%)",
                        affected_rows=negative_volume,
                        percentage=percentage,
                        symbol=symbol,
                        suggested_action=self.config.negative_volume_action,
                        metadata={"threshold": 0},
                    )
                )

        return issues

    def _validate_ohlc_consistency(self, data: pd.DataFrame, symbol: str) -> list[DataQualityIssue]:
        """Validate OHLC relationships"""
        issues = []

        # High should be >= max(Open, Close)
        high_violations = (data["High"] < data[["Open", "Close"]].max(axis=1)).sum()

        # Low should be <= min(Open, Close)
        low_violations = (data["Low"] > data[["Open", "Close"]].min(axis=1)).sum()

        if high_violations > 0:
            percentage = (high_violations / len(data)) * 100
            issues.append(
                DataQualityIssue(
                    issue_type=DataQualityIssueType.INCONSISTENT_OHLC,
                    severity=DataQualitySeverity.HIGH,
                    description=f"High price violations: {high_violations} cases ({percentage:.1f}%)",
                    affected_rows=high_violations,
                    percentage=percentage,
                    symbol=symbol,
                    suggested_action="fix_ohlc_consistency",
                    metadata={"violation_type": "high_price"},
                )
            )

        if low_violations > 0:
            percentage = (low_violations / len(data)) * 100
            issues.append(
                DataQualityIssue(
                    issue_type=DataQualityIssueType.INCONSISTENT_OHLC,
                    severity=DataQualitySeverity.HIGH,
                    description=f"Low price violations: {low_violations} cases ({percentage:.1f}%)",
                    affected_rows=low_violations,
                    percentage=percentage,
                    symbol=symbol,
                    suggested_action="fix_ohlc_consistency",
                    metadata={"violation_type": "low_price"},
                )
            )

        return issues


class TemporalValidator(BaseDataValidator):
    """Validates temporal aspects of data"""

    def validate(self, data: pd.DataFrame, symbol: str) -> list[DataQualityIssue]:
        issues = []

        # Check for duplicate timestamps
        duplicate_indices = data.index.duplicated()
        duplicate_count = duplicate_indices.sum()

        if duplicate_count > 0:
            percentage = (duplicate_count / len(data)) * 100
            issues.append(
                DataQualityIssue(
                    issue_type=DataQualityIssueType.DUPLICATE_TIMESTAMPS,
                    severity=DataQualitySeverity.MEDIUM,
                    description=f"Duplicate timestamps: {duplicate_count} cases ({percentage:.1f}%)",
                    affected_rows=duplicate_count,
                    percentage=percentage,
                    symbol=symbol,
                    suggested_action="remove_duplicates",
                    metadata={"action": "keep_last"},
                )
            )

        # Check for data gaps
        if len(data) > 1:
            issues.extend(self._check_data_gaps(data, symbol))

        return issues

    def _check_data_gaps(self, data: pd.DataFrame, symbol: str) -> list[DataQualityIssue]:
        """Check for gaps in time series data"""
        issues = []

        try:
            # Sort by index to ensure chronological order
            data_sorted = data.sort_index()

            # Calculate time differences
            time_diffs = data_sorted.index.to_series().diff()

            # Expected frequency (assume daily for now)
            pd.Timedelta(days=1)

            # Find gaps larger than expected (accounting for weekends)
            large_gaps = time_diffs[time_diffs > pd.Timedelta(days=self.config.max_gap_days)]

            if len(large_gaps) > 0:
                gap_count = len(large_gaps)
                max_gap = large_gaps.max()

                severity = DataQualitySeverity.LOW
                if max_gap > pd.Timedelta(days=30):
                    severity = DataQualitySeverity.HIGH
                elif max_gap > pd.Timedelta(days=14):
                    severity = DataQualitySeverity.MEDIUM

                issues.append(
                    DataQualityIssue(
                        issue_type=DataQualityIssueType.GAPS_IN_DATA,
                        severity=severity,
                        description=f"Data gaps detected: {gap_count} gaps, max gap: {max_gap}",
                        affected_rows=gap_count,
                        percentage=(gap_count / len(data)) * 100,
                        symbol=symbol,
                        suggested_action="interpolate_gaps",
                        metadata={
                            "max_gap_days": max_gap.days,
                            "gap_locations": [
                                str(ts) for ts in large_gaps.index[:5]
                            ],  # First 5 gaps
                        },
                    )
                )

        except Exception as e:
            logger.warning(f"Error checking data gaps for {symbol}: {str(e)}")

        return issues


class DataCleaner:
    """Cleans data based on validation results and configuration"""

    def __init__(self, config: DataCleaningConfig) -> None:
        self.config = config

    def clean_data(
        self, data: pd.DataFrame, issues: list[DataQualityIssue], symbol: str
    ) -> tuple[pd.DataFrame, list[str]]:
        """Clean data based on identified issues"""
        cleaned_data = data.copy()
        actions_applied = []

        # Sort issues by severity (critical first)
        sorted_issues = sorted(issues, key=lambda x: self._severity_priority(x.severity))

        for issue in sorted_issues:
            if issue.severity == DataQualitySeverity.INFO:
                continue  # Skip info-level issues

            try:
                cleaned_data, action = self._apply_cleaning_action(cleaned_data, issue, symbol)
                if action:
                    actions_applied.append(action)
            except Exception as e:
                logger.warning(f"Error applying cleaning action for {issue.issue_type}: {str(e)}")
                continue

        return cleaned_data, actions_applied

    def _severity_priority(self, severity: DataQualitySeverity) -> int:
        """Get numeric priority for severity (lower = higher priority)"""
        priority_map = {
            DataQualitySeverity.CRITICAL: 0,
            DataQualitySeverity.HIGH: 1,
            DataQualitySeverity.MEDIUM: 2,
            DataQualitySeverity.LOW: 3,
            DataQualitySeverity.INFO: 4,
        }
        return priority_map.get(severity, 5)

    def _apply_cleaning_action(
        self, data: pd.DataFrame, issue: DataQualityIssue, symbol: str
    ) -> tuple[pd.DataFrame, str | None]:
        """Apply specific cleaning action based on issue type"""

        if issue.issue_type == DataQualityIssueType.MISSING_DATA:
            return self._handle_missing_data(data, issue)

        elif issue.issue_type == DataQualityIssueType.DUPLICATE_TIMESTAMPS:
            return self._handle_duplicates(data, issue)

        elif issue.issue_type == DataQualityIssueType.OUTLIERS:
            return self._handle_outliers(data, issue)

        elif issue.issue_type == DataQualityIssueType.INCONSISTENT_OHLC:
            return self._handle_ohlc_inconsistencies(data, issue)

        elif issue.issue_type == DataQualityIssueType.NEGATIVE_PRICES:
            return self._handle_negative_prices(data, issue)

        elif issue.issue_type == DataQualityIssueType.NEGATIVE_VOLUME:
            return self._handle_negative_volume(data, issue)

        return data, None

    def _handle_missing_data(
        self, data: pd.DataFrame, issue: DataQualityIssue
    ) -> tuple[pd.DataFrame, str]:
        """Handle missing data based on configuration"""
        column = issue.metadata.get("column")
        if not column or column not in data.columns:
            return data, None

        if self.config.fill_missing_method == "forward":
            data[column] = data[column].fillna(method="ffill")
            action = f"Forward-filled missing values in {column}"

        elif self.config.fill_missing_method == "backward":
            data[column] = data[column].fillna(method="bfill")
            action = f"Backward-filled missing values in {column}"

        elif self.config.fill_missing_method == "interpolate":
            data[column] = data[column].interpolate()
            action = f"Interpolated missing values in {column}"

        elif self.config.fill_missing_method == "drop":
            data = data.dropna(subset=[column])
            action = f"Dropped rows with missing values in {column}"
        else:
            return data, None

        return data, action

    def _handle_duplicates(
        self, data: pd.DataFrame, issue: DataQualityIssue
    ) -> tuple[pd.DataFrame, str]:
        """Handle duplicate timestamps"""
        data = data[~data.index.duplicated(keep="last")]
        return data, "Removed duplicate timestamps (kept last)"

    def _handle_outliers(
        self, data: pd.DataFrame, issue: DataQualityIssue
    ) -> tuple[pd.DataFrame, str]:
        """Handle outliers based on configuration"""
        column = issue.metadata.get("column")
        if not column or column not in data.columns:
            return data, None

        if self.config.outlier_action == "cap":
            # Cap outliers at 1st and 99th percentiles
            lower_bound = data[column].quantile(0.01)
            upper_bound = data[column].quantile(0.99)

            data[column] = data[column].clip(lower_bound, upper_bound)
            action = f"Capped outliers in {column} to [{lower_bound:.2f}, {upper_bound:.2f}]"

        elif self.config.outlier_action == "remove":
            # Remove outlier rows
            if self.config.outlier_method == "iqr":
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                mask = (data[column] >= lower_bound) & (data[column] <= upper_bound)
                data = data[mask]
                action = f"Removed outlier rows in {column}"
            else:
                return data, None
        else:
            return data, None

        return data, action

    def _handle_ohlc_inconsistencies(
        self, data: pd.DataFrame, issue: DataQualityIssue
    ) -> tuple[pd.DataFrame, str]:
        """Fix OHLC inconsistencies"""
        if not all(col in data.columns for col in ["Open", "High", "Low", "Close"]):
            return data, None

        # Fix High values
        data["High"] = data[["Open", "High", "Close"]].max(axis=1)

        # Fix Low values
        data["Low"] = data[["Open", "Low", "Close"]].min(axis=1)

        return data, "Fixed OHLC inconsistencies"

    def _handle_negative_prices(
        self, data: pd.DataFrame, issue: DataQualityIssue
    ) -> tuple[pd.DataFrame, str]:
        """Handle negative or zero prices"""
        column = issue.metadata.get("column")
        if not column or column not in data.columns:
            return data, None

        # Remove rows with negative or zero prices
        initial_len = len(data)
        data = data[data[column] > 0]
        removed_count = initial_len - len(data)

        return data, f"Removed {removed_count} rows with negative/zero prices in {column}"

    def _handle_negative_volume(
        self, data: pd.DataFrame, issue: DataQualityIssue
    ) -> tuple[pd.DataFrame, str]:
        """Handle negative volume"""
        if "Volume" not in data.columns:
            return data, None

        if self.config.negative_volume_action == "remove":
            initial_len = len(data)
            data = data[data["Volume"] >= 0]
            removed_count = initial_len - len(data)
            action = f"Removed {removed_count} rows with negative volume"

        elif self.config.negative_volume_action == "zero":
            data.loc[data["Volume"] < 0, "Volume"] = 0
            action = "Set negative volume values to zero"
        else:
            return data, None

        return data, action


class DataQualityFramework:
    """Main data quality framework that coordinates validation and cleaning"""

    def __init__(self, config: DataCleaningConfig | None = None) -> None:
        self.config = config or DataCleaningConfig()

        # Initialize validators
        self.validators = [
            MissingDataValidator(self.config),
            OutlierValidator(self.config),
            ConsistencyValidator(self.config),
            TemporalValidator(self.config),
        ]

        self.cleaner = DataCleaner(self.config)

        logger.info("Data Quality Framework initialized")

    def assess_quality(self, data: pd.DataFrame, symbol: str) -> DataQualityReport:
        """Assess data quality and generate comprehensive report"""
        if data.empty:
            return DataQualityReport(
                symbol=symbol,
                total_records=0,
                date_range=(datetime.now(), datetime.now()),
                quality_score=0.0,
                issues=[
                    DataQualityIssue(
                        issue_type=DataQualityIssueType.MISSING_DATA,
                        severity=DataQualitySeverity.CRITICAL,
                        description="Dataset is empty",
                        affected_rows=0,
                        percentage=100.0,
                    )
                ],
                cleaning_applied=[],
                recommendations=["Obtain valid data for this symbol"],
            )

        # Run all validators
        all_issues = []
        for validator in self.validators:
            try:
                issues = validator.validate(data, symbol)
                all_issues.extend(issues)
            except Exception as e:
                logger.warning(
                    f"Validator {validator.__class__.__name__} failed for {symbol}: {str(e)}"
                )
                continue

        # Calculate quality score
        quality_score = self._calculate_quality_score(data, all_issues)

        # Get date range
        date_range = (data.index.min().to_pydatetime(), data.index.max().to_pydatetime())

        # Generate recommendations
        recommendations = self._generate_recommendations(all_issues, quality_score)

        return DataQualityReport(
            symbol=symbol,
            total_records=len(data),
            date_range=date_range,
            quality_score=quality_score,
            issues=all_issues,
            cleaning_applied=[],
            recommendations=recommendations,
            metadata={
                "validation_timestamp": datetime.now().isoformat(),
                "config": {
                    "outlier_method": self.config.outlier_method,
                    "missing_data_method": self.config.fill_missing_method,
                    "min_quality_score": self.config.min_quality_score,
                },
            },
        )

    def clean_and_validate(
        self, data: pd.DataFrame, symbol: str
    ) -> tuple[pd.DataFrame, DataQualityReport]:
        """Clean data and return both cleaned data and quality report"""

        # Initial quality assessment
        initial_report = self.assess_quality(data, symbol)

        # Clean data based on issues
        cleaned_data, actions_applied = self.cleaner.clean_data(data, initial_report.issues, symbol)

        # Re-assess quality after cleaning
        final_report = self.assess_quality(cleaned_data, symbol)
        final_report.cleaning_applied = actions_applied

        # Add cleaning metadata
        final_report.metadata.update(
            {
                "initial_quality_score": initial_report.quality_score,
                "initial_issues_count": len(initial_report.issues),
                "cleaning_applied": actions_applied,
                "improvement": final_report.quality_score - initial_report.quality_score,
            }
        )

        logger.info(
            f"Data cleaning completed for {symbol}: "
            f"Quality improved from {initial_report.quality_score:.1f} to {final_report.quality_score:.1f}"
        )

        return cleaned_data, final_report

    def _calculate_quality_score(self, data: pd.DataFrame, issues: list[DataQualityIssue]) -> float:
        """Calculate overall quality score (0-100)"""
        if len(data) == 0:
            return 0.0

        base_score = 100.0

        # Deduct points based on issue severity and percentage
        severity_weights = {
            DataQualitySeverity.CRITICAL: 30,
            DataQualitySeverity.HIGH: 15,
            DataQualitySeverity.MEDIUM: 8,
            DataQualitySeverity.LOW: 3,
            DataQualitySeverity.INFO: 1,
        }

        for issue in issues:
            weight = severity_weights.get(issue.severity, 1)
            # Scale deduction by percentage affected
            deduction = weight * (issue.percentage / 100)
            base_score -= deduction

        # Ensure score is between 0 and 100
        return max(0.0, min(100.0, base_score))

    def _generate_recommendations(
        self, issues: list[DataQualityIssue], quality_score: float
    ) -> list[str]:
        """Generate recommendations for improving data quality"""
        recommendations = []

        if quality_score < 50:
            recommendations.append("Data quality is poor - consider using alternative data source")
        elif quality_score < 70:
            recommendations.append("Data quality is moderate - apply cleaning before use")

        # Specific recommendations based on issues
        issue_types = {issue.issue_type for issue in issues}

        if DataQualityIssueType.MISSING_DATA in issue_types:
            recommendations.append("Apply missing data interpolation or forward-filling")

        if DataQualityIssueType.OUTLIERS in issue_types:
            recommendations.append("Consider outlier detection and capping/removal")

        if DataQualityIssueType.INCONSISTENT_OHLC in issue_types:
            recommendations.append("Fix OHLC consistency issues before analysis")

        if DataQualityIssueType.GAPS_IN_DATA in issue_types:
            recommendations.append("Fill data gaps using interpolation or external sources")

        # Always recommend monitoring
        if quality_score > 90:
            recommendations.append("Data quality is excellent - suitable for production use")
        else:
            recommendations.append("Monitor data quality over time for degradation")

        return recommendations

    def batch_assess_quality(
        self, datasets: dict[str, pd.DataFrame]
    ) -> dict[str, DataQualityReport]:
        """Assess quality for multiple datasets"""
        reports = {}

        for symbol, data in datasets.items():
            try:
                report = self.assess_quality(data, symbol)
                reports[symbol] = report
                logger.info(
                    f"Quality assessment completed for {symbol}: {report.quality_score:.1f}/100"
                )
            except Exception as e:
                logger.error(f"Quality assessment failed for {symbol}: {str(e)}")
                continue

        return reports

    def generate_quality_summary(self, reports: dict[str, DataQualityReport]) -> dict[str, Any]:
        """Generate summary statistics across multiple quality reports"""
        if not reports:
            return {"error": "No reports provided"}

        quality_scores = [report.quality_score for report in reports.values()]
        usable_datasets = [report for report in reports.values() if report.is_usable]

        issue_type_counts = {}
        for report in reports.values():
            for issue in report.issues:
                issue_type_counts[issue.issue_type.value] = (
                    issue_type_counts.get(issue.issue_type.value, 0) + 1
                )

        return {
            "total_datasets": len(reports),
            "usable_datasets": len(usable_datasets),
            "usable_percentage": (len(usable_datasets) / len(reports)) * 100,
            "average_quality_score": np.mean(quality_scores),
            "min_quality_score": min(quality_scores),
            "max_quality_score": max(quality_scores),
            "quality_distribution": {
                "excellent (90-100)": len([s for s in quality_scores if s >= 90]),
                "good (80-89)": len([s for s in quality_scores if 80 <= s < 90]),
                "moderate (70-79)": len([s for s in quality_scores if 70 <= s < 80]),
                "poor (60-69)": len([s for s in quality_scores if 60 <= s < 70]),
                "critical (<60)": len([s for s in quality_scores if s < 60]),
            },
            "common_issues": dict(
                sorted(issue_type_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
            "recommendations": self._generate_batch_recommendations(reports),
        }

    def _generate_batch_recommendations(self, reports: dict[str, DataQualityReport]) -> list[str]:
        """Generate recommendations for batch of datasets"""
        recommendations = []

        usable_count = len([r for r in reports.values() if r.is_usable])
        total_count = len(reports)

        if usable_count < total_count * 0.5:
            recommendations.append(
                f"Only {usable_count}/{total_count} datasets are usable - review data sources"
            )

        # Find most common issues
        issue_counts = {}
        for report in reports.values():
            for issue in report.issues:
                issue_counts[issue.issue_type] = issue_counts.get(issue.issue_type, 0) + 1

        if issue_counts:
            most_common_issue = max(issue_counts.items(), key=lambda x: x[1])
            recommendations.append(
                f"Most common issue: {most_common_issue[0].value} ({most_common_issue[1]} datasets)"
            )

        recommendations.append("Run automated cleaning pipeline before strategy training")
        recommendations.append("Set up monitoring for ongoing data quality assessment")

        return recommendations


# Factory function for easy initialization
def create_data_quality_framework(
    outlier_method: str = "iqr",
    missing_data_method: str = "forward",
    min_quality_score: float = 70.0,
    **kwargs,
) -> DataQualityFramework:
    """Factory function to create data quality framework"""

    config = DataCleaningConfig(
        outlier_method=outlier_method,
        fill_missing_method=missing_data_method,
        min_quality_score=min_quality_score,
        **kwargs,
    )

    return DataQualityFramework(config)


# Example usage and testing
if __name__ == "__main__":

    def main() -> None:
        """Example usage of Data Quality Framework"""
        print("Data Quality Framework Testing")
        print("=" * 40)

        # Create framework
        framework = create_data_quality_framework(
            outlier_method="iqr", missing_data_method="forward", min_quality_score=75.0
        )

        # Generate sample data with various quality issues
        dates = pd.date_range(start="2022-01-01", end="2023-12-31", freq="D")
        np.random.seed(42)

        data = pd.DataFrame(
            {
                "Open": np.random.lognormal(mean=4.0, sigma=0.02, size=len(dates)),
                "High": np.random.lognormal(mean=4.01, sigma=0.02, size=len(dates)),
                "Low": np.random.lognormal(mean=3.99, sigma=0.02, size=len(dates)),
                "Close": np.random.lognormal(mean=4.0, sigma=0.02, size=len(dates)),
                "Volume": np.random.randint(1000000, 10000000, size=len(dates)),
                "Symbol": "TEST",
                "Source": "test_data",
            },
            index=dates,
        )

        # Introduce quality issues
        # Missing data
        data.loc[data.index[10:15], "Close"] = np.nan
        data.loc[data.index[50:52], "Volume"] = np.nan

        # Outliers
        data.loc[data.index[100], "High"] = data.loc[data.index[100], "High"] * 5  # Price spike
        data.loc[data.index[200], "Volume"] = (
            data.loc[data.index[200], "Volume"] * 20
        )  # Volume spike

        # OHLC inconsistencies
        data.loc[data.index[300], "Low"] = data.loc[data.index[300], "High"] * 1.1  # Low > High

        # Negative prices (shouldn't happen but testing)
        data.loc[data.index[400], "Open"] = -10.0

        print(f"Generated test data with {len(data)} records")
        print(f"Date range: {data.index.min().date()} to {data.index.max().date()}")

        # Assess quality
        print("\n📊 Quality Assessment:")
        quality_report = framework.assess_quality(data, "TEST")

        print(f"   Quality Score: {quality_report.quality_score:.1f}/100")
        print(f"   Total Issues: {len(quality_report.issues)}")
        print(f"   Usable: {'Yes' if quality_report.is_usable else 'No'}")

        # Show issues
        print("\n⚠️  Quality Issues:")
        for issue in quality_report.issues[:5]:  # Show first 5 issues
            print(f"   {issue.severity.value.upper()}: {issue.description}")
            if issue.suggested_action:
                print(f"      → Suggested: {issue.suggested_action}")

        # Clean data
        print("\n🧹 Data Cleaning:")
        cleaned_data, final_report = framework.clean_and_validate(data, "TEST")

        print(f"   Original records: {len(data)}")
        print(f"   Cleaned records: {len(cleaned_data)}")
        print(
            f"   Quality improvement: {final_report.quality_score - quality_report.quality_score:.1f} points"
        )

        # Show cleaning actions
        print("\n✅ Cleaning Actions Applied:")
        for action in final_report.cleaning_applied:
            print(f"   • {action}")

        # Show recommendations
        print("\n💡 Recommendations:")
        for rec in final_report.recommendations:
            print(f"   • {rec}")

        print("\n🚀 Data Quality Framework ready for production datasets!")

    # Run the example
    main()
