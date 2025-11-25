from __future__ import annotations
from typing import Any
import pandas as pd


class DataQualityChecker:
    def __init__(self):
        self.quality_history = []
        self.scores = []

    class QualityResult:
        def __init__(self, acceptable: bool = True, score: float = 1.0):
            self._acceptable = acceptable
            self._score = score

        def overall_score(self) -> float:
            return self._score

        def is_acceptable(self, threshold: float = 0.8) -> bool:
            return self._acceptable and self._score >= threshold

        @property
        def completeness(self) -> float:
            return self._score

    def check_quality(self, data: pd.DataFrame) -> QualityResult:
        if data is None or data.empty:
            result = self.QualityResult(acceptable=False, score=0.0)
            self.scores.append(result._score)
            return result

        if data.isnull().any().any():
            result = self.QualityResult(acceptable=False, score=0.5)
        else:
            result = self.QualityResult(acceptable=True, score=1.0)

        self.quality_history.append(result)
        self.scores.append(result._score)
        return result

    def validate_ohlcv(self, data: pd.DataFrame) -> list[str]:
        issues = []
        if data is None or data.empty:
            return ["Empty dataframe"]

        required_columns = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            issues.append(f"Missing columns: {missing}")

        # Check for negative prices specific columns
        if "open" in data.columns and (data["open"] < 0).any():
            issues.append("Negative open found")
        if "high" in data.columns and (data["high"] < 0).any():
            issues.append("Negative high found")
        if "low" in data.columns and (data["low"] < 0).any():
            issues.append("Negative low found")
        if "close" in data.columns and (data["close"] < 0).any():
            issues.append("Negative close found")
        if "volume" in data.columns and (data["volume"] < 0).any():
            issues.append("Negative volume found")

        # Check high >= low
        if "high" in data.columns and "low" in data.columns and (data["high"] < data["low"]).any():
            issues.append("High < Low")

        return issues

    def get_quality_trend(self) -> dict[str, float]:
        avg_score = 0.0
        if self.scores:
            avg_score = sum(self.scores) / len(self.scores)

        return {"timeliness": 1.0, "completeness": avg_score, "overall": avg_score}  # Mock
