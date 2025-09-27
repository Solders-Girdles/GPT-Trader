from __future__ import annotations

from bot.monitor.performance_monitor import AlertConfig, PerformanceMonitor, PerformanceThresholds


class DummyBroker:
    def get_account(self):
        class A:
            portfolio_value = 100000
            cash = 25000

        return A()

    def get_positions(self):
        return []


def test_turnover_rolling_stats():
    broker = DummyBroker()
    thresholds = PerformanceThresholds()
    alerts = AlertConfig()
    mon = PerformanceMonitor(broker, thresholds, alerts)

    for v in [0.02, 0.05, 0.10, 0.01, 0.03]:
        mon.record_turnover(v)

    summary = mon.get_performance_summary()
    ts = summary.get("turnover_stats", {})
    assert ts.get("count") == 5
    assert ts.get("mean") > 0
    assert ts.get("p95") >= 0.10  # top value present


def test_min_transition_smoothness_threshold_exposed():
    broker = DummyBroker()
    thresholds = PerformanceThresholds(min_transition_smoothness=0.6)
    alerts = AlertConfig()
    mon = PerformanceMonitor(broker, thresholds, alerts)

    # Ensure the threshold is stored and retrievable
    assert mon.thresholds.min_transition_smoothness == 0.6
