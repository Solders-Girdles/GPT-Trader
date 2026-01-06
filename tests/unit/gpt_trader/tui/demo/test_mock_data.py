"""Tests for MockDataGenerator determinism with seeding."""

from gpt_trader.tui.demo.mock_data import MockDataGenerator


class TestMockDataGeneratorSeeding:
    """Tests verifying seeded MockDataGenerator produces deterministic output."""

    def test_same_seed_produces_identical_output(self) -> None:
        """Two generators with the same seed produce identical status output."""
        seed = 42

        gen1 = MockDataGenerator(seed=seed)
        gen2 = MockDataGenerator(seed=seed)

        status1 = gen1.generate_full_status()
        status2 = gen2.generate_full_status()

        # Verify orders list has same length
        assert len(status1["orders"]) == len(status2["orders"])

        # Verify base prices match
        assert status1["market"]["last_prices"] == status2["market"]["last_prices"]

        # Verify position data matches
        assert status1["positions"]["positions"].keys() == status2["positions"]["positions"].keys()

    def test_different_seeds_produce_different_output(self) -> None:
        """Two generators with different seeds produce different output."""
        gen1 = MockDataGenerator(seed=42)
        gen2 = MockDataGenerator(seed=99)

        status1 = gen1.generate_full_status()
        status2 = gen2.generate_full_status()

        # Base prices should differ (initialized with random variance)
        assert status1["market"]["last_prices"] != status2["market"]["last_prices"]

    def test_unseeded_generator_produces_varying_output(self) -> None:
        """Unseeded generators produce different output each time."""
        gen1 = MockDataGenerator(seed=None)
        gen2 = MockDataGenerator(seed=None)

        status1 = gen1.generate_full_status()
        status2 = gen2.generate_full_status()

        # Very likely to differ (random initialization)
        # Note: There's a tiny chance they could match, but base_prices
        # use random.uniform so this is astronomically unlikely
        assert status1["market"]["last_prices"] != status2["market"]["last_prices"]

    def test_multiple_cycles_deterministic(self) -> None:
        """Multiple status generation cycles remain deterministic with same seed."""
        seed = 123

        gen1 = MockDataGenerator(seed=seed)
        gen2 = MockDataGenerator(seed=seed)

        # Generate multiple cycles
        for _ in range(3):
            status1 = gen1.generate_full_status()
            status2 = gen2.generate_full_status()

        # After 3 cycles, outputs should still match
        assert status1["market"]["last_prices"] == status2["market"]["last_prices"]
        assert len(status1["orders"]) == len(status2["orders"])
        assert status1["engine"]["cycle_count"] == status2["engine"]["cycle_count"]


class TestDemoBotSeeding:
    """Tests verifying DemoBot accepts and propagates seed."""

    def test_demo_bot_accepts_seed_parameter(self) -> None:
        """DemoBot can be initialized with a seed parameter."""
        from gpt_trader.tui.demo.demo_bot import DemoBot

        bot = DemoBot(seed=42)

        # Verify the seed was passed to the data generator
        assert bot.engine.status_reporter.data_generator.seed == 42

    def test_demo_bot_without_seed_uses_random(self) -> None:
        """DemoBot without seed creates generator with no seed."""
        from gpt_trader.tui.demo.demo_bot import DemoBot

        bot = DemoBot()

        # Generator should have no seed
        assert bot.engine.status_reporter.data_generator.seed is None

    def test_demo_bot_with_generator_ignores_seed(self) -> None:
        """When data_generator is provided, seed parameter is ignored."""
        from gpt_trader.tui.demo.demo_bot import DemoBot

        custom_gen = MockDataGenerator(seed=99)
        bot = DemoBot(data_generator=custom_gen, seed=42)

        # Should use the provided generator, not create new one with seed=42
        assert bot.engine.status_reporter.data_generator is custom_gen
        assert bot.engine.status_reporter.data_generator.seed == 99


class TestGenerateStrategyDataPerformance:
    """Tests for generate_strategy_data performance fields."""

    def test_generate_strategy_data_includes_performance_dicts(self) -> None:
        """generate_strategy_data includes both performance dicts."""
        gen = MockDataGenerator(seed=42)
        data = gen.generate_strategy_data()

        # Verify performance dict exists with expected keys
        assert "performance" in data
        perf = data["performance"]
        assert "win_rate" in perf
        assert "profit_factor" in perf
        assert "total_return" in perf
        assert "max_drawdown" in perf
        assert "total_trades" in perf

        # Verify backtest_performance dict exists with expected keys
        assert "backtest_performance" in data
        backtest = data["backtest_performance"]
        assert "win_rate" in backtest
        assert "profit_factor" in backtest
        assert "total_trades" in backtest

    def test_generate_strategy_data_performance_is_deterministic(self) -> None:
        """Performance dicts are deterministic with same seed."""
        seed = 42

        gen1 = MockDataGenerator(seed=seed)
        gen2 = MockDataGenerator(seed=seed)

        data1 = gen1.generate_strategy_data()
        data2 = gen2.generate_strategy_data()

        # Performance values should match exactly
        assert data1["performance"]["win_rate"] == data2["performance"]["win_rate"]
        assert data1["performance"]["total_trades"] == data2["performance"]["total_trades"]
        assert (
            data1["backtest_performance"]["win_rate"] == data2["backtest_performance"]["win_rate"]
        )
        assert (
            data1["backtest_performance"]["total_trades"]
            == data2["backtest_performance"]["total_trades"]
        )
