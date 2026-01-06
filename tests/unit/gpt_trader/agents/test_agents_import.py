"""
Stub tests for gpt_trader.agents module.

These tests verify the module can be imported and basic functionality works.
Add more comprehensive tests as the module evolves.
"""


class TestAgentsModuleImport:
    """Test that the agents module can be imported."""

    def test_agents_module_imports(self) -> None:
        """Verify the agents module is importable."""
        import gpt_trader.agents

        assert gpt_trader.agents is not None

    def test_agents_cli_imports(self) -> None:
        """Verify the CLI module is importable."""
        from gpt_trader.agents import cli

        assert cli is not None

    def test_agents_has_docstring(self) -> None:
        """Verify module has documentation."""
        import gpt_trader.agents

        assert gpt_trader.agents.__doc__ is not None
        assert "Agent tooling" in gpt_trader.agents.__doc__
