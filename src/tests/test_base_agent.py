"""
Tests for agents/base.py module.
"""

import logging
import sys
import os
import tempfile

import pytest

# Add src2 to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base import BaseAgent, AgentResult
from config import load_config, AppConfig
from run_manager import RunManager


class ConcreteAgent(BaseAgent[AgentResult]):
    """Concrete implementation for testing."""

    def run(self, **kwargs) -> AgentResult:
        self.log("Running concrete agent")
        return AgentResult(success=True, run_id="test-123")


class CustomNameAgent(BaseAgent[AgentResult]):
    """Agent with custom name."""

    def _get_agent_name(self) -> str:
        return "custom"

    def run(self, **kwargs) -> AgentResult:
        return AgentResult(success=True)


class TestAgentResult:
    """Test AgentResult dataclass."""

    def test_default_values(self):
        result = AgentResult(success=True)
        assert result.success is True
        assert result.error is None
        assert result.run_id is None

    def test_with_error(self):
        result = AgentResult(success=False, error="Something went wrong")
        assert result.success is False
        assert result.error == "Something went wrong"

    def test_with_run_id(self):
        result = AgentResult(success=True, run_id="run-456")
        assert result.run_id == "run-456"


class TestBaseAgent:
    """Test BaseAgent base class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temp directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def config(self):
        """Load test config."""
        return load_config()

    def test_init_with_default_config(self):
        agent = ConcreteAgent()
        assert agent.config is not None
        assert agent.verbose is True

    def test_init_with_custom_config(self, config):
        agent = ConcreteAgent(config=config, verbose=False)
        assert agent.config is config
        assert agent.verbose is False

    def test_init_creates_run_manager(self, config):
        agent = ConcreteAgent(config=config)
        assert agent.run_manager is not None
        assert isinstance(agent.run_manager, RunManager)

    def test_init_uses_provided_run_manager(self, temp_dir):
        custom_manager = RunManager(temp_dir)
        agent = ConcreteAgent(run_manager=custom_manager)
        assert agent.run_manager is custom_manager

    def test_get_agent_name_default(self):
        agent = ConcreteAgent()
        assert agent._get_agent_name() == "concrete"

    def test_get_agent_name_custom(self):
        agent = CustomNameAgent()
        assert agent._get_agent_name() == "custom"

    def test_log_adapter_created(self):
        agent = ConcreteAgent()
        assert agent.log is not None

    def test_run_abstract_method(self):
        agent = ConcreteAgent()
        result = agent.run()
        assert result.success is True
        assert result.run_id == "test-123"

    def test_log_init_logs_details(self, caplog):
        with caplog.at_level(logging.INFO, logger="chatcfd.agents.concrete"):
            agent = ConcreteAgent(verbose=True)
            agent._log_init(foo="bar", baz=123)

        assert "ConcreteAgent initialized" in caplog.text
        assert "foo: bar" in caplog.text
        assert "baz: 123" in caplog.text

    def test_verbose_false_suppresses_init_log(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="chatcfd.agents.concrete"):
            agent = ConcreteAgent(verbose=False)
            agent._log_init(key="value")

        # With verbose=False, _log_init should not log
        assert "key: value" not in caplog.text


class TestMeshingAgentInheritance:
    """Test that MeshingAgent properly inherits from BaseAgent."""

    def test_meshing_agent_is_base_agent(self):
        from agents.meshing import MeshingAgent

        agent = MeshingAgent(verbose=False)
        assert isinstance(agent, BaseAgent)
        assert hasattr(agent, "log")
        assert hasattr(agent, "run_manager")


class TestSolvingAgentInheritance:
    """Test that SolvingAgent properly inherits from BaseAgent."""

    def test_solving_agent_is_base_agent(self):
        from agents.solving import SolvingAgent

        agent = SolvingAgent(verbose=False)
        assert isinstance(agent, BaseAgent)
        assert hasattr(agent, "log")
        assert hasattr(agent, "run_manager")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
