"""
Base class for ChatCFD agents.

Provides common functionality:
- Configuration management
- Logging with verbose mode
- Run manager integration

Usage:
    from agents.base import BaseAgent

    class MyAgent(BaseAgent):
        def run(self, **kwargs) -> MyResult:
            self.log("Starting...")
            ...
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, TypeVar, Generic

from config import AppConfig, load_config
from logging_utils import get_logger, LoggerAdapter
from run_manager import RunManager

# Type variable for agent result types
ResultT = TypeVar("ResultT")


@dataclass
class AgentResult:
    """Base result class for agents."""
    success: bool
    error: Optional[str] = None
    run_id: Optional[str] = None


class BaseAgent(ABC, Generic[ResultT]):
    """
    Base class for ChatCFD agents.

    Provides:
    - Configuration loading
    - Logging with verbose mode
    - Optional run manager integration

    Subclasses must implement:
    - run() method returning their specific result type
    - Optionally override _get_agent_name() for logging
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        run_manager: Optional[RunManager] = None,
        verbose: bool = True,
    ):
        """
        Initialize the agent.

        Args:
            config: Application config. Loads default if None.
            run_manager: RunManager for tracking runs. Creates one if None.
            verbose: Whether to log progress messages.
        """
        self.config = config or load_config()
        self.verbose = verbose

        # Setup logging
        self._logger = get_logger(f"agents.{self._get_agent_name()}")
        self.log = LoggerAdapter(self._logger, verbose=verbose)

        # Setup run manager
        if run_manager is not None:
            self.run_manager = run_manager
        else:
            self.run_manager = RunManager(self.config.paths.runs_dir)

    def _get_agent_name(self) -> str:
        """
        Get the agent name for logging.

        Override in subclasses for custom names.
        Default: lowercase class name without 'Agent' suffix.
        """
        name = self.__class__.__name__
        if name.endswith("Agent"):
            name = name[:-5]
        return name.lower()

    @abstractmethod
    def run(self, **kwargs) -> ResultT:
        """
        Run the agent's main task.

        Must be implemented by subclasses.

        Returns:
            Agent-specific result type.
        """
        pass

    def _log_init(self, **details) -> None:
        """
        Log agent initialization with optional details.

        Args:
            **details: Key-value pairs to log.
        """
        agent_name = self.__class__.__name__
        self.log(f"{agent_name} initialized")
        for key, value in details.items():
            self.log(f"  {key}: {value}")
