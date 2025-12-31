"""
ChatCFD src2: Clean architecture refactor.

This is a reimplementation of ChatCFD following patterns from:
- UK AISI's Inspect AI (TaskState, Solver pattern)
- Tinker Cookbook (Builder pattern for environments)

Key design principles:
1. No global mutable state - all state in CaseState
2. Explicit dependencies via Dependencies container
3. Workflow as explicit state machine
4. Step functions return partial updates, don't mutate

Modules:
- state.py: CaseState, StateUpdate, CaseBuilder, NextStep enum
- config.py: Immutable AppConfig, PathConfig, LLMConfig, OpenFOAMConstants
- llm.py: LLMClient, ConversationManager
- database.py: ReferenceDatabase for OpenFOAM tutorials
- steps.py: Step functions that transform CaseState
- workflow.py: WorkflowRunner and routing logic
- main.py: CLI and programmatic entry points
"""

from state import CaseState, StateUpdate, CaseBuilder, NextStep
from config import AppConfig, load_config
from llm import LLMClient
from database import ReferenceDatabase
from steps import Dependencies
from workflow import run_workflow, WorkflowRunner
from main import run_case

__all__ = [
    # State
    "CaseState",
    "StateUpdate",
    "CaseBuilder",
    "NextStep",
    # Config
    "AppConfig",
    "load_config",
    # LLM
    "LLMClient",
    # Database
    "ReferenceDatabase",
    # Steps
    "Dependencies",
    # Workflow
    "run_workflow",
    "WorkflowRunner",
    # Main
    "run_case",
]
