"""
CaseState: Single source of truth for all mutable state during case execution.

Inspired by Inspect AI's TaskState pattern - explicit state container passed
through all step functions, replacing global variables.

Design:
- CaseState is frozen (immutable) to prevent accidental mutation
- All updates create new instances via apply_update()
- This makes state changes explicit and enables safe parallel execution
"""

from dataclasses import dataclass, field, replace
from typing import Any, TypeVar, TypedDict
from enum import Enum, auto


T = TypeVar("T")


class NextStep(Enum):
    """Explicit workflow routing - what to do next."""
    DETERMINE_FILES = auto()
    CONVERT_MESH = auto()
    EXTRACT_BOUNDARIES = auto()
    GENERATE_FILES = auto()
    WRITE_FILES = auto()
    RUN_SIMULATION = auto()
    ANALYZE_ERROR = auto()
    CORRECT_FILES = auto()
    REFLECT = auto()
    REWRITE_FILE = auto()
    ADD_FILE = auto()
    DONE = auto()
    FAILED = auto()


class StateUpdate(TypedDict, total=False):
    """
    Partial state update returned by step functions.

    Step functions return what changed rather than mutating globals.
    This makes data flow explicit and testable.
    """
    # Case identification
    case_name: str
    solver: str
    turbulence_model: str | None
    description: str

    # File structure
    file_structure: list[str]
    reference_file_name: str
    generated_files: dict[str, str]

    # Mesh state
    grid_boundaries: list[str]
    mesh_converted: bool

    # Error handling
    error_history: list[str]
    correction_trajectory: list[dict[str, list[str]]]
    current_error: str | None
    error_file: str | None

    # Reflection
    reflection_history: list[dict[str, str]]

    # Control flow
    completed: bool
    success: bool
    needs_new_file: bool
    new_file_name: str | None


@dataclass(frozen=True)
class CaseState:
    """
    Immutable state for a single case execution.

    This replaces the 80+ global variables in config.py with a single,
    typed, explicit state container that is passed through all functions.

    Design principles:
    - All state is in one place
    - State is passed explicitly (no hidden dependencies)
    - Updates create new instances (immutable)
    - Can run multiple cases in parallel (each has own state)
    - Changes are explicit and trackable
    """

    # === Case Identification (set once at start) ===
    case_name: str = ""
    solver: str = ""
    turbulence_model: str | None = None
    other_physical_model: list[str] | None = None
    description: str = ""
    output_path: str = ""

    # === File Structure ===
    file_structure: list[str] = field(default_factory=list)
    reference_file_name: str = ""
    generated_files: dict[str, str] = field(default_factory=dict)

    # === Mesh State ===
    grid_path: str = ""
    grid_type: str = "msh"  # "msh" or "polyMesh"
    grid_boundaries: list[str] = field(default_factory=list)
    mesh_converted: bool = False

    # === Error Handling ===
    error_history: list[str] = field(default_factory=list)
    correction_trajectory: list[dict[str, list[str]]] = field(default_factory=list)
    current_error: str | None = None
    error_file: str | None = None

    # === Reflection State ===
    reflection_history: list[dict[str, str]] = field(default_factory=list)

    # === Control Flow ===
    completed: bool = False
    success: bool = False
    attempt_count: int = 0
    max_attempts: int = 30
    needs_new_file: bool = False
    new_file_name: str | None = None

    # === Typed Store for Extension Data ===
    # Like Inspect AI's Store - for arbitrary typed data
    # Note: tuple is used for immutability; convert to dict when accessing
    _store: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    def get(self, key: str, default: T = None) -> T:
        """Get a value from the store with type hint."""
        store_dict = dict(self._store)
        return store_dict.get(key, default)

    def set(self, key: str, value: Any) -> "CaseState":
        """
        Set a value in the store, returning new CaseState.

        Args:
            key: Store key
            value: Value to store

        Returns:
            New CaseState with updated store.
        """
        store_dict = dict(self._store)
        store_dict[key] = value
        return replace(self, _store=tuple(store_dict.items()))

    def apply_update(self, update: StateUpdate) -> "CaseState":
        """
        Apply a partial update to the state.

        Creates a new CaseState instance with updated values.
        This is the only way state should be modified during execution,
        making all changes explicit and trackable.

        Args:
            update: Dictionary of field names to new values.

        Returns:
            New CaseState with updates applied.
        """
        # Separate known fields from store updates
        field_updates = {}
        store_updates = {}

        for key, value in update.items():
            if hasattr(self, key) and key != "_store":
                field_updates[key] = value
            else:
                store_updates[key] = value

        # Create new instance with field updates
        new_state = replace(self, **field_updates) if field_updates else self

        # Apply store updates if any
        if store_updates:
            store_dict = dict(new_state._store)
            store_dict.update(store_updates)
            new_state = replace(new_state, _store=tuple(store_dict.items()))

        return new_state

    def consecutive_same_errors(self) -> int:
        """Count how many times the same error has occurred consecutively."""
        if len(self.error_history) < 2:
            return 1

        count = 1
        current = self.error_history[-1]
        for error in reversed(self.error_history[:-1]):
            # Compare first 100 chars to handle minor variations
            if error[:100] == current[:100]:
                count += 1
            else:
                break
        return count

    def should_stop(self) -> bool:
        """Check if we should stop the workflow."""
        return self.completed or self.attempt_count >= self.max_attempts


@dataclass
class CaseBuilder:
    """
    Immutable configuration for creating case runs.

    Inspired by Tinker's EnvBuilder pattern - configuration is separate
    from runtime state, and each run gets a fresh state object.
    """
    solver: str
    turbulence_model: str | None
    other_physical_model: list[str] | None
    description: str
    grid_path: str
    grid_type: str
    output_dir: str
    max_attempts: int = 30

    def build(self, case_name: str) -> CaseState:
        """Create a fresh CaseState for a new run."""
        import os
        output_path = os.path.join(self.output_dir, case_name)

        return CaseState(
            case_name=case_name,
            solver=self.solver,
            turbulence_model=self.turbulence_model,
            other_physical_model=self.other_physical_model,
            description=self.description,
            output_path=output_path,
            grid_path=self.grid_path,
            grid_type=self.grid_type,
            max_attempts=self.max_attempts,
        )
