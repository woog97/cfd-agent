"""
Tests for state.py module.

Tests:
- CaseState dataclass
- StateUpdate TypedDict
- CaseBuilder
- NextStep enum
"""

import os
import sys
import pytest
from dataclasses import FrozenInstanceError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state import CaseState, StateUpdate, CaseBuilder, NextStep


# =============================================================================
# Test NextStep Enum
# =============================================================================

class TestNextStep:
    """Tests for NextStep enum."""

    def test_all_expected_steps_exist(self):
        """All expected workflow steps are defined."""
        expected = [
            "DETERMINE_FILES", "CONVERT_MESH", "EXTRACT_BOUNDARIES",
            "GENERATE_FILES", "WRITE_FILES", "RUN_SIMULATION",
            "ANALYZE_ERROR", "CORRECT_FILES", "REFLECT",
            "REWRITE_FILE", "ADD_FILE", "DONE", "FAILED",
        ]
        for name in expected:
            assert hasattr(NextStep, name), f"Missing step: {name}"

    def test_no_unexpected_steps(self):
        """No unexpected steps in the enum."""
        expected_count = 13
        assert len(NextStep) == expected_count, f"Expected {expected_count} steps, got {len(NextStep)}"

    def test_steps_are_unique(self):
        """Each step has a unique value."""
        values = [step.value for step in NextStep]
        assert len(values) == len(set(values)), "Duplicate step values found"

    def test_terminal_steps(self):
        """DONE and FAILED are terminal steps."""
        assert NextStep.DONE in NextStep
        assert NextStep.FAILED in NextStep


# =============================================================================
# Test CaseState
# =============================================================================

class TestCaseState:
    """Tests for CaseState dataclass."""

    def test_create_empty_state(self):
        """Can create state with defaults."""
        state = CaseState()

        assert state.case_name == ""
        assert state.solver == ""
        assert state.turbulence_model is None
        assert state.description == ""
        assert state.output_path == ""
        assert state.file_structure == []
        assert state.generated_files == {}
        assert state.grid_boundaries == []
        assert state.error_history == []
        assert state.current_error is None
        assert state.completed is False
        assert state.success is False
        assert state.attempt_count == 0
        assert state.max_attempts == 30

    def test_create_state_with_values(self):
        """Can create state with specific values."""
        state = CaseState(
            case_name="test_case",
            solver="simpleFoam",
            turbulence_model="kOmegaSST",
            description="A test case",
            max_attempts=10,
        )

        assert state.case_name == "test_case"
        assert state.solver == "simpleFoam"
        assert state.turbulence_model == "kOmegaSST"
        assert state.description == "A test case"
        assert state.max_attempts == 10

    def test_lists_are_independent(self):
        """Each state instance has independent lists."""
        state1 = CaseState()
        state2 = CaseState()

        state1.error_history.append("error1")
        state1.file_structure.append("file1")

        assert state2.error_history == []
        assert state2.file_structure == []

    def test_dicts_are_independent(self):
        """Each state instance has independent dicts."""
        state1 = CaseState()
        state2 = CaseState()

        state1.generated_files["test"] = "content"

        assert "test" not in state2.generated_files


class TestCaseStateApplyUpdate:
    """Tests for CaseState.apply_update method."""

    def test_apply_update_single_field(self):
        """apply_update updates a single field."""
        state = CaseState(solver="simpleFoam")

        update: StateUpdate = {"turbulence_model": "kOmegaSST"}
        new_state = state.apply_update(update)

        assert new_state.turbulence_model == "kOmegaSST"
        assert new_state.solver == "simpleFoam"  # Unchanged

    def test_apply_update_multiple_fields(self):
        """apply_update updates multiple fields at once."""
        state = CaseState()

        update: StateUpdate = {
            "solver": "pimpleFoam",
            "turbulence_model": "kEpsilon",
            "file_structure": ["0/U", "0/p"],
        }
        new_state = state.apply_update(update)

        assert new_state.solver == "pimpleFoam"
        assert new_state.turbulence_model == "kEpsilon"
        assert new_state.file_structure == ["0/U", "0/p"]

    def test_apply_update_returns_new_instance(self):
        """apply_update returns a new instance (immutable pattern)."""
        state = CaseState(solver="simpleFoam")
        original_id = id(state)

        new_state = state.apply_update({"turbulence_model": "kOmegaSST"})

        # Immutable: returns different instance
        assert id(new_state) != original_id
        # Original unchanged
        assert state.turbulence_model is None
        # New state has update
        assert new_state.turbulence_model == "kOmegaSST"

    def test_apply_update_overwrites_lists(self):
        """apply_update replaces entire list, doesn't append."""
        state = CaseState(file_structure=["file1"])

        new_state = state.apply_update({"file_structure": ["file2", "file3"]})

        assert new_state.file_structure == ["file2", "file3"]

    def test_apply_update_unknown_key_goes_to_store(self):
        """Unknown keys are stored in _store."""
        state = CaseState()

        new_state = state.apply_update({"custom_key": "custom_value"})

        # Check on new_state, not original (immutable)
        assert new_state.get("custom_key") == "custom_value"
        # Original unchanged
        assert state.get("custom_key") is None


class TestCaseStateStore:
    """Tests for CaseState store functionality."""

    def test_set_and_get(self):
        """Can store and retrieve arbitrary data."""
        state = CaseState()

        # set() returns new state (immutable)
        new_state = state.set("my_key", {"data": 123})
        result = new_state.get("my_key")

        assert result == {"data": 123}
        # Original unchanged
        assert state.get("my_key") is None

    def test_get_default_for_missing_key(self):
        """get returns default for missing keys."""
        state = CaseState()

        result = state.get("missing_key", default="default_value")

        assert result == "default_value"

    def test_get_none_for_missing_key_no_default(self):
        """get returns None if key missing and no default."""
        state = CaseState()

        result = state.get("missing_key")

        assert result is None

    def test_store_complex_objects(self):
        """Can store complex objects like lists and nested dicts."""
        state = CaseState()

        complex_data = {
            "list": [1, 2, 3],
            "nested": {"a": {"b": "c"}},
            "tuple": (1, 2),
        }
        new_state = state.set("complex", complex_data)

        assert new_state.get("complex") == complex_data


class TestCaseStateConsecutiveSameErrors:
    """Tests for CaseState.consecutive_same_errors method."""

    def test_empty_history_returns_1(self):
        """No errors returns 1."""
        state = CaseState()
        assert state.consecutive_same_errors() == 1

    def test_single_error_returns_1(self):
        """Single error returns 1."""
        state = CaseState(error_history=["error A"])
        assert state.consecutive_same_errors() == 1

    def test_different_errors_returns_1(self):
        """Different consecutive errors returns 1."""
        state = CaseState(error_history=["error A", "error B"])
        assert state.consecutive_same_errors() == 1

    def test_two_same_errors_returns_2(self):
        """Two same errors returns 2."""
        state = CaseState(error_history=["error A", "error A"])
        assert state.consecutive_same_errors() == 2

    def test_three_same_errors_returns_3(self):
        """Three same errors returns 3."""
        state = CaseState(error_history=[
            "error A with details",
            "error A with details",
            "error A with details",
        ])
        assert state.consecutive_same_errors() == 3

    def test_compares_first_100_chars(self):
        """Comparison uses first 100 characters only."""
        prefix = "X" * 100
        state = CaseState(error_history=[
            prefix + "different ending 1",
            prefix + "different ending 2",
        ])
        assert state.consecutive_same_errors() == 2

    def test_resets_on_different_error(self):
        """Count resets when a different error appears."""
        state = CaseState(error_history=[
            "error A",
            "error A",
            "error B",  # Different
            "error A",
            "error A",
        ])
        # Should count from end: error A, error A = 2
        assert state.consecutive_same_errors() == 2


class TestCaseStateShouldStop:
    """Tests for CaseState.should_stop method."""

    def test_not_complete_not_at_max(self):
        """Returns False when not complete and under max attempts."""
        state = CaseState(completed=False, attempt_count=5, max_attempts=30)
        assert state.should_stop() is False

    def test_completed_returns_true(self):
        """Returns True when completed."""
        state = CaseState(completed=True, attempt_count=5, max_attempts=30)
        assert state.should_stop() is True

    def test_at_max_attempts_returns_true(self):
        """Returns True when at max attempts."""
        state = CaseState(completed=False, attempt_count=30, max_attempts=30)
        assert state.should_stop() is True

    def test_over_max_attempts_returns_true(self):
        """Returns True when over max attempts."""
        state = CaseState(completed=False, attempt_count=35, max_attempts=30)
        assert state.should_stop() is True


# =============================================================================
# Test CaseBuilder
# =============================================================================

class TestCaseBuilder:
    """Tests for CaseBuilder."""

    def test_create_builder(self):
        """Can create a builder with required parameters."""
        builder = CaseBuilder(
            solver="simpleFoam",
            turbulence_model="kOmegaSST",
            other_physical_model=None,
            description="Test case",
            grid_path="/path/to/mesh.msh",
            grid_type="msh",
            output_dir="/tmp/output",
        )

        assert builder.solver == "simpleFoam"
        assert builder.turbulence_model == "kOmegaSST"
        assert builder.grid_type == "msh"

    def test_build_creates_state(self):
        """build() creates a properly initialized CaseState."""
        builder = CaseBuilder(
            solver="simpleFoam",
            turbulence_model="kOmegaSST",
            other_physical_model=None,
            description="Test case description",
            grid_path="/path/to/mesh.msh",
            grid_type="msh",
            output_dir="/tmp/output",
        )

        state = builder.build("my_case")

        assert state.case_name == "my_case"
        assert state.solver == "simpleFoam"
        assert state.turbulence_model == "kOmegaSST"
        assert state.description == "Test case description"
        assert state.grid_path == "/path/to/mesh.msh"
        assert state.grid_type == "msh"
        assert "/tmp/output/my_case" in state.output_path

    def test_build_creates_independent_states(self):
        """Each build() call creates an independent state."""
        builder = CaseBuilder(
            solver="simpleFoam",
            turbulence_model="kOmegaSST",
            other_physical_model=None,
            description="Test",
            grid_path="/path/to/mesh.msh",
            grid_type="msh",
            output_dir="/tmp/output",
        )

        state1 = builder.build("case1")
        state2 = builder.build("case2")

        state1.file_structure.append("test_file")

        assert state1.case_name == "case1"
        assert state2.case_name == "case2"
        assert "test_file" not in state2.file_structure

    def test_build_with_custom_max_attempts(self):
        """Can override max_attempts in builder."""
        builder = CaseBuilder(
            solver="simpleFoam",
            turbulence_model=None,
            other_physical_model=None,
            description="Test",
            grid_path="/path/to/mesh.msh",
            grid_type="msh",
            output_dir="/tmp/output",
            max_attempts=50,
        )

        state = builder.build("custom_case")

        assert state.max_attempts == 50

    def test_build_with_other_physical_model(self):
        """Can specify other physical models."""
        builder = CaseBuilder(
            solver="interFoam",
            turbulence_model="kOmegaSST",
            other_physical_model=["multiphase", "VOF"],
            description="Multiphase test",
            grid_path="/path/to/mesh.msh",
            grid_type="msh",
            output_dir="/tmp/output",
        )

        state = builder.build("multiphase_case")

        assert state.other_physical_model == ["multiphase", "VOF"]


# =============================================================================
# Test StateUpdate TypedDict
# =============================================================================

class TestStateUpdate:
    """Tests for StateUpdate TypedDict."""

    def test_can_create_partial_update(self):
        """StateUpdate allows partial updates."""
        update: StateUpdate = {
            "solver": "simpleFoam",
        }
        assert update["solver"] == "simpleFoam"

    def test_can_include_multiple_fields(self):
        """StateUpdate can include multiple fields."""
        update: StateUpdate = {
            "solver": "simpleFoam",
            "turbulence_model": "kOmegaSST",
            "file_structure": ["0/U", "0/p"],
            "completed": True,
        }

        assert update["solver"] == "simpleFoam"
        assert update["turbulence_model"] == "kOmegaSST"
        assert update["file_structure"] == ["0/U", "0/p"]
        assert update["completed"] is True


# =============================================================================
# Integration Tests
# =============================================================================

class TestStateIntegration:
    """Integration tests for state module."""

    def test_state_through_multiple_updates(self):
        """State accumulates correctly through multiple updates."""
        state = CaseState(case_name="test", solver="simpleFoam")

        # Simulate workflow
        state = state.apply_update({"file_structure": ["0/U", "0/p"]})
        state = state.apply_update({"mesh_converted": True})
        state = state.apply_update({"grid_boundaries": ["inlet", "outlet", "wall"]})
        state = state.apply_update({
            "generated_files": {"0/U": "content1", "0/p": "content2"}
        })

        assert state.solver == "simpleFoam"
        assert state.file_structure == ["0/U", "0/p"]
        assert state.mesh_converted is True
        assert state.grid_boundaries == ["inlet", "outlet", "wall"]
        assert len(state.generated_files) == 2

    def test_error_history_accumulates(self):
        """Error history accumulates correctly."""
        state = CaseState()

        state = state.apply_update({"error_history": ["error 1"]})
        assert state.error_history == ["error 1"]

        # Add another error by appending
        state = state.apply_update({
            "error_history": state.error_history + ["error 2"]
        })
        assert state.error_history == ["error 1", "error 2"]

    def test_builder_to_state_to_updates(self):
        """Full flow from builder through updates."""
        builder = CaseBuilder(
            solver="pimpleFoam",
            turbulence_model="kOmegaSST",
            other_physical_model=None,
            description="Transient flow",
            grid_path="/tmp/mesh.msh",
            grid_type="msh",
            output_dir="/tmp/out",
        )

        state = builder.build("transient_test")

        # Apply updates as workflow progresses
        state = state.apply_update({
            "file_structure": ["0/U", "0/p", "0/k", "0/omega"],
        })
        state = state.apply_update({
            "error_history": ["Error 1"],
            "current_error": "Error 1",
        })
        # Use apply_update for immutable increment
        state = state.apply_update({"attempt_count": state.attempt_count + 1})

        state = state.apply_update({
            "error_history": state.error_history + ["Error 1"],
        })

        assert state.solver == "pimpleFoam"
        assert state.consecutive_same_errors() == 2
        assert not state.should_stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
