"""
Tests for workflow.py module.

Tests:
- Routing functions
- STEPS registry
- ROUTERS registry
- WorkflowRunner
- run_workflow function
"""

import os
import sys
import pytest
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state import CaseState, NextStep
from workflow import (
    STEPS,
    ROUTERS,
    route_initial,
    route_after_determine_files,
    route_after_convert_mesh,
    route_after_extract_boundaries,
    route_after_generate_files,
    route_after_write_files,
    route_after_run,
    route_after_analyze,
    route_after_correct,
    route_after_reflect,
    route_after_rewrite,
    route_after_add_file,
    WorkflowRunner,
    run_workflow,
)


# =============================================================================
# Test STEPS Registry
# =============================================================================

class TestStepsRegistry:
    """Tests for STEPS registry."""

    def test_all_steps_have_implementations(self):
        """Every non-terminal step has an implementation."""
        terminal_steps = {NextStep.DONE, NextStep.FAILED}

        for step in NextStep:
            if step not in terminal_steps:
                assert step in STEPS, f"No implementation for {step}"

    def test_implementations_are_callable(self):
        """All step implementations are callable."""
        for step, impl in STEPS.items():
            assert callable(impl), f"{step} implementation is not callable"

    def test_expected_steps_registered(self):
        """Expected steps are in registry."""
        expected = [
            NextStep.DETERMINE_FILES,
            NextStep.CONVERT_MESH,
            NextStep.EXTRACT_BOUNDARIES,
            NextStep.GENERATE_FILES,
            NextStep.WRITE_FILES,
            NextStep.RUN_SIMULATION,
            NextStep.ANALYZE_ERROR,
            NextStep.CORRECT_FILES,
            NextStep.REFLECT,
            NextStep.REWRITE_FILE,
            NextStep.ADD_FILE,
        ]

        for step in expected:
            assert step in STEPS, f"Missing step: {step}"


# =============================================================================
# Test ROUTERS Registry
# =============================================================================

class TestRoutersRegistry:
    """Tests for ROUTERS registry."""

    def test_all_steps_have_routers(self):
        """Every non-terminal step has a router."""
        terminal_steps = {NextStep.DONE, NextStep.FAILED}

        for step in NextStep:
            if step not in terminal_steps:
                assert step in ROUTERS, f"No router for {step}"

    def test_routers_are_callable(self):
        """All routers are callable."""
        for step, router in ROUTERS.items():
            assert callable(router), f"{step} router is not callable"

    def test_routers_return_next_step(self):
        """Routers return NextStep enum values."""
        state = CaseState()

        for step, router in ROUTERS.items():
            result = router(state)
            assert isinstance(result, NextStep), \
                f"{step} router returned {type(result)}, expected NextStep"


# =============================================================================
# Test route_initial
# =============================================================================

class TestRouteInitial:
    """Tests for route_initial function."""

    def test_always_returns_determine_files(self):
        """Always starts with DETERMINE_FILES."""
        state = CaseState()
        assert route_initial(state) == NextStep.DETERMINE_FILES

    def test_ignores_state_content(self):
        """Result doesn't depend on state content."""
        state = CaseState(
            solver="simpleFoam",
            completed=True,  # Even if marked complete
            success=True,
        )
        assert route_initial(state) == NextStep.DETERMINE_FILES


# =============================================================================
# Test route_after_determine_files
# =============================================================================

class TestRouteAfterDetermineFiles:
    """Tests for route_after_determine_files function."""

    def test_msh_goes_to_convert_mesh(self):
        """MSH grid type goes to CONVERT_MESH."""
        state = CaseState(grid_type="msh")
        assert route_after_determine_files(state) == NextStep.CONVERT_MESH

    def test_polymesh_skips_conversion(self):
        """polyMesh type goes directly to EXTRACT_BOUNDARIES."""
        state = CaseState(grid_type="polyMesh")
        assert route_after_determine_files(state) == NextStep.EXTRACT_BOUNDARIES

    def test_other_types_skip_conversion(self):
        """Other grid types skip conversion."""
        state = CaseState(grid_type="other")
        assert route_after_determine_files(state) == NextStep.EXTRACT_BOUNDARIES


# =============================================================================
# Test route_after_convert_mesh
# =============================================================================

class TestRouteAfterConvertMesh:
    """Tests for route_after_convert_mesh function."""

    def test_success_goes_to_extract_boundaries(self):
        """Successful conversion goes to EXTRACT_BOUNDARIES."""
        state = CaseState(mesh_converted=True)
        assert route_after_convert_mesh(state) == NextStep.EXTRACT_BOUNDARIES

    def test_failure_goes_to_failed(self):
        """Failed conversion goes to FAILED."""
        state = CaseState(mesh_converted=False)
        assert route_after_convert_mesh(state) == NextStep.FAILED


# =============================================================================
# Test route_after_extract_boundaries
# =============================================================================

class TestRouteAfterExtractBoundaries:
    """Tests for route_after_extract_boundaries function."""

    def test_goes_to_generate_files(self):
        """Always goes to GENERATE_FILES."""
        state = CaseState()
        assert route_after_extract_boundaries(state) == NextStep.GENERATE_FILES


# =============================================================================
# Test route_after_generate_files
# =============================================================================

class TestRouteAfterGenerateFiles:
    """Tests for route_after_generate_files function."""

    def test_goes_to_write_files(self):
        """Always goes to WRITE_FILES."""
        state = CaseState()
        assert route_after_generate_files(state) == NextStep.WRITE_FILES


# =============================================================================
# Test route_after_write_files
# =============================================================================

class TestRouteAfterWriteFiles:
    """Tests for route_after_write_files function."""

    def test_goes_to_run_simulation(self):
        """Always goes to RUN_SIMULATION."""
        state = CaseState()
        assert route_after_write_files(state) == NextStep.RUN_SIMULATION


# =============================================================================
# Test route_after_run
# =============================================================================

class TestRouteAfterRun:
    """Tests for route_after_run function - main decision point."""

    def test_success_goes_to_done(self):
        """Successful run goes to DONE."""
        state = CaseState(success=True)
        assert route_after_run(state) == NextStep.DONE

    def test_max_attempts_goes_to_failed(self):
        """Max attempts reached goes to FAILED."""
        state = CaseState(attempt_count=30, max_attempts=30)
        assert route_after_run(state) == NextStep.FAILED

    def test_completed_goes_to_failed(self):
        """Completed but not successful goes to FAILED."""
        state = CaseState(completed=True, success=False)
        assert route_after_run(state) == NextStep.FAILED

    def test_four_same_errors_goes_to_rewrite(self):
        """4+ same errors goes to REWRITE_FILE."""
        error = "FOAM FATAL ERROR: " + "x" * 100
        state = CaseState(
            error_history=[error, error, error, error],
            current_error=error,
        )
        assert route_after_run(state) == NextStep.REWRITE_FILE

    def test_two_same_errors_goes_to_reflect(self):
        """2-3 same errors goes to REFLECT."""
        error = "FOAM FATAL ERROR: " + "x" * 100
        state = CaseState(
            error_history=[error, error],
            current_error=error,
        )
        assert route_after_run(state) == NextStep.REFLECT

    def test_three_same_errors_goes_to_reflect(self):
        """3 same errors still goes to REFLECT (not REWRITE)."""
        error = "FOAM FATAL ERROR: " + "x" * 100
        state = CaseState(
            error_history=[error, error, error],
            current_error=error,
        )
        assert route_after_run(state) == NextStep.REFLECT

    def test_first_error_goes_to_analyze(self):
        """First error goes to ANALYZE_ERROR."""
        state = CaseState(
            error_history=["Some error"],
            current_error="Some error",
        )
        assert route_after_run(state) == NextStep.ANALYZE_ERROR

    def test_different_errors_stay_on_analyze(self):
        """Different consecutive errors keep analyzing."""
        state = CaseState(
            error_history=["Error type A", "Error type B"],
            current_error="Error type B",
        )
        assert route_after_run(state) == NextStep.ANALYZE_ERROR


# =============================================================================
# Test route_after_analyze
# =============================================================================

class TestRouteAfterAnalyze:
    """Tests for route_after_analyze function."""

    def test_with_error_file_goes_to_correct(self):
        """If error file identified, go to CORRECT_FILES."""
        state = CaseState(error_file="system/fvSolution")
        assert route_after_analyze(state) == NextStep.CORRECT_FILES

    def test_without_error_file_goes_to_reflect(self):
        """If no error file identified, go to REFLECT."""
        state = CaseState(error_file=None)
        assert route_after_analyze(state) == NextStep.REFLECT


# =============================================================================
# Test route_after_correct
# =============================================================================

class TestRouteAfterCorrect:
    """Tests for route_after_correct function."""

    def test_goes_to_run_simulation(self):
        """After correction, try running again."""
        state = CaseState()
        assert route_after_correct(state) == NextStep.RUN_SIMULATION


# =============================================================================
# Test route_after_reflect
# =============================================================================

class TestRouteAfterReflect:
    """Tests for route_after_reflect function."""

    def test_goes_to_analyze_error(self):
        """After reflection, analyze with new insights."""
        state = CaseState()
        assert route_after_reflect(state) == NextStep.ANALYZE_ERROR


# =============================================================================
# Test route_after_rewrite
# =============================================================================

class TestRouteAfterRewrite:
    """Tests for route_after_rewrite function."""

    def test_goes_to_run_simulation(self):
        """After rewrite, try running again."""
        state = CaseState()
        assert route_after_rewrite(state) == NextStep.RUN_SIMULATION


# =============================================================================
# Test route_after_add_file
# =============================================================================

class TestRouteAfterAddFile:
    """Tests for route_after_add_file function."""

    def test_goes_to_run_simulation(self):
        """After adding file, try running again."""
        state = CaseState()
        assert route_after_add_file(state) == NextStep.RUN_SIMULATION


# =============================================================================
# Test WorkflowRunner
# =============================================================================

class TestWorkflowRunner:
    """Tests for WorkflowRunner class."""

    def test_init(self):
        """Can create WorkflowRunner."""
        mock_deps = Mock()
        runner = WorkflowRunner(mock_deps, verbose=False)

        assert runner.deps == mock_deps
        assert runner.verbose is False

    def test_stops_at_done(self):
        """Runner stops when reaching DONE."""
        mock_deps = Mock()

        # Mock all steps to set success=True
        with patch.dict(STEPS, {
            NextStep.DETERMINE_FILES: lambda s, d: {"file_structure": ["test"]},
            NextStep.EXTRACT_BOUNDARIES: lambda s, d: {"grid_boundaries": ["inlet"]},
            NextStep.GENERATE_FILES: lambda s, d: {"generated_files": {"test": "content"}},
            NextStep.WRITE_FILES: lambda s, d: {},
            NextStep.RUN_SIMULATION: lambda s, d: {"success": True, "completed": True},
        }):
            runner = WorkflowRunner(mock_deps, verbose=False)
            state = CaseState(grid_type="polyMesh")

            final_state = runner.run(state)

            assert final_state.success is True

    def test_stops_at_max_attempts(self):
        """Runner stops when max attempts reached."""
        mock_deps = Mock()

        state = CaseState(
            solver="simpleFoam",
            attempt_count=30,
            max_attempts=30,
        )

        assert state.should_stop() is True

    def test_increments_attempt_count(self):
        """Runner increments attempt count on simulation."""
        mock_deps = Mock()

        # Mock steps
        with patch.dict(STEPS, {
            NextStep.DETERMINE_FILES: lambda s, d: {},
            NextStep.EXTRACT_BOUNDARIES: lambda s, d: {},
            NextStep.GENERATE_FILES: lambda s, d: {},
            NextStep.WRITE_FILES: lambda s, d: {},
            NextStep.RUN_SIMULATION: lambda s, d: {"success": True, "completed": True},
        }):
            runner = WorkflowRunner(mock_deps, verbose=False)
            state = CaseState(grid_type="polyMesh")

            final_state = runner.run(state)

            assert final_state.attempt_count == 1


# =============================================================================
# Test run_workflow Function
# =============================================================================

class TestRunWorkflow:
    """Tests for run_workflow convenience function."""

    def test_creates_runner_and_runs(self):
        """run_workflow creates runner and executes."""
        mock_deps = Mock()

        with patch.dict(STEPS, {
            NextStep.DETERMINE_FILES: lambda s, d: {},
            NextStep.EXTRACT_BOUNDARIES: lambda s, d: {},
            NextStep.GENERATE_FILES: lambda s, d: {},
            NextStep.WRITE_FILES: lambda s, d: {},
            NextStep.RUN_SIMULATION: lambda s, d: {"success": True, "completed": True},
        }):
            state = CaseState(grid_type="polyMesh")

            final_state = run_workflow(state, mock_deps, verbose=False)

            assert final_state.success is True


# =============================================================================
# Integration Tests
# =============================================================================

class TestWorkflowIntegration:
    """Integration tests for workflow module."""

    def test_error_correction_flow(self):
        """Test workflow through error correction cycle."""
        mock_deps = Mock()

        call_count = {"run": 0, "analyze": 0, "correct": 0}

        def mock_run(state, deps):
            call_count["run"] += 1
            if call_count["run"] >= 2:
                return {"success": True, "completed": True}
            return {
                "current_error": "Test error",
                "error_history": state.error_history + ["Test error"],
            }

        def mock_analyze(state, deps):
            call_count["analyze"] += 1
            return {"error_file": "0/U"}

        def mock_correct(state, deps):
            call_count["correct"] += 1
            return {}

        with patch.dict(STEPS, {
            NextStep.DETERMINE_FILES: lambda s, d: {},
            NextStep.EXTRACT_BOUNDARIES: lambda s, d: {},
            NextStep.GENERATE_FILES: lambda s, d: {},
            NextStep.WRITE_FILES: lambda s, d: {},
            NextStep.RUN_SIMULATION: mock_run,
            NextStep.ANALYZE_ERROR: mock_analyze,
            NextStep.CORRECT_FILES: mock_correct,
        }):
            state = CaseState(grid_type="polyMesh")

            final_state = run_workflow(state, mock_deps, verbose=False)

            assert final_state.success is True
            assert call_count["run"] == 2
            assert call_count["analyze"] == 1
            assert call_count["correct"] == 1

    def test_workflow_graph_consistency(self):
        """Verify workflow graph has no dead ends."""
        state = CaseState()

        # Check all routers can be called
        for step, router in ROUTERS.items():
            next_step = router(state)
            # Next step should either be terminal or have implementation
            assert next_step in {NextStep.DONE, NextStep.FAILED} or next_step in STEPS, \
                f"Router for {step} returns {next_step} which has no implementation"

    def test_routing_logic_completeness(self):
        """All step transitions are defined."""
        terminal_steps = {NextStep.DONE, NextStep.FAILED}

        for step in NextStep:
            if step not in terminal_steps:
                # Should have both implementation and router
                assert step in STEPS, f"{step} missing implementation"
                assert step in ROUTERS, f"{step} missing router"


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestWorkflowEdgeCases:
    """Edge case tests for workflow."""

    def test_immediate_success(self):
        """Workflow handles immediate success on first run."""
        mock_deps = Mock()

        with patch.dict(STEPS, {
            NextStep.DETERMINE_FILES: lambda s, d: {},
            NextStep.EXTRACT_BOUNDARIES: lambda s, d: {},
            NextStep.GENERATE_FILES: lambda s, d: {},
            NextStep.WRITE_FILES: lambda s, d: {},
            NextStep.RUN_SIMULATION: lambda s, d: {"success": True, "completed": True},
        }):
            state = CaseState(grid_type="polyMesh")

            final_state = run_workflow(state, mock_deps, verbose=False)

            assert final_state.success is True
            assert final_state.attempt_count == 1

    def test_mesh_conversion_failure(self):
        """Workflow handles mesh conversion failure."""
        mock_deps = Mock()

        with patch.dict(STEPS, {
            NextStep.DETERMINE_FILES: lambda s, d: {},
            NextStep.CONVERT_MESH: lambda s, d: {"mesh_converted": False},
        }):
            state = CaseState(grid_type="msh")

            final_state = run_workflow(state, mock_deps, verbose=False)

            # Should fail due to mesh conversion
            assert final_state.success is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
