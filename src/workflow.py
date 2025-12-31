"""
Workflow: Explicit state machine for CFD case execution.

The workflow is defined as a graph of steps with explicit transitions.
This makes the control flow visible and modifiable.

Design principles:
- Workflow structure is data, not implicit in code
- Routing decisions are explicit functions
- Steps don't know about each other (decoupled)
- Easy to add/remove/reorder steps
"""

from enum import Enum, auto
from typing import Callable

from state import CaseState, StateUpdate, NextStep
from steps import (
    Dependencies,
    determine_file_structure,
    convert_mesh,
    extract_boundaries,
    generate_files,
    write_files,
    run_simulation,
    analyze_error,
    correct_file,
    reflect_on_errors,
    check_needs_new_file,
    add_new_file,
    rewrite_file,
)


# Type alias for step functions
StepFunction = Callable[[CaseState, Dependencies], StateUpdate]


# =============================================================================
# Step Registry
# =============================================================================

STEPS: dict[NextStep, StepFunction] = {
    NextStep.DETERMINE_FILES: determine_file_structure,
    NextStep.CONVERT_MESH: convert_mesh,
    NextStep.EXTRACT_BOUNDARIES: extract_boundaries,
    NextStep.GENERATE_FILES: generate_files,
    NextStep.WRITE_FILES: write_files,
    NextStep.RUN_SIMULATION: run_simulation,
    NextStep.ANALYZE_ERROR: analyze_error,
    NextStep.CORRECT_FILES: correct_file,
    NextStep.REFLECT: reflect_on_errors,
    NextStep.REWRITE_FILE: rewrite_file,
    NextStep.ADD_FILE: add_new_file,
}


# =============================================================================
# Routing Logic
# =============================================================================

def route_initial(state: CaseState) -> NextStep:
    """Determine the first step to execute."""
    return NextStep.DETERMINE_FILES


def route_after_determine_files(state: CaseState) -> NextStep:
    """After determining file structure, process mesh."""
    if state.grid_type == "msh":
        return NextStep.CONVERT_MESH
    return NextStep.EXTRACT_BOUNDARIES


def route_after_convert_mesh(state: CaseState) -> NextStep:
    """After mesh conversion, extract boundaries."""
    if not state.mesh_converted:
        return NextStep.FAILED  # Can't proceed without mesh
    return NextStep.EXTRACT_BOUNDARIES


def route_after_extract_boundaries(state: CaseState) -> NextStep:
    """After boundary extraction, generate files."""
    return NextStep.GENERATE_FILES


def route_after_generate_files(state: CaseState) -> NextStep:
    """After file generation, write to disk."""
    return NextStep.WRITE_FILES


def route_after_write_files(state: CaseState) -> NextStep:
    """After writing files, run simulation."""
    return NextStep.RUN_SIMULATION


def route_after_run(state: CaseState) -> NextStep:
    """
    After running simulation, determine next action.

    This is the main decision point for error handling.
    """
    # Success!
    if state.success:
        return NextStep.DONE

    # Check if we should stop
    if state.should_stop():
        return NextStep.FAILED

    # Check for missing file errors first
    update = check_needs_new_file(state, None)  # Quick check, no deps needed
    if update.get("needs_new_file"):
        return NextStep.ADD_FILE

    # Check for repeated errors
    consecutive = state.consecutive_same_errors()

    if consecutive >= 4:
        # Same error 4+ times -> rewrite the file completely
        return NextStep.REWRITE_FILE
    elif consecutive >= 2:
        # Same error 2-3 times -> reflect on what's going wrong
        return NextStep.REFLECT

    # Normal error -> analyze and correct
    return NextStep.ANALYZE_ERROR


def route_after_analyze(state: CaseState) -> NextStep:
    """After error analysis, correct the identified file."""
    if state.error_file:
        return NextStep.CORRECT_FILES
    # Couldn't identify error file -> try reflection
    return NextStep.REFLECT


def route_after_correct(state: CaseState) -> NextStep:
    """After correction, try running again."""
    return NextStep.RUN_SIMULATION


def route_after_reflect(state: CaseState) -> NextStep:
    """After reflection, analyze error with new insights."""
    return NextStep.ANALYZE_ERROR


def route_after_rewrite(state: CaseState) -> NextStep:
    """After rewriting file, try running again."""
    return NextStep.RUN_SIMULATION


def route_after_add_file(state: CaseState) -> NextStep:
    """After adding missing file, try running again."""
    return NextStep.RUN_SIMULATION


# Routing table: maps each step to its router function
ROUTERS: dict[NextStep, Callable[[CaseState], NextStep]] = {
    NextStep.DETERMINE_FILES: route_after_determine_files,
    NextStep.CONVERT_MESH: route_after_convert_mesh,
    NextStep.EXTRACT_BOUNDARIES: route_after_extract_boundaries,
    NextStep.GENERATE_FILES: route_after_generate_files,
    NextStep.WRITE_FILES: route_after_write_files,
    NextStep.RUN_SIMULATION: route_after_run,
    NextStep.ANALYZE_ERROR: route_after_analyze,
    NextStep.CORRECT_FILES: route_after_correct,
    NextStep.REFLECT: route_after_reflect,
    NextStep.REWRITE_FILE: route_after_rewrite,
    NextStep.ADD_FILE: route_after_add_file,
}


# =============================================================================
# Workflow Runner
# =============================================================================

class WorkflowRunner:
    """
    Executes the workflow state machine.

    Usage:
        deps = Dependencies(config=config, llm=llm, database=db)
        runner = WorkflowRunner(deps)

        state = builder.build("my_case")
        final_state = runner.run(state)
    """

    def __init__(self, deps: Dependencies, verbose: bool = True):
        """
        Initialize workflow runner.

        Args:
            deps: External dependencies (config, LLM, database).
            verbose: Print step transitions.
        """
        self.deps = deps
        self.verbose = verbose

    def run(self, state: CaseState) -> CaseState:
        """
        Execute workflow until completion or failure.

        Args:
            state: Initial case state.

        Returns:
            Final case state.
        """
        current_step = route_initial(state)

        while current_step not in (NextStep.DONE, NextStep.FAILED):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Step: {current_step.name}")
                print(f"Attempt: {state.attempt_count}")
                print(f"{'='*60}")

            # Execute step
            step_fn = STEPS.get(current_step)
            if step_fn is None:
                print(f"Error: No implementation for step {current_step}")
                break

            # Special case: check_needs_new_file needs deps for full analysis
            if current_step == NextStep.RUN_SIMULATION:
                update = step_fn(state, self.deps)
                state = state.apply_update(update)
                state = state.apply_update({"attempt_count": state.attempt_count + 1})

                # Do the needs_new_file check with deps
                new_file_update = check_needs_new_file(state, self.deps)
                state = state.apply_update(new_file_update)
            else:
                update = step_fn(state, self.deps)
                state = state.apply_update(update)

            # Get next step
            router = ROUTERS.get(current_step)
            if router is None:
                print(f"Error: No router for step {current_step}")
                break

            current_step = router(state)

        # Final status
        if self.verbose:
            print(f"\n{'='*60}")
            if state.success:
                print("Workflow completed successfully!")
            else:
                print(f"Workflow ended: {current_step.name}")
                print(f"Total attempts: {state.attempt_count}")
            print(f"{'='*60}")

        return state


def run_workflow(state: CaseState, deps: Dependencies, verbose: bool = True) -> CaseState:
    """
    Convenience function to run workflow.

    Args:
        state: Initial case state.
        deps: External dependencies.
        verbose: Print progress.

    Returns:
        Final case state.
    """
    runner = WorkflowRunner(deps, verbose=verbose)
    return runner.run(state)


# =============================================================================
# Workflow Visualization (for debugging)
# =============================================================================

def print_workflow_graph():
    """Print a text representation of the workflow graph."""
    print("\nWorkflow Graph:")
    print("-" * 40)

    # Generate all possible transitions
    transitions = []

    for step in NextStep:
        if step in (NextStep.DONE, NextStep.FAILED):
            continue

        router = ROUTERS.get(step)
        if router:
            # We can't actually call the router without state,
            # but we can list the step and its router
            transitions.append(f"{step.name} -> (depends on state)")

    for t in transitions:
        print(t)

    print("-" * 40)
    print("Terminal states: DONE, FAILED")


if __name__ == "__main__":
    print_workflow_graph()
