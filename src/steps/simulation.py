"""
Simulation execution step.

Orchestrates running OpenFOAM solvers via local or Docker execution.
"""

import os

from state import CaseState, StateUpdate
from steps.common import Dependencies
from steps.tools import (
    run_solver_local,
    run_solver_docker,
    run_mesh_converter_docker,
    extract_solver_from_controldict,
    read_log_tail,
)


def run_simulation(state: CaseState, deps: Dependencies) -> StateUpdate:
    """
    Run the OpenFOAM simulation.

    Supports both local execution and Docker-based execution.

    Returns:
        StateUpdate with error info if failed, or completed=True if succeeded.
    """
    # Get solver from controlDict
    control_dict_path = os.path.join(state.output_path, "system/controlDict")
    solver = extract_solver_from_controldict(control_dict_path)

    if not solver:
        solver = state.solver

    # Run via Docker or local
    if deps.config.docker.enabled:
        print(f"Running {solver} via Docker...")
        returncode, stdout, stderr = run_solver_docker(
            solver=solver,
            case_path=state.output_path,
            file_structure=state.file_structure,
            docker_config=deps.config.docker,
        )
    else:
        returncode, stdout, stderr = run_solver_local(
            solver=solver,
            case_path=state.output_path,
            file_structure=state.file_structure,
        )

    # Check for success
    combined_output = stdout + stderr
    has_fatal_error = "FOAM FATAL" in combined_output

    if returncode == 0 and not has_fatal_error:
        print("Simulation completed successfully")
        return {
            "completed": True,
            "success": True,
            "current_error": None,
        }
    else:
        # Extract error message
        error_msg = stderr or stdout or "Unknown error"

        # For local runs, try reading from log file
        if not deps.config.docker.enabled and not error_msg:
            log_file = os.path.join(state.output_path, "case_run.log")
            error_msg = read_log_tail(log_file)

        # Extract FOAM FATAL section if present
        if "FOAM FATAL" in error_msg:
            fatal_idx = error_msg.find("FOAM FATAL")
            error_msg = error_msg[fatal_idx:fatal_idx + 1000]

        print(f"Simulation failed: {error_msg[:200]}...")
        return {
            "current_error": error_msg,
            "error_history": state.error_history + [error_msg],
        }


# Re-export mesh conversion for backwards compatibility
def run_mesh_conversion_docker(case_path: str, mesh_file: str, deps: Dependencies) -> bool:
    """
    Convert mesh file to OpenFOAM format using Docker.

    This is a convenience wrapper that extracts config from deps.
    """
    return run_mesh_converter_docker(
        mesh_file=mesh_file,
        case_path=case_path,
        docker_config=deps.config.docker,
    )
