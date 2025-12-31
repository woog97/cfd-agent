"""
File generation steps: structure determination, LLM generation, and writing.
"""

import json
import os
import time

from state import CaseState, StateUpdate
from database import reference_files_to_json
from steps.common import Dependencies, clean_llm_file_response, fix_dimensions
from steps.prompts import PROMPT_GENERATE_FILE
from steps.tools import write_openfoam_file, validate_safe_path, PathTraversalError


def determine_file_structure(state: CaseState, deps: Dependencies) -> StateUpdate:
    """
    Determine which files are needed for this case based on solver and turbulence model.

    Returns:
        StateUpdate with file_structure and reference_file_name.
    """
    solver = state.solver
    turbulence_model = state.turbulence_model

    # Load file requirements from database
    solver_files_path = os.path.join(
        deps.config.paths.database_dir,
        "final_OF_solver_required_files.json"
    )
    turbulence_files_path = os.path.join(
        deps.config.paths.database_dir,
        "final_OF_turbulence_required_files.json"
    )

    required_files = set()
    reference_name = ""

    # Load solver-required files
    # Format: {"simpleFoam": ["file1", "file2", ...]}
    if os.path.exists(solver_files_path):
        with open(solver_files_path, 'r') as f:
            solver_data = json.load(f)
            if solver in solver_data:
                solver_files = solver_data[solver]
                if isinstance(solver_files, list):
                    required_files.update(solver_files)
                elif isinstance(solver_files, dict):
                    # Handle case where it might be {"required_files": [...]}
                    required_files.update(solver_files.get("required_files", []))
                    reference_name = solver_files.get("reference_case", "")

    # Load turbulence-required files
    # Format: {"kEpsilon": ["file1", "file2", ...]}
    if turbulence_model and os.path.exists(turbulence_files_path):
        with open(turbulence_files_path, 'r') as f:
            turbulence_data = json.load(f)
            if turbulence_model in turbulence_data:
                turb_files = turbulence_data[turbulence_model]
                if isinstance(turb_files, list):
                    required_files.update(turb_files)
                elif isinstance(turb_files, dict):
                    required_files.update(turb_files.get("required_files", []))

    # Always need basic system files
    required_files.update([
        "system/controlDict",
        "system/fvSchemes",
        "system/fvSolution",
    ])

    return {
        "file_structure": sorted(list(required_files)),
        "reference_file_name": reference_name,
    }


def generate_files(state: CaseState, deps: Dependencies, verbose: bool = True) -> StateUpdate:
    """
    Generate OpenFOAM case files using LLM.

    Args:
        state: Current case state.
        deps: Dependencies (config, llm, database).
        verbose: If True, print progress as files are generated.

    Returns:
        StateUpdate with generated_files dict.
    """
    generated = {}
    total = len(state.file_structure)

    for i, file_path in enumerate(state.file_structure, 1):
        if verbose:
            print(f"  [{i}/{total}] Generating {file_path}...", end="", flush=True)
        start = time.time()

        content = _generate_single_file(
            file_path=file_path,
            state=state,
            deps=deps,
        )

        elapsed = time.time() - start
        if content:
            generated[file_path] = content
            if verbose:
                print(f" done ({elapsed:.1f}s)")
        elif verbose:
            print(f" FAILED ({elapsed:.1f}s)")

    return {"generated_files": generated}


def _generate_single_file(
    file_path: str,
    state: CaseState,
    deps: Dependencies,
) -> str:
    """Generate a single OpenFOAM file using LLM."""
    # Get reference files from database
    reference_files = deps.database.find_reference_files(
        target_file=file_path,
        solver=state.solver,
        turbulence_model=state.turbulence_model,
        other_physical_model=state.other_physical_model,
    )

    prompt = PROMPT_GENERATE_FILE.format(
        file_path=file_path,
        description=state.description,
        solver=state.solver,
        turbulence_model=state.turbulence_model or "laminar",
        boundaries=", ".join(state.grid_boundaries),
        reference_files=reference_files_to_json(reference_files),
    )

    response = deps.llm.ask_reasoning(prompt)

    # Clean up response (remove markdown code blocks if present)
    content = clean_llm_file_response(response)

    # Fix dimensions if needed
    content = fix_dimensions(
        content=content,
        file_path=file_path,
        solver=state.solver,
        incompressible_solvers=deps.config.openfoam.incompressible_solvers,
        database_dir=deps.config.paths.database_dir,
    )

    return content


def write_files(state: CaseState, deps: Dependencies) -> StateUpdate:
    """
    Write generated files to disk.

    Returns:
        Empty StateUpdate (side effect only).

    Raises:
        PathTraversalError: If any file path attempts to escape output directory.
    """
    # Ensure output directory exists
    os.makedirs(state.output_path, exist_ok=True)

    for rel_path, content in state.generated_files.items():
        # Validate path doesn't escape output directory
        try:
            full_path = validate_safe_path(state.output_path, rel_path)
        except PathTraversalError as e:
            print(f"Security error: {e}")
            raise

        try:
            write_openfoam_file(full_path, content)
            print(f"Wrote: {rel_path}")
        except Exception as e:
            print(f"Error writing {rel_path}: {e}")

    return {}
