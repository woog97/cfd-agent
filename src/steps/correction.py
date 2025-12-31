"""
Error correction steps: analysis, correction, reflection, and file addition.
"""

import json
import os

from state import CaseState, StateUpdate
from database import reference_files_to_json
from steps.common import Dependencies, clean_llm_file_response, fix_dimensions
from steps.prompts import (
    PROMPT_ANALYZE_ERROR,
    PROMPT_CORRECT_FILE,
    PROMPT_REFLECT,
    PROMPT_CHECK_NEW_FILE,
    PROMPT_ADD_FILE,
    PROMPT_REWRITE_FILE,
)
from steps.tools import write_openfoam_file, read_file_safe, validate_safe_path


def analyze_error(state: CaseState, deps: Dependencies) -> StateUpdate:
    """
    Analyze simulation error to identify problematic files.

    Returns:
        StateUpdate with error_file and analysis info.
    """
    if not state.current_error:
        return {}

    prompt = PROMPT_ANALYZE_ERROR.format(
        error=state.current_error,
        file_structure=state.file_structure,
    )

    response = deps.llm.ask_instruct(prompt)
    error_file = response.strip().strip('"').strip("'")

    # Validate it's in our file list
    if error_file not in state.file_structure:
        # Try to find closest match
        for f in state.file_structure:
            if error_file in f or f in error_file:
                error_file = f
                break

    return {"error_file": error_file}


def correct_file(state: CaseState, deps: Dependencies) -> StateUpdate:
    """
    Correct the identified error file.

    Returns:
        StateUpdate with updated generated_files and correction_trajectory.
    """
    if not state.error_file:
        return {}

    file_path = state.error_file
    # Validate path doesn't escape output directory
    full_path = validate_safe_path(state.output_path, file_path)

    # Read current file content
    current_content = read_file_safe(
        full_path,
        default=state.generated_files.get(file_path, "")
    )

    # Get reference files
    reference_files = deps.database.find_reference_files(
        target_file=file_path,
        solver=state.solver,
        turbulence_model=state.turbulence_model,
    )

    prompt = PROMPT_CORRECT_FILE.format(
        error=state.current_error,
        file_path=file_path,
        current_content=current_content,
        reference_files=reference_files_to_json(reference_files),
        boundaries=state.grid_boundaries,
    )

    response = deps.llm.ask_reasoning(prompt)
    new_content = clean_llm_file_response(response)

    # Write corrected file
    write_openfoam_file(full_path, new_content)

    # Update state
    new_generated = dict(state.generated_files)
    new_generated[file_path] = new_content

    new_trajectory = list(state.correction_trajectory)
    new_trajectory.append({file_path: [current_content, new_content]})

    return {
        "generated_files": new_generated,
        "correction_trajectory": new_trajectory,
    }


def reflect_on_errors(state: CaseState, deps: Dependencies) -> StateUpdate:
    """
    Reflect on repeated errors to find better solutions.

    Returns:
        StateUpdate with reflection_history.
    """
    if len(state.error_history) < 2:
        return {}

    # Build trajectory string
    trajectory_str = ""
    for i, trial in enumerate(state.correction_trajectory[-3:]):
        trajectory_str += f"\n### Attempt {i + 1}:\n"
        for file_name, changes in trial.items():
            if len(changes) > 1:
                trajectory_str += f"Modified {file_name}:\n"
                trajectory_str += f"Before: {changes[0][:500]}...\n"
                trajectory_str += f"After: {changes[1][:500]}...\n"

    prompt = PROMPT_REFLECT.format(
        error=state.current_error,
        trajectory=trajectory_str,
        file_structure=state.file_structure,
    )

    response = deps.llm.ask_reasoning(prompt)

    new_history = list(state.reflection_history)
    new_history.append({
        "error": state.current_error,
        "reflection": response,
    })

    return {"reflection_history": new_history}


def check_needs_new_file(state: CaseState, deps: Dependencies) -> StateUpdate:
    """
    Check if the error indicates a missing file.

    Returns:
        StateUpdate with needs_new_file flag and new_file_name.
    """
    if not state.current_error:
        return {"needs_new_file": False}

    # Check for "cannot find file" errors
    if "cannot find file" not in state.current_error.lower():
        return {"needs_new_file": False}

    prompt = PROMPT_CHECK_NEW_FILE.format(error=state.current_error)

    response = deps.llm.ask_instruct(prompt).strip().lower()

    if response == "no" or not response:
        return {"needs_new_file": False}

    # Validate file path format
    if "/" in response and response.split("/")[0] in ("0", "constant", "system"):
        return {
            "needs_new_file": True,
            "new_file_name": response,
        }

    return {"needs_new_file": False}


def add_new_file(state: CaseState, deps: Dependencies) -> StateUpdate:
    """
    Add a new file that's missing from the case.

    Returns:
        StateUpdate with updated file_structure and generated_files.
    """
    if not state.new_file_name:
        return {}

    file_path = state.new_file_name

    # Get reference files
    reference_files = deps.database.find_reference_files(
        target_file=file_path,
        solver=state.solver,
        turbulence_model=state.turbulence_model,
    )

    # Read existing case files for context
    existing_files = {}
    for f in state.file_structure[:5]:  # Limit to avoid huge prompts
        full_path = os.path.join(state.output_path, f)
        content = read_file_safe(full_path)
        if content:
            existing_files[f] = content[:1000]  # Truncate

    prompt = PROMPT_ADD_FILE.format(
        file_path=file_path,
        description=state.description,
        solver=state.solver,
        turbulence_model=state.turbulence_model or "laminar",
        boundaries=state.grid_boundaries,
        existing_files=json.dumps(existing_files, indent=2),
        reference_files=reference_files_to_json(reference_files),
    )

    response = deps.llm.ask_reasoning(prompt)
    content = clean_llm_file_response(response)
    content = fix_dimensions(
        content=content,
        file_path=file_path,
        solver=state.solver,
        incompressible_solvers=deps.config.openfoam.incompressible_solvers,
        database_dir=deps.config.paths.database_dir,
    )

    # Write file (validate path doesn't escape output directory)
    full_path = validate_safe_path(state.output_path, file_path)
    write_openfoam_file(full_path, content)

    # Update state
    new_structure = list(state.file_structure)
    if file_path not in new_structure:
        new_structure.append(file_path)

    new_generated = dict(state.generated_files)
    new_generated[file_path] = content

    return {
        "file_structure": new_structure,
        "generated_files": new_generated,
        "needs_new_file": False,
        "new_file_name": None,
    }


def rewrite_file(state: CaseState, deps: Dependencies) -> StateUpdate:
    """
    Completely rewrite a file that has persistent errors.

    Returns:
        StateUpdate with updated generated_files.
    """
    if not state.error_file:
        return {}

    file_path = state.error_file
    # Validate path doesn't escape output directory
    full_path = validate_safe_path(state.output_path, file_path)

    # Get reference files
    reference_files = deps.database.find_reference_files(
        target_file=file_path,
        solver=state.solver,
        turbulence_model=state.turbulence_model,
    )

    prompt = PROMPT_REWRITE_FILE.format(
        file_path=file_path,
        solver=state.solver,
        turbulence_model=state.turbulence_model or "laminar",
        boundaries=state.grid_boundaries,
        description=state.description,
        reference_files=reference_files_to_json(reference_files),
    )

    response = deps.llm.ask_reasoning(prompt)
    new_content = clean_llm_file_response(response)
    new_content = fix_dimensions(
        content=new_content,
        file_path=file_path,
        solver=state.solver,
        incompressible_solvers=deps.config.openfoam.incompressible_solvers,
        database_dir=deps.config.paths.database_dir,
    )

    # Write file
    write_openfoam_file(full_path, new_content)

    new_generated = dict(state.generated_files)
    new_generated[file_path] = new_content

    return {"generated_files": new_generated}
