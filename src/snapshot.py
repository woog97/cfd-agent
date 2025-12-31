"""
Snapshot: Save and load case state for iterative testing.

This module allows separating the expensive LLM file generation phase
from the OpenFOAM execution and correction phases.

Usage:
    # After generating files (slow, ~10 min)
    save_snapshot(state, "my_case_snapshot.json")

    # Later, for iteration (fast)
    state = load_snapshot("my_case_snapshot.json")
    # Run OpenFOAM, correct errors, repeat...
"""

import json
import os
from datetime import datetime
from dataclasses import asdict
from typing import Any

from state import CaseState


def save_snapshot(state: CaseState, filepath: str) -> None:
    """
    Save case state to a JSON snapshot file.

    Saves everything needed to resume from this point:
    - Case configuration (solver, turbulence model, etc.)
    - Generated files content
    - File structure
    - Boundary information

    Args:
        state: The CaseState to save.
        filepath: Path to save the snapshot JSON.
    """
    snapshot = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "snapshot_version": "1.0",
        },
        "case_config": {
            "case_name": state.case_name,
            "solver": state.solver,
            "turbulence_model": state.turbulence_model,
            "other_physical_model": state.other_physical_model,
            "description": state.description,
            "output_path": state.output_path,
        },
        "mesh_info": {
            "grid_path": state.grid_path,
            "grid_type": state.grid_type,
            "grid_boundaries": state.grid_boundaries,
            "mesh_converted": state.mesh_converted,
        },
        "file_structure": state.file_structure,
        "generated_files": state.generated_files,
        "execution_state": {
            "attempt_count": state.attempt_count,
            "max_attempts": state.max_attempts,
            "completed": state.completed,
            "success": state.success,
        },
        "error_history": state.error_history,
        "correction_trajectory": state.correction_trajectory,
        "reflection_history": state.reflection_history,
    }

    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)

    print(f"Snapshot saved: {filepath}")
    print(f"  - {len(state.generated_files)} generated files")
    print(f"  - {len(state.file_structure)} files in structure")


def load_snapshot(filepath: str) -> CaseState:
    """
    Load case state from a JSON snapshot file.

    Args:
        filepath: Path to the snapshot JSON.

    Returns:
        CaseState restored from the snapshot.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        snapshot = json.load(f)

    case_config = snapshot["case_config"]
    mesh_info = snapshot["mesh_info"]
    exec_state = snapshot.get("execution_state", {})

    state = CaseState(
        # Case config
        case_name=case_config["case_name"],
        solver=case_config["solver"],
        turbulence_model=case_config.get("turbulence_model"),
        other_physical_model=case_config.get("other_physical_model"),
        description=case_config.get("description", ""),
        output_path=case_config["output_path"],
        # Mesh info
        grid_path=mesh_info.get("grid_path", ""),
        grid_type=mesh_info.get("grid_type", "msh"),
        grid_boundaries=mesh_info.get("grid_boundaries", []),
        mesh_converted=mesh_info.get("mesh_converted", False),
        # File structure
        file_structure=snapshot.get("file_structure", []),
        generated_files=snapshot.get("generated_files", {}),
        # Execution state
        attempt_count=exec_state.get("attempt_count", 0),
        max_attempts=exec_state.get("max_attempts", 30),
        completed=exec_state.get("completed", False),
        success=exec_state.get("success", False),
        # Error history
        error_history=snapshot.get("error_history", []),
        correction_trajectory=snapshot.get("correction_trajectory", []),
        reflection_history=snapshot.get("reflection_history", []),
    )

    print(f"Snapshot loaded: {filepath}")
    print(f"  - Case: {state.case_name}")
    print(f"  - Solver: {state.solver}")
    print(f"  - {len(state.generated_files)} generated files")

    return state


def save_generated_files_only(generated_files: dict[str, str], filepath: str) -> None:
    """
    Save only the generated files content (minimal snapshot).

    Useful when you just want to cache LLM outputs without full state.

    Args:
        generated_files: Dict mapping file paths to content.
        filepath: Path to save the JSON.
    """
    snapshot = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "type": "generated_files_only",
        },
        "generated_files": generated_files,
    }

    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)

    print(f"Generated files saved: {filepath} ({len(generated_files)} files)")


def load_generated_files_only(filepath: str) -> dict[str, str]:
    """
    Load only generated files from a minimal snapshot.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Dict mapping file paths to content.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        snapshot = json.load(f)

    return snapshot.get("generated_files", {})


def write_files_from_snapshot(snapshot_path: str, output_dir: str) -> list[str]:
    """
    Write generated files from a snapshot to disk.

    Args:
        snapshot_path: Path to snapshot JSON.
        output_dir: Directory to write files to.

    Returns:
        List of file paths that were written.
    """
    with open(snapshot_path, 'r', encoding='utf-8') as f:
        snapshot = json.load(f)

    generated_files = snapshot.get("generated_files", {})
    written = []

    for rel_path, content in generated_files.items():
        full_path = os.path.join(output_dir, rel_path)

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        # Write file
        try:
            # Handle escape sequences
            processed = content.encode('latin-1').decode('unicode_escape')
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(processed)
        except Exception:
            # Fallback: write as-is
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)

        written.append(rel_path)
        print(f"  Wrote: {rel_path}")

    print(f"Wrote {len(written)} files to {output_dir}")
    return written


def update_snapshot_after_correction(
    snapshot_path: str,
    file_path: str,
    new_content: str,
    error_msg: str | None = None,
) -> None:
    """
    Update a snapshot file after a correction was made.

    This allows persisting corrections without regenerating everything.

    Args:
        snapshot_path: Path to snapshot JSON.
        file_path: The file that was corrected (e.g., "0/U").
        new_content: The corrected file content.
        error_msg: Optional error message that led to this correction.
    """
    with open(snapshot_path, 'r', encoding='utf-8') as f:
        snapshot = json.load(f)

    # Update generated files
    if "generated_files" in snapshot:
        old_content = snapshot["generated_files"].get(file_path, "")
        snapshot["generated_files"][file_path] = new_content

        # Track correction in trajectory
        if "correction_trajectory" not in snapshot:
            snapshot["correction_trajectory"] = []

        snapshot["correction_trajectory"].append({
            file_path: [old_content[:500], new_content[:500]],  # Truncate for readability
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
        })

    # Update metadata
    snapshot["metadata"]["last_modified"] = datetime.now().isoformat()

    with open(snapshot_path, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)

    print(f"Snapshot updated: {file_path} corrected")
