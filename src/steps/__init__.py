"""
Steps: Workflow step functions for OpenFOAM case generation.

Each step function follows the pattern:
    def step_name(state: CaseState, deps: Dependencies) -> StateUpdate

This package is organized into modules by workflow phase:
- prompts: LLM prompt templates
- common: Shared dependencies and utilities
- tools: External execution (Docker, subprocess)
- mesh: Mesh conversion and boundary extraction
- generation: File structure, generation, and writing
- simulation: Solver execution
- correction: Error analysis, correction, and reflection
"""

# Core types
from steps.common import Dependencies, clean_llm_file_response, fix_dimensions

# Mesh processing
from steps.mesh import (
    convert_mesh,
    extract_boundaries,
    _extract_boundaries_from_msh,
    _extract_boundaries_from_polymesh,
)

# File generation
from steps.generation import (
    determine_file_structure,
    generate_files,
    write_files,
)

# Simulation execution
from steps.simulation import run_simulation, run_mesh_conversion_docker

# Tool functions (for direct access when needed)
from steps.tools import run_mesh_converter_local

# Security utilities
from steps.tools import (
    validate_solver_name,
    validate_safe_path,
    CommandInjectionError,
    PathTraversalError,
)

# Error correction
from steps.correction import (
    analyze_error,
    correct_file,
    reflect_on_errors,
    check_needs_new_file,
    add_new_file,
    rewrite_file,
)

# Backwards compatibility aliases for private helpers (used by tests)
_clean_llm_file_response = clean_llm_file_response


def _fix_dimensions(content: str, file_path: str, state, deps) -> str:
    """
    Backwards-compatible wrapper for fix_dimensions.

    Old signature: _fix_dimensions(content, file_path, state, deps)
    New signature: fix_dimensions(content, file_path, solver, incompressible_solvers, database_dir)
    """
    return fix_dimensions(
        content=content,
        file_path=file_path,
        solver=state.solver,
        incompressible_solvers=deps.config.openfoam.incompressible_solvers,
        database_dir=deps.config.paths.database_dir,
    )

# Public API
__all__ = [
    # Types
    "Dependencies",
    # Helpers (public)
    "clean_llm_file_response",
    "fix_dimensions",
    # Security
    "validate_solver_name",
    "validate_safe_path",
    "CommandInjectionError",
    "PathTraversalError",
    # Mesh
    "convert_mesh",
    "extract_boundaries",
    # Generation
    "determine_file_structure",
    "generate_files",
    "write_files",
    # Simulation
    "run_simulation",
    "run_mesh_conversion_docker",
    "run_mesh_converter_local",
    # Correction
    "analyze_error",
    "correct_file",
    "reflect_on_errors",
    "check_needs_new_file",
    "add_new_file",
    "rewrite_file",
    # Private helpers (for backwards compat)
    "_extract_boundaries_from_msh",
    "_extract_boundaries_from_polymesh",
    "_clean_llm_file_response",
    "_fix_dimensions",
]
