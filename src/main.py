"""
Main entry point for ChatCFD.

This module wires together all components and provides both CLI and programmatic interfaces.

Usage:
    # Command line
    python main.py --case_description case.pdf --grid mesh.msh

    # Programmatic
    from main import run_case
    result = run_case(description="...", solver="simpleFoam", grid_path="mesh.msh")
"""

import argparse
import os
import sys
from pathlib import Path

from config import AppConfig, load_config, ensure_directories, load_openfoam_environment
from state import CaseState, CaseBuilder
from llm import LLMClient
from database import ReferenceDatabase
from steps import Dependencies
from workflow import run_workflow


def setup_environment(config: AppConfig) -> bool:
    """
    Set up the runtime environment.

    Args:
        config: Application configuration.

    Returns:
        True if setup successful, False otherwise.
    """
    # Ensure directories exist
    ensure_directories(config)

    # Load OpenFOAM environment
    openfoam_path = os.path.dirname(config.paths.openfoam_tutorials_dir)
    if not load_openfoam_environment(openfoam_path):
        print("Warning: OpenFOAM environment not loaded. Simulations will fail.")
        return False

    return True


def create_dependencies(config: AppConfig, log_dir: str | None = None) -> Dependencies:
    """
    Create the dependencies container.

    Args:
        config: Application configuration.
        log_dir: Optional directory for LLM logs.

    Returns:
        Dependencies instance with all external dependencies.
    """
    # Create LLM client
    log_path = os.path.join(log_dir, "llm_logs.jsonl") if log_dir else None
    llm = LLMClient(config.llm, log_path=log_path)

    # Load reference database
    database = ReferenceDatabase.load(config.paths.database_dir)
    print(f"Loaded {len(database)} reference cases from database")

    return Dependencies(
        config=config,
        llm=llm,
        database=database,
    )


def run_case(
    description: str,
    solver: str,
    grid_path: str,
    turbulence_model: str | None = None,
    other_physical_model: list[str] | None = None,
    case_name: str | None = None,
    output_dir: str | None = None,
    config: AppConfig | None = None,
    max_attempts: int = 30,
    verbose: bool = True,
) -> CaseState:
    """
    Run a CFD case from start to finish.

    This is the main programmatic interface.

    Args:
        description: Text description of the case to simulate.
        solver: OpenFOAM solver name (e.g., "simpleFoam").
        grid_path: Path to mesh file (.msh) or polyMesh directory.
        turbulence_model: Turbulence model name (optional).
        other_physical_model: Additional physical models (optional).
        case_name: Name for the case (defaults to solver name).
        output_dir: Output directory (defaults to config.paths.output_dir).
        config: Application configuration (loads default if None).
        max_attempts: Maximum error correction attempts.
        verbose: Print progress information.

    Returns:
        Final CaseState with success/failure status and all generated files.
    """
    # Load configuration
    if config is None:
        config = load_config()

    # Setup environment
    setup_environment(config)

    # Determine output location
    if output_dir is None:
        output_dir = config.paths.output_dir

    # Determine grid type
    grid_type = "polyMesh" if os.path.isdir(grid_path) else "msh"

    # Create case builder
    if case_name is None:
        case_name = solver

    builder = CaseBuilder(
        solver=solver,
        turbulence_model=turbulence_model,
        other_physical_model=other_physical_model,
        description=description,
        grid_path=grid_path,
        grid_type=grid_type,
        output_dir=output_dir,
        max_attempts=max_attempts,
    )

    # Build initial state
    state = builder.build(case_name)

    # Create dependencies
    deps = create_dependencies(config, log_dir=state.output_path)

    # Run workflow
    if verbose:
        print(f"\n{'='*60}")
        print(f"Starting CFD Case: {case_name}")
        print(f"Solver: {solver}")
        print(f"Turbulence: {turbulence_model or 'laminar'}")
        print(f"Grid: {grid_path}")
        print(f"Output: {state.output_path}")
        print(f"{'='*60}\n")

    final_state = run_workflow(state, deps, verbose=verbose)

    return final_state


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="ChatCFD: LLM-driven CFD simulation automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --description "Flow around a cylinder at Re=100" --solver simpleFoam --grid mesh.msh
    python main.py --case_file case.txt --grid polyMesh/ --turbulence kOmegaSST
        """
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--description",
        type=str,
        help="Case description text"
    )
    input_group.add_argument(
        "--case_file",
        type=str,
        help="Path to case description file (PDF or TXT)"
    )

    # Required arguments
    parser.add_argument(
        "--solver",
        type=str,
        required=True,
        help="OpenFOAM solver name (e.g., simpleFoam, pimpleFoam)"
    )
    parser.add_argument(
        "--grid",
        type=str,
        required=True,
        help="Path to mesh file (.msh) or polyMesh directory"
    )

    # Optional arguments
    parser.add_argument(
        "--turbulence",
        type=str,
        default=None,
        help="Turbulence model (e.g., kOmegaSST, kEpsilon)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Case name (defaults to solver name)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=30,
        help="Maximum error correction attempts (default: 30)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config JSON file"
    )

    args = parser.parse_args()

    # Get case description
    if args.description:
        description = args.description
    else:
        # Read from file
        case_file = Path(args.case_file)
        if not case_file.exists():
            print(f"Error: Case file not found: {args.case_file}")
            sys.exit(1)

        if case_file.suffix.lower() == ".pdf":
            # Would need to import PDF processing here
            print("PDF processing not yet implemented in src2")
            print("Please provide a text description with --description")
            sys.exit(1)
        else:
            description = case_file.read_text()

    # Validate grid path
    grid_path = Path(args.grid)
    if not grid_path.exists():
        print(f"Error: Grid file/directory not found: {args.grid}")
        sys.exit(1)

    # Load config
    config = load_config(args.config) if args.config else load_config()

    # Run the case
    try:
        final_state = run_case(
            description=description,
            solver=args.solver,
            grid_path=str(grid_path),
            turbulence_model=args.turbulence,
            case_name=args.name,
            output_dir=args.output,
            config=config,
            max_attempts=args.max_attempts,
            verbose=not args.quiet,
        )

        # Report results
        if final_state.success:
            print(f"\nSuccess! Case output: {final_state.output_path}")
            sys.exit(0)
        else:
            print(f"\nCase did not complete successfully after {final_state.attempt_count} attempts")
            print(f"Last error: {final_state.current_error[:200] if final_state.current_error else 'Unknown'}...")
            print(f"Case files at: {final_state.output_path}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
