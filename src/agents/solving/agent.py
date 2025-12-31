"""
SolvingAgent: High-level agent for OpenFOAM case generation and simulation.

Wraps the existing workflow components (WorkflowRunner, steps) with a clean interface.
"""

import os
from dataclasses import dataclass
from typing import Callable, Optional

from config import AppConfig, load_config
from state import CaseState
from database import ReferenceDatabase
from llm import LLMClient
from steps import Dependencies
from workflow import WorkflowRunner
from snapshot import save_snapshot, load_snapshot
from agents.base import BaseAgent
from run_manager import RunManager


@dataclass
class SolvingResult:
    """Result of a solving agent run."""
    success: bool
    case_path: str
    attempts: int
    errors: list[str]
    generated_files: list[str]
    snapshot_path: str | None = None


class SolvingAgent(BaseAgent[SolvingResult]):
    """
    Agent for generating and running OpenFOAM simulations.

    This agent:
    1. Determines required files based on solver/turbulence model
    2. Generates OpenFOAM case files using LLM
    3. Converts mesh if needed
    4. Runs simulation with error correction loop

    Usage:
        agent = SolvingAgent(config)
        result = agent.run(
            solver="simpleFoam",
            turbulence_model="kOmegaSST",
            description="Incompressible flow over airfoil",
            mesh_path="/path/to/mesh.msh",
            boundaries=["inlet", "outlet", "wall"],
        )

        # Or from a saved snapshot:
        result = agent.run_from_snapshot("/path/to/snapshot.json")
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        run_manager: Optional[RunManager] = None,
        verbose: bool = True,
        log_dir: Optional[str] = None,
    ):
        """
        Initialize the solving agent.

        Args:
            config: Application config. Loads default if None.
            run_manager: RunManager for tracking runs. Creates one if None.
            verbose: Whether to log progress.
            log_dir: Directory for LLM call logs. Uses output_dir if None.
        """
        super().__init__(config=config, run_manager=run_manager, verbose=verbose)
        self.log_dir = log_dir

        # Initialize components
        self.database = ReferenceDatabase.load(self.config.paths.database_dir)

        self._log_init(
            database=f"{len(self.database)} reference cases",
            docker="enabled" if self.config.docker.enabled else "disabled",
        )

    def run(
        self,
        solver: str,
        turbulence_model: str,
        description: str,
        mesh_path: str,
        boundaries: list[str],
        case_name: str | None = None,
        output_dir: str | None = None,
        max_attempts: int = 10,
        save_snapshot_path: str | None = None,
        on_step: Callable[[str, CaseState], None] | None = None,
    ) -> SolvingResult:
        """
        Run the solving agent to generate and simulate an OpenFOAM case.

        Args:
            solver: OpenFOAM solver name (e.g., "simpleFoam").
            turbulence_model: Turbulence model (e.g., "kOmegaSST").
            description: Natural language description of the case.
            mesh_path: Path to mesh file (.msh) or polyMesh directory.
            boundaries: List of boundary names in the mesh.
            case_name: Name for the case. Auto-generated if None.
            output_dir: Output directory. Uses config default if None.
            max_attempts: Maximum correction attempts.
            save_snapshot_path: Path to save snapshot after completion.
            on_step: Optional callback called after each step.

        Returns:
            SolvingResult with success status and case details.
        """
        # Determine case name and output path
        if case_name is None:
            case_name = f"{solver}_{turbulence_model}_case"

        if output_dir is None:
            output_dir = self.config.paths.output_dir

        case_path = os.path.join(output_dir, case_name)
        os.makedirs(case_path, exist_ok=True)

        # Determine mesh type
        mesh_type = "polyMesh" if os.path.isdir(mesh_path) else "msh"

        # Create initial state
        state = CaseState(
            case_name=case_name,
            solver=solver,
            turbulence_model=turbulence_model,
            description=description,
            output_path=case_path,
            grid_path=mesh_path,
            grid_type=mesh_type,
            grid_boundaries=boundaries,
            max_attempts=max_attempts,
        )

        return self._run_workflow(state, save_snapshot_path, on_step)

    def run_from_snapshot(
        self,
        snapshot_path: str,
        max_attempts: int | None = None,
        on_step: Callable[[str, CaseState], None] | None = None,
    ) -> SolvingResult:
        """
        Run the solving agent from a saved snapshot.

        This is useful for:
        - Resuming a failed run
        - Iterating on corrections without regenerating files
        - Testing with pre-generated files

        Args:
            snapshot_path: Path to snapshot JSON file.
            max_attempts: Override max attempts from snapshot.
            on_step: Optional callback called after each step.

        Returns:
            SolvingResult with success status and case details.
        """
        state = load_snapshot(snapshot_path)

        if max_attempts is not None:
            state = state.apply_update({"max_attempts": max_attempts})

        # Reset execution state for fresh run
        state = state.apply_update({
            "attempt_count": 0,
            "completed": False,
            "success": False,
            "current_error": None,
        })

        save_path = snapshot_path.replace(".json", "_result.json")
        return self._run_workflow(state, save_path, on_step)

    def _run_workflow(
        self,
        state: CaseState,
        save_snapshot_path: str | None,
        on_step: Callable[[str, CaseState], None] | None,
    ) -> SolvingResult:
        """Run the workflow with given state."""
        # Create LLM client
        log_path = None
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            log_path = os.path.join(self.log_dir, "llm_calls.jsonl")

        llm = LLMClient(self.config.llm, log_path=log_path)

        # Create dependencies
        deps = Dependencies(
            config=self.config,
            llm=llm,
            database=self.database,
        )

        # Create and run workflow
        runner = WorkflowRunner(deps, verbose=self.verbose)

        if on_step:
            # Wrap runner to call callback
            original_run_step = runner._run_step
            def wrapped_run_step(step_name, state):
                result = original_run_step(step_name, state)
                on_step(step_name.name, result)
                return result
            runner._run_step = wrapped_run_step

        final_state = runner.run(state)

        # Save snapshot if requested
        snapshot_path = None
        if save_snapshot_path:
            save_snapshot(final_state, save_snapshot_path)
            snapshot_path = save_snapshot_path

        return SolvingResult(
            success=final_state.success,
            case_path=final_state.output_path,
            attempts=final_state.attempt_count,
            errors=final_state.error_history,
            generated_files=list(final_state.generated_files.keys()),
            snapshot_path=snapshot_path,
        )

    def run_simulation_only(
        self,
        snapshot_path: str,
        max_attempts: int = 5,
    ) -> SolvingResult:
        """
        Run only the simulation + correction loop from a snapshot.

        Skips file generation - assumes files are already generated.
        Useful for fast iteration on simulation errors.

        Args:
            snapshot_path: Path to snapshot with generated files.
            max_attempts: Maximum correction attempts.

        Returns:
            SolvingResult with success status.
        """
        from steps import (
            run_simulation, analyze_error, correct_file,
            reflect_on_errors, write_files, convert_mesh,
            run_mesh_conversion_docker, Dependencies,
        )
        from snapshot import save_snapshot, load_snapshot, write_files_from_snapshot

        state = load_snapshot(snapshot_path)

        # Create LLM and dependencies
        llm = LLMClient(self.config.llm)
        deps = Dependencies(
            config=self.config,
            llm=llm,
            database=self.database,
        )

        # Write files from snapshot
        self.log("Writing files from snapshot...")
        write_files_from_snapshot(snapshot_path, state.output_path)

        # Convert mesh if needed
        if not state.mesh_converted and state.grid_path:
            self.log("Converting mesh...")
            if self.config.docker.enabled:
                success = run_mesh_conversion_docker(
                    state.output_path, state.grid_path, deps
                )
            else:
                from steps import run_mesh_converter_local
                success = run_mesh_converter_local(
                    mesh_file=state.grid_path,
                    case_path=state.output_path,
                )
            state = state.apply_update({"mesh_converted": success})

        # Run simulation with correction loop
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            self.log(f"=== Attempt {attempt}/{max_attempts} ===")

            update = run_simulation(state, deps)
            state = state.apply_update(update)

            if state.success:
                self.log("Simulation succeeded!")
                break

            if not state.current_error:
                self.log("No error captured")
                break

            self.log(f"Error: {state.current_error[:100]}...")

            # Check for repeated errors
            consecutive = state.consecutive_same_errors()
            if consecutive >= 4:
                self.log("Same error 4+ times - stopping")
                break
            elif consecutive >= 2:
                self.log("Reflecting on repeated errors...")
                update = reflect_on_errors(state, deps)
                state = state.apply_update(update)

            # Analyze and correct
            self.log("Analyzing error...")
            update = analyze_error(state, deps)
            state = state.apply_update(update)

            if state.error_file:
                self.log(f"Correcting {state.error_file}...")
                update = correct_file(state, deps)
                state = state.apply_update(update)

        # Save result
        result_path = snapshot_path.replace(".json", "_simulated.json")
        save_snapshot(state, result_path)

        return SolvingResult(
            success=state.success,
            case_path=state.output_path,
            attempts=attempt,
            errors=state.error_history,
            generated_files=list(state.generated_files.keys()),
            snapshot_path=result_path,
        )

    def generate_only(
        self,
        solver: str,
        turbulence_model: str,
        description: str,
        boundaries: list[str],
        output_dir: str | None = None,
        case_name: str | None = None,
        save_snapshot_path: str | None = None,
    ) -> tuple[CaseState, str]:
        """
        Generate OpenFOAM files without running simulation.

        Useful for:
        - Generating files to inspect before running
        - Creating snapshots for later execution
        - Testing file generation separately from simulation

        Args:
            solver: OpenFOAM solver name.
            turbulence_model: Turbulence model.
            description: Case description.
            boundaries: Boundary names.
            output_dir: Output directory.
            case_name: Case name.
            save_snapshot_path: Path to save snapshot.

        Returns:
            Tuple of (final_state, case_path).
        """
        from steps import determine_file_structure, generate_files, write_files

        # Setup
        if case_name is None:
            case_name = f"{solver}_{turbulence_model}_generated"
        if output_dir is None:
            output_dir = self.config.paths.output_dir

        case_path = os.path.join(output_dir, case_name)
        os.makedirs(case_path, exist_ok=True)

        state = CaseState(
            case_name=case_name,
            solver=solver,
            turbulence_model=turbulence_model,
            description=description,
            output_path=case_path,
            grid_boundaries=boundaries,
        )

        # Create dependencies
        llm = LLMClient(self.config.llm)
        deps = Dependencies(
            config=self.config,
            llm=llm,
            database=self.database,
        )

        # Run generation steps only
        self.log("Determining file structure...")
        update = determine_file_structure(state, deps)
        state = state.apply_update(update)

        self.log(f"Generating {len(state.file_structure)} files...")
        update = generate_files(state, deps)
        state = state.apply_update(update)

        self.log("Writing files...")
        update = write_files(state, deps)
        state = state.apply_update(update)

        # Save snapshot
        if save_snapshot_path:
            save_snapshot(state, save_snapshot_path)

        return state, case_path
