"""
Orchestrator: Coordinates multiple agents in a CFD pipeline.

The orchestrator manages the flow between agents:
    CAD Geometry → MeshingAgent → Mesh → SolvingAgent → Results

Usage:
    from agents.orchestrator import Pipeline, run_pipeline

    # Simple pipeline
    result = run_pipeline(
        geometry_path="/path/to/geometry.step",
        solver="simpleFoam",
        description="Flow simulation...",
    )

    # Or build custom pipeline
    pipeline = Pipeline(config)
    pipeline.add_stage("mesh", meshing_agent, {...})
    pipeline.add_stage("solve", solving_agent, {...})
    result = pipeline.run()
"""

import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

from config import AppConfig, load_config
from agents.solving import SolvingAgent
from agents.meshing import MeshingAgent


class StageStatus(Enum):
    """Status of a pipeline stage."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass
class StageResult:
    """Result from a single pipeline stage."""
    name: str
    status: StageStatus
    result: Any = None
    error: str | None = None
    duration_s: float = 0.0


@dataclass
class PipelineResult:
    """Result from complete pipeline run."""
    success: bool
    stages: list[StageResult] = field(default_factory=list)
    final_output: Any = None

    @property
    def failed_stage(self) -> str | None:
        """Name of first failed stage, if any."""
        for stage in self.stages:
            if stage.status == StageStatus.FAILED:
                return stage.name
        return None


class Pipeline:
    """
    Configurable pipeline for chaining agents.

    Example:
        pipeline = Pipeline(config)

        # Add meshing stage (optional, skip if mesh already exists)
        pipeline.add_stage(
            name="mesh",
            agent=MeshingAgent(config),
            run_method="run",
            kwargs={"geometry_path": "/path/to/cad.step", ...},
            skip_if=lambda: os.path.exists("/path/to/mesh"),
        )

        # Add solving stage
        pipeline.add_stage(
            name="solve",
            agent=SolvingAgent(config),
            run_method="run",
            kwargs={"solver": "simpleFoam", ...},
            depends_on=["mesh"],  # Uses output from mesh stage
        )

        result = pipeline.run()
    """

    def __init__(self, config: AppConfig | None = None, verbose: bool = True):
        self.config = config or load_config()
        self.verbose = verbose
        self._stages: list[dict] = []
        self._results: dict[str, StageResult] = {}

    def add_stage(
        self,
        name: str,
        agent: Any,
        run_method: str = "run",
        kwargs: dict | None = None,
        skip_if: Callable[[], bool] | None = None,
        depends_on: list[str] | None = None,
        output_mapper: Callable[[Any], dict] | None = None,
    ) -> "Pipeline":
        """
        Add a stage to the pipeline.

        Args:
            name: Unique name for this stage.
            agent: Agent instance to run.
            run_method: Method name to call on agent.
            kwargs: Arguments to pass to the method.
            skip_if: Callable that returns True to skip this stage.
            depends_on: List of stage names this depends on.
            output_mapper: Function to extract kwargs for next stage from result.

        Returns:
            Self for chaining.
        """
        self._stages.append({
            "name": name,
            "agent": agent,
            "run_method": run_method,
            "kwargs": kwargs or {},
            "skip_if": skip_if,
            "depends_on": depends_on or [],
            "output_mapper": output_mapper,
        })
        return self

    def run(self) -> PipelineResult:
        """
        Run all stages in order.

        Returns:
            PipelineResult with all stage results.
        """
        import time

        results = []
        accumulated_kwargs: dict[str, Any] = {}

        for stage in self._stages:
            name = stage["name"]

            if self.verbose:
                print(f"\n{'='*50}")
                print(f"Stage: {name}")
                print('='*50)

            # Check skip condition
            if stage["skip_if"] and stage["skip_if"]():
                if self.verbose:
                    print(f"Skipping {name} (skip condition met)")
                results.append(StageResult(
                    name=name,
                    status=StageStatus.SKIPPED,
                ))
                continue

            # Check dependencies
            for dep in stage["depends_on"]:
                if dep not in self._results:
                    raise ValueError(f"Stage '{name}' depends on '{dep}' which hasn't run")
                if self._results[dep].status == StageStatus.FAILED:
                    if self.verbose:
                        print(f"Skipping {name} (dependency '{dep}' failed)")
                    results.append(StageResult(
                        name=name,
                        status=StageStatus.SKIPPED,
                        error=f"Dependency '{dep}' failed",
                    ))
                    continue

            # Merge kwargs from dependencies
            merged_kwargs = {**stage["kwargs"], **accumulated_kwargs}

            # Run the stage
            start_time = time.time()
            try:
                agent = stage["agent"]
                method = getattr(agent, stage["run_method"])
                result = method(**merged_kwargs)

                duration = time.time() - start_time
                stage_result = StageResult(
                    name=name,
                    status=StageStatus.COMPLETED,
                    result=result,
                    duration_s=duration,
                )

                # Map outputs for next stage
                if stage["output_mapper"]:
                    mapped = stage["output_mapper"](result)
                    accumulated_kwargs.update(mapped)

                if self.verbose:
                    print(f"Stage {name} completed in {duration:.1f}s")

            except NotImplementedError as e:
                duration = time.time() - start_time
                stage_result = StageResult(
                    name=name,
                    status=StageStatus.FAILED,
                    error=str(e),
                    duration_s=duration,
                )
                if self.verbose:
                    print(f"Stage {name} not implemented: {e}")

            except Exception as e:
                duration = time.time() - start_time
                stage_result = StageResult(
                    name=name,
                    status=StageStatus.FAILED,
                    error=str(e),
                    duration_s=duration,
                )
                if self.verbose:
                    print(f"Stage {name} failed: {e}")

            results.append(stage_result)
            self._results[name] = stage_result

            # Stop on failure
            if stage_result.status == StageStatus.FAILED:
                break

        # Determine overall success
        success = all(
            r.status in (StageStatus.COMPLETED, StageStatus.SKIPPED)
            for r in results
        )

        final_output = None
        if results and results[-1].status == StageStatus.COMPLETED:
            final_output = results[-1].result

        return PipelineResult(
            success=success,
            stages=results,
            final_output=final_output,
        )


def run_pipeline(
    geometry_path: str | None = None,
    mesh_path: str | None = None,
    solver: str = "simpleFoam",
    turbulence_model: str = "kOmegaSST",
    description: str = "",
    boundaries: list[str] | None = None,
    output_dir: str | None = None,
    config: AppConfig | None = None,
    verbose: bool = True,
) -> PipelineResult:
    """
    Run the full CFD pipeline from geometry or mesh to results.

    This is a convenience function that sets up a standard pipeline.

    Args:
        geometry_path: Path to CAD geometry (triggers meshing).
        mesh_path: Path to existing mesh (skips meshing).
        solver: OpenFOAM solver name.
        turbulence_model: Turbulence model.
        description: Case description for LLM.
        boundaries: Boundary names (required if using mesh_path).
        output_dir: Output directory.
        config: Application config.
        verbose: Print progress.

    Returns:
        PipelineResult with all stage results.

    Example:
        # From geometry (meshing + solving)
        result = run_pipeline(
            geometry_path="/path/to/cad.step",
            solver="simpleFoam",
            description="External flow over vehicle",
        )

        # From existing mesh (solving only)
        result = run_pipeline(
            mesh_path="/path/to/mesh.msh",
            solver="simpleFoam",
            boundaries=["inlet", "outlet", "walls"],
            description="External flow simulation",
        )
    """
    config = config or load_config()
    pipeline = Pipeline(config, verbose=verbose)

    # Determine if we need meshing
    if geometry_path and not mesh_path:
        # Add meshing stage
        meshing_agent = MeshingAgent(config, verbose=verbose)
        pipeline.add_stage(
            name="mesh",
            agent=meshing_agent,
            run_method="run",
            kwargs={
                "geometry_path": geometry_path,
                "boundaries": {},  # Would need surface mapping
                "output_dir": output_dir,
            },
            output_mapper=lambda r: {
                "mesh_path": r.mesh_path,
                "boundaries": r.boundaries,
            },
        )

    # Add solving stage
    solving_agent = SolvingAgent(config, verbose=verbose)

    solve_kwargs = {
        "solver": solver,
        "turbulence_model": turbulence_model,
        "description": description,
        "output_dir": output_dir,
    }

    if mesh_path:
        solve_kwargs["mesh_path"] = mesh_path
    if boundaries:
        solve_kwargs["boundaries"] = boundaries

    pipeline.add_stage(
        name="solve",
        agent=solving_agent,
        run_method="run",
        kwargs=solve_kwargs,
        depends_on=["mesh"] if geometry_path else [],
    )

    return pipeline.run()
