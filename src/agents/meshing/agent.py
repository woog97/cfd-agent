"""
MeshingAgent: Generates OpenFOAM mesh from CAD geometry.

Supports:
- STEP, STL, IGES, GEO files via Gmsh
- Automatic or manual boundary mapping
- Export to OpenFOAM polyMesh format
- Mesh quality checking
- Run management and mesh deduplication

Requires: pip install gmsh
"""

import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from config import AppConfig, load_config
from run_manager import RunManager, Run, RunStatus
from agents.base import BaseAgent
from agents.meshing.steps import (
    MeshingState,
    MeshingDependencies,
    detect_geometry_type,
    read_geometry,
    generate_mesh_gmsh,
    generate_mesh_with_box,
    convert_to_openfoam,
    check_mesh_quality,
    create_box_geometry,
    create_cylinder_geometry,
)


class MeshingMethod(Enum):
    """Available meshing methods."""
    GMSH = auto()           # Gmsh mesher (default)
    GMSH_WITH_BOX = auto()  # Gmsh with bounding box domain
    SNAPPY_HEX_MESH = auto()  # OpenFOAM's snappyHexMesh (not implemented)


@dataclass
class MeshingResult:
    """Result of a meshing agent run."""
    success: bool
    mesh_path: str
    boundaries: list[str]
    cell_count: int = 0
    quality_metrics: dict | None = None
    error: str | None = None
    run_id: str | None = None  # Associated run ID


class MeshingAgent(BaseAgent[MeshingResult]):
    """
    Agent for generating OpenFOAM meshes from CAD geometry.

    Usage:
        agent = MeshingAgent(config)

        # Mesh a STEP file (creates a run automatically)
        result = agent.run(
            geometry_path="/path/to/geometry.step",
            mesh_size_min=0.01,
            mesh_size_max=0.5,
        )
        print(f"Run ID: {result.run_id}")

        # Reuse mesh from previous run
        result = agent.run(
            geometry_path="/path/to/geometry.step",
            reuse_mesh=True,  # Will find matching run
        )

        # Mesh with external domain (for aerodynamics)
        result = agent.run(
            geometry_path="/path/to/airfoil.step",
            method=MeshingMethod.GMSH_WITH_BOX,
            domain_bounds=(-5, -5, -0.1, 10, 5, 0.1),
        )
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        run_manager: RunManager | None = None,
        verbose: bool = True,
    ):
        """
        Initialize the meshing agent.

        Args:
            config: Application config. Loads default if None.
            run_manager: RunManager for tracking runs. Creates one if None.
            verbose: Whether to log progress.
        """
        super().__init__(config=config, run_manager=run_manager, verbose=verbose)

        # Check dependencies
        self.deps = MeshingDependencies(config=self.config)

        self._log_init(
            gmsh_available=self.deps.gmsh_available,
            docker="enabled" if self.config.docker.enabled else "disabled",
            runs_dir=self.config.paths.runs_dir,
        )

    def run(
        self,
        geometry_path: str,
        method: MeshingMethod = MeshingMethod.GMSH,
        mesh_size_min: float = 0.5,
        mesh_size_max: float = 2.0,
        domain_bounds: tuple[float, ...] | None = None,
        boundary_mapping: dict[str, list] | None = None,
        convert_to_openfoam: bool = True,
        check_quality: bool = True,
        reuse_mesh: bool = True,
        run: Run | None = None,
    ) -> MeshingResult:
        """
        Generate mesh from CAD geometry.

        Args:
            geometry_path: Path to CAD file (.step, .stl, .iges, .geo).
            method: Meshing method to use.
            mesh_size_min: Minimum mesh element size.
            mesh_size_max: Maximum mesh element size.
            domain_bounds: Bounding box for external domain.
            boundary_mapping: Map boundary names to surface IDs.
            convert_to_openfoam: Whether to convert to OpenFOAM format.
            check_quality: Whether to run checkMesh.
            reuse_mesh: If True, check for existing matching mesh.
            run: Existing Run to use. Creates new one if None.

        Returns:
            MeshingResult with mesh details and run_id.
        """
        if not self.deps.gmsh_available:
            return MeshingResult(
                success=False,
                mesh_path="",
                boundaries=[],
                error="Gmsh not installed. Run: pip install gmsh",
            )

        # Check for existing matching mesh
        if reuse_mesh and run is None:
            existing = self.run_manager.find_matching_mesh(
                geometry_path=geometry_path,
                mesh_size_min=mesh_size_min,
                mesh_size_max=mesh_size_max,
                mesh_method=method.name.lower(),
            )
            if existing and existing.has_mesh():
                self.log(f"Reusing mesh from run: {existing.run_id}")

                # Create new run that references existing mesh
                run = self.run_manager.create_run(
                    geometry_path=geometry_path,
                    mesh_size_min=mesh_size_min,
                    mesh_size_max=mesh_size_max,
                    mesh_method=method.name.lower(),
                )
                run.copy_mesh_from(existing)

                # If we need OpenFOAM format, copy that too
                if convert_to_openfoam and existing.has_polymesh():
                    run.copy_polymesh_from(existing)
                    return MeshingResult(
                        success=True,
                        mesh_path=run.polymesh_path,
                        boundaries=[],  # TODO: load from existing
                        cell_count=0,
                        run_id=run.run_id,
                    )

                # Otherwise continue with conversion
                return self._convert_existing_mesh(run, check_quality)

        # Create new run if not provided
        if run is None:
            run = self.run_manager.create_run(
                geometry_path=geometry_path,
                mesh_size_min=mesh_size_min,
                mesh_size_max=mesh_size_max,
                mesh_method=method.name.lower(),
            )

        run.set_status(RunStatus.MESHING)

        # Use run's cache directory for output
        output_path = run.cache_dir

        # Create initial state
        state = MeshingState(
            geometry_path=geometry_path,
            geometry_type=detect_geometry_type(geometry_path),
            output_path=output_path,
            mesh_size_min=mesh_size_min,
            mesh_size_max=mesh_size_max,
            domain_bounds=domain_bounds,
            boundary_mapping=boundary_mapping,
        )

        # Step 1: Read geometry
        self.log(f"Reading geometry: {geometry_path}")

        update = read_geometry(state, self.deps)
        state = state.apply_update(update)

        if state.error:
            run.set_status(RunStatus.FAILED, error=state.error)
            return MeshingResult(
                success=False,
                mesh_path="",
                boundaries=[],
                error=state.error,
                run_id=run.run_id,
            )

        self.log(f"  Type: {state.geometry_type}")
        self.log(f"  Bounds: {state.domain_bounds}")

        # Step 2: Generate mesh
        self.log(f"Generating mesh (method: {method.name})...")

        if method == MeshingMethod.GMSH_WITH_BOX:
            if not domain_bounds:
                # Auto-calculate domain bounds (10x geometry size)
                if state.domain_bounds:
                    xmin, ymin, zmin, xmax, ymax, zmax = state.domain_bounds[:6]
                    dx = xmax - xmin
                    dy = ymax - ymin
                    dz = zmax - zmin
                    state = state.apply_update({
                        "domain_bounds": (
                            xmin - 5*dx, ymin - 5*dy, zmin - 5*dz,
                            xmax + 10*dx, ymax + 5*dy, zmax + 5*dz
                        )
                    })
            update = generate_mesh_with_box(state, self.deps)
        else:
            update = generate_mesh_gmsh(state, self.deps)

        state = state.apply_update(update)

        if state.error:
            run.set_status(RunStatus.FAILED, error=state.error)
            return MeshingResult(
                success=False,
                mesh_path="",
                boundaries=state.boundaries or [],
                error=state.error,
                run_id=run.run_id,
            )

        self.log(f"  Cells: {state.cell_count}")
        self.log(f"  Output: {state.mesh_path}")

        # Update run with mesh hash for deduplication
        run.metadata.mesh_hash = self.run_manager._hash_file(state.mesh_path)
        run.save()

        # Step 3: Convert to OpenFOAM format
        if convert_to_openfoam:
            self.log("Converting to OpenFOAM format...")

            # Update output path to case directory for polyMesh
            state = state.apply_update({"output_path": run.case_dir})

            from agents.meshing.steps import convert_to_openfoam as convert_fn
            update = convert_fn(state, self.deps)
            state = state.apply_update(update)

            if state.error:
                run.set_status(RunStatus.FAILED, error=state.error)
                return MeshingResult(
                    success=False,
                    mesh_path=state.mesh_path,
                    boundaries=state.boundaries or [],
                    cell_count=state.cell_count,
                    error=state.error,
                    run_id=run.run_id,
                )

            self.log(f"  polyMesh: {state.mesh_path}")

        # Step 4: Check mesh quality
        if check_quality and convert_to_openfoam:
            self.log("Checking mesh quality...")

            update = check_mesh_quality(state, self.deps)
            state = state.apply_update(update)

            if state.quality_metrics:
                for k, v in state.quality_metrics.items():
                    self.log(f"  {k}: {v}")

        # Mark run as completed
        run.set_status(RunStatus.COMPLETED)

        return MeshingResult(
            success=state.success,
            mesh_path=state.mesh_path,
            boundaries=state.boundaries or [],
            cell_count=state.cell_count,
            quality_metrics=state.quality_metrics,
            error=state.error,
            run_id=run.run_id,
        )

    def _convert_existing_mesh(self, run: Run, check_quality: bool) -> MeshingResult:
        """Convert an existing mesh in a run to OpenFOAM format."""
        if not run.has_mesh():
            return MeshingResult(
                success=False,
                mesh_path="",
                boundaries=[],
                error="No mesh file in run",
                run_id=run.run_id,
            )

        state = MeshingState(
            geometry_path="",
            output_path=run.case_dir,
            mesh_path=run.mesh_path,
        )

        self.log("Converting existing mesh to OpenFOAM format...")

        from agents.meshing.steps import convert_to_openfoam as convert_fn
        update = convert_fn(state, self.deps)
        state = state.apply_update(update)

        if state.error:
            run.set_status(RunStatus.FAILED, error=state.error)
            return MeshingResult(
                success=False,
                mesh_path=state.mesh_path,
                boundaries=[],
                error=state.error,
                run_id=run.run_id,
            )

        if check_quality:
            update = check_mesh_quality(state, self.deps)
            state = state.apply_update(update)

        run.set_status(RunStatus.COMPLETED)

        return MeshingResult(
            success=state.success,
            mesh_path=state.mesh_path,
            boundaries=state.boundaries or [],
            cell_count=state.cell_count,
            quality_metrics=state.quality_metrics,
            error=state.error,
            run_id=run.run_id,
        )

    def create_test_mesh(
        self,
        geometry: str = "box",
        **mesh_kwargs,
    ) -> MeshingResult:
        """
        Create and mesh a simple test geometry.

        Args:
            geometry: Type of geometry ("box" or "cylinder").
            **mesh_kwargs: Additional arguments passed to run().

        Returns:
            MeshingResult with mesh details.
        """
        # Create a run for the test
        run = self.run_manager.create_run(
            prefix=f"test_{geometry}",
        )

        # Create geometry in run's cache
        if geometry == "box":
            geo_path = create_box_geometry(run.cache_dir)
        elif geometry == "cylinder":
            geo_path = create_cylinder_geometry(run.cache_dir)
        else:
            run.set_status(RunStatus.FAILED, error=f"Unknown geometry: {geometry}")
            return MeshingResult(
                success=False,
                mesh_path="",
                boundaries=[],
                error=f"Unknown geometry type: {geometry}",
                run_id=run.run_id,
            )

        self.log(f"Created test geometry: {geo_path}")

        # Run meshing with the created run
        return self.run(
            geometry_path=geo_path,
            run=run,
            reuse_mesh=False,  # Don't try to dedupe test geometries
            **mesh_kwargs,
        )

    def find_runs_for_geometry(self, geometry_path: str) -> list[Run]:
        """Find all runs that used a specific geometry file."""
        return self.run_manager.find_runs_by_geometry(geometry_path)

    def get_run(self, run_id: str) -> Optional[Run]:
        """Get a specific run by ID."""
        return self.run_manager.get_run(run_id)

    def list_runs(self, status: RunStatus = None, limit: int = None) -> list[Run]:
        """List meshing runs."""
        return self.run_manager.list_runs(status=status, limit=limit)

    def get_supported_formats(self) -> list[str]:
        """Get list of supported geometry file formats."""
        return [".step", ".stp", ".stl", ".iges", ".igs", ".geo", ".brep"]
