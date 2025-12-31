"""
RunManager: Manages run workspaces for agent rollouts.

Each run is a self-contained workspace with:
- Cached intermediate artifacts (mesh, snapshots)
- Final OpenFOAM case
- Logs and results

Supports deduplication by referencing artifacts from previous runs.
"""

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional


class RunStatus(str, Enum):
    """Status of a run."""
    PENDING = "pending"
    MESHING = "meshing"
    GENERATING = "generating"
    SIMULATING = "simulating"
    CORRECTING = "correcting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RunConfig:
    """Configuration for a run."""
    # Input sources
    geometry_path: Optional[str] = None
    mesh_path: Optional[str] = None
    case_description_path: Optional[str] = None

    # Mesh parameters (for deduplication matching)
    mesh_size_min: float = 0.5
    mesh_size_max: float = 2.0
    mesh_method: str = "gmsh"

    # Simulation parameters
    solver: Optional[str] = None
    end_time: Optional[float] = None

    # Deduplication references
    mesh_from: Optional[str] = None      # run_id to copy mesh from
    snapshot_from: Optional[str] = None  # "run_id:step_name" to resume from

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict) -> "RunConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class RunMetadata:
    """Metadata for a run."""
    run_id: str
    created: str
    status: RunStatus
    config: RunConfig
    parent_run: Optional[str] = None
    error: Optional[str] = None
    completed: Optional[str] = None

    # Artifact hashes for deduplication
    geometry_hash: Optional[str] = None
    mesh_hash: Optional[str] = None

    def to_dict(self) -> dict:
        data = {
            "run_id": self.run_id,
            "created": self.created,
            "status": self.status.value if isinstance(self.status, RunStatus) else self.status,
            "config": self.config.to_dict(),
        }
        if self.parent_run:
            data["parent_run"] = self.parent_run
        if self.error:
            data["error"] = self.error
        if self.completed:
            data["completed"] = self.completed
        if self.geometry_hash:
            data["geometry_hash"] = self.geometry_hash
        if self.mesh_hash:
            data["mesh_hash"] = self.mesh_hash
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "RunMetadata":
        config = RunConfig.from_dict(data.get("config", {}))
        status = data.get("status", "pending")
        if isinstance(status, str):
            status = RunStatus(status)
        return cls(
            run_id=data["run_id"],
            created=data["created"],
            status=status,
            config=config,
            parent_run=data.get("parent_run"),
            error=data.get("error"),
            completed=data.get("completed"),
            geometry_hash=data.get("geometry_hash"),
            mesh_hash=data.get("mesh_hash"),
        )


class Run:
    """
    A single run workspace.

    Directory structure:
        {run_id}/
            run.json         # metadata
            cache/           # intermediate artifacts
                mesh.msh
                snapshots/
            case/            # OpenFOAM case
                constant/polyMesh
                0/
                system/
            logs/
            results/
    """

    def __init__(self, run_dir: str, metadata: RunMetadata):
        self.run_dir = run_dir
        self.metadata = metadata

    @property
    def run_id(self) -> str:
        return self.metadata.run_id

    @property
    def status(self) -> RunStatus:
        return self.metadata.status

    @property
    def cache_dir(self) -> str:
        return os.path.join(self.run_dir, "cache")

    @property
    def case_dir(self) -> str:
        return os.path.join(self.run_dir, "case")

    @property
    def logs_dir(self) -> str:
        return os.path.join(self.run_dir, "logs")

    @property
    def results_dir(self) -> str:
        return os.path.join(self.run_dir, "results")

    @property
    def snapshots_dir(self) -> str:
        return os.path.join(self.cache_dir, "snapshots")

    @property
    def mesh_path(self) -> str:
        """Path to cached mesh file."""
        return os.path.join(self.cache_dir, "mesh.msh")

    @property
    def polymesh_path(self) -> str:
        """Path to OpenFOAM polyMesh."""
        return os.path.join(self.case_dir, "constant", "polyMesh")

    def ensure_dirs(self) -> None:
        """Create all run directories."""
        for dir_path in [self.cache_dir, self.case_dir, self.logs_dir,
                         self.results_dir, self.snapshots_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def save(self) -> None:
        """Save run metadata to disk."""
        run_file = os.path.join(self.run_dir, "run.json")
        with open(run_file, "w") as f:
            json.dump(self.metadata.to_dict(), f, indent=2)

    def set_status(self, status: RunStatus, error: str = None) -> None:
        """Update run status."""
        self.metadata.status = status
        if error:
            self.metadata.error = error
        if status == RunStatus.COMPLETED:
            self.metadata.completed = datetime.now().isoformat()
        self.save()

    def has_mesh(self) -> bool:
        """Check if mesh exists in cache."""
        return os.path.exists(self.mesh_path)

    def has_polymesh(self) -> bool:
        """Check if OpenFOAM polyMesh exists."""
        return os.path.exists(os.path.join(self.polymesh_path, "points"))

    def get_snapshots(self) -> list[str]:
        """List available snapshots."""
        if not os.path.exists(self.snapshots_dir):
            return []
        return sorted([
            f for f in os.listdir(self.snapshots_dir)
            if f.endswith(".json")
        ])

    def copy_mesh_from(self, source_run: "Run") -> bool:
        """Copy mesh from another run."""
        if not source_run.has_mesh():
            return False
        os.makedirs(self.cache_dir, exist_ok=True)
        shutil.copy2(source_run.mesh_path, self.mesh_path)
        self.metadata.config.mesh_from = source_run.run_id
        self.metadata.mesh_hash = source_run.metadata.mesh_hash
        self.save()
        return True

    def copy_polymesh_from(self, source_run: "Run") -> bool:
        """Copy polyMesh from another run."""
        if not source_run.has_polymesh():
            return False
        dest = self.polymesh_path
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(source_run.polymesh_path, dest)
        return True


class RunManager:
    """
    Manages run workspaces.

    Usage:
        manager = RunManager("/path/to/runs")

        # Create new run
        run = manager.create_run(
            geometry_path="inputs/geometry/part.step",
            mesh_size_min=0.5,
        )

        # Find existing run with same mesh
        existing = manager.find_matching_mesh(
            geometry_path="inputs/geometry/part.step",
            mesh_size_min=0.5,
        )
        if existing:
            run.copy_mesh_from(existing)

        # Load existing run
        run = manager.get_run("20241226_143052_abc123")

        # List all runs
        for run in manager.list_runs():
            print(f"{run.run_id}: {run.status}")
    """

    def __init__(self, runs_dir: str):
        """
        Initialize RunManager.

        Args:
            runs_dir: Base directory for all runs.
        """
        self.runs_dir = runs_dir
        os.makedirs(runs_dir, exist_ok=True)

    def _generate_run_id(self, prefix: str = None) -> str:
        """Generate unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:6]
        if prefix:
            return f"{timestamp}_{prefix}_{suffix}"
        return f"{timestamp}_{suffix}"

    def _hash_file(self, file_path: str) -> str:
        """Compute hash of a file for deduplication."""
        if not os.path.exists(file_path):
            return None
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def create_run(
        self,
        geometry_path: str = None,
        mesh_path: str = None,
        case_description_path: str = None,
        prefix: str = None,
        **config_kwargs,
    ) -> Run:
        """
        Create a new run workspace.

        Args:
            geometry_path: Path to CAD geometry file.
            mesh_path: Path to existing mesh (skips meshing).
            case_description_path: Path to case description PDF/txt.
            prefix: Optional prefix for run ID (e.g., geometry name).
            **config_kwargs: Additional RunConfig parameters.

        Returns:
            New Run instance.
        """
        # Generate run ID
        if prefix is None and geometry_path:
            prefix = os.path.splitext(os.path.basename(geometry_path))[0]
        run_id = self._generate_run_id(prefix)

        # Create run directory
        run_dir = os.path.join(self.runs_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        # Build config
        config = RunConfig(
            geometry_path=geometry_path,
            mesh_path=mesh_path,
            case_description_path=case_description_path,
            **config_kwargs,
        )

        # Build metadata
        metadata = RunMetadata(
            run_id=run_id,
            created=datetime.now().isoformat(),
            status=RunStatus.PENDING,
            config=config,
        )

        # Hash geometry for deduplication
        if geometry_path and os.path.exists(geometry_path):
            metadata.geometry_hash = self._hash_file(geometry_path)

        # Create run
        run = Run(run_dir, metadata)
        run.ensure_dirs()
        run.save()

        return run

    def get_run(self, run_id: str) -> Optional[Run]:
        """
        Load an existing run by ID.

        Args:
            run_id: Run identifier.

        Returns:
            Run instance or None if not found.
        """
        run_dir = os.path.join(self.runs_dir, run_id)
        run_file = os.path.join(run_dir, "run.json")

        if not os.path.exists(run_file):
            return None

        with open(run_file) as f:
            data = json.load(f)

        metadata = RunMetadata.from_dict(data)
        return Run(run_dir, metadata)

    def list_runs(
        self,
        status: RunStatus = None,
        limit: int = None,
    ) -> list[Run]:
        """
        List all runs, optionally filtered by status.

        Args:
            status: Filter by status.
            limit: Maximum number of runs to return.

        Returns:
            List of Run instances, newest first.
        """
        runs = []

        if not os.path.exists(self.runs_dir):
            return runs

        for run_id in sorted(os.listdir(self.runs_dir), reverse=True):
            run = self.get_run(run_id)
            if run is None:
                continue
            if status and run.status != status:
                continue
            runs.append(run)
            if limit and len(runs) >= limit:
                break

        return runs

    def find_matching_mesh(
        self,
        geometry_path: str,
        mesh_size_min: float = None,
        mesh_size_max: float = None,
        mesh_method: str = None,
    ) -> Optional[Run]:
        """
        Find a run with matching mesh parameters for reuse.

        Args:
            geometry_path: Path to geometry file.
            mesh_size_min: Minimum mesh size.
            mesh_size_max: Maximum mesh size.
            mesh_method: Meshing method.

        Returns:
            Run with matching mesh, or None.
        """
        if not os.path.exists(geometry_path):
            return None

        target_hash = self._hash_file(geometry_path)

        for run in self.list_runs():
            # Check geometry hash matches
            if run.metadata.geometry_hash != target_hash:
                continue

            # Check mesh parameters match
            cfg = run.metadata.config
            if mesh_size_min is not None and cfg.mesh_size_min != mesh_size_min:
                continue
            if mesh_size_max is not None and cfg.mesh_size_max != mesh_size_max:
                continue
            if mesh_method is not None and cfg.mesh_method != mesh_method:
                continue

            # Check mesh exists
            if run.has_mesh() or run.has_polymesh():
                return run

        return None

    def find_runs_by_geometry(self, geometry_path: str) -> list[Run]:
        """Find all runs using a specific geometry file."""
        if not os.path.exists(geometry_path):
            return []

        target_hash = self._hash_file(geometry_path)
        return [
            run for run in self.list_runs()
            if run.metadata.geometry_hash == target_hash
        ]

    def cleanup_failed_runs(self, keep_logs: bool = True) -> int:
        """
        Remove failed run directories.

        Args:
            keep_logs: If True, only remove cache, keep logs.

        Returns:
            Number of runs cleaned up.
        """
        count = 0
        for run in self.list_runs(status=RunStatus.FAILED):
            if keep_logs:
                # Only remove cache
                if os.path.exists(run.cache_dir):
                    shutil.rmtree(run.cache_dir)
            else:
                # Remove entire run
                shutil.rmtree(run.run_dir)
            count += 1
        return count
