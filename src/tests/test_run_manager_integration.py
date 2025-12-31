"""
Integration tests for RunManager with the solving workflow.

Tests:
- Creating runs from existing snapshots
- Running simulation from a run workspace
- Deduplication between runs
"""

import os
import sys
import json
import shutil
import tempfile

import pytest

# Add src2 to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config, AppConfig, DockerConfig
from run_manager import RunManager, Run, RunStatus
from snapshot import load_snapshot, save_snapshot, write_files_from_snapshot


# Paths
SRC2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(SRC2_DIR)
EXISTING_SNAPSHOT = os.path.join(
    ROOT_DIR, "run_chatcfd", "correction_loop_test", "case_snapshot.json"
)
MESH_PATH = os.path.join(ROOT_DIR, "inputs", "grids", "naca0012.msh")
DOCKER_IMAGE = "openfoam/openfoam11-paraview510:latest"


def check_docker_available() -> bool:
    """Check if Docker is available."""
    import subprocess
    try:
        result = subprocess.run(
            ["docker", "images", "-q", DOCKER_IMAGE],
            capture_output=True, text=True, timeout=10
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def create_docker_config() -> AppConfig:
    """Create config with Docker enabled."""
    base = load_config()
    return AppConfig(
        paths=base.paths,
        llm=base.llm,
        openfoam=base.openfoam,
        docker=DockerConfig(
            enabled=True,
            image=DOCKER_IMAGE,
            openfoam_path="/opt/openfoam11",
            timeout=300,
        ),
        pdf_chunk_distance=base.pdf_chunk_distance,
        sentence_transformer_path=base.sentence_transformer_path,
        max_correction_attempts=base.max_correction_attempts,
    )


@pytest.fixture
def config():
    return create_docker_config()


@pytest.fixture
def manager(config):
    """Create RunManager with the configured runs directory."""
    return RunManager(config.paths.runs_dir)


class TestRunManagerWithSnapshot:
    """Test RunManager integration with existing snapshots."""

    @pytest.mark.skipif(
        not os.path.exists(EXISTING_SNAPSHOT),
        reason="No existing snapshot - run test_correction_loop.py --generate first"
    )
    def test_create_run_from_snapshot(self, manager, config):
        """Test creating a run and loading an existing snapshot."""
        # Create a new run
        run = manager.create_run(
            case_description_path="test_naca0012",
            mesh_path=MESH_PATH,
            prefix="naca0012_from_snapshot",
        )

        assert run.status == RunStatus.PENDING
        assert os.path.exists(run.run_dir)

        # Copy snapshot to run's snapshots directory
        snapshot_dest = os.path.join(run.snapshots_dir, "initial.json")
        shutil.copy(EXISTING_SNAPSHOT, snapshot_dest)

        # Load and verify snapshot
        state = load_snapshot(snapshot_dest)
        assert state is not None
        assert state.solver == "simpleFoam"

        # Update run status
        run.set_status(RunStatus.SIMULATING)
        assert run.status == RunStatus.SIMULATING

        # Verify snapshot is accessible
        snapshots = run.get_snapshots()
        assert "initial.json" in snapshots

        print(f"\nCreated run: {run.run_id}")
        print(f"Snapshots: {snapshots}")

    @pytest.mark.skipif(
        not os.path.exists(EXISTING_SNAPSHOT),
        reason="No existing snapshot"
    )
    def test_write_case_files_to_run(self, manager, config):
        """Test writing case files from snapshot to run's case directory."""
        # Create run
        run = manager.create_run(
            mesh_path=MESH_PATH,
            prefix="naca0012_case_files",
        )

        # Copy snapshot to run first
        snapshot_in_run = os.path.join(run.snapshots_dir, "initial.json")
        shutil.copy(EXISTING_SNAPSHOT, snapshot_in_run)

        # Write files to run's case directory
        written_files = write_files_from_snapshot(snapshot_in_run, run.case_dir)

        # Verify files exist
        assert os.path.exists(os.path.join(run.case_dir, "system", "controlDict"))
        assert os.path.exists(os.path.join(run.case_dir, "system", "fvSchemes"))

        # List what was created
        print(f"\nCase directory: {run.case_dir}")
        print(f"Written {len(written_files)} files")
        for root, dirs, files in os.walk(run.case_dir):
            level = root.replace(run.case_dir, '').count(os.sep)
            indent = '  ' * level
            print(f"{indent}{os.path.basename(root)}/")
            for f in files[:5]:  # Limit files shown
                print(f"{indent}  {f}")
            if len(files) > 5:
                print(f"{indent}  ... and {len(files) - 5} more files")

    @pytest.mark.skipif(
        not os.path.exists(EXISTING_SNAPSHOT),
        reason="No existing snapshot"
    )
    def test_run_deduplication(self, manager, config):
        """Test that runs with same config can share artifacts."""
        # Create first run and mark as having mesh
        run1 = manager.create_run(
            mesh_path=MESH_PATH,
            prefix="naca0012_v1",
        )
        # Simulate having a mesh
        with open(run1.mesh_path, "w") as f:
            f.write("mesh content")
        run1.metadata.mesh_hash = "abc123"
        run1.save()

        # Create second run with same mesh path
        run2 = manager.create_run(
            mesh_path=MESH_PATH,
            prefix="naca0012_v2",
        )

        # Find matching run (by mesh path in this case)
        # In real usage, we'd match by geometry_hash for CAD files
        runs_with_mesh = [r for r in manager.list_runs() if r.has_mesh()]
        assert len(runs_with_mesh) >= 1

        # Copy mesh from first run
        success = run2.copy_mesh_from(run1)
        assert success
        assert run2.has_mesh()
        assert run2.metadata.config.mesh_from == run1.run_id

        print(f"\nRun 1: {run1.run_id}")
        print(f"Run 2: {run2.run_id} (mesh copied from run 1)")


class TestRunManagerWithDocker:
    """Test RunManager with actual Docker execution."""

    @pytest.mark.skipif(
        not check_docker_available(),
        reason="Docker not available"
    )
    @pytest.mark.skipif(
        not os.path.exists(EXISTING_SNAPSHOT),
        reason="No existing snapshot"
    )
    def test_run_simulation_in_run_workspace(self, manager, config):
        """Test setting up a run workspace for simulation."""
        # Create run
        run = manager.create_run(
            mesh_path=MESH_PATH,
            prefix="naca0012_docker_test",
        )
        run.set_status(RunStatus.SIMULATING)

        # Copy snapshot to run and write files
        snapshot_in_run = os.path.join(run.snapshots_dir, "initial.json")
        shutil.copy(EXISTING_SNAPSHOT, snapshot_in_run)
        write_files_from_snapshot(snapshot_in_run, run.case_dir)

        # Load state and update output_path
        state = load_snapshot(snapshot_in_run)
        state = state.apply_update({"output_path": run.case_dir})

        # Verify case structure is ready for simulation
        assert os.path.exists(os.path.join(run.case_dir, "system", "controlDict"))
        assert os.path.exists(os.path.join(run.case_dir, "0", "U"))
        assert state.output_path == run.case_dir

        # Simulate a result and update status
        run.set_status(RunStatus.CORRECTING)

        # Save snapshot showing state transition
        final_snapshot_path = os.path.join(run.snapshots_dir, "after_setup.json")
        save_snapshot(state, final_snapshot_path)

        print(f"\nRun workspace ready: {run.run_id}")
        print(f"Case dir: {run.case_dir}")
        print(f"Status: {run.status.value}")
        print(f"Snapshots: {run.get_snapshots()}")


class TestRunWorkflow:
    """Test complete workflow using RunManager."""

    @pytest.mark.skipif(
        not os.path.exists(EXISTING_SNAPSHOT),
        reason="No existing snapshot"
    )
    def test_full_run_lifecycle(self, manager, config):
        """Test the full lifecycle of a run."""
        # 1. Create run
        run = manager.create_run(
            mesh_path=MESH_PATH,
            prefix="lifecycle_test",
        )
        assert run.status == RunStatus.PENDING

        # 2. Import snapshot
        snapshot_path = os.path.join(run.snapshots_dir, "step_001_initial.json")
        shutil.copy(EXISTING_SNAPSHOT, snapshot_path)

        # 3. Transition through states
        run.set_status(RunStatus.GENERATING)
        assert run.status == RunStatus.GENERATING

        run.set_status(RunStatus.SIMULATING)
        assert run.status == RunStatus.SIMULATING

        # 4. Mark as complete
        run.set_status(RunStatus.COMPLETED)
        assert run.status == RunStatus.COMPLETED
        assert run.metadata.completed is not None

        # 5. Verify run can be reloaded
        reloaded = manager.get_run(run.run_id)
        assert reloaded.status == RunStatus.COMPLETED
        assert reloaded.get_snapshots() == ["step_001_initial.json"]

        print(f"\nRun lifecycle test passed: {run.run_id}")
        print(f"  Created: {run.metadata.created}")
        print(f"  Completed: {run.metadata.completed}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
