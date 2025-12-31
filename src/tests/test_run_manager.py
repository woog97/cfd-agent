"""
Tests for RunManager.

These tests verify:
- Run creation and loading
- Status management
- Deduplication (finding matching meshes)
- Artifact copying between runs
"""

import os
import sys
import json
import shutil
import tempfile

import pytest

# Add src2 to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_manager import RunManager, Run, RunStatus, RunConfig, RunMetadata


@pytest.fixture
def temp_runs_dir():
    """Create a temporary directory for test runs."""
    tmpdir = tempfile.mkdtemp(prefix="test_runs_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def temp_geometry_file(temp_runs_dir):
    """Create a temporary geometry file for testing."""
    geo_path = os.path.join(temp_runs_dir, "test_geometry.step")
    with open(geo_path, "w") as f:
        f.write("STEP geometry content for testing")
    return geo_path


@pytest.fixture
def manager(temp_runs_dir):
    """Create a RunManager with temporary directory."""
    return RunManager(temp_runs_dir)


class TestRunConfig:
    """Test RunConfig dataclass."""

    def test_default_values(self):
        config = RunConfig()
        assert config.mesh_size_min == 0.5
        assert config.mesh_size_max == 2.0
        assert config.mesh_method == "gmsh"

    def test_to_dict(self):
        config = RunConfig(geometry_path="/path/to/geo.step", mesh_size_min=0.1)
        data = config.to_dict()
        assert data["geometry_path"] == "/path/to/geo.step"
        assert data["mesh_size_min"] == 0.1
        assert "mesh_from" not in data  # None values excluded

    def test_from_dict(self):
        data = {"geometry_path": "/path/to/geo.step", "mesh_size_min": 0.1}
        config = RunConfig.from_dict(data)
        assert config.geometry_path == "/path/to/geo.step"
        assert config.mesh_size_min == 0.1


class TestRunMetadata:
    """Test RunMetadata dataclass."""

    def test_to_dict(self):
        config = RunConfig(geometry_path="/path/to/geo.step")
        metadata = RunMetadata(
            run_id="test_123",
            created="2024-12-26T12:00:00",
            status=RunStatus.PENDING,
            config=config,
        )
        data = metadata.to_dict()
        assert data["run_id"] == "test_123"
        assert data["status"] == "pending"
        assert data["config"]["geometry_path"] == "/path/to/geo.step"

    def test_from_dict(self):
        data = {
            "run_id": "test_123",
            "created": "2024-12-26T12:00:00",
            "status": "pending",
            "config": {"geometry_path": "/path/to/geo.step"},
        }
        metadata = RunMetadata.from_dict(data)
        assert metadata.run_id == "test_123"
        assert metadata.status == RunStatus.PENDING
        assert metadata.config.geometry_path == "/path/to/geo.step"


class TestRunManager:
    """Test RunManager class."""

    def test_create_run(self, manager, temp_geometry_file):
        run = manager.create_run(geometry_path=temp_geometry_file)

        assert run.run_id is not None
        assert run.status == RunStatus.PENDING
        assert os.path.exists(run.run_dir)
        assert os.path.exists(run.cache_dir)
        assert os.path.exists(run.case_dir)

    def test_create_run_with_prefix(self, manager, temp_geometry_file):
        run = manager.create_run(
            geometry_path=temp_geometry_file,
            prefix="MyGeometry",
        )
        assert "MyGeometry" in run.run_id

    def test_get_run(self, manager, temp_geometry_file):
        created = manager.create_run(geometry_path=temp_geometry_file)
        loaded = manager.get_run(created.run_id)

        assert loaded is not None
        assert loaded.run_id == created.run_id
        assert loaded.metadata.config.geometry_path == temp_geometry_file

    def test_get_nonexistent_run(self, manager):
        result = manager.get_run("nonexistent_run_id")
        assert result is None

    def test_list_runs(self, manager, temp_geometry_file):
        # Create multiple runs
        run1 = manager.create_run(geometry_path=temp_geometry_file)
        run2 = manager.create_run(geometry_path=temp_geometry_file)

        runs = manager.list_runs()
        assert len(runs) == 2
        # Both runs should be in the list
        run_ids = {r.run_id for r in runs}
        assert run1.run_id in run_ids
        assert run2.run_id in run_ids

    def test_list_runs_by_status(self, manager, temp_geometry_file):
        run1 = manager.create_run(geometry_path=temp_geometry_file)
        run2 = manager.create_run(geometry_path=temp_geometry_file)
        run2.set_status(RunStatus.COMPLETED)

        pending = manager.list_runs(status=RunStatus.PENDING)
        completed = manager.list_runs(status=RunStatus.COMPLETED)

        assert len(pending) == 1
        assert len(completed) == 1
        assert pending[0].run_id == run1.run_id
        assert completed[0].run_id == run2.run_id

    def test_list_runs_with_limit(self, manager, temp_geometry_file):
        for _ in range(5):
            manager.create_run(geometry_path=temp_geometry_file)

        runs = manager.list_runs(limit=3)
        assert len(runs) == 3


class TestRun:
    """Test Run class."""

    def test_status_update(self, manager, temp_geometry_file):
        run = manager.create_run(geometry_path=temp_geometry_file)
        assert run.status == RunStatus.PENDING

        run.set_status(RunStatus.MESHING)
        assert run.status == RunStatus.MESHING

        # Reload and verify persisted
        loaded = manager.get_run(run.run_id)
        assert loaded.status == RunStatus.MESHING

    def test_status_completed_sets_timestamp(self, manager, temp_geometry_file):
        run = manager.create_run(geometry_path=temp_geometry_file)
        assert run.metadata.completed is None

        run.set_status(RunStatus.COMPLETED)
        assert run.metadata.completed is not None

    def test_status_failed_with_error(self, manager, temp_geometry_file):
        run = manager.create_run(geometry_path=temp_geometry_file)
        run.set_status(RunStatus.FAILED, error="Something went wrong")

        assert run.status == RunStatus.FAILED
        assert run.metadata.error == "Something went wrong"

    def test_has_mesh(self, manager, temp_geometry_file):
        run = manager.create_run(geometry_path=temp_geometry_file)
        assert not run.has_mesh()

        # Create fake mesh
        with open(run.mesh_path, "w") as f:
            f.write("fake mesh content")

        assert run.has_mesh()

    def test_has_polymesh(self, manager, temp_geometry_file):
        run = manager.create_run(geometry_path=temp_geometry_file)
        assert not run.has_polymesh()

        # Create fake polyMesh
        os.makedirs(run.polymesh_path, exist_ok=True)
        with open(os.path.join(run.polymesh_path, "points"), "w") as f:
            f.write("fake points")

        assert run.has_polymesh()

    def test_get_snapshots(self, manager, temp_geometry_file):
        run = manager.create_run(geometry_path=temp_geometry_file)
        assert run.get_snapshots() == []

        # Create fake snapshots
        for i in range(3):
            with open(os.path.join(run.snapshots_dir, f"step_{i:03d}.json"), "w") as f:
                f.write("{}")

        snapshots = run.get_snapshots()
        assert len(snapshots) == 3
        assert snapshots[0] == "step_000.json"


class TestDeduplication:
    """Test deduplication features."""

    def test_find_matching_mesh(self, manager, temp_geometry_file):
        # Create run with mesh
        run1 = manager.create_run(
            geometry_path=temp_geometry_file,
            mesh_size_min=0.5,
            mesh_size_max=2.0,
        )
        with open(run1.mesh_path, "w") as f:
            f.write("mesh content")

        # Find matching
        match = manager.find_matching_mesh(
            geometry_path=temp_geometry_file,
            mesh_size_min=0.5,
            mesh_size_max=2.0,
        )
        assert match is not None
        assert match.run_id == run1.run_id

    def test_find_matching_mesh_different_params(self, manager, temp_geometry_file):
        # Create run with mesh
        run1 = manager.create_run(
            geometry_path=temp_geometry_file,
            mesh_size_min=0.5,
        )
        with open(run1.mesh_path, "w") as f:
            f.write("mesh content")

        # Different mesh size should not match
        match = manager.find_matching_mesh(
            geometry_path=temp_geometry_file,
            mesh_size_min=0.1,  # Different
        )
        assert match is None

    def test_find_matching_mesh_different_geometry(self, manager, temp_runs_dir):
        # Create geometry 1
        geo1 = os.path.join(temp_runs_dir, "geo1.step")
        with open(geo1, "w") as f:
            f.write("geometry 1 content")

        # Create geometry 2
        geo2 = os.path.join(temp_runs_dir, "geo2.step")
        with open(geo2, "w") as f:
            f.write("geometry 2 content")

        # Create run with geo1
        run1 = manager.create_run(geometry_path=geo1)
        with open(run1.mesh_path, "w") as f:
            f.write("mesh content")

        # Search with geo2 should not match
        match = manager.find_matching_mesh(geometry_path=geo2)
        assert match is None

    def test_copy_mesh_from(self, manager, temp_geometry_file):
        # Create source run with mesh
        source = manager.create_run(geometry_path=temp_geometry_file)
        with open(source.mesh_path, "w") as f:
            f.write("mesh content")
        source.metadata.mesh_hash = "abc123"
        source.save()

        # Create target run
        target = manager.create_run(geometry_path=temp_geometry_file)
        assert not target.has_mesh()

        # Copy mesh
        success = target.copy_mesh_from(source)
        assert success
        assert target.has_mesh()
        assert target.metadata.config.mesh_from == source.run_id

    def test_find_runs_by_geometry(self, manager, temp_geometry_file):
        # Create multiple runs with same geometry
        run1 = manager.create_run(geometry_path=temp_geometry_file)
        run2 = manager.create_run(geometry_path=temp_geometry_file)

        runs = manager.find_runs_by_geometry(temp_geometry_file)
        assert len(runs) == 2


class TestCleanup:
    """Test cleanup features."""

    def test_cleanup_failed_runs(self, manager, temp_geometry_file):
        # Create runs
        run1 = manager.create_run(geometry_path=temp_geometry_file)
        run2 = manager.create_run(geometry_path=temp_geometry_file)
        run2.set_status(RunStatus.FAILED)

        # Create some cache content
        with open(os.path.join(run2.cache_dir, "test.txt"), "w") as f:
            f.write("test")

        # Cleanup
        count = manager.cleanup_failed_runs(keep_logs=True)
        assert count == 1

        # Cache should be removed but logs kept
        assert not os.path.exists(run2.cache_dir)
        assert os.path.exists(run2.logs_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
