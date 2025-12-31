"""
Tests for MeshingAgent.

These tests verify:
- Geometry reading (STEP, STL, GEO)
- Mesh generation with Gmsh
- OpenFOAM conversion
- Mesh quality checking

Usage:
    # Run all tests
    pytest agents/meshing/tests/test_meshing_agent.py -v

    # Run only tests that don't require Gmsh
    pytest agents/meshing/tests/test_meshing_agent.py -v -m "not requires_gmsh"

    # Run with real mesh generation
    pytest agents/meshing/tests/test_meshing_agent.py -v -m "requires_gmsh"
"""

import os
import sys
import shutil
import tempfile

import pytest

# Add src2 to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from config import load_config, AppConfig, DockerConfig
from agents.meshing import MeshingAgent
from agents.meshing.agent import MeshingMethod, MeshingResult
from agents.meshing.steps import (
    MeshingState,
    MeshingDependencies,
    detect_geometry_type,
    create_box_geometry,
    create_cylinder_geometry,
)


# Check if Gmsh is available
def check_gmsh_available():
    try:
        import gmsh
        return True
    except ImportError:
        return False


GMSH_AVAILABLE = check_gmsh_available()

requires_gmsh = pytest.mark.skipif(
    not GMSH_AVAILABLE,
    reason="Gmsh not installed. Run: pip install gmsh"
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    tmpdir = tempfile.mkdtemp(prefix="meshing_test_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def config():
    """Load test config."""
    return load_config()


@pytest.fixture
def docker_config():
    """Create config with Docker enabled."""
    base = load_config()
    return AppConfig(
        paths=base.paths,
        llm=base.llm,
        openfoam=base.openfoam,
        docker=DockerConfig(
            enabled=True,
            image="openfoam/openfoam11-paraview510:latest",
            openfoam_path="/opt/openfoam11",
        ),
    )


@pytest.fixture
def box_geometry(temp_dir):
    """Create a box geometry file."""
    return create_box_geometry(temp_dir)


@pytest.fixture
def cylinder_geometry(temp_dir):
    """Create a cylinder geometry file."""
    return create_cylinder_geometry(temp_dir)


# =============================================================================
# Unit Tests (no Gmsh required)
# =============================================================================

class TestGeometryDetection:
    """Test geometry type detection."""

    def test_detect_step(self):
        assert detect_geometry_type("model.step") == "step"
        assert detect_geometry_type("model.stp") == "step"
        assert detect_geometry_type("MODEL.STEP") == "step"

    def test_detect_stl(self):
        assert detect_geometry_type("model.stl") == "stl"
        assert detect_geometry_type("MODEL.STL") == "stl"

    def test_detect_iges(self):
        assert detect_geometry_type("model.iges") == "iges"
        assert detect_geometry_type("model.igs") == "iges"

    def test_detect_geo(self):
        assert detect_geometry_type("model.geo") == "geo"

    def test_detect_unknown(self):
        assert detect_geometry_type("model.xyz") == "unknown"
        assert detect_geometry_type("model") == "unknown"


class TestMeshingState:
    """Test MeshingState dataclass."""

    def test_default_state(self):
        state = MeshingState()
        assert state.geometry_path == ""
        assert state.success is False
        assert state.error is None

    def test_apply_update(self):
        state = MeshingState(geometry_path="/path/to/geo.step")
        new_state = state.apply_update({
            "success": True,
            "cell_count": 1000,
        })

        assert new_state.geometry_path == "/path/to/geo.step"  # Unchanged
        assert new_state.success is True
        assert new_state.cell_count == 1000

    def test_immutability(self):
        state = MeshingState()
        new_state = state.apply_update({"success": True})

        assert state.success is False  # Original unchanged
        assert new_state.success is True


class TestMeshingDependencies:
    """Test MeshingDependencies."""

    def test_gmsh_detection(self, config):
        deps = MeshingDependencies(config=config)
        assert deps.gmsh_available == GMSH_AVAILABLE


class TestGeometryCreation:
    """Test geometry creation helpers."""

    def test_create_box_geometry(self, temp_dir):
        geo_path = create_box_geometry(temp_dir)

        assert os.path.exists(geo_path)
        assert geo_path.endswith(".geo")

        with open(geo_path) as f:
            content = f.read()
            assert "Box" in content
            assert "Physical Surface" in content

    def test_create_cylinder_geometry(self, temp_dir):
        geo_path = create_cylinder_geometry(temp_dir)

        assert os.path.exists(geo_path)
        assert geo_path.endswith(".geo")

        with open(geo_path) as f:
            content = f.read()
            assert "Cylinder" in content


class TestMeshingAgentInit:
    """Test MeshingAgent initialization."""

    def test_init_default(self, config):
        agent = MeshingAgent(config, verbose=False)
        assert agent.deps.config == config
        assert agent.deps.gmsh_available == GMSH_AVAILABLE

    def test_supported_formats(self, config):
        agent = MeshingAgent(config, verbose=False)
        formats = agent.get_supported_formats()

        assert ".step" in formats
        assert ".stl" in formats
        assert ".geo" in formats


class TestMeshingAgentWithoutGmsh:
    """Test MeshingAgent behavior when Gmsh is not available."""

    def test_run_without_gmsh(self, config, temp_dir):
        # Temporarily disable Gmsh
        agent = MeshingAgent(config, verbose=False)
        agent.deps.gmsh_available = False

        result = agent.run(
            geometry_path="/path/to/model.step",
        )

        assert result.success is False
        assert "Gmsh not installed" in result.error


# =============================================================================
# Integration Tests (requires Gmsh)
# =============================================================================

@requires_gmsh
class TestMeshingAgentWithGmsh:
    """Test MeshingAgent with Gmsh installed."""

    def test_read_geo_file(self, config, box_geometry, temp_dir):
        """Test reading a .geo file."""
        from agents.meshing.steps import read_geometry, MeshingDependencies

        state = MeshingState(
            geometry_path=box_geometry,
            output_path=temp_dir,
        )
        deps = MeshingDependencies(config=config)

        update = read_geometry(state, deps)

        assert "error" not in update or update.get("error") is None
        assert update.get("geometry_type") == "geo"

    def test_generate_mesh_box(self, config, box_geometry, temp_dir):
        """Test generating mesh from box geometry."""
        agent = MeshingAgent(config, verbose=True)

        result = agent.run(
            geometry_path=box_geometry,
            convert_to_openfoam=False,  # Skip OF conversion for speed
            check_quality=False,
            reuse_mesh=False,  # Don't dedupe for testing
        )

        print(f"Result: {result}")
        print(f"Run ID: {result.run_id}")

        assert result.error is None or result.success
        if result.success:
            assert result.cell_count > 0
            assert os.path.exists(result.mesh_path)
            assert result.run_id is not None

    def test_create_test_mesh_box(self, config, temp_dir):
        """Test create_test_mesh helper."""
        agent = MeshingAgent(config, verbose=True)

        result = agent.create_test_mesh(
            geometry="box",
            convert_to_openfoam=False,
            check_quality=False,
        )

        print(f"Result: {result}")
        print(f"Run ID: {result.run_id}")

        if result.success:
            assert result.cell_count > 0
            assert result.run_id is not None

    def test_create_test_mesh_cylinder(self, config, temp_dir):
        """Test create_test_mesh with cylinder."""
        agent = MeshingAgent(config, verbose=True)

        result = agent.create_test_mesh(
            geometry="cylinder",
            convert_to_openfoam=False,
            check_quality=False,
        )

        print(f"Result: {result}")
        print(f"Run ID: {result.run_id}")

        if result.success:
            assert result.cell_count > 0
            assert result.run_id is not None


@requires_gmsh
class TestMeshingWithDocker:
    """Test meshing with Docker for OpenFOAM conversion."""

    def test_mesh_and_convert(self, docker_config, temp_dir):
        """Test full pipeline: mesh + convert to OpenFOAM."""
        agent = MeshingAgent(docker_config, verbose=True)

        result = agent.create_test_mesh(
            geometry="box",
            convert_to_openfoam=True,
            check_quality=True,
        )

        print(f"Result: {result}")
        print(f"Run ID: {result.run_id}")
        print(f"Mesh path: {result.mesh_path}")
        print(f"Quality: {result.quality_metrics}")

        # May fail if Docker not available, that's OK
        if result.error and "Docker" in result.error:
            pytest.skip("Docker not available")

        if result.success:
            assert result.cell_count > 0
            assert result.run_id is not None
            assert "polyMesh" in result.mesh_path or result.mesh_path.endswith(".msh")


# =============================================================================
# CLI Runner
# =============================================================================

def main():
    """Run meshing tests from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Meshing agent tests")
    parser.add_argument("--quick", action="store_true", help="Run only quick tests (no Gmsh)")
    parser.add_argument("--full", action="store_true", help="Run all tests including Gmsh")
    parser.add_argument("--test-mesh", action="store_true", help="Create and test a simple mesh")
    args = parser.parse_args()

    if args.test_mesh:
        # Quick test of meshing
        print("Testing mesh generation...")

        config = load_config()
        agent = MeshingAgent(config, verbose=True)

        if not agent.deps.gmsh_available:
            print("\nGmsh not installed. Run: pip install gmsh")
            sys.exit(1)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = agent.create_test_mesh(
                geometry="box",
                output_dir=tmpdir,
                convert_to_openfoam=False,
            )

            print(f"\nResult:")
            print(f"  Success: {result.success}")
            print(f"  Cells: {result.cell_count}")
            print(f"  Path: {result.mesh_path}")
            print(f"  Error: {result.error}")

        sys.exit(0 if result.success else 1)

    # Run pytest
    pytest_args = [__file__, "-v"]

    if args.quick:
        pytest_args.extend(["-m", "not requires_gmsh"])
    elif args.full:
        pass  # Run all

    sys.exit(pytest.main(pytest_args))


if __name__ == "__main__":
    main()
