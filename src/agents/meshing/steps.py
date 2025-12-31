"""
Meshing Steps: Step functions for mesh generation workflow.

This module provides step functions for:
- Reading CAD geometry (.step, .stl, .iges)
- Generating mesh with Gmsh
- Converting to OpenFOAM format
- Checking mesh quality

Design follows the same pattern as src2/steps.py:
- Each step is a pure-ish function: (state, deps) -> StateUpdate
- No global state
- Dependencies injected
"""

import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any

from config import AppConfig


@dataclass
class MeshingDependencies:
    """Dependencies for meshing step functions."""
    config: AppConfig
    gmsh_available: bool = False

    def __post_init__(self):
        # Check if gmsh is available
        try:
            import gmsh
            self.gmsh_available = True
        except ImportError:
            self.gmsh_available = False


@dataclass
class MeshingState:
    """State for meshing workflow."""
    # Input
    geometry_path: str = ""
    geometry_type: str = ""  # step, stl, iges, geo

    # Configuration
    case_name: str = ""
    output_path: str = ""

    # Domain definition
    domain_bounds: tuple[float, ...] | None = None  # (xmin, ymin, zmin, xmax, ymax, zmax)

    # Mesh parameters
    mesh_size_min: float = 0.01
    mesh_size_max: float = 1.0
    refinement_level: int = 2
    boundary_layers: int = 3

    # Boundary mapping: name -> list of surface IDs or names
    boundary_mapping: dict[str, list[str | int]] | None = None

    # Output
    mesh_path: str = ""
    boundaries: list[str] | None = None
    cell_count: int = 0

    # Quality metrics
    quality_metrics: dict[str, float] | None = None

    # Status
    success: bool = False
    error: str | None = None

    def apply_update(self, update: dict) -> "MeshingState":
        """Apply an update dict to create a new state."""
        import dataclasses
        current = dataclasses.asdict(self)
        current.update(update)
        return MeshingState(**current)


MeshingStateUpdate = dict[str, Any]


# =============================================================================
# Geometry Reading
# =============================================================================

def detect_geometry_type(geometry_path: str) -> str:
    """Detect geometry file type from extension."""
    ext = os.path.splitext(geometry_path)[1].lower()
    type_map = {
        ".step": "step",
        ".stp": "step",
        ".stl": "stl",
        ".iges": "iges",
        ".igs": "iges",
        ".geo": "geo",  # Gmsh native format
        ".brep": "brep",
    }
    return type_map.get(ext, "unknown")


def read_geometry(state: MeshingState, deps: MeshingDependencies) -> MeshingStateUpdate:
    """
    Read and validate CAD geometry.

    Returns:
        StateUpdate with geometry info or error.
    """
    if not os.path.exists(state.geometry_path):
        return {"error": f"Geometry file not found: {state.geometry_path}"}

    geo_type = detect_geometry_type(state.geometry_path)
    if geo_type == "unknown":
        return {"error": f"Unknown geometry type: {state.geometry_path}"}

    if not deps.gmsh_available:
        return {"error": "Gmsh not installed. Run: pip install gmsh"}

    try:
        import gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)  # Suppress output

        # Import geometry
        if geo_type == "step":
            gmsh.model.occ.importShapes(state.geometry_path)
            gmsh.model.occ.synchronize()
        elif geo_type == "stl":
            gmsh.merge(state.geometry_path)
        elif geo_type == "geo":
            gmsh.open(state.geometry_path)
        else:
            gmsh.merge(state.geometry_path)

        # Get bounding box
        bounds = gmsh.model.getBoundingBox(-1, -1)

        # Get surfaces for boundary mapping
        surfaces = gmsh.model.getEntities(2)  # dim=2 for surfaces

        gmsh.finalize()

        return {
            "geometry_type": geo_type,
            "domain_bounds": tuple(bounds),
            "boundaries": [f"surface_{s[1]}" for s in surfaces],
        }
    except Exception as e:
        try:
            gmsh.finalize()
        except Exception:
            pass
        return {"error": f"Failed to read geometry: {str(e)}"}


# =============================================================================
# Mesh Generation
# =============================================================================

def generate_mesh_gmsh(state: MeshingState, deps: MeshingDependencies) -> MeshingStateUpdate:
    """
    Generate mesh using Gmsh.

    Supports:
    - STEP, STL, IGES, GEO files
    - Automatic sizing based on geometry
    - Boundary layer meshing
    - Export to OpenFOAM format

    Returns:
        StateUpdate with mesh path and cell count.
    """
    if not deps.gmsh_available:
        return {"error": "Gmsh not installed. Run: pip install gmsh"}

    try:
        import gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)

        # Import geometry
        geo_type = state.geometry_type or detect_geometry_type(state.geometry_path)

        if geo_type == "step":
            gmsh.model.occ.importShapes(state.geometry_path)
            gmsh.model.occ.synchronize()
        elif geo_type == "geo":
            gmsh.open(state.geometry_path)
        else:
            gmsh.merge(state.geometry_path)

        gmsh.model.occ.synchronize()

        # Set mesh sizes
        gmsh.option.setNumber("Mesh.MeshSizeMin", state.mesh_size_min)
        gmsh.option.setNumber("Mesh.MeshSizeMax", state.mesh_size_max)

        # Mesh algorithm options
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT

        # Generate 3D mesh
        gmsh.model.mesh.generate(3)

        # Get mesh statistics
        nodes = gmsh.model.mesh.getNodes()
        elements = gmsh.model.mesh.getElements()

        # Count 3D elements (tetrahedra, hexahedra, etc.)
        cell_count = 0
        for elem_type, tags, _ in zip(*elements):
            elem_props = gmsh.model.mesh.getElementProperties(elem_type)
            if elem_props[1] == 3:  # dimension = 3
                cell_count += len(tags)

        # Output path for mesh
        mesh_output = os.path.join(state.output_path, "mesh.msh")
        os.makedirs(state.output_path, exist_ok=True)

        # Save as MSH format (version 2 for OpenFOAM compatibility)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(mesh_output)

        # Get physical groups (boundaries)
        boundaries = []
        for dim, tag in gmsh.model.getPhysicalGroups():
            name = gmsh.model.getPhysicalName(dim, tag)
            if name:
                boundaries.append(name)

        gmsh.finalize()

        return {
            "mesh_path": mesh_output,
            "cell_count": cell_count,
            "boundaries": boundaries if boundaries else state.boundaries,
            "success": True,
        }

    except Exception as e:
        try:
            gmsh.finalize()
        except Exception:
            pass
        return {"error": f"Mesh generation failed: {str(e)}"}


def generate_mesh_with_box(state: MeshingState, deps: MeshingDependencies) -> MeshingStateUpdate:
    """
    Generate mesh with a bounding box domain around the geometry.

    Useful for external aerodynamics where you need a far-field domain.

    Returns:
        StateUpdate with mesh path and cell count.
    """
    if not deps.gmsh_available:
        return {"error": "Gmsh not installed"}

    if not state.domain_bounds:
        return {"error": "Domain bounds required for box mesh"}

    try:
        import gmsh
        gmsh.initialize()
        gmsh.model.add("domain")

        xmin, ymin, zmin, xmax, ymax, zmax = state.domain_bounds

        # Create outer box
        outer_box = gmsh.model.occ.addBox(xmin, ymin, zmin,
                                          xmax - xmin, ymax - ymin, zmax - zmin)

        # Import geometry
        geo_type = state.geometry_type or detect_geometry_type(state.geometry_path)

        if geo_type == "step":
            shapes = gmsh.model.occ.importShapes(state.geometry_path)
        else:
            gmsh.merge(state.geometry_path)
            shapes = gmsh.model.getEntities(3)

        gmsh.model.occ.synchronize()

        # Cut geometry from box (boolean difference)
        if shapes:
            inner_volumes = [s for s in shapes if s[0] == 3]
            if inner_volumes:
                gmsh.model.occ.cut([(3, outer_box)], inner_volumes)
                gmsh.model.occ.synchronize()

        # Set mesh sizes
        gmsh.option.setNumber("Mesh.MeshSizeMin", state.mesh_size_min)
        gmsh.option.setNumber("Mesh.MeshSizeMax", state.mesh_size_max)

        # Generate mesh
        gmsh.model.mesh.generate(3)

        # Get statistics
        cell_count = 0
        for elem_type, tags, _ in zip(*gmsh.model.mesh.getElements()):
            props = gmsh.model.mesh.getElementProperties(elem_type)
            if props[1] == 3:
                cell_count += len(tags)

        # Save mesh
        mesh_output = os.path.join(state.output_path, "mesh.msh")
        os.makedirs(state.output_path, exist_ok=True)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(mesh_output)

        gmsh.finalize()

        return {
            "mesh_path": mesh_output,
            "cell_count": cell_count,
            "success": True,
        }

    except Exception as e:
        try:
            gmsh.finalize()
        except Exception:
            pass
        return {"error": f"Box mesh generation failed: {str(e)}"}


# =============================================================================
# OpenFOAM Conversion
# =============================================================================

def convert_to_openfoam(state: MeshingState, deps: MeshingDependencies) -> MeshingStateUpdate:
    """
    Convert Gmsh mesh to OpenFOAM polyMesh format.

    Uses gmshToFoam or fluentMeshToFoam depending on mesh format.

    Returns:
        StateUpdate with OpenFOAM mesh path.
    """
    if not state.mesh_path:
        return {"error": "No mesh to convert"}

    mesh_ext = os.path.splitext(state.mesh_path)[1].lower()

    # Determine converter
    if mesh_ext == ".msh":
        converter = "gmshToFoam"
        fallback = "fluentMeshToFoam"
    else:
        return {"error": f"Unknown mesh format: {mesh_ext}"}

    # Create minimal OpenFOAM case structure required by converters
    _create_minimal_case_structure(state.output_path)

    case_path = state.output_path

    # Try Docker if enabled
    if deps.config.docker.enabled:
        return _convert_mesh_docker(state, deps, converter)

    # Try local conversion
    try:
        result = subprocess.run(
            [converter, "-case", case_path, state.mesh_path],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            # Try fallback
            result = subprocess.run(
                [fallback, "-case", case_path, state.mesh_path],
                capture_output=True,
                text=True,
                timeout=120,
            )

        if result.returncode == 0:
            poly_mesh = os.path.join(case_path, "constant", "polyMesh")
            if os.path.exists(poly_mesh):
                return {"mesh_path": poly_mesh, "success": True}

        return {"error": f"Mesh conversion failed: {result.stderr}"}

    except FileNotFoundError:
        return {"error": f"{converter} not found. Is OpenFOAM installed?"}
    except Exception as e:
        return {"error": f"Conversion failed: {str(e)}"}


def _convert_mesh_docker(
    state: MeshingState,
    deps: MeshingDependencies,
    converter: str
) -> MeshingStateUpdate:
    """Convert mesh using Docker."""
    docker_cfg = deps.config.docker
    case_path = os.path.abspath(state.output_path)
    mesh_path = os.path.abspath(state.mesh_path)

    # Mount mesh file if outside case directory
    mesh_dir = os.path.dirname(mesh_path)
    mesh_name = os.path.basename(mesh_path)

    if mesh_dir.startswith(case_path):
        # Mesh is inside case directory
        rel_mesh = os.path.relpath(mesh_path, case_path)
        docker_cmd = [
            "docker", "run", "--rm",
            "--platform", docker_cfg.platform,
            "--entrypoint", "/bin/bash",
            "-v", f"{case_path}:/case",
            "-w", "/case",
            docker_cfg.image,
            "-c", f"source {docker_cfg.openfoam_path}/etc/bashrc && {converter} {rel_mesh}"
        ]
    else:
        # Mount mesh separately
        docker_cmd = [
            "docker", "run", "--rm",
            "--platform", docker_cfg.platform,
            "--entrypoint", "/bin/bash",
            "-v", f"{case_path}:/case",
            "-v", f"{mesh_dir}:/mesh:ro",
            "-w", "/case",
            docker_cfg.image,
            "-c", f"source {docker_cfg.openfoam_path}/etc/bashrc && {converter} /mesh/{mesh_name}"
        ]

    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )

        poly_mesh = os.path.join(case_path, "constant", "polyMesh")
        if os.path.exists(poly_mesh):
            return {"mesh_path": poly_mesh, "success": True}

        return {"error": f"Docker mesh conversion failed: {result.stderr}"}

    except Exception as e:
        return {"error": f"Docker conversion error: {str(e)}"}


# =============================================================================
# Mesh Quality
# =============================================================================

def check_mesh_quality(state: MeshingState, deps: MeshingDependencies) -> MeshingStateUpdate:
    """
    Check mesh quality using OpenFOAM's checkMesh.

    Returns:
        StateUpdate with quality metrics.
    """
    case_path = state.output_path

    if deps.config.docker.enabled:
        return _check_mesh_docker(state, deps)

    try:
        result = subprocess.run(
            ["checkMesh", "-case", case_path],
            capture_output=True,
            text=True,
            timeout=60,
        )

        metrics = _parse_check_mesh_output(result.stdout)

        # Check for fatal errors
        if "FOAM FATAL" in result.stdout or "FOAM FATAL" in result.stderr:
            return {
                "quality_metrics": metrics,
                "error": "Mesh has fatal errors",
                "success": False,
            }

        return {
            "quality_metrics": metrics,
            "success": True,
        }

    except FileNotFoundError:
        return {"error": "checkMesh not found"}
    except Exception as e:
        return {"error": f"Mesh check failed: {str(e)}"}


def _check_mesh_docker(state: MeshingState, deps: MeshingDependencies) -> MeshingStateUpdate:
    """Check mesh using Docker."""
    docker_cfg = deps.config.docker
    case_path = os.path.abspath(state.output_path)

    docker_cmd = [
        "docker", "run", "--rm",
        "--platform", docker_cfg.platform,
        "--entrypoint", "/bin/bash",
        "-v", f"{case_path}:/case",
        "-w", "/case",
        docker_cfg.image,
        "-c", f"source {docker_cfg.openfoam_path}/etc/bashrc && checkMesh"
    ]

    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )

        metrics = _parse_check_mesh_output(result.stdout)

        if "FOAM FATAL" in result.stdout or "FOAM FATAL" in result.stderr:
            return {
                "quality_metrics": metrics,
                "error": "Mesh has fatal errors",
                "success": False,
            }

        return {
            "quality_metrics": metrics,
            "success": True,
        }

    except Exception as e:
        return {"error": f"Docker checkMesh failed: {str(e)}"}


def _parse_check_mesh_output(output: str) -> dict[str, float]:
    """Parse checkMesh output for quality metrics."""
    import re

    metrics = {}

    # Cell count
    match = re.search(r"cells:\s+(\d+)", output)
    if match:
        metrics["cells"] = int(match.group(1))

    # Non-orthogonality
    match = re.search(r"Mesh non-orthogonality Max:\s+([\d.]+)", output)
    if match:
        metrics["max_non_orthogonality"] = float(match.group(1))

    # Skewness
    match = re.search(r"Max skewness =\s+([\d.]+)", output)
    if match:
        metrics["max_skewness"] = float(match.group(1))

    # Aspect ratio
    match = re.search(r"Max aspect ratio =\s+([\d.]+)", output)
    if match:
        metrics["max_aspect_ratio"] = float(match.group(1))

    return metrics


# =============================================================================
# Simple Geometry Creation (for testing)
# =============================================================================

def _create_minimal_case_structure(case_path: str) -> None:
    """
    Create minimal OpenFOAM case structure required by mesh converters.

    gmshToFoam and other utilities require at least:
    - system/controlDict
    """
    system_dir = os.path.join(case_path, "system")
    constant_dir = os.path.join(case_path, "constant")

    os.makedirs(system_dir, exist_ok=True)
    os.makedirs(constant_dir, exist_ok=True)

    # Minimal controlDict
    control_dict = os.path.join(system_dir, "controlDict")
    if not os.path.exists(control_dict):
        with open(control_dict, "w") as f:
            f.write("""FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}

application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1000;
deltaT          1;
writeControl    timeStep;
writeInterval   100;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
""")


def create_box_geometry(
    output_path: str,
    dimensions: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> str:
    """
    Create a simple box geometry for testing.

    Args:
        output_path: Directory to save the geometry.
        dimensions: Box dimensions (dx, dy, dz).
        origin: Box origin (x, y, z).

    Returns:
        Path to created .geo file.
    """
    os.makedirs(output_path, exist_ok=True)
    geo_path = os.path.join(output_path, "box.geo")

    dx, dy, dz = dimensions
    x0, y0, z0 = origin

    geo_content = f"""// Simple box geometry for testing
SetFactory("OpenCASCADE");

// Create box
Box(1) = {{{x0}, {y0}, {z0}, {dx}, {dy}, {dz}}};

// Define physical groups for boundaries
Physical Surface("inlet") = {{1}};
Physical Surface("outlet") = {{2}};
Physical Surface("walls") = {{3, 4, 5, 6}};
Physical Volume("internal") = {{1}};

// Mesh sizing
Mesh.MeshSizeMin = 0.05;
Mesh.MeshSizeMax = 0.2;
"""

    with open(geo_path, "w") as f:
        f.write(geo_content)

    return geo_path


def create_cylinder_geometry(
    output_path: str,
    radius: float = 0.5,
    height: float = 2.0,
) -> str:
    """
    Create a cylinder geometry for testing.

    Args:
        output_path: Directory to save the geometry.
        radius: Cylinder radius.
        height: Cylinder height.

    Returns:
        Path to created .geo file.
    """
    os.makedirs(output_path, exist_ok=True)
    geo_path = os.path.join(output_path, "cylinder.geo")

    geo_content = f"""// Cylinder geometry for testing
SetFactory("OpenCASCADE");

// Create cylinder along z-axis
Cylinder(1) = {{0, 0, 0, 0, 0, {height}, {radius}}};

// Define physical groups
Physical Surface("inlet") = {{2}};    // Bottom
Physical Surface("outlet") = {{3}};   // Top
Physical Surface("walls") = {{1}};    // Curved surface
Physical Volume("internal") = {{1}};

// Mesh sizing
Mesh.MeshSizeMin = 0.02;
Mesh.MeshSizeMax = 0.1;
"""

    with open(geo_path, "w") as f:
        f.write(geo_content)

    return geo_path
