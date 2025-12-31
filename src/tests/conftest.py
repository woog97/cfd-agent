"""
Shared test fixtures and utilities for src2 tests.

This module provides:
- Dependency checking (PyFoam, OpenFOAM, etc.)
- Shared fixtures for database, config, state
- Temporary directory fixtures
- Mock LLM client fixtures
"""

import os
import sys
import json
import shutil
import tempfile
import pytest
from unittest.mock import Mock, MagicMock
from dataclasses import FrozenInstanceError

# Add src2 to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Dependency Checking
# =============================================================================

def check_pyfoam_available():
    """Check if PyFoam is available."""
    try:
        from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
        return True
    except ImportError:
        return False


def check_openfoam_available():
    """Check if OpenFOAM is available."""
    return shutil.which("simpleFoam") is not None


def check_tiktoken_available():
    """Check if tiktoken is available."""
    try:
        import tiktoken
        return True
    except ImportError:
        return False


# Check dependencies at import time
PYFOAM_AVAILABLE = check_pyfoam_available()
OPENFOAM_AVAILABLE = check_openfoam_available()
TIKTOKEN_AVAILABLE = check_tiktoken_available()

# Skip markers
requires_pyfoam = pytest.mark.skipif(
    not PYFOAM_AVAILABLE,
    reason="PyFoam not installed"
)

requires_openfoam = pytest.mark.skipif(
    not OPENFOAM_AVAILABLE,
    reason="OpenFOAM not installed"
)

requires_tiktoken = pytest.mark.skipif(
    not TIKTOKEN_AVAILABLE,
    reason="tiktoken not installed"
)


# =============================================================================
# Path Constants
# =============================================================================

SRC2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(SRC2_DIR)
DATABASE_DIR = os.path.join(ROOT_DIR, "database_OFv24")
CONFIG_PATH = os.path.join(ROOT_DIR, "inputs", "chatcfd_config.json")

# Check if real resources exist
DATABASE_EXISTS = os.path.exists(os.path.join(DATABASE_DIR, "processed_merged_OF_cases.json"))
CONFIG_EXISTS = os.path.exists(CONFIG_PATH)


# =============================================================================
# Fixtures: Temporary Files and Directories
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory that's cleaned up after the test."""
    tmpdir = tempfile.mkdtemp(prefix="chatcfd_test_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def temp_openfoam_case(temp_dir):
    """Create a temporary OpenFOAM case directory structure."""
    # Create standard directories
    os.makedirs(os.path.join(temp_dir, "0"))
    os.makedirs(os.path.join(temp_dir, "constant"))
    os.makedirs(os.path.join(temp_dir, "system"))

    # Create minimal files
    u_content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);
boundaryField
{
    inlet
    {
        type    fixedValue;
        value   uniform (1 0 0);
    }
    outlet
    {
        type    zeroGradient;
    }
    wall
    {
        type    noSlip;
    }
}
"""
    with open(os.path.join(temp_dir, "0", "U"), "w") as f:
        f.write(u_content)

    p_content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    inlet
    {
        type    zeroGradient;
    }
    outlet
    {
        type    fixedValue;
        value   uniform 0;
    }
    wall
    {
        type    zeroGradient;
    }
}
"""
    with open(os.path.join(temp_dir, "0", "p"), "w") as f:
        f.write(p_content)

    control_dict = """FoamFile
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
endTime         100;
deltaT          1;
writeControl    timeStep;
writeInterval   50;
"""
    with open(os.path.join(temp_dir, "system", "controlDict"), "w") as f:
        f.write(control_dict)

    return temp_dir


@pytest.fixture
def temp_msh_file(temp_dir):
    """Create a temporary MSH file with zone sections."""
    msh_content = '''(0 "Zone Sections")
(39 (1 fluid fluid)())
(39 (2 wall WALL)())
(39 (3 inlet INLET)())
(39 (4 outlet OUTLET)())
(39 (5 interior interior_FLUID)())
'''
    msh_path = os.path.join(temp_dir, "test_mesh.msh")
    with open(msh_path, "w") as f:
        f.write(msh_content)
    return msh_path


# =============================================================================
# Fixtures: Config
# =============================================================================

@pytest.fixture
def mock_path_config():
    """Create a mock PathConfig for testing."""
    from config import PathConfig
    return PathConfig(
        root_dir="/tmp/test_root",
        src_dir="/tmp/test_root/src2",
        output_dir="/tmp/test_root/run_chatcfd",
        database_dir="/tmp/test_root/database_OFv24",
        temp_dir="/tmp/test_root/temp",
        runs_dir="/tmp/test_root/runs",
        inputs_dir="/tmp/test_root/inputs",
        openfoam_tutorials_dir="/usr/lib/openfoam/tutorials",
    )


@pytest.fixture
def mock_llm_config():
    """Create a mock LLMConfig for testing."""
    from config import LLMConfig
    return LLMConfig(
        base_url="https://test.api.com",
        instruct_model="test-instruct",
        reasoning_model="test-reasoning",
        instruct_temperature=0.7,
        reasoning_temperature=0.9,
    )


@pytest.fixture
def mock_openfoam_constants():
    """Create OpenFOAMConstants for testing."""
    from config import OpenFOAMConstants
    return OpenFOAMConstants()


@pytest.fixture
def mock_app_config(mock_path_config, mock_llm_config, mock_openfoam_constants):
    """Create a mock AppConfig for testing."""
    from config import AppConfig
    return AppConfig(
        paths=mock_path_config,
        llm=mock_llm_config,
        openfoam=mock_openfoam_constants,
    )


@pytest.fixture
def real_path_config():
    """Load the real PathConfig (requires database to exist)."""
    if not DATABASE_EXISTS:
        pytest.skip("Database not found")
    from config import PathConfig
    return PathConfig.from_defaults(ROOT_DIR)


# =============================================================================
# Fixtures: State
# =============================================================================

@pytest.fixture
def empty_state():
    """Create an empty CaseState."""
    from state import CaseState
    return CaseState()


@pytest.fixture
def basic_state():
    """Create a CaseState with basic configuration."""
    from state import CaseState
    return CaseState(
        case_name="test_case",
        solver="simpleFoam",
        turbulence_model="kOmegaSST",
        description="Test case for unit testing",
        grid_boundaries=["inlet", "outlet", "wall"],
    )


@pytest.fixture
def state_with_errors():
    """Create a CaseState with error history."""
    from state import CaseState
    return CaseState(
        case_name="error_case",
        solver="simpleFoam",
        error_history=[
            "FOAM FATAL ERROR: Cannot find file",
            "FOAM FATAL ERROR: Cannot find file",
        ],
        current_error="FOAM FATAL ERROR: Cannot find file",
        attempt_count=2,
    )


@pytest.fixture
def state_with_files(temp_openfoam_case):
    """Create a CaseState with output path and files."""
    from state import CaseState
    return CaseState(
        case_name="file_case",
        solver="simpleFoam",
        output_path=temp_openfoam_case,
        file_structure=["0/U", "0/p", "system/controlDict"],
        generated_files={
            "0/U": "velocity file content",
            "0/p": "pressure file content",
        },
    )


# =============================================================================
# Fixtures: Database
# =============================================================================

@pytest.fixture
def empty_database():
    """Create an empty ReferenceDatabase."""
    from database import ReferenceDatabase
    return ReferenceDatabase({})


@pytest.fixture
def mock_database():
    """Create a mock ReferenceDatabase with sample data."""
    from database import ReferenceDatabase, CaseReference

    cases = {
        "incompressible/simpleFoam/pitzDaily": CaseReference(
            path="incompressible/simpleFoam/pitzDaily",
            solver="simpleFoam",
            turbulence_type="RAS",
            turbulence_model="kEpsilon",
            other_physical_model=None,
            configuration_files={
                "0/U": """FoamFile { version 2.0; } dimensions [0 1 -1 0 0 0 0]; internalField uniform (0 0 0);""",
                "0/p": """FoamFile { version 2.0; } dimensions [0 2 -2 0 0 0 0]; internalField uniform 0;""",
                "0/k": """FoamFile { version 2.0; } dimensions [0 2 -2 0 0 0 0]; internalField uniform 0.1;""",
                "0/epsilon": """FoamFile { version 2.0; } dimensions [0 2 -3 0 0 0 0]; internalField uniform 0.1;""",
                "system/fvSolution": """FoamFile { version 2.0; } solvers { p { solver GAMG; } }""",
            },
            required_fields=["U", "p", "k", "epsilon"],
            is_single_phase=True,
            is_particle_flow=False,
            is_reacting_flow=False,
            boundary_types=["inlet", "outlet", "wall"],
        ),
        "incompressible/simpleFoam/motorBike": CaseReference(
            path="incompressible/simpleFoam/motorBike",
            solver="simpleFoam",
            turbulence_type="RAS",
            turbulence_model="kOmegaSST",
            other_physical_model=None,
            configuration_files={
                "0/U": """FoamFile { version 2.0; } dimensions [0 1 -1 0 0 0 0]; internalField uniform (20 0 0);""",
                "0/p": """FoamFile { version 2.0; } dimensions [0 2 -2 0 0 0 0]; internalField uniform 0;""",
                "0/k": """FoamFile { version 2.0; } dimensions [0 2 -2 0 0 0 0]; internalField uniform 0.24;""",
                "0/omega": """FoamFile { version 2.0; } dimensions [0 0 -1 0 0 0 0]; internalField uniform 1.78;""",
            },
            required_fields=["U", "p", "k", "omega"],
            is_single_phase=True,
            is_particle_flow=False,
            is_reacting_flow=False,
            boundary_types=["inlet", "outlet", "wall", "symmetry"],
        ),
    }

    return ReferenceDatabase(cases)


@pytest.fixture
def real_database():
    """Load the real database (requires database files to exist)."""
    if not DATABASE_EXISTS:
        pytest.skip("Database not found")
    from database import ReferenceDatabase
    return ReferenceDatabase.load(DATABASE_DIR)


# =============================================================================
# Fixtures: LLM
# =============================================================================

@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    mock_client = Mock()
    mock_completion = Mock()
    mock_completion.choices = [Mock(message=Mock(content="Mock response"))]
    mock_completion.usage = Mock(prompt_tokens=10, completion_tokens=5)
    mock_client.chat.completions.create.return_value = mock_completion
    return mock_client


@pytest.fixture
def mock_llm_client(mock_llm_config, mock_openai_client):
    """Create a mock LLMClient."""
    from unittest.mock import patch

    with patch('llm.OpenAI', return_value=mock_openai_client):
        from llm import LLMClient
        client = LLMClient(mock_llm_config)
        return client


# =============================================================================
# Fixtures: Dependencies
# =============================================================================

@pytest.fixture
def mock_dependencies(mock_app_config, mock_database):
    """Create mock Dependencies for step functions."""
    from steps import Dependencies

    mock_llm = Mock()
    mock_llm.ask_instruct.return_value = "Mock LLM response"
    mock_llm.ask_reasoning.return_value = "Mock reasoning response"

    return Dependencies(
        config=mock_app_config,
        llm=mock_llm,
        database=mock_database,
    )


@pytest.fixture
def real_dependencies():
    """Create real Dependencies (requires database and config)."""
    if not DATABASE_EXISTS:
        pytest.skip("Database not found")

    from config import load_config
    from database import ReferenceDatabase
    from steps import Dependencies

    config = load_config(CONFIG_PATH) if CONFIG_EXISTS else None
    if config is None:
        pytest.skip("Config not found")

    # Mock LLM since we don't want real API calls in tests
    mock_llm = Mock()
    mock_llm.ask_instruct.return_value = "system/fvSolution"
    mock_llm.ask_reasoning.return_value = """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object fvSolution;
}
"""

    return Dependencies(
        config=config,
        llm=mock_llm,
        database=ReferenceDatabase.load(DATABASE_DIR),
    )


# =============================================================================
# Helper Functions
# =============================================================================

def mock_pyfoam_modules():
    """Mock PyFoam modules when not available."""
    if not PYFOAM_AVAILABLE:
        sys.modules['PyFoam'] = MagicMock()
        sys.modules['PyFoam.RunDictionary'] = MagicMock()
        sys.modules['PyFoam.RunDictionary.ParsedParameterFile'] = MagicMock()
        sys.modules['PyFoam.Basics'] = MagicMock()
        sys.modules['PyFoam.Basics.DataStructures'] = MagicMock()


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
