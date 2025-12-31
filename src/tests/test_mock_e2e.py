"""
Mock E2E test using recorded LLM responses.

This test replays a successful correction loop run without calling real LLMs,
making it fast and deterministic. It uses the SAME assertions as the original
e2e tests (test_e2e_workflow.py, test_correction_loop.py) plus additional
content quality checks.

The recorded responses come from a real run that:
- Generated OpenFOAM files for NACA 0012 airfoil case
- Fixed errors through correction iterations
- Successfully ran simpleFoam simulation

Usage:
    python -m pytest tests/test_mock_e2e.py -v
    python -m pytest tests/test_mock_e2e.py::test_mock_workflow_with_replay -v -s
"""

import json
import os
import sys
from unittest.mock import Mock, MagicMock
from difflib import SequenceMatcher

import pytest

# Add src2 to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state import CaseState
from steps import Dependencies, determine_file_structure, generate_files, write_files

# Paths
SRC2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES_DIR = os.path.join(SRC2_DIR, "tests", "fixtures")
RECORDED_CALLS_PATH = os.path.join(FIXTURES_DIR, "recorded_llm_calls.json")


def validate_openfoam_file(content: str, file_path: str) -> list[str]:
    """
    Validate OpenFOAM file content.

    Returns list of validation errors (empty if valid).
    Same checks that would cause OpenFOAM to fail at runtime.
    """
    errors = []

    # All OpenFOAM files should have FoamFile header
    if "FoamFile" not in content:
        errors.append(f"{file_path}: missing FoamFile header")

    # Field files (0/*) should have dimensions and boundaryField
    if file_path.startswith("0/"):
        if "dimensions" not in content:
            errors.append(f"{file_path}: missing dimensions")
        if "boundaryField" not in content:
            errors.append(f"{file_path}: missing boundaryField")
        if "internalField" not in content:
            errors.append(f"{file_path}: missing internalField")

    # System files checks
    if file_path == "system/controlDict":
        if "application" not in content:
            errors.append(f"{file_path}: missing application")
        if "endTime" not in content and "stopAt" not in content:
            errors.append(f"{file_path}: missing endTime/stopAt")

    if file_path == "system/fvSchemes":
        if "ddtSchemes" not in content:
            errors.append(f"{file_path}: missing ddtSchemes")
        if "gradSchemes" not in content:
            errors.append(f"{file_path}: missing gradSchemes")

    if file_path == "system/fvSolution":
        if "solvers" not in content:
            errors.append(f"{file_path}: missing solvers block")

    # Constant files checks
    if file_path == "constant/turbulenceProperties":
        if "simulationType" not in content and "RAS" not in content:
            errors.append(f"{file_path}: missing simulationType/RAS")

    return errors


def load_recorded_calls():
    """Load recorded LLM calls from fixture."""
    if not os.path.exists(RECORDED_CALLS_PATH):
        pytest.skip(f"Recorded calls not found: {RECORDED_CALLS_PATH}")

    with open(RECORDED_CALLS_PATH) as f:
        return json.load(f)


def prompt_similarity(prompt1: str, prompt2: str) -> float:
    """Calculate similarity between two prompts (0-1)."""
    # Compare first 500 chars to avoid reference file noise
    return SequenceMatcher(None, prompt1[:500], prompt2[:500]).ratio()


def find_matching_response(prompt: str, recorded_calls: list, model_type: str) -> str:
    """Find the best matching recorded response for a prompt."""
    candidates = [c for c in recorded_calls if c['model_type'] == model_type]

    if not candidates:
        raise ValueError(f"No recorded {model_type} calls available")

    # Find best match by similarity
    best_match = None
    best_score = 0

    for call in candidates:
        score = prompt_similarity(prompt, call['prompt'])
        if score > best_score:
            best_score = score
            best_match = call

    if best_score < 0.5:
        # Show what we were looking for
        raise ValueError(
            f"No matching {model_type} response found (best score: {best_score:.2f})\n"
            f"Prompt preview: {prompt[:200]}..."
        )

    return best_match['response']


def create_replay_mock(recorded_calls: list):
    """Create a mock LLM client that replays recorded responses."""
    mock_llm = Mock()

    # Track which calls have been made for debugging
    mock_llm._call_count = {'reasoning': 0, 'instruct': 0}

    def replay_reasoning(prompt):
        mock_llm._call_count['reasoning'] += 1
        return find_matching_response(prompt, recorded_calls, 'reasoning')

    def replay_instruct(prompt):
        mock_llm._call_count['instruct'] += 1
        return find_matching_response(prompt, recorded_calls, 'instruct')

    mock_llm.ask_reasoning.side_effect = replay_reasoning
    mock_llm.ask_instruct.side_effect = replay_instruct

    # Mock stats
    mock_llm.stats = Mock()
    mock_llm.stats.instruct_calls = 0
    mock_llm.stats.reasoning_calls = 0

    return mock_llm


class TestMockE2E:
    """
    Mock E2E tests using recorded LLM responses.

    These tests mirror the assertions from:
    - test_e2e_workflow.py::TestE2EWorkflow::test_generate_files
    - test_correction_loop.py::TestCorrectionLoop::test_generate_and_save

    Plus additional content quality validation.
    """

    @pytest.fixture
    def recorded_calls(self):
        """Load recorded LLM calls."""
        return load_recorded_calls()

    @pytest.fixture
    def mock_llm(self, recorded_calls):
        """Create mock LLM with recorded responses."""
        return create_replay_mock(recorded_calls)

    @pytest.fixture
    def mock_deps(self, mock_llm, tmp_path):
        """Create mock dependencies."""
        from database import ReferenceDatabase
        from config import load_config

        config = load_config()
        database = ReferenceDatabase.load(config.paths.database_dir)

        return Dependencies(
            config=config,
            llm=mock_llm,
            database=database,
        )

    def test_file_generation_with_replay(self, mock_deps, tmp_path):
        """
        Test file generation using recorded LLM responses.

        Same assertions as test_e2e_workflow.py::test_generate_files:
        1. files_written == len(state.generated_files)
        2. All files exist on disk

        Plus content quality validation.
        """
        state = CaseState(
            case_name="naca0012_mock_test",
            solver="simpleFoam",
            turbulence_model="kOmegaSST",
            description="Incompressible turbulent flow over a NACA 0012 airfoil at 5 degrees angle of attack. Reynolds number ~1e6. Use standard wall functions.",
            output_path=str(tmp_path / "case"),
            grid_boundaries=["inlet", "outlet", "airfoil", "frontAndBackPlanes"],
        )

        # Step 1: Determine file structure
        update = determine_file_structure(state, mock_deps)
        state = state.apply_update(update)

        assert len(state.file_structure) > 0, "Should have files to generate"
        print(f"Files to generate: {state.file_structure}")

        # Step 2: Generate files using recorded LLM responses
        update = generate_files(state, mock_deps, verbose=True)
        state = state.apply_update(update)

        assert len(state.generated_files) > 0, "Should have generated files"
        print(f"Generated {len(state.generated_files)} files")

        # Step 3: Write files
        write_files(state, mock_deps)

        # === SAME ASSERTION AS ORIGINAL E2E TEST ===
        # From test_e2e_workflow.py line 279-280
        files_written = 0
        for rel_path in state.generated_files:
            full_path = tmp_path / "case" / rel_path
            if full_path.exists():
                files_written += 1
                size = os.path.getsize(full_path)
                print(f"  {rel_path}: {size} bytes")

        assert files_written == len(state.generated_files), \
            f"Only {files_written}/{len(state.generated_files)} files written"

        # === ADDITIONAL CONTENT QUALITY CHECKS ===
        all_validation_errors = []
        for rel_path, content in state.generated_files.items():
            errors = validate_openfoam_file(content, rel_path)
            all_validation_errors.extend(errors)

        if all_validation_errors:
            print("\nValidation errors:")
            for err in all_validation_errors:
                print(f"  - {err}")

        assert len(all_validation_errors) == 0, \
            f"Content validation failed:\n" + "\n".join(all_validation_errors)

        # Verify LLM was actually called (not just returning cached/empty)
        call_count = mock_deps.llm._call_count
        print(f"LLM calls: {call_count}")
        assert call_count['reasoning'] > 0, "Should have made reasoning calls"
        assert call_count['reasoning'] == len(state.file_structure), \
            f"Should call LLM once per file: {call_count['reasoning']} != {len(state.file_structure)}"


def test_mock_workflow_with_replay(tmp_path):
    """
    Full workflow test with recorded LLM responses.

    This is the main test that validates file generation using
    pre-recorded responses from a successful run.

    Mirrors assertions from:
    - test_e2e_workflow.py::test_generate_files
    - test_correction_loop.py::test_generate_and_save

    Plus content quality validation.
    """
    recorded_calls = load_recorded_calls()
    mock_llm = create_replay_mock(recorded_calls)

    from config import load_config
    from database import ReferenceDatabase

    config = load_config()
    database = ReferenceDatabase.load(config.paths.database_dir)

    deps = Dependencies(
        config=config,
        llm=mock_llm,
        database=database,
    )

    # Create initial state matching the recorded run
    state = CaseState(
        case_name="naca0012_replay_test",
        solver="simpleFoam",
        turbulence_model="kOmegaSST",
        description="Incompressible turbulent flow over a NACA 0012 airfoil at 5 degrees angle of attack. Reynolds number ~1e6. Use standard wall functions.",
        output_path=str(tmp_path / "case"),
        grid_boundaries=["inlet", "outlet", "airfoil", "frontAndBackPlanes"],
        max_attempts=10,
    )

    # Run file structure determination
    update = determine_file_structure(state, deps)
    state = state.apply_update(update)

    assert len(state.file_structure) > 0, "Should determine files to generate"

    print(f"\n{'='*60}")
    print("Mock E2E Test: File Generation with Recorded Responses")
    print(f"{'='*60}")
    print(f"Case: {state.case_name}")
    print(f"Solver: {state.solver}")
    print(f"Files to generate: {len(state.file_structure)}")

    # Generate files
    update = generate_files(state, deps, verbose=True)
    state = state.apply_update(update)

    assert len(state.generated_files) > 0, "Should generate files"

    # Write files
    write_files(state, deps)

    # === SAME ASSERTION AS ORIGINAL E2E TEST ===
    # From test_e2e_workflow.py line 279-280
    files_written = 0
    for rel_path in state.generated_files:
        full_path = tmp_path / "case" / rel_path
        if full_path.exists():
            files_written += 1
            size = os.path.getsize(full_path)
            print(f"  {rel_path}: {size} bytes")

    assert files_written == len(state.generated_files), \
        f"Only {files_written}/{len(state.generated_files)} files written"

    # === CONTENT QUALITY VALIDATION ===
    all_validation_errors = []
    for rel_path, content in state.generated_files.items():
        errors = validate_openfoam_file(content, rel_path)
        all_validation_errors.extend(errors)

    assert len(all_validation_errors) == 0, \
        f"Content validation failed:\n" + "\n".join(all_validation_errors)

    # === VERIFY EXPECTED FILES ===
    expected_files = ["0/U", "0/p", "0/k", "0/omega", "0/nut",
                      "system/controlDict", "system/fvSchemes", "system/fvSolution"]

    for expected in expected_files:
        if expected in state.file_structure:
            assert expected in state.generated_files, f"Missing from generated: {expected}"
            file_path = tmp_path / "case" / expected
            assert file_path.exists(), f"Not written to disk: {expected}"

    # === VERIFY LLM WAS CALLED ===
    assert mock_llm._call_count['reasoning'] > 0, "Should have made reasoning calls"
    assert mock_llm._call_count['reasoning'] == len(state.file_structure), \
        f"Should call LLM once per file"

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Generated files: {len(state.generated_files)}")
    print(f"Files written: {files_written}")
    print(f"LLM reasoning calls: {mock_llm._call_count['reasoning']}")
    print(f"Validation errors: {len(all_validation_errors)}")
    print("\nTest PASSED - all files generated and validated")


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        test_mock_workflow_with_replay(tmp)
