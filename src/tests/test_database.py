"""
Tests for database.py module.

Tests:
- CaseReference dataclass
- ReferenceDatabase class
- reference_files_to_json function
- load_dimensions_dict function
- load_boundary_entries function
- _build_indices defensive checks
"""

import logging
import os
import sys
import json
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import (
    CaseReference,
    ReferenceDatabase,
    reference_files_to_json,
    load_dimensions_dict,
    load_boundary_entries,
)

# Path constants - defined here to avoid conftest import issues
SRC2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(SRC2_DIR)
DATABASE_DIR = os.path.join(ROOT_DIR, "database_OFv24")
DATABASE_EXISTS = os.path.exists(os.path.join(DATABASE_DIR, "processed_merged_OF_cases.json"))


# =============================================================================
# Test CaseReference
# =============================================================================

class TestCaseReference:
    """Tests for CaseReference dataclass."""

    def test_create_case_reference(self):
        """Can create CaseReference with all fields."""
        ref = CaseReference(
            path="incompressible/simpleFoam/pitzDaily",
            solver="simpleFoam",
            turbulence_type="RAS",
            turbulence_model="kEpsilon",
            other_physical_model=None,
            configuration_files={
                "0/U": "U content",
                "0/p": "p content",
            },
            required_fields=["U", "p", "k", "epsilon"],
            is_single_phase=True,
            is_particle_flow=False,
            is_reacting_flow=False,
            boundary_types=["inlet", "outlet", "wall"],
        )

        assert ref.path == "incompressible/simpleFoam/pitzDaily"
        assert ref.solver == "simpleFoam"
        assert ref.turbulence_model == "kEpsilon"
        assert len(ref.configuration_files) == 2
        assert ref.is_single_phase is True

    def test_case_reference_is_frozen(self):
        """CaseReference is immutable."""
        ref = CaseReference(
            path="test/path",
            solver="simpleFoam",
            turbulence_type="RAS",
            turbulence_model="kEpsilon",
            other_physical_model=None,
            configuration_files={},
            required_fields=[],
            is_single_phase=True,
            is_particle_flow=False,
            is_reacting_flow=False,
            boundary_types=[],
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            ref.solver = "pimpleFoam"

    def test_case_reference_with_optional_fields(self):
        """CaseReference handles optional fields."""
        ref = CaseReference(
            path="test/path",
            solver="simpleFoam",
            turbulence_type=None,
            turbulence_model=None,
            other_physical_model=["multiphase"],
            configuration_files={},
            required_fields=[],
            is_single_phase=True,
            is_particle_flow=False,
            is_reacting_flow=False,
            boundary_types=[],
        )

        assert ref.turbulence_type is None
        assert ref.turbulence_model is None
        assert ref.other_physical_model == ["multiphase"]


# =============================================================================
# Test ReferenceDatabase
# =============================================================================

class TestReferenceDatabaseBasic:
    """Basic tests for ReferenceDatabase."""

    def test_create_empty_database(self):
        """Can create empty database."""
        db = ReferenceDatabase({})

        assert len(db) == 0

    def test_create_with_cases(self):
        """Can create database with cases."""
        cases = {
            "path1": CaseReference(
                path="path1",
                solver="simpleFoam",
                turbulence_type="RAS",
                turbulence_model="kEpsilon",
                other_physical_model=None,
                configuration_files={"0/U": "content"},
                required_fields=["U"],
                is_single_phase=True,
                is_particle_flow=False,
                is_reacting_flow=False,
                boundary_types=[],
            ),
            "path2": CaseReference(
                path="path2",
                solver="pimpleFoam",
                turbulence_type="RAS",
                turbulence_model="kOmegaSST",
                other_physical_model=None,
                configuration_files={"0/U": "content"},
                required_fields=["U"],
                is_single_phase=True,
                is_particle_flow=False,
                is_reacting_flow=False,
                boundary_types=[],
            ),
        }

        db = ReferenceDatabase(cases)

        assert len(db) == 2

    def test_iterate_cases(self):
        """Can iterate over cases."""
        cases = {
            "path1": CaseReference(
                path="path1",
                solver="simpleFoam",
                turbulence_type="RAS",
                turbulence_model="kEpsilon",
                other_physical_model=None,
                configuration_files={},
                required_fields=[],
                is_single_phase=True,
                is_particle_flow=False,
                is_reacting_flow=False,
                boundary_types=[],
            ),
        }

        db = ReferenceDatabase(cases)

        for case in db:
            assert isinstance(case, CaseReference)

    def test_get_case(self):
        """Can get case by path."""
        cases = {
            "path1": CaseReference(
                path="path1",
                solver="simpleFoam",
                turbulence_type="RAS",
                turbulence_model="kEpsilon",
                other_physical_model=None,
                configuration_files={},
                required_fields=[],
                is_single_phase=True,
                is_particle_flow=False,
                is_reacting_flow=False,
                boundary_types=[],
            ),
        }

        db = ReferenceDatabase(cases)

        case = db.get_case("path1")
        assert case is not None
        assert case.solver == "simpleFoam"

    def test_get_case_missing(self):
        """get_case returns None for missing path."""
        db = ReferenceDatabase({})

        case = db.get_case("nonexistent")
        assert case is None


class TestBuildIndicesDefensiveChecks:
    """Tests for _build_indices defensive checks."""

    def test_handles_invalid_case_values(self, caplog):
        """_build_indices warns about non-CaseReference values."""
        cases = {
            "valid": CaseReference(
                path="valid",
                solver="simpleFoam",
                turbulence_type=None,
                turbulence_model=None,
                other_physical_model=None,
                configuration_files={},
                required_fields=[],
                is_single_phase=True,
                is_particle_flow=False,
                is_reacting_flow=False,
                boundary_types=[],
            ),
        }
        db = ReferenceDatabase(cases)

        # Manually inject an invalid entry
        db._cases["invalid"] = "not a CaseReference"
        db._by_solver = {}

        with caplog.at_level(logging.WARNING, logger="chatcfd.database"):
            db._build_indices()

        assert "Skipping invalid case" in caplog.text
        # Valid case should still be indexed
        assert "simpleFoam" in db._by_solver

    def test_handles_non_dict_cases(self, caplog):
        """_build_indices handles _cases being non-dict."""
        db = ReferenceDatabase({})

        # Manually set _cases to wrong type
        db._cases = "not a dict"
        db._by_solver = {}

        with caplog.at_level(logging.ERROR, logger="chatcfd.database"):
            db._build_indices()

        assert "_cases is not a dict" in caplog.text
        assert db._cases == {}

    def test_load_logs_warning_for_missing_file(self, caplog):
        """load() logs warning when database file doesn't exist."""
        with caplog.at_level(logging.WARNING, logger="chatcfd.database"):
            db = ReferenceDatabase.load("/nonexistent/path/to/database")

        assert "Database file not found" in caplog.text
        assert len(db) == 0


class TestReferenceDatabaseLoad:
    """Tests for ReferenceDatabase.load method."""

    def test_load_nonexistent_path(self):
        """load returns empty database for nonexistent path."""
        db = ReferenceDatabase.load("/nonexistent/path")

        assert len(db) == 0

    def test_load_from_json(self, tmp_path):
        """Can load database from JSON file."""
        data = {
            "incompressible/simpleFoam/pitzDaily": {
                "solver": "simpleFoam",
                "turbulence_type": "RAS",
                "turbulence_model": "kEpsilon",
                "other_physical_model": None,
                "configuration_files": {
                    "0/U": "velocity content",
                    "0/p": "pressure content",
                },
                "required_field": ["U", "p"],
                "singlePhase": True,
                "particle_flow": False,
                "reacting_flow": False,
                "boundary_type": ["inlet", "outlet"],
            },
        }

        json_path = tmp_path / "processed_merged_OF_cases.json"
        with open(json_path, "w") as f:
            json.dump(data, f)

        db = ReferenceDatabase.load(str(tmp_path))

        assert len(db) == 1
        case = db.get_case("incompressible/simpleFoam/pitzDaily")
        assert case is not None
        assert case.solver == "simpleFoam"
        assert case.turbulence_model == "kEpsilon"

    @pytest.mark.skipif(not DATABASE_EXISTS, reason="Real database not found")
    def test_load_real_database(self):
        """Can load the real OpenFOAM tutorial database."""
        db = ReferenceDatabase.load(DATABASE_DIR)

        # Should have many cases
        assert len(db) > 100, f"Expected 100+ cases, got {len(db)}"


class TestReferenceDatabaseQueries:
    """Tests for ReferenceDatabase query methods."""

    @pytest.fixture
    def sample_database(self):
        """Create a sample database for testing."""
        cases = {
            "incompressible/simpleFoam/pitzDaily": CaseReference(
                path="incompressible/simpleFoam/pitzDaily",
                solver="simpleFoam",
                turbulence_type="RAS",
                turbulence_model="kEpsilon",
                other_physical_model=None,
                configuration_files={
                    "0/U": "pitzDaily U content",
                    "0/p": "pitzDaily p content",
                    "0/k": "pitzDaily k content",
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
                    "0/U": "motorBike U content",
                    "0/p": "motorBike p content",
                    "0/k": "motorBike k content",
                    "0/omega": "motorBike omega content",
                },
                required_fields=["U", "p", "k", "omega"],
                is_single_phase=True,
                is_particle_flow=False,
                is_reacting_flow=False,
                boundary_types=["inlet", "outlet", "wall", "symmetry"],
            ),
            "compressible/rhoSimpleFoam/aerofoilNACA0012": CaseReference(
                path="compressible/rhoSimpleFoam/aerofoilNACA0012",
                solver="rhoSimpleFoam",
                turbulence_type="RAS",
                turbulence_model="kOmegaSST",
                other_physical_model=None,
                configuration_files={
                    "0/U": "aerofoil U content",
                    "0/p": "aerofoil p content",
                },
                required_fields=["U", "p", "T"],
                is_single_phase=True,
                is_particle_flow=False,
                is_reacting_flow=False,
                boundary_types=["inlet", "outlet", "wall"],
            ),
        }
        return ReferenceDatabase(cases)

    def test_get_all_solvers(self, sample_database):
        """get_all_solvers returns all unique solvers."""
        solvers = sample_database.get_all_solvers()

        assert "simpleFoam" in solvers
        assert "rhoSimpleFoam" in solvers
        assert len(solvers) == 2

    def test_get_all_turbulence_models(self, sample_database):
        """get_all_turbulence_models returns all unique models."""
        models = sample_database.get_all_turbulence_models()

        assert "kEpsilon" in models
        assert "kOmegaSST" in models
        assert len(models) == 2

    def test_get_cases_by_solver(self, sample_database):
        """get_cases_by_solver returns matching cases."""
        cases = sample_database.get_cases_by_solver("simpleFoam")

        assert len(cases) == 2
        for case in cases:
            assert case.solver == "simpleFoam"

    def test_get_cases_by_solver_no_match(self, sample_database):
        """get_cases_by_solver returns empty for unknown solver."""
        cases = sample_database.get_cases_by_solver("nonexistentFoam")

        assert len(cases) == 0


class TestFindReferenceFiles:
    """Tests for ReferenceDatabase.find_reference_files method."""

    @pytest.fixture
    def sample_database(self):
        """Create a sample database for testing."""
        cases = {
            "incompressible/simpleFoam/pitzDaily": CaseReference(
                path="incompressible/simpleFoam/pitzDaily",
                solver="simpleFoam",
                turbulence_type="RAS",
                turbulence_model="kEpsilon",
                other_physical_model=None,
                configuration_files={
                    "0/U": "pitzDaily U content",
                    "0/p": "pitzDaily p content",
                },
                required_fields=["U", "p"],
                is_single_phase=True,
                is_particle_flow=False,
                is_reacting_flow=False,
                boundary_types=[],
            ),
            "incompressible/simpleFoam/motorBike": CaseReference(
                path="incompressible/simpleFoam/motorBike",
                solver="simpleFoam",
                turbulence_type="RAS",
                turbulence_model="kOmegaSST",
                other_physical_model=None,
                configuration_files={
                    "0/U": "motorBike U content",
                    "0/p": "motorBike p content",
                },
                required_fields=["U", "p"],
                is_single_phase=True,
                is_particle_flow=False,
                is_reacting_flow=False,
                boundary_types=[],
            ),
        }
        return ReferenceDatabase(cases)

    def test_find_reference_files_by_solver(self, sample_database):
        """find_reference_files finds files by solver."""
        refs = sample_database.find_reference_files(
            target_file="0/U",
            solver="simpleFoam",
        )

        assert len(refs) > 0
        # Should have content from one of the simpleFoam cases
        assert any("U content" in v for v in refs.values())

    def test_find_reference_files_by_solver_and_turbulence(self, sample_database):
        """find_reference_files filters by turbulence model."""
        refs = sample_database.find_reference_files(
            target_file="0/U",
            solver="simpleFoam",
            turbulence_model="kOmegaSST",
        )

        # Should only find motorBike case
        assert len(refs) > 0
        for content in refs.values():
            assert "motorBike" in content

    def test_find_reference_files_max_results(self, sample_database):
        """find_reference_files respects max_results."""
        refs = sample_database.find_reference_files(
            target_file="0/U",
            solver="simpleFoam",
            max_results=1,
        )

        assert len(refs) <= 1

    def test_find_reference_files_no_match(self, sample_database):
        """find_reference_files returns empty for no match."""
        refs = sample_database.find_reference_files(
            target_file="0/nonexistent",
            solver="simpleFoam",
        )

        assert len(refs) == 0

    def test_find_reference_files_respects_content_length(self, sample_database):
        """find_reference_files filters by content length."""
        refs = sample_database.find_reference_files(
            target_file="0/U",
            solver="simpleFoam",
            max_content_length=5,  # Very small - should exclude all
        )

        # All content is longer than 5 chars, so should be empty
        assert len(refs) == 0

    @pytest.mark.skipif(not DATABASE_EXISTS, reason="Real database not found")
    def test_find_reference_files_real_database(self):
        """find_reference_files works with real database."""
        db = ReferenceDatabase.load(DATABASE_DIR)

        refs = db.find_reference_files(
            target_file="0/U",
            solver="simpleFoam",
        )

        assert len(refs) > 0
        for content in refs.values():
            assert "FoamFile" in content or "boundaryField" in content


# =============================================================================
# Test Helper Functions
# =============================================================================

class TestReferenceFilesToJson:
    """Tests for reference_files_to_json function."""

    def test_empty_dict(self):
        """Handles empty dict."""
        result = reference_files_to_json({})
        assert result == "{}"

    def test_single_entry(self):
        """Converts single entry to JSON."""
        refs = {"sample_0": "file content"}
        result = reference_files_to_json(refs)

        parsed = json.loads(result)
        assert parsed["sample_0"] == "file content"

    def test_multiple_entries(self):
        """Converts multiple entries to JSON."""
        refs = {
            "sample_0": "content 1",
            "sample_1": "content 2",
        }
        result = reference_files_to_json(refs)

        parsed = json.loads(result)
        assert len(parsed) == 2

    def test_preserves_special_chars(self):
        """Preserves special characters in content."""
        refs = {"sample_0": "Line 1\nLine 2\tTabbed"}
        result = reference_files_to_json(refs)

        parsed = json.loads(result)
        assert "\n" in parsed["sample_0"]
        assert "\t" in parsed["sample_0"]


class TestLoadDimensionsDict:
    """Tests for load_dimensions_dict function."""

    def test_nonexistent_path(self):
        """Returns empty dict for nonexistent path."""
        result = load_dimensions_dict("/nonexistent/path")
        assert result == {}

    def test_load_from_file(self, tmp_path):
        """Loads dimensions from JSON file."""
        dims = {
            "0/U": "[0 1 -1 0 0 0 0]",
            "0/p": "[0 2 -2 0 0 0 0]",
            "0/k": "[0 2 -2 0 0 0 0]",
        }

        with open(tmp_path / "OF_case_dimensions.json", "w") as f:
            json.dump(dims, f)

        result = load_dimensions_dict(str(tmp_path))

        assert result["0/U"] == "[0 1 -1 0 0 0 0]"
        assert result["0/p"] == "[0 2 -2 0 0 0 0]"

    @pytest.mark.skipif(not DATABASE_EXISTS, reason="Real database not found")
    def test_load_real_dimensions(self):
        """Loads real dimensions dictionary."""
        result = load_dimensions_dict(DATABASE_DIR)

        # Should have common fields
        assert "0/U" in result or "0/p" in result


class TestLoadBoundaryEntries:
    """Tests for load_boundary_entries function."""

    def test_nonexistent_path(self):
        """Returns empty list for nonexistent path."""
        result = load_boundary_entries("/nonexistent/path")
        assert result == []

    def test_load_from_file(self, tmp_path):
        """Loads boundary entries from JSON file."""
        entries = [
            {"type": "fixedValue", "required": ["value"]},
            {"type": "zeroGradient", "required": []},
        ]

        with open(tmp_path / "OF_bc_entry.json", "w") as f:
            json.dump(entries, f)

        result = load_boundary_entries(str(tmp_path))

        assert len(result) == 2
        assert result[0]["type"] == "fixedValue"


# =============================================================================
# Integration Tests
# =============================================================================

class TestDatabaseIntegration:
    """Integration tests for database module."""

    @pytest.mark.skipif(not DATABASE_EXISTS, reason="Real database not found")
    def test_full_database_workflow(self):
        """Test complete database workflow."""
        # Load database
        db = ReferenceDatabase.load(DATABASE_DIR)
        assert len(db) > 0

        # Get all solvers
        solvers = db.get_all_solvers()
        assert "simpleFoam" in solvers

        # Find reference files
        refs = db.find_reference_files(
            target_file="0/U",
            solver="simpleFoam",
            turbulence_model="kEpsilon",
        )

        # Convert to JSON for prompts
        json_str = reference_files_to_json(refs)
        parsed = json.loads(json_str)
        assert len(parsed) > 0

    @pytest.mark.skipif(not DATABASE_EXISTS, reason="Real database not found")
    def test_database_has_common_solvers(self):
        """Real database has common solvers."""
        db = ReferenceDatabase.load(DATABASE_DIR)
        solvers = db.get_all_solvers()

        expected_solvers = ["simpleFoam", "pimpleFoam", "interFoam"]
        for solver in expected_solvers:
            assert solver in solvers, f"Missing solver: {solver}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
