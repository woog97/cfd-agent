"""
Tests for config.py module.

Tests:
- PathConfig
- LLMConfig
- OpenFOAMConstants
- AppConfig
- load_config function
- ensure_directories function
"""

import os
import sys
import json
import tempfile
import pytest
from dataclasses import FrozenInstanceError
from unittest.mock import patch, Mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    PathConfig,
    LLMConfig,
    OpenFOAMConstants,
    AppConfig,
    load_config,
    ensure_directories,
    load_openfoam_environment,
)


# =============================================================================
# Test PathConfig
# =============================================================================

class TestPathConfig:
    """Tests for PathConfig dataclass."""

    def test_create_with_values(self):
        """Can create PathConfig with specific values."""
        config = PathConfig(
            root_dir="/test/root",
            src_dir="/test/root/src",
            output_dir="/test/root/output",
            database_dir="/test/root/database",
            temp_dir="/test/root/temp",
            runs_dir="/test/root/runs",
            inputs_dir="/test/root/inputs",
            openfoam_tutorials_dir="/opt/openfoam/tutorials",
        )

        assert config.root_dir == "/test/root"
        assert config.src_dir == "/test/root/src"
        assert config.output_dir == "/test/root/output"
        assert config.database_dir == "/test/root/database"
        assert config.runs_dir == "/test/root/runs"
        assert config.inputs_dir == "/test/root/inputs"

    def test_from_defaults(self):
        """from_defaults creates config with sensible defaults."""
        config = PathConfig.from_defaults()

        assert config.root_dir != ""
        assert config.src_dir != ""
        assert "run_chatcfd" in config.output_dir
        assert "database_OFv24" in config.database_dir

    def test_from_defaults_with_custom_root(self):
        """from_defaults respects custom root directory."""
        config = PathConfig.from_defaults(root_dir="/custom/root")

        assert config.root_dir == "/custom/root"
        assert "/custom/root" in config.output_dir
        assert "/custom/root" in config.database_dir

    def test_frozen_raises_frozeninstanceerror(self):
        """Frozen config raises FrozenInstanceError on mutation attempt."""
        config = PathConfig.from_defaults()

        with pytest.raises(FrozenInstanceError):
            config.root_dir = "/new/path"

    def test_all_paths_are_absolute(self):
        """All paths from defaults are absolute."""
        config = PathConfig.from_defaults()

        assert os.path.isabs(config.root_dir)
        assert os.path.isabs(config.src_dir)
        assert os.path.isabs(config.output_dir)
        assert os.path.isabs(config.database_dir)
        assert os.path.isabs(config.temp_dir)


# =============================================================================
# Test LLMConfig
# =============================================================================

class TestLLMConfig:
    """Tests for LLMConfig dataclass."""

    def test_create_with_values(self):
        """Can create LLMConfig with specific values."""
        config = LLMConfig(
            base_url="https://api.example.com",
            instruct_model="gpt-4",
            reasoning_model="gpt-4-reasoning",
            instruct_temperature=0.5,
            reasoning_temperature=0.8,
        )

        assert config.base_url == "https://api.example.com"
        assert config.instruct_model == "gpt-4"
        assert config.reasoning_model == "gpt-4-reasoning"
        assert config.instruct_temperature == 0.5
        assert config.reasoning_temperature == 0.8

    def test_default_temperatures(self):
        """Default temperatures are set correctly."""
        config = LLMConfig(
            base_url="https://api.example.com",
            instruct_model="test",
            reasoning_model="test",
        )

        assert config.instruct_temperature == 0.7
        assert config.reasoning_temperature == 0.9

    def test_from_env(self):
        """from_env reads from environment variables."""
        with patch.dict(os.environ, {
            "OPENAI_BASE_URL": "https://custom.api.com",
            "INSTRUCT_MODEL_NAME": "custom-instruct",
            "REASONING_MODEL_NAME": "custom-reasoning",
            "INSTRUCT_TEMPERATURE": "0.3",
            "REASONING_TEMPERATURE": "0.6",
        }):
            config = LLMConfig.from_env()

            assert config.base_url == "https://custom.api.com"
            assert config.instruct_model == "custom-instruct"
            assert config.reasoning_model == "custom-reasoning"
            assert config.instruct_temperature == 0.3
            assert config.reasoning_temperature == 0.6

    def test_from_env_with_defaults(self):
        """from_env uses defaults when env vars not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Clear specific vars
            for key in ["OPENAI_BASE_URL", "INSTRUCT_MODEL_NAME", "REASONING_MODEL_NAME"]:
                os.environ.pop(key, None)

            config = LLMConfig.from_env()

            assert "openrouter" in config.base_url or config.base_url != ""

    def test_frozen_raises_frozeninstanceerror(self):
        """Frozen config raises FrozenInstanceError on mutation."""
        config = LLMConfig(
            base_url="https://api.example.com",
            instruct_model="test",
            reasoning_model="test",
        )

        with pytest.raises(FrozenInstanceError):
            config.base_url = "https://other.api.com"


# =============================================================================
# Test OpenFOAMConstants
# =============================================================================

class TestOpenFOAMConstants:
    """Tests for OpenFOAMConstants dataclass."""

    def test_solver_keywords_populated(self):
        """Solver keywords are populated."""
        constants = OpenFOAMConstants()

        assert len(constants.solver_keywords) > 50
        assert "simpleFoam" in constants.solver_keywords
        assert "pimpleFoam" in constants.solver_keywords
        assert "interFoam" in constants.solver_keywords
        assert "rhoSimpleFoam" in constants.solver_keywords

    def test_steady_solvers_populated(self):
        """Steady-state solvers are defined."""
        constants = OpenFOAMConstants()

        assert "simpleFoam" in constants.steady_solvers
        assert "laplacianFoam" in constants.steady_solvers
        # Transient solver should NOT be in steady
        assert "pisoFoam" not in constants.steady_solvers

    def test_incompressible_solvers_populated(self):
        """Incompressible solvers are defined."""
        constants = OpenFOAMConstants()

        assert "simpleFoam" in constants.incompressible_solvers
        assert "pimpleFoam" in constants.incompressible_solvers
        assert "icoFoam" in constants.incompressible_solvers
        assert "pisoFoam" in constants.incompressible_solvers

    def test_compressible_solvers_populated(self):
        """Compressible solvers are defined."""
        constants = OpenFOAMConstants()

        assert "rhoSimpleFoam" in constants.compressible_solvers
        assert "rhoPimpleFoam" in constants.compressible_solvers
        assert "sonicFoam" in constants.compressible_solvers

    def test_turbulence_models_populated(self):
        """Turbulence models are populated."""
        constants = OpenFOAMConstants()

        assert "kOmegaSST" in constants.turbulence_model_keywords
        assert "kEpsilon" in constants.turbulence_model_keywords
        assert "SpalartAllmaras" in constants.turbulence_model_keywords
        assert "realizableKE" in constants.turbulence_model_keywords

    def test_turbulence_types_populated(self):
        """Turbulence types are populated."""
        constants = OpenFOAMConstants()

        assert "laminar" in constants.turbulence_type_keywords
        assert "RAS" in constants.turbulence_type_keywords
        assert "LES" in constants.turbulence_type_keywords

    def test_boundary_types_populated(self):
        """Boundary types are populated."""
        constants = OpenFOAMConstants()

        assert "fixedValue" in constants.boundary_type_keywords
        assert "zeroGradient" in constants.boundary_type_keywords
        assert "noSlip" in constants.boundary_type_keywords
        assert "inletOutlet" in constants.boundary_type_keywords
        assert "symmetry" in constants.boundary_type_keywords

    def test_solver_keywords_str(self):
        """solver_keywords_str returns comma-separated string."""
        constants = OpenFOAMConstants()
        result = constants.solver_keywords_str()

        assert isinstance(result, str)
        assert "simpleFoam" in result
        assert ", " in result

    def test_turbulence_model_str(self):
        """turbulence_model_str returns comma-separated string."""
        constants = OpenFOAMConstants()
        result = constants.turbulence_model_str()

        assert isinstance(result, str)
        assert "kOmegaSST" in result

    def test_sets_are_frozenset(self):
        """All keyword sets are frozensets (immutable)."""
        constants = OpenFOAMConstants()

        assert isinstance(constants.solver_keywords, frozenset)
        assert isinstance(constants.steady_solvers, frozenset)
        assert isinstance(constants.incompressible_solvers, frozenset)
        assert isinstance(constants.turbulence_model_keywords, frozenset)


# =============================================================================
# Test AppConfig
# =============================================================================

class TestAppConfig:
    """Tests for AppConfig dataclass."""

    def test_create_with_components(self):
        """Can create AppConfig with component configs."""
        paths = PathConfig.from_defaults()
        llm = LLMConfig(
            base_url="https://api.example.com",
            instruct_model="test",
            reasoning_model="test",
        )
        openfoam = OpenFOAMConstants()

        config = AppConfig(
            paths=paths,
            llm=llm,
            openfoam=openfoam,
        )

        assert config.paths == paths
        assert config.llm == llm
        assert config.openfoam == openfoam

    def test_default_values(self):
        """Default values are set correctly."""
        paths = PathConfig.from_defaults()
        llm = LLMConfig(
            base_url="https://api.example.com",
            instruct_model="test",
            reasoning_model="test",
        )
        openfoam = OpenFOAMConstants()

        config = AppConfig(paths=paths, llm=llm, openfoam=openfoam)

        assert config.pdf_chunk_distance == 1.5
        assert config.max_correction_attempts == 30
        assert config.reference_case_search_rounds == 10

    def test_frozen_raises_error(self):
        """Frozen config raises error on mutation."""
        paths = PathConfig.from_defaults()
        llm = LLMConfig(
            base_url="https://api.example.com",
            instruct_model="test",
            reasoning_model="test",
        )
        config = AppConfig(paths=paths, llm=llm, openfoam=OpenFOAMConstants())

        with pytest.raises(FrozenInstanceError):
            config.max_correction_attempts = 50


# =============================================================================
# Test load_config Function
# =============================================================================

class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_no_file(self):
        """load_config works even when config file doesn't exist."""
        config = load_config("/nonexistent/path/config.json")

        assert isinstance(config, AppConfig)
        assert isinstance(config.paths, PathConfig)
        assert isinstance(config.llm, LLMConfig)
        assert isinstance(config.openfoam, OpenFOAMConstants)

    def test_load_config_from_json(self, tmp_path):
        """load_config reads values from JSON file."""
        config_data = {
            "OPENAI_BASE_URL": "https://custom.api.com",
            "INSTRUCT_MODEL_NAME": "custom-model",
            "REASONING_MODEL_NAME": "custom-reasoning",
            "pdf_chunk_d": 2.0,
            "max_running_test_round": 50,
        }

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        # Clear env vars to ensure we read from file
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_BASE_URL", None)
            os.environ.pop("INSTRUCT_MODEL_NAME", None)
            os.environ.pop("REASONING_MODEL_NAME", None)

            config = load_config(str(config_path))

            # Check values from file were loaded
            assert config.pdf_chunk_distance == 2.0
            assert config.max_correction_attempts == 50

    def test_load_config_env_overrides_file(self, tmp_path):
        """Environment variables override config file values."""
        config_data = {
            "OPENAI_BASE_URL": "https://file.api.com",
        }

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        with patch.dict(os.environ, {
            "OPENAI_BASE_URL": "https://env.api.com",
        }):
            config = load_config(str(config_path))

            # Environment should override file
            assert config.llm.base_url == "https://env.api.com"

    def test_load_config_sets_openfoam_paths(self, tmp_path):
        """load_config handles OpenFOAM path configuration."""
        config_data = {
            "OpenFOAM_path": "/custom/openfoam",
            "OpenFOAM_tutorials_path": "/custom/openfoam/tutorials",
        }

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        config = load_config(str(config_path))

        assert config.paths.openfoam_tutorials_dir == "/custom/openfoam/tutorials"


# =============================================================================
# Test ensure_directories Function
# =============================================================================

class TestEnsureDirectories:
    """Tests for ensure_directories function."""

    def test_creates_missing_directories(self, tmp_path):
        """ensure_directories creates missing directories."""
        paths = PathConfig(
            root_dir=str(tmp_path),
            src_dir=str(tmp_path / "src"),
            output_dir=str(tmp_path / "output"),
            database_dir=str(tmp_path / "database"),
            temp_dir=str(tmp_path / "temp"),
            runs_dir=str(tmp_path / "runs"),
            inputs_dir=str(tmp_path / "inputs"),
            openfoam_tutorials_dir="/opt/openfoam/tutorials",
        )
        llm = LLMConfig(
            base_url="https://api.example.com",
            instruct_model="test",
            reasoning_model="test",
        )
        config = AppConfig(paths=paths, llm=llm, openfoam=OpenFOAMConstants())

        ensure_directories(config)

        assert os.path.exists(paths.output_dir)
        assert os.path.exists(paths.database_dir)
        assert os.path.exists(paths.temp_dir)
        assert os.path.exists(paths.runs_dir)

    def test_handles_existing_directories(self, tmp_path):
        """ensure_directories doesn't fail on existing directories."""
        # Create directories first
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        paths = PathConfig(
            root_dir=str(tmp_path),
            src_dir=str(tmp_path / "src"),
            output_dir=str(output_dir),
            database_dir=str(tmp_path / "database"),
            temp_dir=str(tmp_path / "temp"),
            runs_dir=str(tmp_path / "runs"),
            inputs_dir=str(tmp_path / "inputs"),
            openfoam_tutorials_dir="/opt/openfoam/tutorials",
        )
        llm = LLMConfig(
            base_url="https://api.example.com",
            instruct_model="test",
            reasoning_model="test",
        )
        config = AppConfig(paths=paths, llm=llm, openfoam=OpenFOAMConstants())

        # Should not raise - and should create all required directories
        ensure_directories(config)

        # Verify all required directories exist after the call
        assert output_dir.exists(), "Pre-existing output_dir should still exist"
        assert os.path.exists(paths.database_dir), "database_dir should be created"
        assert os.path.exists(paths.temp_dir), "temp_dir should be created"
        assert os.path.exists(paths.runs_dir), "runs_dir should be created"


# =============================================================================
# Test load_openfoam_environment Function
# =============================================================================

class TestLoadOpenfoamEnvironment:
    """Tests for load_openfoam_environment function."""

    def test_returns_false_on_missing_path(self):
        """Returns False when OpenFOAM path doesn't exist."""
        result = load_openfoam_environment("/nonexistent/openfoam/path")
        assert result is False

    def test_returns_bool(self):
        """Function returns a boolean value."""
        result = load_openfoam_environment()
        assert isinstance(result, bool)


# =============================================================================
# Integration Tests
# =============================================================================

class TestConfigIntegration:
    """Integration tests for config module."""

    def test_full_config_creation(self):
        """Can create complete config with all components."""
        config = load_config()

        # All components should be present
        assert config.paths is not None
        assert config.llm is not None
        assert config.openfoam is not None

        # Should be able to access nested values
        assert isinstance(config.paths.root_dir, str)
        assert isinstance(config.llm.base_url, str)
        assert len(config.openfoam.solver_keywords) > 0

    def test_config_is_fully_immutable(self):
        """All config components are immutable."""
        config = load_config()

        # Top level
        with pytest.raises(FrozenInstanceError):
            config.max_correction_attempts = 100

        # Nested PathConfig
        with pytest.raises(FrozenInstanceError):
            config.paths.root_dir = "/new/path"

        # Nested LLMConfig
        with pytest.raises(FrozenInstanceError):
            config.llm.base_url = "https://new.api.com"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
