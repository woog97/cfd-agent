"""
AppConfig: Immutable application configuration.

This module contains ONLY static configuration that doesn't change during execution.
All runtime state belongs in CaseState (see state.py).

Design principles:
- Frozen dataclasses prevent accidental mutation
- Configuration loaded once at startup
- Environment variables can override config file values
- No global mutable state
"""

import os
import json
import subprocess
from dataclasses import dataclass, field
from typing import FrozenSet


@dataclass(frozen=True)
class PathConfig:
    """Immutable path configuration."""
    root_dir: str
    src_dir: str
    output_dir: str
    database_dir: str
    temp_dir: str
    runs_dir: str
    inputs_dir: str
    openfoam_tutorials_dir: str

    @classmethod
    def from_defaults(cls, root_dir: str | None = None) -> "PathConfig":
        """Create PathConfig with default paths based on root directory."""
        src_dir = os.path.dirname(os.path.abspath(__file__))
        if root_dir is None:
            root_dir = os.path.dirname(src_dir)

        return cls(
            root_dir=root_dir,
            src_dir=src_dir,
            output_dir=os.path.join(root_dir, "run_chatcfd"),
            database_dir=os.path.join(root_dir, "database_OFv24"),
            temp_dir=os.path.join(root_dir, "temp"),
            runs_dir=os.path.join(root_dir, "runs"),
            inputs_dir=os.path.join(root_dir, "inputs"),
            openfoam_tutorials_dir="/usr/lib/openfoam/openfoam2406/tutorials",
        )


@dataclass(frozen=True)
class LLMConfig:
    """Immutable LLM configuration."""
    base_url: str
    instruct_model: str
    reasoning_model: str
    instruct_temperature: float = 0.7
    reasoning_temperature: float = 0.9

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Create LLMConfig from environment variables."""
        return cls(
            base_url=os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
            instruct_model=os.environ.get("INSTRUCT_MODEL_NAME", ""),
            reasoning_model=os.environ.get("REASONING_MODEL_NAME", ""),
            instruct_temperature=float(os.environ.get("INSTRUCT_TEMPERATURE", "0.7")),
            reasoning_temperature=float(os.environ.get("REASONING_TEMPERATURE", "0.9")),
        )


@dataclass(frozen=True)
class OpenFOAMConstants:
    """Immutable OpenFOAM-specific constants and keyword lists."""

    # Solver categories
    steady_solvers: FrozenSet[str] = field(default_factory=lambda: frozenset({
        "laplacianFoam", "overLaplacianDyMFoam", "potentialFoam", "overPotentialFoam",
        "scalarTransportFoam", "adjointShapeOptimizationFoam", "boundaryFoam",
        "simpleFoam", "overSimpleFoam", "porousSimpleFoam", "SRFSimpleFoam",
        "rhoSimpleFoam", "overRhoSimpleFoam", "rhoPorousSimpleFoam", "interFoam",
        "interMixingFoam", "interIsoFoam", "interPhaseChangeFoam", "MPPICInterFoam",
        "multiphaseInterFoam", "potentialFreeSurfaceFoam", "potentialFreeSurfaceDyMFoam",
        "buoyantBoussinesqSimpleFoam", "buoyantFoam", "buoyantSimpleFoam",
        "chtMultiRegionSimpleFoam", "thermoFoam", "icoUncoupledKinematicParcelFoam",
        "simpleReactingParcelFoam", "simpleCoalParcelFoam", "simpleSprayFoam",
        "uncoupledKinematicParcelFoam", "solidEquilibriumDisplacementFoam", "financialFoam"
    }))

    incompressible_solvers: FrozenSet[str] = field(default_factory=lambda: frozenset({
        "simpleFoam", "pimpleFoam", "pisoFoam", "icoFoam",
        "adjointOptimisationFoam", "adjointShapeOptimizationFoam", "boundaryFoam",
        "lumpedPointMotion", "nonNewtonianlcoFoam", "overPimpleDyMFoam",
        "overSimpleFoam", "porousSimpleFoam", "shallowWaterFoam",
        "SRFPimpleFoam", "SRFSimpleFoam", "potentialFoam"
    }))

    compressible_solvers: FrozenSet[str] = field(default_factory=lambda: frozenset({
        "rhoSimpleFoam", "rhoPimpleFoam", "rhoCentralFoam", "sonicFoam",
        "sonicLiquidFoam", "acousticFoam", "overRhoPimpleDyMFoam",
        "overRhoSimpleFoam", "rhoPimpleAdiabaticFoam", "rhoPorousSimpleFoam",
        "sonicDyMFoam", "reactingFoam", "interFoam", "compressibleInterFoam",
        "twoPhaseCompressibleFoam", "compressibleInterIsoFoam",
        "MPPICInterFoam", "overCompressibleInterDyMFoam"
    }))

    # All valid solvers
    solver_keywords: FrozenSet[str] = field(default_factory=lambda: frozenset({
        'buoyantBoussinesqSimpleFoam', 'overInterDyMFoam', 'kinematicParcelFoam',
        'buoyantSimpleFoam', 'reactingFoam', 'rhoReactingFoam',
        'compressibleInterDyMFoam', 'XiEngineFoam', 'pimpleFoam',
        'cavitatingFoam', 'adjointOptimisationFoam', 'overPimpleDyMFoam',
        'twoPhaseEulerFoam', 'interMixingFoam', 'compressibleInterFoam',
        'multiphaseInterFoam', 'porousSimpleFoam', 'overRhoSimpleFoam',
        'overLaplacianDyMFoam', 'interFoam', 'MPPICInterFoam', 'icoFoam',
        'overRhoPimpleDyMFoam', 'interPhaseChangeDyMFoam', 'sonicDyMFoam',
        'chtMultiRegionTwoPhaseEulerFoam', 'adjointShapeOptimizationFoam',
        'laplacianFoam', 'fireFoam', 'rhoSimpleFoam', 'overCompressibleInterDyMFoam',
        'shallowWaterFoam', 'simpleFoam', 'snappyHexMesh', 'sonicLiquidFoam',
        'sonicFoam', 'icoUncoupledKinematicParcelFoam', 'overSimpleFoam',
        'driftFluxFoam', 'interIsoFoam', 'uncoupledKinematicParcelDyMFoam',
        'sprayFoam', 'buoyantPimpleFoam', 'reactingHeterogenousParcelFoam',
        'chemFoam', 'acousticFoam', 'nonNewtonianIcoFoam', 'simpleReactingParcelFoam',
        'overInterPhaseChangeDyMFoam', 'boundaryFoam', 'compressibleMultiphaseInterFoam',
        'coalChemistryFoam', 'coldEngineFoam', 'rhoPimpleAdiabaticFoam',
        'MPPICDyMFoam', 'icoReactingMultiphaseInterFoam', 'SRFPimpleFoam',
        'overBuoyantPimpleDyMFoam', 'solidFoam', 'reactingParcelFoam',
        'icoUncoupledKinematicParcelDyMFoam', 'compressibleInterIsoFoam',
        'potentialFreeSurfaceFoam', 'chtMultiRegionFoam', 'XiDyMFoam',
        'multiphaseEulerFoam', 'overPotentialFoam', 'interCondensatingEvaporatingFoam',
        'potentialFreeSurfaceDyMFoam', 'subsetMesh', 'twoLiquidMixingFoam',
        'rhoPimpleFoam', 'MPPICFoam', 'pisoFoam', 'potentialFoam',
        'reactingTwoPhaseEulerFoam', 'reactingMultiphaseEulerFoam',
        'rhoPorousSimpleFoam', 'rhoCentralFoam', 'SRFSimpleFoam',
        'PDRFoam', 'interPhaseChangeFoam', 'buoyantBoussinesqPimpleFoam',
        'XiFoam', 'dnsFoam', 'chtMultiRegionSimpleFoam', 'buoyantFoam'
    }))

    turbulence_type_keywords: FrozenSet[str] = field(default_factory=lambda: frozenset({
        'laminar', 'RAS', 'LES', 'twoPhaseTransport'
    }))

    turbulence_model_keywords: FrozenSet[str] = field(default_factory=lambda: frozenset({
        'SpalartAllmarasDDES', 'Smagorinsky', 'SpalartAllmaras',
        'SpalartAllmarasIDDES', 'kOmegaSST', 'buoyantKEpsilon',
        'kkLOmega', 'RNGkEpsilon', 'WALE', 'LaunderSharmaKE',
        'realizableKE', 'PDRkEpsilon', 'dynamicKEqn',
        'kOmegaSSTLM', 'kEqn', 'kEpsilon', 'kEpsilonPhitF'
    }))

    boundary_type_keywords: FrozenSet[str] = field(default_factory=lambda: frozenset({
        'overset', 'zeroGradient', 'fixedValue', 'movingWallVelocity',
        'inletOutlet', 'symmetryPlane', 'symmetry', 'empty', 'uniformFixedValue',
        'noSlip', 'cyclicAMI', 'mappedField', 'calculated', 'waveTransmissive',
        'compressible::alphatWallFunction', 'supersonicFreestream',
        'epsilonWallFunction', 'kqRWallFunction', 'nutkWallFunction', 'slip',
        'turbulentIntensityKineticEnergyInlet', 'turbulentMixingLengthDissipationRateInlet',
        'flowRateInletVelocity', 'freestreamPressure', 'omegaWallFunction',
        'freestreamVelocity', 'pressureInletOutletVelocity', 'nutUWallFunction',
        'totalPressure', 'wedge', 'totalTemperature', 'turbulentInlet',
        'fixedMean', 'plenumPressure', 'pressureInletVelocity',
        'fluxCorrectedVelocity', 'mixed', 'uniformTotalPressure',
        'nutUSpaldingWallFunction', 'fixedFluxPressure', 'fixedGradient',
        'processor', 'prghPressure', 'prghTotalPressure', 'cyclic', 'freestream'
    }))

    thermodynamic_model_keywords: FrozenSet[str] = field(default_factory=lambda: frozenset({
        'hePsiThermo', 'heRhoThermo', 'heSolidThermo'
    }))

    # Helper methods for string formatting (for prompts)
    def solver_keywords_str(self) -> str:
        return ", ".join(sorted(self.solver_keywords))

    def turbulence_model_str(self) -> str:
        return ", ".join(sorted(self.turbulence_model_keywords))

    def turbulence_type_str(self) -> str:
        return ", ".join(sorted(self.turbulence_type_keywords))


@dataclass(frozen=True)
class DockerConfig:
    """Docker execution configuration."""
    enabled: bool = False
    image: str = "openfoam/openfoam11-paraview510:latest"
    openfoam_path: str = "/opt/openfoam11"
    timeout: int = 300  # seconds
    platform: str = "linux/amd64"


@dataclass(frozen=True)
class AppConfig:
    """
    Complete immutable application configuration.

    This is the single source of truth for all static configuration.
    Create once at startup and pass to functions that need it.
    """
    paths: PathConfig
    llm: LLMConfig
    openfoam: OpenFOAMConstants
    docker: DockerConfig = DockerConfig()

    # PDF processing
    pdf_chunk_distance: float = 1.5
    sentence_transformer_path: str = "sentence-transformers/all-mpnet-base-v2"

    # Execution limits
    max_correction_attempts: int = 30
    reference_case_search_rounds: int = 10


def load_config(config_path: str | None = None) -> AppConfig:
    """
    Load configuration from JSON file and environment variables.

    Priority: Environment variables > Config file > Defaults

    Args:
        config_path: Path to JSON config file. If None, uses default location.

    Returns:
        Immutable AppConfig instance.
    """
    paths = PathConfig.from_defaults()

    if config_path is None:
        config_path = os.path.join(paths.root_dir, "inputs", "chatcfd_config.json")

    # Load from JSON if exists
    config_data = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")

    # Set environment variables from config (env vars take priority)
    _set_env_if_not_exists("OPENAI_API_KEY", config_data.get("OPENAI_API_KEY", ""))
    _set_env_if_not_exists("OPENAI_BASE_URL", config_data.get("OPENAI_BASE_URL", ""))
    _set_env_if_not_exists("INSTRUCT_MODEL_NAME", config_data.get("INSTRUCT_MODEL_NAME", ""))
    _set_env_if_not_exists("REASONING_MODEL_NAME", config_data.get("REASONING_MODEL_NAME", ""))

    # Update paths if OpenFOAM path specified
    openfoam_path = config_data.get("OpenFOAM_path", "/usr/lib/openfoam/openfoam2406")
    tutorials_path = config_data.get("OpenFOAM_tutorials_path", os.path.join(openfoam_path, "tutorials"))

    paths = PathConfig(
        root_dir=paths.root_dir,
        src_dir=paths.src_dir,
        output_dir=paths.output_dir,
        database_dir=paths.database_dir,
        temp_dir=paths.temp_dir,
        runs_dir=paths.runs_dir,
        inputs_dir=paths.inputs_dir,
        openfoam_tutorials_dir=tutorials_path,
    )

    # Docker configuration
    docker_config = DockerConfig(
        enabled=config_data.get("docker_enabled", False),
        image=config_data.get("docker_image", "openfoam/openfoam11-paraview510:latest"),
        openfoam_path=config_data.get("docker_openfoam_path", "/opt/openfoam11"),
        timeout=config_data.get("docker_timeout", 300),
    )

    return AppConfig(
        paths=paths,
        llm=LLMConfig.from_env(),
        openfoam=OpenFOAMConstants(),
        docker=docker_config,
        pdf_chunk_distance=config_data.get("pdf_chunk_d", 1.5),
        sentence_transformer_path=config_data.get("sentence_transformer_path", "sentence-transformers/all-mpnet-base-v2"),
        max_correction_attempts=config_data.get("max_running_test_round", 30),
    )


def _set_env_if_not_exists(key: str, value: str) -> None:
    """Set environment variable only if not already set and value is non-empty."""
    if key not in os.environ or not os.environ[key]:
        if value:
            os.environ[key] = value


def ensure_directories(config: AppConfig) -> None:
    """Ensure all required directories exist."""
    directories = [
        config.paths.database_dir,
        config.paths.temp_dir,
        config.paths.output_dir,
        config.paths.runs_dir,
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")


def load_openfoam_environment(openfoam_path: str = "/usr/lib/openfoam/openfoam2406") -> bool:
    """
    Load OpenFOAM environment variables into the current process.

    Returns:
        True if successful, False otherwise.
    """
    try:
        command = f'source {openfoam_path}/etc/bashrc && env'
        output = subprocess.run(
            command,
            shell=True,
            executable="/bin/bash",
            check=True,
            text=True,
            capture_output=True,
        )
        for line in output.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                os.environ[key] = value
        return True
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to load OpenFOAM environment: {e.stderr}")
        return False
    except Exception as e:
        print(f"Warning: Could not load OpenFOAM environment: {e}")
        return False
