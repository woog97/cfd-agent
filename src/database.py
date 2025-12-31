"""
Database: OpenFOAM tutorial case reference database.

This module provides access to pre-processed OpenFOAM tutorial cases
for finding reference files during case generation and error correction.

Design principles:
- Immutable database loaded once
- No global state
- Pure functions for querying
"""

import json
import os
import random
from dataclasses import dataclass
from typing import Iterator

from logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class CaseReference:
    """A reference case from the OpenFOAM tutorials."""
    path: str
    solver: str
    turbulence_type: str | None
    turbulence_model: str | None
    other_physical_model: list[str] | None
    configuration_files: dict[str, str]
    required_fields: list[str]
    is_single_phase: bool
    is_particle_flow: bool
    is_reacting_flow: bool
    boundary_types: list[str]


class ReferenceDatabase:
    """
    OpenFOAM tutorial case reference database.

    Usage:
        db = ReferenceDatabase.load(config.paths.database_dir)

        # Find reference files for a specific file type
        refs = db.find_reference_files(
            target_file="0/U",
            solver="simpleFoam",
            turbulence_model="kOmegaSST",
        )
    """

    def __init__(self, cases: dict[str, CaseReference]):
        """
        Initialize database with case references.

        Args:
            cases: Dictionary mapping case path to CaseReference.
        """
        self._cases = cases
        self._by_solver: dict[str, list[str]] = {}
        self._build_indices()

    def _build_indices(self) -> None:
        """Build indices for fast lookup."""
        if not isinstance(self._cases, dict):
            logger.error(f"_cases is not a dict: {type(self._cases)}")
            self._cases = {}
            return

        for path, case in self._cases.items():
            if not isinstance(case, CaseReference):
                logger.warning(f"Skipping invalid case at {path}: {type(case)}")
                continue
            if case.solver:
                if case.solver not in self._by_solver:
                    self._by_solver[case.solver] = []
                self._by_solver[case.solver].append(path)

    @classmethod
    def load(cls, database_dir: str) -> "ReferenceDatabase":
        """
        Load database from pre-processed JSON file.

        Args:
            database_dir: Path to database directory containing processed_merged_OF_cases.json

        Returns:
            Loaded ReferenceDatabase instance.
        """
        json_path = os.path.join(database_dir, "processed_merged_OF_cases.json")

        if not os.path.exists(json_path):
            logger.warning(f"Database file not found at {json_path}")
            return cls({})

        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        cases = {}
        for path, data in raw_data.items():
            cases[path] = CaseReference(
                path=path,
                solver=data.get("solver"),
                turbulence_type=data.get("turbulence_type"),
                turbulence_model=data.get("turbulence_model"),
                other_physical_model=data.get("other_physical_model"),
                configuration_files=data.get("configuration_files", {}),
                required_fields=data.get("required_field", []),
                is_single_phase=data.get("singlePhase", True),
                is_particle_flow=data.get("particle_flow", False),
                is_reacting_flow=data.get("reacting_flow", False),
                boundary_types=data.get("boundary_type", []),
            )

        return cls(cases)

    def __len__(self) -> int:
        return len(self._cases)

    def __iter__(self) -> Iterator[CaseReference]:
        return iter(self._cases.values())

    def get_case(self, path: str) -> CaseReference | None:
        """Get a case by its path."""
        return self._cases.get(path)

    def find_reference_files(
        self,
        target_file: str,
        solver: str,
        turbulence_model: str | None = None,
        other_physical_model: list[str] | None = None,
        max_results: int = 3,
        max_content_length: int = 10000,
    ) -> dict[str, str]:
        """
        Find reference files from tutorial cases.

        Search priority:
        1. Cases matching solver + turbulence_model + other_physical_model
        2. Cases matching solver + turbulence_model
        3. Cases matching solver only
        4. Cases in the same domain (e.g., compressible)

        Args:
            target_file: File to search for (e.g., "0/U", "system/fvSolution")
            solver: OpenFOAM solver name
            turbulence_model: Turbulence model name (optional)
            other_physical_model: Additional physical models (optional)
            max_results: Maximum number of reference files to return
            max_content_length: Skip files longer than this

        Returns:
            Dictionary mapping sample names to file contents.
        """
        results: dict[str, str] = {}

        # Level 1: Match solver + turbulence + other physical model
        if turbulence_model:
            for path, case in self._cases.items():
                if solver not in path:
                    continue
                if case.turbulence_model != turbulence_model:
                    continue

                # Check other physical model match
                if other_physical_model:
                    case_other = case.other_physical_model or []
                    if set(other_physical_model) != set(case_other):
                        continue
                else:
                    case_other = case.other_physical_model or []
                    if case_other and case_other != ["common"]:
                        continue

                content = case.configuration_files.get(target_file, "")
                if content and len(content) <= max_content_length:
                    results[f"sample_{len(results)}"] = content

        # Level 2: Match solver + turbulence only
        if not results and turbulence_model:
            for path, case in self._cases.items():
                if solver not in path:
                    continue
                if case.turbulence_model != turbulence_model:
                    continue

                content = case.configuration_files.get(target_file, "")
                if content and len(content) <= max_content_length:
                    results[f"sample_{len(results)}"] = content

        # Level 3: Match solver only
        if not results:
            for path, case in self._cases.items():
                if solver not in path:
                    continue

                content = case.configuration_files.get(target_file, "")
                if content and len(content) <= max_content_length:
                    results[f"sample_{len(results)}"] = content

        # Level 4: Match domain type
        if not results:
            domain_type = self._get_domain_type(solver)
            if domain_type:
                for path, case in self._cases.items():
                    if path.split('/')[0] == domain_type:
                        content = case.configuration_files.get(target_file, "")
                        if content and len(content) <= max_content_length:
                            results[path] = content

        # Limit and randomize results
        if len(results) > max_results:
            keys = random.sample(list(results.keys()), max_results)
            results = {k: results[k] for k in keys}

        return results

    def _get_domain_type(self, solver: str) -> str | None:
        """Get the domain type (e.g., 'compressible') for a solver."""
        for path in self._cases:
            if solver in path:
                return path.split('/')[0]
        return None

    def get_cases_by_solver(self, solver: str) -> list[CaseReference]:
        """Get all cases using a specific solver."""
        paths = self._by_solver.get(solver, [])
        return [self._cases[p] for p in paths]

    def get_all_solvers(self) -> set[str]:
        """Get set of all solvers in the database."""
        return set(self._by_solver.keys())

    def get_all_turbulence_models(self) -> set[str]:
        """Get set of all turbulence models in the database."""
        models = set()
        for case in self._cases.values():
            if case.turbulence_model:
                models.add(case.turbulence_model)
        return models


def reference_files_to_json(refs: dict[str, str]) -> str:
    """Convert reference files dict to JSON string for prompts."""
    return json.dumps(refs, ensure_ascii=False, indent=2)


def load_dimensions_dict(database_dir: str) -> dict[str, str]:
    """
    Load the dimensions dictionary for OpenFOAM fields.

    Args:
        database_dir: Path to database directory.

    Returns:
        Dictionary mapping field names to dimension strings.
    """
    path = os.path.join(database_dir, "OF_case_dimensions.json")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def load_boundary_entries(database_dir: str) -> list[dict]:
    """
    Load boundary condition entry requirements.

    Args:
        database_dir: Path to database directory.

    Returns:
        List of boundary type requirements.
    """
    path = os.path.join(database_dir, "OF_bc_entry.json")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []
