"""
Mesh processing steps: conversion and boundary extraction.
"""

import os
import re

from state import CaseState, StateUpdate
from steps.common import Dependencies
from steps.tools import run_mesh_converter_local, run_mesh_converter_docker


def convert_mesh(state: CaseState, deps: Dependencies) -> StateUpdate:
    """
    Convert Fluent mesh to OpenFOAM format if needed.

    Supports both local and Docker-based conversion.

    Returns:
        StateUpdate with mesh_converted flag.
    """
    if state.grid_type != "msh":
        # Already OpenFOAM format
        return {"mesh_converted": True}

    if deps.config.docker.enabled:
        success = run_mesh_converter_docker(
            mesh_file=state.grid_path,
            case_path=state.output_path,
            docker_config=deps.config.docker,
        )
        return {"mesh_converted": success}
    else:
        success = run_mesh_converter_local(
            mesh_file=state.grid_path,
            case_path=state.output_path,
        )
        return {"mesh_converted": success}


def extract_boundaries(state: CaseState, deps: Dependencies) -> StateUpdate:
    """
    Extract boundary names from mesh file.

    Returns:
        StateUpdate with grid_boundaries list.
    """
    boundaries = []

    if state.grid_type == "msh":
        boundaries = _extract_boundaries_from_msh(state.grid_path)
    else:
        boundaries = _extract_boundaries_from_polymesh(state.grid_path)

    return {"grid_boundaries": boundaries}


def _extract_boundaries_from_msh(filepath: str) -> list[str]:
    """Extract boundary names from Fluent .msh file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read().splitlines()

        # Find Zone Sections marker
        start_index = -1
        for i in range(len(content) - 1, -1, -1):
            if content[i].strip() == '(0 "Zone Sections")':
                start_index = i
                break

        if start_index == -1:
            return []

        pattern = re.compile(r'\(\d+\s+\(\d+\s+\S+\s+(\S+)\)\(\)\)')
        results = []

        for line in content[start_index + 1:]:
            line = line.strip()
            if not line.startswith('(39'):
                continue

            match = pattern.match(line)
            if match:
                value = match.group(1)
                # Filter out FLUID and SOLID zones (case-insensitive)
                # Also filter interior zones
                if not re.search(
                    r'^(fluid|solid|interior|\w+[-_]fluid|\w+[-_]solid|\w+[-_]interior)$',
                    value,
                    re.IGNORECASE
                ):
                    results.append(value)

        return results
    except Exception as e:
        print(f"Error extracting boundaries from msh: {e}")
        return []


def _extract_boundaries_from_polymesh(dirpath: str) -> list[str]:
    """Extract boundary names from OpenFOAM polyMesh."""
    boundary_file = os.path.join(dirpath, "boundary")
    if not os.path.exists(boundary_file):
        return []

    try:
        with open(boundary_file, 'r') as f:
            content = f.read()

        # Parse boundary names (simplified)
        pattern = r'^\s*(\w+)\s*\n\s*\{'
        matches = re.findall(pattern, content, re.MULTILINE)
        return [m for m in matches if m not in ('FoamFile',)]
    except Exception as e:
        print(f"Error extracting boundaries from polyMesh: {e}")
        return []
