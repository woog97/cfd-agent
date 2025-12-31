"""
Common utilities shared across step modules.

Contains:
- Dependencies dataclass for dependency injection
- Shared helper functions for file processing
"""

import re
from dataclasses import dataclass

from config import AppConfig
from llm import LLMClient
from database import ReferenceDatabase, load_dimensions_dict


@dataclass
class Dependencies:
    """
    All external dependencies for step functions.

    Injected once at workflow start, passed to all steps.
    Makes testing easy - just mock these.
    """
    config: AppConfig
    llm: LLMClient
    database: ReferenceDatabase


def clean_llm_file_response(response: str) -> str:
    """
    Remove markdown formatting from LLM response.

    Handles:
    - Code block markers (```foam, ```cpp, etc.)
    - Trailing explanations after OpenFOAM end markers
    """
    content = response.strip()

    # Remove markdown code blocks
    if content.startswith("```"):
        lines = content.split('\n')
        # Remove first line (```foam or ```cpp or ```)
        lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = '\n'.join(lines)

    # Truncate after OpenFOAM file ending comment if present
    # This removes any trailing explanations the LLM might add
    end_marker = "// *****"
    if end_marker in content:
        # Find the last occurrence of the end marker line
        lines = content.split('\n')
        last_marker_idx = -1
        for i, line in enumerate(lines):
            if line.strip().startswith("// ***"):
                last_marker_idx = i
        if last_marker_idx > 0:
            content = '\n'.join(lines[:last_marker_idx + 1])

    return content


def fix_dimensions(
    content: str,
    file_path: str,
    solver: str,
    incompressible_solvers: set[str],
    database_dir: str,
) -> str:
    """
    Fix dimensions in generated file based on known correct values.

    Args:
        content: File content to fix
        file_path: Path like "0/p" or "0/U"
        solver: Solver name for incompressible check
        incompressible_solvers: Set of incompressible solver names
        database_dir: Path to database directory
    """
    dimensions_dict = load_dimensions_dict(database_dir)

    if file_path not in dimensions_dict:
        return content

    if "dimensions" not in content:
        return content

    correct_dim = dimensions_dict[file_path]

    # Handle special cases for pressure fields
    if file_path in ["0/p", "0/p_rgh", "0/alphat"]:
        # Check if there's an incompressible variant
        incomp_key = file_path + "_"
        if incomp_key in dimensions_dict:
            if solver in incompressible_solvers:
                correct_dim = dimensions_dict[incomp_key]

    # Replace dimension line
    dimension_pattern = r'dimensions\s+\[.*?\];'
    replacement = f'dimensions      {correct_dim};'
    content = re.sub(dimension_pattern, replacement, content)

    return content
