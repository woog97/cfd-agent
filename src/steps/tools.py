"""
External tool execution: Docker, subprocess, OpenFOAM CLI.

This module handles all external command execution, keeping
step functions focused on workflow logic.
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Optional

from config import DockerConfig, OpenFOAMConstants


# =============================================================================
# Security Validation
# =============================================================================

class CommandInjectionError(Exception):
    """Raised when command injection is detected."""
    pass


class PathTraversalError(Exception):
    """Raised when path traversal is detected."""
    pass


# Cache OpenFOAM constants for validation
_OPENFOAM_CONSTANTS = OpenFOAMConstants()


def validate_solver_name(solver: str) -> str:
    """
    Validate solver name against known OpenFOAM solvers.

    Args:
        solver: Solver name to validate

    Returns:
        The validated solver name

    Raises:
        CommandInjectionError: If solver name is invalid or contains dangerous characters
    """
    # Check for shell metacharacters
    dangerous_chars = set(";|&$`\\\"'<>(){}[]!#*?~")
    if any(c in solver for c in dangerous_chars):
        raise CommandInjectionError(
            f"Invalid solver name contains shell metacharacters: {solver!r}"
        )

    # Check against whitelist of known solvers
    if solver not in _OPENFOAM_CONSTANTS.solver_keywords:
        raise CommandInjectionError(
            f"Unknown solver: {solver!r}. "
            f"Must be one of the known OpenFOAM solvers."
        )

    return solver


def validate_safe_path(base_path: str, rel_path: str) -> str:
    """
    Validate that a relative path doesn't escape the base directory.

    Args:
        base_path: The base directory that must contain the result
        rel_path: The relative path to validate

    Returns:
        The validated absolute path

    Raises:
        PathTraversalError: If the path would escape base_path
    """
    # Resolve to absolute paths
    base = Path(base_path).resolve()
    full = (base / rel_path).resolve()

    # Check that the resolved path is within base
    try:
        full.relative_to(base)
    except ValueError:
        raise PathTraversalError(
            f"Path traversal detected: {rel_path!r} escapes {base_path!r}"
        )

    return str(full)


# =============================================================================
# Solver Execution
# =============================================================================

def run_solver_local(
    solver: str,
    case_path: str,
    file_structure: list[str],
) -> tuple[int, str, str]:
    """
    Run OpenFOAM solver locally via subprocess.

    Args:
        solver: Solver name (e.g., "simpleFoam")
        case_path: Path to case directory
        file_structure: List of case files (to check for setFieldsDict)

    Returns:
        Tuple of (return_code, stdout, stderr)

    Raises:
        CommandInjectionError: If solver name is invalid
    """
    # Validate solver name before shell execution
    solver = validate_solver_name(solver)

    log_file = os.path.join(case_path, "case_run.log")

    # Check if setFields is needed
    if "system/setFieldsDict" in file_structure:
        command = f"cd {case_path} && setFields && {solver} > {log_file} 2>&1"
    else:
        command = f"cd {case_path} && {solver} > {log_file} 2>&1"

    try:
        result = subprocess.run(
            command,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)


def run_solver_docker(
    solver: str,
    case_path: str,
    file_structure: list[str],
    docker_config: DockerConfig,
) -> tuple[int, str, str]:
    """
    Run OpenFOAM solver via Docker container.

    Args:
        solver: Solver name (e.g., "simpleFoam")
        case_path: Path to case directory
        file_structure: List of case files (to check for setFieldsDict)
        docker_config: Docker configuration

    Returns:
        Tuple of (return_code, stdout, stderr)

    Raises:
        CommandInjectionError: If solver name is invalid
    """
    # Validate solver name before execution
    solver = validate_solver_name(solver)

    # Build OpenFOAM command to run inside Docker
    if "system/setFieldsDict" in file_structure:
        of_command = f"setFields && {solver}"
    else:
        of_command = solver

    # Build Docker command
    case_path = os.path.abspath(case_path)
    docker_cmd = [
        "docker", "run", "--rm",
        "--platform", docker_config.platform,
        "--entrypoint", "/bin/bash",
        "-v", f"{case_path}:/case",
        "-w", "/case",
        docker_config.image,
        "-c", f"source {docker_config.openfoam_path}/etc/bashrc && {of_command}"
    ]

    log_file = os.path.join(case_path, "case_run.log")

    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=docker_config.timeout,
        )

        # Write combined output to log file
        full_output = f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        with open(log_file, 'w') as f:
            f.write(full_output)

        return result.returncode, result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        return 1, "", f"Docker execution timed out after {docker_config.timeout}s"
    except Exception as e:
        return 1, "", f"Docker execution error: {str(e)}"


# =============================================================================
# Mesh Conversion
# =============================================================================

def run_mesh_converter_local(mesh_file: str, case_path: str) -> bool:
    """
    Convert mesh locally using fluentMeshToFoam.

    Args:
        mesh_file: Path to mesh file
        case_path: Path to case directory

    Returns:
        True if successful, False otherwise
    """
    try:
        command = ["fluentMeshToFoam", "-case", case_path, mesh_file]
        subprocess.run(command, check=True, capture_output=True)
        print(f"Mesh conversion completed: {mesh_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Mesh conversion failed: {e}")
        return False
    except FileNotFoundError:
        print("fluentMeshToFoam not found - OpenFOAM environment not loaded")
        return False


def run_mesh_converter_docker(
    mesh_file: str,
    case_path: str,
    docker_config: DockerConfig,
) -> bool:
    """
    Convert mesh file to OpenFOAM format using Docker.

    Args:
        mesh_file: Path to mesh file (relative to case_path or absolute)
        case_path: Path to case directory
        docker_config: Docker configuration

    Returns:
        True if conversion successful, False otherwise
    """
    case_path = os.path.abspath(case_path)

    # Handle mesh file path
    if os.path.isabs(mesh_file):
        # Mount mesh file separately
        mesh_dir = os.path.dirname(mesh_file)
        mesh_name = os.path.basename(mesh_file)
        docker_cmd = [
            "docker", "run", "--rm",
            "--platform", docker_config.platform,
            "--entrypoint", "/bin/bash",
            "-v", f"{case_path}:/case",
            "-v", f"{mesh_dir}:/mesh:ro",
            "-w", "/case",
            docker_config.image,
            "-c", f"source {docker_config.openfoam_path}/etc/bashrc && fluentMeshToFoam /mesh/{mesh_name}"
        ]
    else:
        # Mesh file is in case directory
        docker_cmd = [
            "docker", "run", "--rm",
            "--platform", docker_config.platform,
            "--entrypoint", "/bin/bash",
            "-v", f"{case_path}:/case",
            "-w", "/case",
            docker_config.image,
            "-c", f"source {docker_config.openfoam_path}/etc/bashrc && fluentMeshToFoam {mesh_file}"
        ]

    try:
        print(f"Converting mesh via Docker: {mesh_file}")
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            print("Mesh conversion completed successfully")
            return True
        else:
            print(f"Mesh conversion failed: {result.stderr[:500]}")
            return False
    except Exception as e:
        print(f"Mesh conversion error: {e}")
        return False


# =============================================================================
# File Utilities
# =============================================================================

def extract_solver_from_controldict(path: str) -> Optional[str]:
    """Extract application name from controlDict."""
    try:
        with open(path, 'r') as f:
            content = f.read()
        match = re.search(r'application\s+(\w+);', content)
        return match.group(1) if match else None
    except OSError:
        return None


def read_log_tail(path: str, lines: int = 50) -> str:
    """Read the last N lines of a log file."""
    try:
        with open(path, 'r') as f:
            all_lines = f.readlines()
            return ''.join(all_lines[-lines:])
    except OSError:
        return ""


def write_openfoam_file(path: str, content: str) -> None:
    """
    Write content to an OpenFOAM file with proper encoding handling.

    Handles escape sequences that may be in LLM-generated content.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    try:
        # Try to handle escape sequences
        processed = content.encode('latin-1').decode('unicode_escape')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(processed)
    except (UnicodeDecodeError, UnicodeEncodeError):
        # Fall back to writing raw content
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)


def read_file_safe(path: str, default: str = "") -> str:
    """Read file content, returning default if file doesn't exist."""
    try:
        with open(path, 'r') as f:
            return f.read()
    except OSError:
        return default
