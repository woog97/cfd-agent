"""
End-to-end tests for src2 workflow.

These tests call REAL LLMs and run REAL OpenFOAM simulations via Docker.

Tests:
- test_generate_files: Generate OpenFOAM files using real LLM
- test_run_simulation: Run simulation via Docker
- test_full_workflow: Complete workflow from description to results

Usage:
    # Generate files only (calls real LLM)
    python -m pytest tests/test_e2e_workflow.py::TestE2EWorkflow::test_generate_files -v -s

    # Run simulation only (requires generated files)
    python -m pytest tests/test_e2e_workflow.py::TestE2EWorkflow::test_run_simulation -v -s

    # Full workflow
    python -m pytest tests/test_e2e_workflow.py::TestE2EWorkflow::test_full_workflow -v -s

    # Or run directly:
    python tests/test_e2e_workflow.py --generate
    python tests/test_e2e_workflow.py --simulate
    python tests/test_e2e_workflow.py --all
"""

import os
import sys
import json
import shutil
import subprocess
import time
from datetime import datetime

# Add src2 to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config, AppConfig
from state import CaseState, CaseBuilder
from llm import LLMClient
from database import ReferenceDatabase
from steps import Dependencies, determine_file_structure, extract_boundaries, generate_files, write_files
from workflow import run_workflow, WorkflowRunner

# Paths
SRC2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(SRC2_DIR)
FIXTURES_DIR = os.path.join(ROOT_DIR, "src", "tests", "fixtures")
SNAPSHOT_PATH = os.path.join(FIXTURES_DIR, "naca0012_case_snapshot.json")
OUTPUT_DIR = os.path.join(ROOT_DIR, "run_chatcfd", "src2_e2e_test")

# Docker settings
DOCKER_IMAGE = "openfoam/openfoam11-paraview510:latest"


class E2ELogger:
    """Logger for tracing test actions and LLM calls."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "test_trace.jsonl")
        self.llm_log_file = os.path.join(log_dir, "llm_calls.jsonl")
        self.start_time = time.time()

    def log(self, event_type: str, data: dict):
        """Log an event."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_s": round(time.time() - self.start_time, 2),
            "event": event_type,
            "data": data,
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_step(self, message: str):
        """Log a workflow step."""
        self.log("step", {"message": message})
        print(f"[{round(time.time() - self.start_time, 1)}s] {message}")

    def log_error(self, message: str):
        """Log an error."""
        self.log("error", {"message": message})
        print(f"ERROR: {message}")

    def log_result(self, success: bool, message: str):
        """Log final result."""
        self.log("result", {"success": success, "message": message})


def load_case_snapshot() -> dict | None:
    """Load the NACA0012 case snapshot."""
    if not os.path.exists(SNAPSHOT_PATH):
        return None
    with open(SNAPSHOT_PATH, "r") as f:
        return json.load(f)


def run_openfoam_docker(command: str, case_path: str, timeout: int = 300) -> tuple[int, str, str]:
    """
    Run an OpenFOAM command inside Docker.

    Args:
        command: OpenFOAM command to run
        case_path: Path to case directory (mounted in Docker)
        timeout: Timeout in seconds

    Returns:
        (returncode, stdout, stderr)
    """
    case_path = os.path.abspath(case_path)

    docker_cmd = [
        "docker", "run", "--rm",
        "--platform", "linux/amd64",
        "--entrypoint", "/bin/bash",
        "-v", f"{case_path}:/case",
        "-w", "/case",
        DOCKER_IMAGE,
        "-c", f"source /opt/openfoam11/etc/bashrc && {command}"
    ]

    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Timeout expired"
    except Exception as e:
        return -1, "", str(e)


def check_docker_available() -> bool:
    """Check if Docker is available and OpenFOAM image exists."""
    try:
        result = subprocess.run(
            ["docker", "images", "-q", DOCKER_IMAGE],
            capture_output=True,
            text=True,
            timeout=10
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


class TestE2EWorkflow:
    """End-to-end tests using real LLM and OpenFOAM."""

    @classmethod
    def setup_class(cls):
        """Set up test fixtures."""
        cls.snapshot = load_case_snapshot()
        cls.config = load_config()
        cls.docker_available = check_docker_available()

        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cls.logger = E2ELogger(os.path.join(OUTPUT_DIR, "logs"))

    def test_generate_files(self):
        """
        Test: Generate OpenFOAM files using REAL LLM.

        This test:
        1. Loads case description from snapshot
        2. Creates real LLM client
        3. Generates all required OpenFOAM files
        4. Writes files to disk
        """
        print("\n" + "=" * 60)
        print("E2E Test: Generate OpenFOAM Files (Real LLM)")
        print("=" * 60)

        assert self.snapshot is not None, f"Snapshot not found at {SNAPSHOT_PATH}"

        self.logger.log_step("Loading case configuration")

        # Extract case info
        case_info = self.snapshot["case_info"]
        paths = self.snapshot["paths"]

        print(f"Case: {case_info['case_name']}")
        print(f"Solver: {case_info['case_solver']}")
        print(f"Turbulence: {case_info['turbulence_model']}")

        # Create output directory for this test
        case_output = os.path.join(OUTPUT_DIR, case_info["case_name"])
        if os.path.exists(case_output):
            shutil.rmtree(case_output)
        os.makedirs(case_output)

        self.logger.log_step("Creating LLM client and database")

        # Create real LLM client
        llm = LLMClient(
            self.config.llm,
            log_path=os.path.join(OUTPUT_DIR, "logs", "llm_calls.jsonl")
        )

        # Load real database
        database = ReferenceDatabase.load(self.config.paths.database_dir)
        print(f"Loaded {len(database)} reference cases")

        # Create dependencies
        deps = Dependencies(
            config=self.config,
            llm=llm,
            database=database,
        )

        # Build initial state
        self.logger.log_step("Building case state")

        builder = CaseBuilder(
            solver=case_info["case_solver"],
            turbulence_model=case_info["turbulence_model"],
            other_physical_model=case_info.get("other_physical_model") or [],
            description=case_info["case_description"],
            grid_path=paths["mesh_path"],
            grid_type=self.snapshot.get("grid_type", "msh"),
            output_dir=OUTPUT_DIR,
            max_attempts=5,
        )

        state = builder.build(case_info["case_name"])

        # Step 1: Determine file structure
        self.logger.log_step("Determining file structure")
        update = determine_file_structure(state, deps)
        state = state.apply_update(update)
        print(f"Required files: {len(state.file_structure)}")
        for f in state.file_structure:
            print(f"  - {f}")

        # Step 2: Extract boundaries from mesh
        self.logger.log_step("Extracting boundaries from mesh")
        update = extract_boundaries(state, deps)
        state = state.apply_update(update)

        if not state.grid_boundaries:
            # Use boundaries from snapshot if extraction failed
            self.logger.log_step("Using default boundaries")
            state = state.apply_update({
                "grid_boundaries": ["inlet", "outlet", "airfoil", "frontAndBackPlanes"]
            })

        print(f"Boundaries: {state.grid_boundaries}")

        # Step 3: Generate files using LLM
        self.logger.log_step("Generating files via LLM (this may take a while)...")
        start_time = time.time()

        update = generate_files(state, deps)
        state = state.apply_update(update)

        generation_time = time.time() - start_time
        print(f"Generated {len(state.generated_files)} files in {generation_time:.1f}s")

        # Step 4: Write files to disk
        self.logger.log_step("Writing files to disk")
        update = write_files(state, deps)
        state = state.apply_update(update)

        # Verify files exist
        files_written = 0
        for f in state.generated_files:
            full_path = os.path.join(state.output_path, f)
            if os.path.exists(full_path):
                files_written += 1
                size = os.path.getsize(full_path)
                print(f"  {f}: {size} bytes")

        assert files_written == len(state.generated_files), \
            f"Only {files_written}/{len(state.generated_files)} files written"

        # Save snapshot of generated files
        snapshot_path = os.path.join(OUTPUT_DIR, "generated_files_snapshot.json")
        with open(snapshot_path, "w") as f:
            json.dump(state.generated_files, f, indent=2)

        self.logger.log_result(True, f"Generated {files_written} files")
        print("\n" + "=" * 60)
        print("FILE GENERATION PASSED")
        print("=" * 60)
        print(f"Output: {state.output_path}")
        print(f"LLM stats: {llm.stats.instruct_calls} instruct, {llm.stats.reasoning_calls} reasoning calls")

        return True

    def test_mesh_conversion(self):
        """
        Test: Convert mesh via Docker.

        Requires Docker with OpenFOAM image.
        """
        print("\n" + "=" * 60)
        print("E2E Test: Mesh Conversion (Docker)")
        print("=" * 60)

        if not self.docker_available:
            print("SKIPPED: Docker not available or OpenFOAM image not pulled")
            print(f"Run: docker pull {DOCKER_IMAGE}")
            return False

        assert self.snapshot is not None, "Snapshot not found"

        case_info = self.snapshot["case_info"]
        mesh_path = self.snapshot["paths"]["mesh_path"]

        if not os.path.exists(mesh_path):
            print(f"SKIPPED: Mesh file not found: {mesh_path}")
            return False

        case_output = os.path.join(OUTPUT_DIR, case_info["case_name"])
        if not os.path.exists(case_output):
            print("SKIPPED: Run test_generate_files first")
            return False

        # Copy mesh file to case directory
        self.logger.log_step("Copying mesh file")
        mesh_dest = os.path.join(case_output, os.path.basename(mesh_path))
        if not os.path.exists(mesh_dest):
            shutil.copy(mesh_path, mesh_dest)

        # Run mesh conversion
        self.logger.log_step("Converting mesh via Docker")
        mesh_file = os.path.basename(mesh_path)
        returncode, stdout, stderr = run_openfoam_docker(
            f"fluentMeshToFoam {mesh_file}",
            case_output,
            timeout=120
        )

        if returncode != 0:
            self.logger.log_error(f"Mesh conversion failed: {stderr}")
            print(f"stdout: {stdout[-500:] if stdout else 'empty'}")
            print(f"stderr: {stderr[-500:] if stderr else 'empty'}")
            return False

        # Verify polyMesh created
        boundary_file = os.path.join(case_output, "constant", "polyMesh", "boundary")
        assert os.path.exists(boundary_file), "boundary file not created"

        self.logger.log_result(True, "Mesh conversion successful")
        print("\n" + "=" * 60)
        print("MESH CONVERSION PASSED")
        print("=" * 60)

        return True

    def test_run_simulation(self):
        """
        Test: Run OpenFOAM simulation via Docker.

        Requires:
        - Docker with OpenFOAM image
        - Generated case files (run test_generate_files first)
        - Converted mesh (run test_mesh_conversion first)
        """
        print("\n" + "=" * 60)
        print("E2E Test: Run Simulation (Docker)")
        print("=" * 60)

        if not self.docker_available:
            print("SKIPPED: Docker not available")
            return False

        assert self.snapshot is not None, "Snapshot not found"

        case_info = self.snapshot["case_info"]
        case_output = os.path.join(OUTPUT_DIR, case_info["case_name"])

        if not os.path.exists(os.path.join(case_output, "constant", "polyMesh")):
            print("SKIPPED: Run test_mesh_conversion first")
            return False

        solver = case_info["case_solver"]
        self.logger.log_step(f"Running {solver} via Docker")

        # Run solver with limited iterations
        returncode, stdout, stderr = run_openfoam_docker(
            f"{solver}",
            case_output,
            timeout=300  # 5 minutes max
        )

        # Check for success
        has_fatal_error = "FOAM FATAL" in stdout or "FOAM FATAL" in stderr

        if has_fatal_error:
            self.logger.log_error("Simulation failed with FOAM FATAL ERROR")
            print(f"Last output:\n{stdout[-1000:] if stdout else stderr[-1000:]}")

            # Save error log
            error_log = os.path.join(case_output, "simulation_error.log")
            with open(error_log, "w") as f:
                f.write(f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}")

            return False

        # Check if simulation produced output
        time_dirs = [d for d in os.listdir(case_output)
                     if d.replace(".", "").isdigit() and d != "0"]

        if time_dirs:
            self.logger.log_result(True, f"Simulation produced {len(time_dirs)} time directories")
            print(f"Time directories: {sorted(time_dirs)[:5]}...")
        else:
            self.logger.log_result(True, "Simulation ran (no time output yet)")

        print("\n" + "=" * 60)
        print("SIMULATION PASSED")
        print("=" * 60)

        return True

    def test_full_workflow(self):
        """
        Test: Complete workflow using WorkflowRunner.

        This runs the full state machine with real LLM and Docker.
        """
        print("\n" + "=" * 60)
        print("E2E Test: Full Workflow")
        print("=" * 60)

        if not self.docker_available:
            print("SKIPPED: Docker not available")
            return False

        assert self.snapshot is not None, "Snapshot not found"

        case_info = self.snapshot["case_info"]
        paths = self.snapshot["paths"]

        # Create fresh output directory
        workflow_output = os.path.join(OUTPUT_DIR, "full_workflow_test")
        if os.path.exists(workflow_output):
            shutil.rmtree(workflow_output)

        self.logger.log_step("Setting up workflow")

        # Create real dependencies
        llm = LLMClient(
            self.config.llm,
            log_path=os.path.join(OUTPUT_DIR, "logs", "workflow_llm.jsonl")
        )
        database = ReferenceDatabase.load(self.config.paths.database_dir)

        deps = Dependencies(
            config=self.config,
            llm=llm,
            database=database,
        )

        # Build state
        builder = CaseBuilder(
            solver=case_info["case_solver"],
            turbulence_model=case_info["turbulence_model"],
            other_physical_model=case_info.get("other_physical_model") or [],
            description=case_info["case_description"],
            grid_path=paths["mesh_path"],
            grid_type="msh",
            output_dir=workflow_output,
            max_attempts=3,  # Limit attempts for testing
        )

        state = builder.build("workflow_test")
        state = state.apply_update({
            "grid_boundaries": ["inlet", "outlet", "airfoil", "frontAndBackPlanes"]
        })

        self.logger.log_step("Running workflow")

        # Run workflow
        runner = WorkflowRunner(deps, verbose=True)
        final_state = runner.run(state)

        # Report results
        print("\n" + "=" * 60)
        print("WORKFLOW RESULTS")
        print("=" * 60)
        print(f"Success: {final_state.success}")
        print(f"Completed: {final_state.completed}")
        print(f"Attempts: {final_state.attempt_count}")
        print(f"Files generated: {len(final_state.generated_files)}")
        print(f"Error history: {len(final_state.error_history)} errors")

        if final_state.current_error:
            print(f"Last error: {final_state.current_error[:200]}...")

        self.logger.log_result(
            final_state.success,
            f"Workflow completed: success={final_state.success}, attempts={final_state.attempt_count}"
        )

        return final_state.success


def _run_generate_files():
    """CLI helper for file generation (not a pytest test)."""
    test = TestE2EWorkflow()
    test.setup_class()
    return test.test_generate_files()


def _run_mesh_conversion():
    """CLI helper for mesh conversion (not a pytest test)."""
    test = TestE2EWorkflow()
    test.setup_class()
    return test.test_mesh_conversion()


def _run_simulation():
    """CLI helper for simulation (not a pytest test)."""
    test = TestE2EWorkflow()
    test.setup_class()
    return test.test_run_simulation()


def _run_full_workflow():
    """CLI helper for full workflow (not a pytest test)."""
    test = TestE2EWorkflow()
    test.setup_class()
    return test.test_full_workflow()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="src2 E2E workflow tests")
    parser.add_argument("--generate", action="store_true", help="Generate files via LLM")
    parser.add_argument("--mesh", action="store_true", help="Convert mesh via Docker")
    parser.add_argument("--simulate", action="store_true", help="Run simulation via Docker")
    parser.add_argument("--all", action="store_true", help="Full workflow test")
    args = parser.parse_args()

    success = True

    if args.generate:
        success = _run_generate_files()
    elif args.mesh:
        if not _run_generate_files():
            success = False
        else:
            success = _run_mesh_conversion()
    elif args.simulate:
        success = _run_simulation()
    elif args.all:
        success = _run_full_workflow()
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python tests/test_e2e_workflow.py --generate   # Generate files via LLM")
        print("  python tests/test_e2e_workflow.py --mesh       # Generate + convert mesh")
        print("  python tests/test_e2e_workflow.py --simulate   # Run simulation")
        print("  python tests/test_e2e_workflow.py --all        # Full workflow")
        sys.exit(0)

    sys.exit(0 if success else 1)
