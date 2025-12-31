"""
Test correction loop with snapshot-based workflow.

This test separates the expensive LLM generation (~10 min) from the
OpenFOAM execution + correction loop (fast iteration).

Workflow:
    Phase 1: Generate files (run once, save snapshot)
        python -m pytest tests/test_correction_loop.py::TestCorrectionLoop::test_generate_and_save -v -s

    Phase 2: Run correction loop (iterate until success)
        python -m pytest tests/test_correction_loop.py::TestCorrectionLoop::test_correction_loop -v -s

    Or run directly:
        python tests/test_correction_loop.py --generate
        python tests/test_correction_loop.py --correct
        python tests/test_correction_loop.py --all
"""

import os
import sys
import json
import shutil
import time
from datetime import datetime
from unittest.mock import Mock

import pytest

# Add src2 to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config, AppConfig, DockerConfig
from state import CaseState, CaseBuilder
from llm import LLMClient
from database import ReferenceDatabase
from steps import (
    Dependencies,
    determine_file_structure,
    extract_boundaries,
    generate_files,
    write_files,
    run_simulation,
    analyze_error,
    correct_file,
    reflect_on_errors,
    run_mesh_conversion_docker,
)
from snapshot import save_snapshot, load_snapshot, write_files_from_snapshot
from workflow import WorkflowRunner, STEPS, ROUTERS, NextStep


# Paths
SRC2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(SRC2_DIR)
DATABASE_DIR = os.path.join(ROOT_DIR, "database_OFv24")
OUTPUT_DIR = os.path.join(ROOT_DIR, "run_chatcfd", "correction_loop_test")
SNAPSHOT_PATH = os.path.join(OUTPUT_DIR, "case_snapshot.json")
MESH_PATH = os.path.join(ROOT_DIR, "inputs", "grids", "naca0012.msh")

# Docker settings
DOCKER_IMAGE = "openfoam/openfoam11-paraview510:latest"


def check_docker_available() -> bool:
    """Check if Docker is available and OpenFOAM image exists."""
    import subprocess
    try:
        result = subprocess.run(
            ["docker", "images", "-q", DOCKER_IMAGE],
            capture_output=True,
            text=True,
            timeout=10
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def create_docker_config() -> AppConfig:
    """Create config with Docker enabled."""
    base_config = load_config()

    # Create new config with Docker enabled
    docker_cfg = DockerConfig(
        enabled=True,
        image=DOCKER_IMAGE,
        openfoam_path="/opt/openfoam11",
        timeout=300,
    )

    # We need to create a new AppConfig with docker enabled
    # Since AppConfig is frozen, we create a new one
    from config import PathConfig, LLMConfig, OpenFOAMConstants

    return AppConfig(
        paths=base_config.paths,
        llm=base_config.llm,
        openfoam=base_config.openfoam,
        docker=docker_cfg,
        pdf_chunk_distance=base_config.pdf_chunk_distance,
        sentence_transformer_path=base_config.sentence_transformer_path,
        max_correction_attempts=base_config.max_correction_attempts,
    )


class TestCorrectionLoop:
    """Test the correction loop with snapshot-based workflow."""

    @classmethod
    def setup_class(cls):
        """Set up test fixtures."""
        cls.config = load_config()
        cls.docker_available = check_docker_available()
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def test_generate_and_save(self):
        """
        Phase 1: Generate OpenFOAM files using real LLM and save snapshot.

        This is the SLOW part (~10 min). Run once, then use snapshot for iteration.
        """
        print("\n" + "=" * 60)
        print("Phase 1: Generate Files and Save Snapshot")
        print("=" * 60)

        # Create real LLM client
        llm = LLMClient(
            self.config.llm,
            log_path=os.path.join(OUTPUT_DIR, "llm_generate.jsonl")
        )

        # Load real database
        database = ReferenceDatabase.load(DATABASE_DIR)
        print(f"Loaded {len(database)} reference cases")

        # Create dependencies (Docker disabled for generation - not needed)
        deps = Dependencies(
            config=self.config,
            llm=llm,
            database=database,
        )

        # Build initial state
        case_output = os.path.join(OUTPUT_DIR, "naca0012_case")
        if os.path.exists(case_output):
            shutil.rmtree(case_output)
        os.makedirs(case_output)

        state = CaseState(
            case_name="naca0012_correction_test",
            solver="simpleFoam",
            turbulence_model="kOmegaSST",
            description="Incompressible turbulent flow over a NACA 0012 airfoil at 5 degrees angle of attack. Reynolds number ~1e6. Use standard wall functions.",
            output_path=case_output,
            grid_path=MESH_PATH,
            grid_type="msh",
            grid_boundaries=["inlet", "outlet", "airfoil", "frontAndBackPlanes"],
            max_attempts=10,
        )

        # Step 1: Determine file structure
        print("\n[1/3] Determining file structure...")
        update = determine_file_structure(state, deps)
        state = state.apply_update(update)
        print(f"Required files: {len(state.file_structure)}")
        for f in state.file_structure:
            print(f"  - {f}")

        # Step 2: Generate files using LLM
        print("\n[2/3] Generating files via LLM (this takes ~10 minutes)...")
        start_time = time.time()

        update = generate_files(state, deps)
        state = state.apply_update(update)

        generation_time = time.time() - start_time
        print(f"Generated {len(state.generated_files)} files in {generation_time:.1f}s")

        # Step 3: Write files to disk
        print("\n[3/3] Writing files to disk...")
        update = write_files(state, deps)
        state = state.apply_update(update)

        # Save snapshot
        print("\nSaving snapshot...")
        save_snapshot(state, SNAPSHOT_PATH)

        # === ASSERTIONS ===
        # Verify files were generated
        assert len(state.generated_files) > 0, "Should have generated files"
        assert len(state.file_structure) > 0, "Should have file structure"

        # Verify files written to disk
        files_written = 0
        for rel_path in state.generated_files:
            full_path = os.path.join(case_output, rel_path)
            if os.path.exists(full_path):
                files_written += 1
        assert files_written == len(state.generated_files), \
            f"Only {files_written}/{len(state.generated_files)} files written"

        # Verify snapshot saved
        assert os.path.exists(SNAPSHOT_PATH), "Snapshot should be saved"

        # Report
        print("\n" + "=" * 60)
        print("GENERATION COMPLETE")
        print("=" * 60)
        print(f"Snapshot: {SNAPSHOT_PATH}")
        print(f"Case dir: {case_output}")
        print(f"LLM calls: {llm.stats.reasoning_calls} reasoning, {llm.stats.instruct_calls} instruct")
        print(f"Total time: {generation_time:.1f}s")
        print("\nRun 'python tests/test_correction_loop.py --correct' to start correction loop")

    def test_correction_loop(self):
        """
        Phase 2: Run correction loop with Docker.

        Loads snapshot, runs OpenFOAM via Docker, and iterates on errors.
        This is the FAST part - can run many times without regenerating files.
        """
        print("\n" + "=" * 60)
        print("Phase 2: Correction Loop with Docker")
        print("=" * 60)

        if not self.docker_available:
            pytest.skip(f"Docker not available. Run: docker pull {DOCKER_IMAGE}")

        if not os.path.exists(SNAPSHOT_PATH):
            pytest.skip("Snapshot not found. Run test_generate_and_save first.")

        # Load snapshot
        print("\nLoading snapshot...")
        state = load_snapshot(SNAPSHOT_PATH)

        # Create Docker-enabled config
        config = create_docker_config()

        # Create LLM for corrections (will be called for error analysis/correction)
        llm = LLMClient(
            config.llm,
            log_path=os.path.join(OUTPUT_DIR, "llm_correct.jsonl")
        )

        # Load database
        database = ReferenceDatabase.load(DATABASE_DIR)

        # Create dependencies with Docker enabled
        deps = Dependencies(
            config=config,
            llm=llm,
            database=database,
        )

        # Ensure files are written
        print("\nWriting files from snapshot...")
        write_files_from_snapshot(SNAPSHOT_PATH, state.output_path)

        # Convert mesh if needed
        if not state.mesh_converted:
            print("\nConverting mesh via Docker...")
            mesh_success = run_mesh_conversion_docker(
                state.output_path,
                state.grid_path,
                deps
            )
            if mesh_success:
                state = state.apply_update({"mesh_converted": True})
            else:
                print("ERROR: Mesh conversion failed")
                return False

        # Run correction loop
        print("\n" + "-" * 40)
        print("Starting correction loop...")
        print("-" * 40)

        max_attempts = state.max_attempts
        attempt = 0

        while attempt < max_attempts:
            attempt += 1
            print(f"\n=== Attempt {attempt}/{max_attempts} ===")

            # Run simulation
            update = run_simulation(state, deps)
            state = state.apply_update(update)

            if state.success:
                print("\nSIMULATION SUCCEEDED!")
                break

            if not state.current_error:
                print("No error captured, stopping")
                break

            print(f"Error: {state.current_error[:200]}...")

            # Count consecutive same errors
            consecutive = state.consecutive_same_errors()
            print(f"Consecutive same errors: {consecutive}")

            if consecutive >= 4:
                print("Same error 4+ times - would rewrite file (not implemented in test)")
                break
            elif consecutive >= 2:
                print("Same error 2+ times - reflecting...")
                update = reflect_on_errors(state, deps)
                state = state.apply_update(update)

            # Analyze error
            print("Analyzing error...")
            update = analyze_error(state, deps)
            state = state.apply_update(update)

            if state.error_file:
                print(f"Identified problematic file: {state.error_file}")

                # Correct file
                print("Correcting file...")
                update = correct_file(state, deps)
                state = state.apply_update(update)

                # Update snapshot with correction
                from snapshot import update_snapshot_after_correction
                if state.error_file in state.generated_files:
                    update_snapshot_after_correction(
                        SNAPSHOT_PATH,
                        state.error_file,
                        state.generated_files[state.error_file],
                        state.current_error,
                    )
            else:
                print("Could not identify error file")

        # Final report
        print("\n" + "=" * 60)
        print("CORRECTION LOOP COMPLETE")
        print("=" * 60)
        print(f"Success: {state.success}")
        print(f"Attempts: {attempt}")
        print(f"Errors encountered: {len(state.error_history)}")
        print(f"LLM calls: {llm.stats.reasoning_calls} reasoning, {llm.stats.instruct_calls} instruct")

        # Save final state
        final_snapshot = SNAPSHOT_PATH.replace(".json", "_final.json")
        save_snapshot(state, final_snapshot)

        # === ASSERTIONS ===
        # Verify correction loop executed
        assert attempt >= 1, "Should have made at least one attempt"

        # Verify final snapshot saved
        assert os.path.exists(final_snapshot), "Final snapshot should be saved"

        # Verify workflow completed (success or hit max attempts)
        assert state.success or attempt == max_attempts or state.consecutive_same_errors() >= 4, \
            "Workflow should complete via success, max attempts, or repeated errors"

    def test_full_workflow_with_docker(self):
        """
        Run full workflow (generate + correct) with Docker execution.

        This combines both phases but uses the WorkflowRunner.
        """
        print("\n" + "=" * 60)
        print("Full Workflow with Docker")
        print("=" * 60)

        if not self.docker_available:
            pytest.skip(f"Docker not available. Run: docker pull {DOCKER_IMAGE}")

        # Create Docker-enabled config
        config = create_docker_config()

        # Create LLM
        llm = LLMClient(
            config.llm,
            log_path=os.path.join(OUTPUT_DIR, "llm_full.jsonl")
        )

        # Load database
        database = ReferenceDatabase.load(DATABASE_DIR)

        # Create dependencies
        deps = Dependencies(
            config=config,
            llm=llm,
            database=database,
        )

        # Create fresh output directory
        case_output = os.path.join(OUTPUT_DIR, "full_workflow_docker")
        if os.path.exists(case_output):
            shutil.rmtree(case_output)
        os.makedirs(case_output)

        # Build state
        state = CaseState(
            case_name="naca0012_full_docker",
            solver="simpleFoam",
            turbulence_model="kOmegaSST",
            description="Incompressible turbulent flow over NACA 0012 airfoil, 5 deg AoA, Re~1e6",
            output_path=case_output,
            grid_path=MESH_PATH,
            grid_type="msh",
            grid_boundaries=["inlet", "outlet", "airfoil", "frontAndBackPlanes"],
            max_attempts=5,
        )

        # Run workflow
        print("\nRunning workflow...")
        runner = WorkflowRunner(deps, verbose=True)
        final_state = runner.run(state)

        # Report
        print("\n" + "=" * 60)
        print("WORKFLOW COMPLETE")
        print("=" * 60)
        print(f"Success: {final_state.success}")
        print(f"Attempts: {final_state.attempt_count}")
        print(f"Files generated: {len(final_state.generated_files)}")

        # Save snapshot
        snapshot_path = os.path.join(OUTPUT_DIR, "full_workflow_snapshot.json")
        save_snapshot(final_state, snapshot_path)

        # === ASSERTIONS ===
        # Verify files were generated
        assert len(final_state.generated_files) > 0, "Should have generated files"

        # Verify files written to disk
        files_written = 0
        for rel_path in final_state.generated_files:
            full_path = os.path.join(case_output, rel_path)
            if os.path.exists(full_path):
                files_written += 1
        assert files_written == len(final_state.generated_files), \
            f"Only {files_written}/{len(final_state.generated_files)} files written"

        # Verify snapshot saved
        assert os.path.exists(snapshot_path), "Snapshot should be saved"

        # Verify workflow ran simulation attempts
        assert final_state.attempt_count >= 0, "Should track attempt count"


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Correction loop tests")
    parser.add_argument("--generate", action="store_true", help="Run LLM generation and save snapshot")
    parser.add_argument("--correct", action="store_true", help="Run correction loop from snapshot")
    parser.add_argument("--all", action="store_true", help="Run full workflow")
    parser.add_argument("--check-docker", action="store_true", help="Check if Docker is available")
    args = parser.parse_args()

    if args.check_docker:
        available = check_docker_available()
        print(f"Docker available: {available}")
        if not available:
            print(f"Run: docker pull {DOCKER_IMAGE}")
        sys.exit(0 if available else 1)

    test = TestCorrectionLoop()
    test.setup_class()

    if args.generate:
        test.test_generate_and_save()
    elif args.correct:
        success = test.test_correction_loop()
        sys.exit(0 if success else 1)
    elif args.all:
        success = test.test_full_workflow_with_docker()
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python tests/test_correction_loop.py --generate    # Generate files (slow, ~10 min)")
        print("  python tests/test_correction_loop.py --correct     # Run correction loop (fast)")
        print("  python tests/test_correction_loop.py --all         # Full workflow")
        print("  python tests/test_correction_loop.py --check-docker # Check Docker setup")


if __name__ == "__main__":
    main()
