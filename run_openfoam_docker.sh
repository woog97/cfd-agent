#!/bin/bash
# Run ChatCFD OpenFOAM execution test in Docker container
#
# Usage:
#   ./run_openfoam_docker.sh              # Run the full test
#   ./run_openfoam_docker.sh --mesh-only  # Only convert mesh
#   ./run_openfoam_docker.sh --shell      # Open interactive shell
#
# This spawns a NEW container (won't interfere with existing containers)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="openfoam/openfoam11-paraview510:latest"
CONTAINER_NAME="chatcfd-openfoam-$$"

echo "=== ChatCFD OpenFOAM Docker Runner ==="
echo "Project dir: $SCRIPT_DIR"
echo "Container: $CONTAINER_NAME"
echo ""

# Check arguments
MODE="${1:-full}"

case "$MODE" in
    --mesh-only)
        echo "Mode: Mesh conversion only"
        docker run --rm --platform linux/amd64 \
            --name "$CONTAINER_NAME" \
            --entrypoint /bin/bash \
            -v "$SCRIPT_DIR:/home/openfoam/ChatCFD" \
            -w /home/openfoam/ChatCFD \
            "$IMAGE" \
            -c '
                source /opt/openfoam11/etc/bashrc
                echo "Converting mesh: grids/naca0012.msh"

                rm -rf run_chatcfd/docker_mesh_test
                mkdir -p run_chatcfd/docker_mesh_test/system
                mkdir -p run_chatcfd/docker_mesh_test/constant

                cat > run_chatcfd/docker_mesh_test/system/controlDict << EOF
FoamFile
{
    format      ascii;
    class       dictionary;
    object      controlDict;
}
application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1000;
deltaT          1;
writeControl    timeStep;
writeInterval   100;
EOF

                cd run_chatcfd/docker_mesh_test
                fluentMeshToFoam ../../grids/naca0012.msh

                echo ""
                echo "=== Boundary file ==="
                cat constant/polyMesh/boundary
            '
        ;;

    --shell)
        echo "Mode: Interactive shell"
        echo "OpenFOAM environment will be sourced automatically."
        echo "Type 'exit' to quit."
        echo ""
        docker run --rm -it --platform linux/amd64 \
            --name "$CONTAINER_NAME" \
            --entrypoint /bin/bash \
            -v "$SCRIPT_DIR:/home/openfoam/ChatCFD" \
            -w /home/openfoam/ChatCFD \
            "$IMAGE" \
            -c 'source /opt/openfoam11/etc/bashrc && exec bash'
        ;;

    --full|*)
        echo "Mode: Full OpenFOAM execution test"
        echo ""

        # Check if snapshot exists
        SNAPSHOT="$SCRIPT_DIR/src/tests/fixtures/naca0012_case_snapshot.json"
        if [ ! -f "$SNAPSHOT" ]; then
            echo "ERROR: Snapshot not found at: $SNAPSHOT"
            echo "Run 'python src/tests/test_e2e_naca0012.py' first to generate it."
            exit 1
        fi

        echo "Using snapshot: $SNAPSHOT"
        echo ""

        docker run --rm --platform linux/amd64 \
            --name "$CONTAINER_NAME" \
            --entrypoint /bin/bash \
            -v "$SCRIPT_DIR:/home/openfoam/ChatCFD" \
            -w /home/openfoam/ChatCFD \
            -e OPENFOAM_PATH=/opt/openfoam11 \
            "$IMAGE" \
            -c '
                source /opt/openfoam11/etc/bashrc

                echo "=== OpenFOAM Environment ==="
                echo "WM_PROJECT_DIR: $WM_PROJECT_DIR"
                echo "simpleFoam: $(which simpleFoam)"
                echo ""

                # Install Python dependencies if needed
                if ! python3 -c "import openai" 2>/dev/null; then
                    echo "Installing Python dependencies..."
                    pip3 install openai pdfplumber PyFoam sentence-transformers --quiet
                fi

                echo "=== Running OpenFOAM execution test ==="
                cd /home/openfoam/ChatCFD

                # Update config to use OpenFOAM 11 path
                python3 -c "
import json
with open(\"inputs/chatcfd_config.json\", \"r\") as f:
    config = json.load(f)
config[\"OpenFOAM_path\"] = \"/opt/openfoam11\"
config[\"OpenFOAM_tutorials_path\"] = \"/opt/openfoam11/tutorials\"
with open(\"inputs/chatcfd_config.json\", \"w\") as f:
    json.dump(config, f, indent=4)
print(\"Updated config to use OpenFOAM 11\")
"

                python3 src/tests/test_openfoam_execution.py --runs 1
            '
        ;;
esac

echo ""
echo "Container $CONTAINER_NAME finished."
