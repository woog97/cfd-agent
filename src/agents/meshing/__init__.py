"""
Meshing Agent: Generates OpenFOAM mesh from CAD geometry.

This agent handles:
- Reading CAD geometry (.step, .stl, .iges)
- Generating blockMeshDict for background mesh
- Generating snappyHexMeshDict for geometry-conforming mesh
- Running mesh generation with quality checks
- Error correction for meshing failures

Status: PLACEHOLDER - Not yet implemented.

Usage (future):
    from agents.meshing import MeshingAgent

    agent = MeshingAgent(config)
    result = agent.run(
        geometry_path="/path/to/geometry.step",
        boundaries={
            "inlet": ["face1", "face2"],
            "outlet": ["face3"],
            "walls": ["face4", "face5"],
        },
        refinement_level=2,
    )
"""

from agents.meshing.agent import MeshingAgent

__all__ = ["MeshingAgent"]
