"""
Solving Agent: Generates OpenFOAM case files and runs simulations.

This agent handles:
- Determining required files based on solver/turbulence model
- Generating OpenFOAM configuration files via LLM
- Running simulations (local or Docker)
- Error correction loop

Usage:
    from agents.solving import SolvingAgent

    agent = SolvingAgent(config)
    result = agent.run(
        solver="simpleFoam",
        turbulence_model="kOmegaSST",
        description="Flow over airfoil...",
        mesh_path="/path/to/mesh.msh",
        boundaries=["inlet", "outlet", "wall"],
    )
"""

from agents.solving.agent import SolvingAgent

__all__ = ["SolvingAgent"]
