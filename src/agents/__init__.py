"""
Agents: Specialized agents for different stages of CFD workflow.

Available agents:
- SolvingAgent: Generates OpenFOAM case files and runs simulations
- MeshingAgent: Generates mesh from CAD geometry (placeholder)

Usage:
    from agents import SolvingAgent, MeshingAgent
    from agents.orchestrator import run_pipeline
"""

from agents.solving import SolvingAgent
from agents.meshing import MeshingAgent

__all__ = ["SolvingAgent", "MeshingAgent"]
