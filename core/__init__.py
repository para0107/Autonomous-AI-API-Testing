"""
core — Pipeline orchestration package.

Exports the two main pipeline classes and the agent manager.
"""

from core.engine import CoreEngine
from core.pipeline import TestGenerationPipeline
from core.agent_manager import AgentManager

__all__ = ['CoreEngine', 'TestGenerationPipeline', 'AgentManager']