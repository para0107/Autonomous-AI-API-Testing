"""
LLM Orchestrator - Single source of truth for LLM agent coordination.

Fixes:
- LlamaClient uses __aenter__/__aexit__ (async context manager), NOT initialize()/close()
- Uses AgentManager internally (no dual orchestration path)
- Properly passes llama_client to AgentManager
- Consistent async context manager
"""

import logging
from typing import Dict, List, Any, Optional

from llm.llama_client import LlamaClient
from core.agent_manager import AgentManager

logger = logging.getLogger(__name__)


class LlamaOrchestrator:
    """
    Orchestrates LLM-based test generation.

    This is the single entry point for agent-based generation.
    Uses AgentManager internally to coordinate agents.
    """

    def __init__(self):
        self.client = LlamaClient()
        self.agent_manager = None

    async def __aenter__(self):
        """Initialize client and agents."""
        # FIX: LlamaClient uses __aenter__, not initialize()
        await self.client.__aenter__()
        # FIX: Pass the client so agents can actually use the LLM
        self.agent_manager = AgentManager(llama_client=self.client)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up."""
        # FIX: LlamaClient uses __aexit__, not close()
        if self.client:
            await self.client.__aexit__(exc_type, exc_val, exc_tb)

    async def generate_test_suite(self, api_spec: Dict[str, Any],
                                  context: Dict[str, Any],
                                  config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate a complete test suite using coordinated agents.

        Args:
            api_spec: Parsed API specification
            context: RAG context (similar_tests, edge_cases, validation_patterns)
            config: Generation config (max_tests, include_edge_cases, etc.)

        Returns:
            Dict with test_cases, edge_cases, analysis, test_data
        """
        if self.agent_manager is None:
            raise RuntimeError(
                "LlamaOrchestrator not initialized. Use 'async with' context manager."
            )

        config = config or {}
        logger.info("Starting LLM test suite generation")

        # Use agent manager for coordinated generation
        result = await self.agent_manager.orchestrate(api_spec, context)

        # Apply config limits
        max_tests = config.get('max_tests', 50)
        test_cases = result.get('test_cases', [])
        edge_cases = result.get('edge_cases', [])

        if len(test_cases) + len(edge_cases) > max_tests:
            # Prioritize: keep all high-priority, trim medium/low
            test_cases = self._apply_limit(test_cases, max_tests * 2 // 3)
            remaining = max_tests - len(test_cases)
            edge_cases = self._apply_limit(edge_cases, remaining)

        if not config.get('include_edge_cases', True):
            edge_cases = []

        result['test_cases'] = test_cases
        result['edge_cases'] = edge_cases

        logger.info(
            f"Generated {len(test_cases)} tests + {len(edge_cases)} edge cases"
        )

        return result

    @staticmethod
    def _apply_limit(tests: List[Dict], limit: int) -> List[Dict]:
        """Limit test count, keeping high-priority tests first."""
        if len(tests) <= limit:
            return tests

        # Sort by priority: high first, then medium, then low
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        sorted_tests = sorted(
            tests,
            key=lambda t: priority_order.get(t.get('priority', 'medium'), 1)
        )
        return sorted_tests[:limit]