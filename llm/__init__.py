"""
LLM integration module for test generation
"""
import asyncio
from .agents.data_generator import DataGeneratorAgent
from .agents.edge_case_agent import EdgeCaseAgent
from .agents.report_writer import ReportWriterAgent
from .agents.test_designer import TestDesignerAgent
from .llama_client import LlamaClient
from .agents.analyzer_agent import AnalyzerAgent
import logging

logger = logging.getLogger(__name__)

class LlamaOrchestrator:
    """Main orchestrator for LLM-based test generation"""

    def __init__(self):
        self.client = None
        self.agents = {}

    async def __aenter__(self):
        """Initialize client with context manager"""
        self.client = LlamaClient()
        await self.client.__aenter__()
        self.agents = {
            'analyzer': AnalyzerAgent(self.client),
            'test_designer': TestDesignerAgent(self.client),
            'edge_case': EdgeCaseAgent(self.client),
            'data_generator': DataGeneratorAgent(self.client),
            'report_writer': ReportWriterAgent(self.client)
        }
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup client session"""
        if self.client:
            await self.client.__aexit__(exc_type, exc_val, exc_tb)

    async def generate_test_suite(self, api_spec, context, config):
        """Generate complete test suite using multiple agents"""
        if not self.client or not self.client.session:
            raise RuntimeError("LlamaOrchestrator must be used as async context manager")

        try:
            # Analyze API
            logger.info("Calling analyzer agent...")
            analysis = await asyncio.wait_for(
                self.agents['analyzer'].analyze(api_spec, context),
                timeout=60
            )

            # Design tests
            logger.info("Calling test_designer agent...")
            test_designer_result = await asyncio.wait_for(
                self.agents['test_designer'].design_tests(analysis, context),
                timeout=180
            )

            # ✅ FIX: Extract list from dict
            test_cases = test_designer_result.get('happy_path_tests', [])

            logger.info(f"Generated {len(test_cases)} test cases")

            # Generate edge cases (optional - can skip if too slow)
            edge_cases = []
            try:
                logger.info("Calling edge_case agent...")
                edge_cases = await asyncio.wait_for(
                    self.agents['edge_case'].generate_edge_cases(api_spec, analysis),
                    timeout=120
                )
                logger.info(f"Generated {len(edge_cases)} edge cases")
            except asyncio.TimeoutError:
                logger.warning("Edge case generation timed out, skipping")
            except Exception as e:
                logger.warning(f"Edge case generation failed: {e}, skipping")

            # ✅ FIX: Now both are lists, can add them
            all_tests = test_cases + edge_cases

            # Skip data generator for now (not critical)
            logger.info("Calling data_generator agent...")
            test_data = {}
            try:
                test_data = await asyncio.wait_for(
                    self.agents['data_generator'].generate_data(all_tests, api_spec),
                    timeout=60
                )
            except Exception as e:
                logger.warning(f"Data generation failed: {e}, skipping")

            return {
                'analysis': analysis,
                'test_cases': test_cases,
                'edge_cases': edge_cases,
                'test_data': test_data
            }

        except asyncio.TimeoutError as e:
            logger.error(f"LLM agent timed out: {e}")
            raise RuntimeError(f"LLM generation timed out. Check if LM Studio is running")
        except Exception as e:
            logger.error(f"Error in test suite generation: {e}")
            raise


__all__ = [
    'LlamaOrchestrator',
    'LlamaClient',
    'AnalyzerAgent',
    'TestDesignerAgent',
    'EdgeCaseAgent',
    'DataGeneratorAgent',
    'ReportWriterAgent'
]