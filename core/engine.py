"""
Core Engine - Main orchestrator for API test generation

This module was missing from the original codebase despite being imported
by core/__init__.py and run_agent_extended_tests.py.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from config import settings
from input_processing import InputProcessor
from llm import LlamaOrchestrator
from rag import RAGSystem
from reinforcement_learning import RLOptimizer
from test_execution.executor import TestExecutor
from feedback.feedback_loop import FeedbackLoop
from output.report_generator import ReportGenerator
from utils.validators import is_valid_test_case, is_valid_api_spec

logger = logging.getLogger(__name__)


@dataclass
class APITestRequest:
    """Request object for API test generation"""
    code_files: List[str]
    language: str
    endpoint_url: str
    test_types: Optional[List[str]] = None
    max_tests: int = 50
    include_edge_cases: bool = True
    auth_token: Optional[str] = None
    use_ssl: bool = False


class CoreEngine:
    """
    Core engine that orchestrates the full API testing workflow.

    Steps:
        1. Parse source code
        2. Build API specification
        3. Retrieve RAG context
        4. Generate tests via LLM agents
        5. Optimize with RL
        6. Execute tests
        7. Process feedback
    """

    def __init__(self):
        logger.info("Initializing Core Engine")

        self.input_processor = InputProcessor()
        self.rag_system = RAGSystem()
        self.rl_optimizer = RLOptimizer()
        self.test_executor = TestExecutor()
        self.report_generator = ReportGenerator()

        # Wire up feedback loop
        self.feedback_loop = FeedbackLoop()
        self.feedback_loop.set_rag_system(self.rag_system)
        self.feedback_loop.set_rl_optimizer(self.rl_optimizer)
        self.feedback_loop.set_knowledge_base(self.rag_system.knowledge_base)

    async def process_api(self, request: APITestRequest) -> Dict[str, Any]:
        """
        Process an API test request through the full pipeline.

        Args:
            request: APITestRequest with code files, language, endpoint URL, etc.

        Returns:
            Dict with status, results, and metrics.
        """
        logger.info(f"Processing API test request for {request.language}")
        start_time = datetime.now()
        stages_completed = []

        try:
            # Step 1: Parse source code
            logger.info("Step 1: Parsing source code...")
            parsed_data = self.input_processor.parse_code(
                request.code_files, request.language
            )
            stages_completed.append("parsing")

            # Step 2: Build API specification
            logger.info("Step 2: Building API specification...")
            if 'endpoints' in parsed_data and 'results' not in parsed_data:
                wrapped_data = {'results': [parsed_data]}
            else:
                wrapped_data = parsed_data

            api_spec = self.input_processor.build_specification(wrapped_data)
            validation_rules = self.input_processor.extract_validation_rules(wrapped_data)
            api_spec['validation_rules'] = validation_rules
            api_spec['business_logic'] = self.input_processor.extract_business_logic(wrapped_data)
            stages_completed.append("specification")

            if not is_valid_api_spec(api_spec):
                logger.warning("API specification validation failed, continuing anyway...")

            logger.info(
                f"Spec: {len(api_spec.get('endpoints', []))} endpoints, "
                f"{len(validation_rules)} rules"
            )

            # Step 3: Retrieve RAG context
            logger.info("Step 3: Retrieving RAG context...")
            context = await self._retrieve_context(api_spec, request)
            stages_completed.append("retrieval")

            # Step 4: Generate tests via LLM
            logger.info("Step 4: Generating tests via LLM agents...")
            test_suite = await self._generate_tests(api_spec, context, request)
            stages_completed.append("generation")

            all_tests = test_suite.get('test_cases', []) + test_suite.get('edge_cases', [])
            valid_tests = [t for t in all_tests if is_valid_test_case(t)]
            logger.info(f"Generated {len(valid_tests)} valid tests")

            # Step 5: Optimize with RL
            logger.info("Step 5: Optimizing with RL...")
            try:
                optimized_tests = await self.rl_optimizer.optimize(
                    self.rl_optimizer.create_state(valid_tests, api_spec),
                    valid_tests
                )
                stages_completed.append("optimization")
            except Exception as e:
                logger.warning(f"RL optimization failed: {e}, using unoptimized tests")
                optimized_tests = valid_tests

            # Step 6: Execute tests
            logger.info("Step 6: Executing tests...")
            execution_results = await self._execute_tests(
                optimized_tests, request
            )
            stages_completed.append("execution")

            # Step 7: Process feedback
            logger.info("Step 7: Processing feedback...")
            try:
                await self.feedback_loop.process_feedback(execution_results)
                stages_completed.append("feedback")
            except Exception as e:
                logger.warning(f"Feedback processing failed: {e}")

            # Build result
            duration = (datetime.now() - start_time).total_seconds()
            passed = sum(1 for r in execution_results if r.get('passed'))

            return {
                'status': 'success',
                'stages_completed': stages_completed,
                'results': {
                    'api_specification': api_spec,
                    'test_cases': optimized_tests,
                    'execution_results': execution_results,
                    'analysis': test_suite.get('analysis', {}),
                },
                'metrics': {
                    'total_duration': duration,
                    'stages_completed': len(stages_completed),
                    'total_stages': 7,
                    'tests_generated': len(valid_tests),
                    'tests_executed': len(execution_results),
                    'tests_passed': passed,
                    'test_pass_rate': passed / len(execution_results) if execution_results else 0,
                }
            }

        except Exception as e:
            logger.error(f"CoreEngine failed: {e}", exc_info=True)
            duration = (datetime.now() - start_time).total_seconds()
            return {
                'status': 'error',
                'error': str(e),
                'stages_completed': stages_completed,
                'metrics': {'total_duration': duration},
            }

    async def _retrieve_context(
        self, api_spec: Dict[str, Any], request: APITestRequest
    ) -> Dict[str, Any]:
        """Retrieve RAG context for the given API spec."""
        context = {
            'similar_tests': [],
            'edge_cases': [],
            'validation_patterns': [],
        }

        try:
            search_parts = []
            for ep in api_spec.get('endpoints', [])[:5]:
                search_parts.append(
                    f"{ep.get('http_method', '')} {ep.get('path', '')}"
                )
            if not search_parts:
                import os
                filename = os.path.basename(request.code_files[0]) if request.code_files else 'api'
                search_parts.append(filename)

            search_text = ' '.join(search_parts)
            embeddings = await self.rag_system.generate_embeddings(search_text)

            for retriever_method, key in [
                (self.rag_system.retrieve_similar_tests, 'similar_tests'),
                (self.rag_system.retrieve_edge_cases, 'edge_cases'),
                (self.rag_system.retrieve_validation_patterns, 'validation_patterns'),
            ]:
                try:
                    results = await retriever_method(embeddings, k=10)
                    if results:
                        context[key] = results
                except Exception as e:
                    logger.warning(f"RAG retrieval for {key} failed: {e}")

        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")

        total = sum(len(v) for v in context.values())
        logger.info(f"RAG context: {total} items retrieved")
        return context

    async def _generate_tests(
        self,
        api_spec: Dict[str, Any],
        context: Dict[str, Any],
        request: APITestRequest,
    ) -> Dict[str, Any]:
        """Generate tests using LLM orchestrator."""
        config = {
            'max_tests': request.max_tests,
            'include_edge_cases': request.include_edge_cases,
        }

        async with LlamaOrchestrator() as orchestrator:
            if not await orchestrator.client.check_connection():
                raise RuntimeError(
                    "Cannot connect to LLM API. "
                    "Check GROQ_API_KEY and base URL in config."
                )
            return await orchestrator.generate_test_suite(api_spec, context, config)

    async def _execute_tests(
        self,
        test_cases: List[Dict[str, Any]],
        request: APITestRequest,
    ) -> List[Dict[str, Any]]:
        """Execute test cases against the target API."""
        if not test_cases:
            logger.warning("No test cases to execute")
            return []

        results = []
        for idx, test in enumerate(test_cases, 1):
            try:
                if request.auth_token:
                    test['auth_token'] = request.auth_token

                logger.info(
                    f"Executing test {idx}/{len(test_cases)}: "
                    f"{test.get('name', 'unknown')}"
                )
                result = await self.test_executor.execute_test(
                    test, request.endpoint_url
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Test execution failed: {e}")
                results.append({
                    'test': test,
                    'name': test.get('name', 'unknown'),
                    'passed': False,
                    'error': str(e),
                })

        passed = sum(1 for r in results if r.get('passed'))
        logger.info(f"Execution complete: {passed}/{len(results)} passed")
        return results