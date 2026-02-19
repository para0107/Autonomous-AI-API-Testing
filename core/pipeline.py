"""
Test generation pipeline orchestrator
"""

import logging
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import asyncio
import json

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
class PipelineStage:
    """Represents a stage in the test generation pipeline"""
    name: str
    function: callable
    required: bool = True
    timeout: int = 60


class TestGenerationPipeline:
    """Manages the complete test generation pipeline"""

    def __init__(self):
        logger.info("Initializing Test Generation Pipeline")

        # Own all components directly
        self.input_processor = InputProcessor()
        self.rag_system = RAGSystem()
        self.rl_optimizer = RLOptimizer()
        self.test_executor = TestExecutor()
        self.report_generator = ReportGenerator()

        # FIX: Wire up feedback loop with actual dependencies
        self.feedback_loop = FeedbackLoop()
        self.feedback_loop.set_rag_system(self.rag_system)
        self.feedback_loop.set_rl_optimizer(self.rl_optimizer)
        self.feedback_loop.set_knowledge_base(self.rag_system.knowledge_base)

        self.stages = self._define_stages()
        self.stage_results = {}
        self.pipeline_metrics = {}

    def _define_stages(self) -> List[PipelineStage]:
        """Define pipeline stages"""
        return [
            PipelineStage("validation", self._validate_input, required=True, timeout=10),
            PipelineStage("parsing", self._parse_code, required=True, timeout=30),
            PipelineStage("analysis", self._analyze_api, required=True, timeout=60),
            PipelineStage("retrieval", self._retrieve_context, required=True, timeout=30),
            PipelineStage("generation", self._generate_tests, required=True, timeout=180),
            PipelineStage("optimization", self._optimize_tests, required=False, timeout=60),
            PipelineStage("execution", self._execute_tests, required=True, timeout=300),
            PipelineStage("feedback", self._process_feedback, required=False, timeout=30),
            PipelineStage("reporting", self._generate_report, required=True, timeout=30)
        ]

    async def run(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute complete pipeline"""
        logger.info("Starting test generation pipeline")

        self._initialize_pipeline(request)

        start_time = datetime.now()
        stages_completed = []

        try:
            for stage in self.stages:
                try:
                    logger.info(f"Executing stage: {stage.name}")

                    await asyncio.wait_for(
                        self._execute_stage(stage, request),
                        timeout=stage.timeout
                    )

                    stages_completed.append(stage.name)

                except asyncio.TimeoutError:
                    error_msg = f"Stage {stage.name} timed out after {stage.timeout}s"
                    logger.error(error_msg)
                    if stage.required:
                        raise Exception(error_msg)
                    logger.warning(f"Skipping optional stage: {stage.name}")

                except Exception as e:
                    logger.error(f"Stage {stage.name} failed: {str(e)}")
                    if stage.required:
                        raise
                    logger.warning(f"Skipping optional stage: {stage.name}")

            self._calculate_metrics(start_time)
            results = self._finalize_pipeline()

            return {
                'status': 'success',
                'stages_completed': stages_completed,
                'results': results,
                'metrics': self.pipeline_metrics
            }

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'stages_completed': stages_completed,
                'metrics': self.pipeline_metrics
            }

    async def _execute_stage(self, stage: PipelineStage, request: Dict[str, Any]):
        """Execute a single pipeline stage"""
        stage_start = datetime.now()

        try:
            result = await stage.function(request)
            self.stage_results[stage.name] = result
            duration = (datetime.now() - stage_start).total_seconds()
            logger.info(f"Stage {stage.name} completed in {duration:.2f}s")

        except Exception as e:
            logger.error(f"Stage {stage.name} error: {str(e)}", exc_info=True)
            raise

    async def _validate_input(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data"""
        required_fields = ['code_files', 'language', 'endpoint_url']
        for field_name in required_fields:
            if field_name not in request:
                raise ValueError(f"Missing required field: {field_name}")

        if request['language'] not in settings.SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {request['language']}")

        for file_path in request['code_files']:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Code file not found: {file_path}")

        return {'validation': 'passed'}

    async def _parse_code(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Parse source code"""
        logger.info("Analyzing API code")
        parsed_data = self.input_processor.parse_code(
            request['code_files'],
            request['language']
        )
        return parsed_data

    async def _analyze_api(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze API specification"""
        parsed_data = self.stage_results.get('parsing', {})

        # Wrap parsed_data if needed
        if 'endpoints' in parsed_data and 'results' not in parsed_data:
            wrapped_data = {'results': [parsed_data]}
        else:
            wrapped_data = parsed_data

        api_spec = self.input_processor.build_specification(wrapped_data)
        validation_rules = self.input_processor.extract_validation_rules(wrapped_data)
        api_spec['validation_rules'] = validation_rules
        api_spec['business_logic'] = self.input_processor.extract_business_logic(wrapped_data)

        logger.info(
            f"API Spec: {len(api_spec.get('endpoints', []))} endpoints, "
            f"{len(validation_rules)} rules, "
            f"{len(api_spec.get('models', []))} models"
        )

        if not is_valid_api_spec(api_spec):
            logger.warning("API specification validation failed, but continuing...")

        return api_spec

    async def _retrieve_context(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve context from RAG"""
        api_spec = self.stage_results.get('analysis', {})

        context = {
            'similar_tests': [],
            'edge_cases': [],
            'validation_patterns': []
        }

        try:
            search_parts = []

            if api_spec.get('endpoints'):
                for ep in api_spec['endpoints'][:5]:
                    search_parts.append(f"{ep.get('http_method', '')} {ep.get('path', '')}")

            if api_spec.get('controllers'):
                for ctrl in api_spec['controllers']:
                    search_parts.append(ctrl.get('name', ''))

            if api_spec.get('models'):
                for model in api_spec['models'][:3]:
                    search_parts.append(model.get('name', ''))

            if not search_parts:
                code_file = request.get('code_files', [''])[0]
                filename = os.path.basename(code_file).replace('.cs', '').replace('Controller', '')
                search_parts.append(filename)

            search_text = ' '.join(search_parts)
            logger.info(f"RAG search text: {search_text[:150]}...")

            embeddings = await self.rag_system.generate_embeddings(search_text)

            retrieval_map = {
                'similar_tests': self.rag_system.retrieve_similar_tests,
                'edge_cases': self.rag_system.retrieve_edge_cases,
                'validation_patterns': self.rag_system.retrieve_validation_patterns,
            }

            for key, retriever_method in retrieval_map.items():
                try:
                    results = await retriever_method(query_embedding=embeddings, k=10)  # ← FIXED
                    if results:
                        context[key] = results
                        logger.info(f"Retrieved {len(results)} {key}")
                    else:
                        logger.warning(f"No {key} found")
                except Exception as e:
                    logger.warning(f"Retrieval for {key} failed: {e}")

        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}", exc_info=True)

        total = sum(len(v) for v in context.values())
        logger.info(f"RAG retrieval complete: {total} total items")
        return context

    async def _generate_tests(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate test cases using LLM agents"""
        api_spec = self.stage_results.get('analysis', {})
        context = self.stage_results.get('retrieval', {})

        config = {
            'max_tests': request.get('max_tests', 50),
            'include_edge_cases': request.get('include_edge_cases', True)
        }

        async with LlamaOrchestrator() as orchestrator:
            logger.info("Checking LLM API connection...")
            if not await orchestrator.client.check_connection():
                raise RuntimeError(
                    "Cannot connect to the LLM API. "
                    "Check that GROQ_API_KEY is set correctly."
                )
            logger.info("LLM API connection successful")

            result = await orchestrator.generate_test_suite(api_spec, context, config)

        test_cases = result.get('test_cases', [])
        edge_cases = result.get('edge_cases', [])

        # Filter malformed edge cases
        valid_edge_cases = []
        for ec in edge_cases:
            if 'test_case' in ec and 'method' not in ec:
                logger.warning(f"Skipping malformed edge case: {ec.get('test_case', 'unknown')}")
                continue
            valid_edge_cases.append(ec)

        all_tests = test_cases + valid_edge_cases

        # Store analysis and test data for later use
        if 'analysis' in result:
            self.stage_results['llm_analysis'] = result['analysis']
        if 'test_data' in result:
            self.stage_results['test_data'] = result['test_data']

        valid_tests = [t for t in all_tests if is_valid_test_case(t)]

        logger.info(f"Generated {len(test_cases)} tests + {len(valid_edge_cases)} edge cases")
        logger.info(f"Total valid tests: {len(valid_tests)}")

        return valid_tests

    async def _optimize_tests(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Optimize test cases with RL"""
        test_cases = self.stage_results.get('generation', [])

        if not test_cases:
            logger.warning("No test cases to optimize")
            return []

        api_spec = self.stage_results.get('analysis', {})

        try:
            state = self.rl_optimizer.create_state(test_cases, api_spec)
            # FIX: optimize is async, always await it
            optimized_tests = await self.rl_optimizer.optimize(state, test_cases)
            logger.info(f"Optimized {len(optimized_tests)} tests using RL")
            return optimized_tests

        except Exception as e:
            logger.warning(f"RL optimization failed: {e}, using unoptimized tests")
            return test_cases

    async def _execute_tests(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute test cases"""
        # Get tests from optimization or generation
        test_cases = self.stage_results.get('optimization')
        if test_cases is None:
            test_cases = self.stage_results.get('generation', [])

        if not isinstance(test_cases, list):
            logger.error(f"Invalid test_cases type: {type(test_cases)}")
            test_cases = []

        if not test_cases:
            logger.warning("No test cases to execute")
            return []

        # Set auth if provided
        if request.get('auth_token'):
            self.test_executor.auth_token = request['auth_token']

        if not request.get('use_ssl', False):
            self.test_executor.ssl_verify = False

        logger.info(f"Executing {len(test_cases)} test cases...")
        results = []

        for idx, test in enumerate(test_cases, 1):
            try:
                logger.info(f"Executing test {idx}/{len(test_cases)}: {test.get('name', 'unknown')}")
                result = await self.test_executor.execute_test(test, request['endpoint_url'])
                results.append(result)
            except Exception as e:
                logger.error(f"Test execution failed for {test.get('name', 'unknown')}: {e}")
                results.append({
                    'test': test,
                    'name': test.get('name', 'unknown'),
                    'passed': False,
                    'error': str(e)
                })

        passed = sum(1 for r in results if r.get('passed'))
        logger.info(f"Execution complete: {passed}/{len(results)} tests passed")
        return results

    async def _process_feedback(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process execution feedback"""
        execution_results = self.stage_results.get('execution', [])

        if not execution_results:
            logger.warning("No execution results to process")
            return {'feedback_processed': False}

        # FIX: Use the properly wired feedback loop
        try:
            await self.feedback_loop.process_feedback(execution_results)
            logger.info("Feedback loop completed: RAG + RL updated")
        except Exception as e:
            logger.warning(f"Feedback processing failed: {e}")

        return {'feedback_processed': True}

    async def _generate_report(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final report"""
        execution_results = self.stage_results.get('execution', [])

        try:
            report = await self.report_generator.generate(
                execution_results,
                self.stage_results.get('analysis', {})
            )
            logger.info("Report generated successfully")
        except Exception as e:
            logger.warning(f"Report generation failed: {e}")
            report = {
                'summary': 'Report generation failed',
                'error': str(e)
            }

        return report

    def _initialize_pipeline(self, request: Dict[str, Any]):
        """Initialize pipeline state"""
        self.stage_results = {}
        self.pipeline_metrics = {}
        logger.info(f"Pipeline initialized for {request.get('language', 'unknown')} API")

    def _finalize_pipeline(self) -> Dict[str, Any]:
        """Finalize pipeline and prepare results"""
        return {
            'api_specification': self.stage_results.get('analysis'),
            'test_cases': self.stage_results.get('optimization',
                                                 self.stage_results.get('generation')),
            'execution_results': self.stage_results.get('execution'),
            'report': self.stage_results.get('reporting')
        }

    def _calculate_metrics(self, start_time: datetime):
        """Calculate pipeline metrics"""
        total_duration = (datetime.now() - start_time).total_seconds()

        self.pipeline_metrics = {
            'total_duration': total_duration,
            'stages_completed': len(self.stage_results),
            'total_stages': len(self.stages),
            'success_rate': len(self.stage_results) / len(self.stages),
            'tests_generated': len(self.stage_results.get('generation', [])),
            'tests_executed': len(self.stage_results.get('execution', [])),
        }

        execution_results = self.stage_results.get('execution', [])
        if execution_results:
            passed = sum(1 for r in execution_results if r.get('passed'))
            self.pipeline_metrics['test_pass_rate'] = (
                passed / len(execution_results) if execution_results else 0
            )