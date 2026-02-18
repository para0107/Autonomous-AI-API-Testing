"""
Feedback loop for continuous learning
"""

import logging
from typing import Dict, List, Any
import asyncio

logger = logging.getLogger(__name__)


class FeedbackLoop:
    """Manages feedback from test execution for continuous improvement"""

    def __init__(self):
        self.rag_system = None
        self.rl_optimizer = None
        self.knowledge_base = None

    def set_rag_system(self, rag_system):
        """Set RAG system for updates"""
        self.rag_system = rag_system

    def set_rl_optimizer(self, rl_optimizer):
        """Set RL optimizer for training"""
        self.rl_optimizer = rl_optimizer

    def set_knowledge_base(self, knowledge_base):
        """Set knowledge base for updates"""
        self.knowledge_base = knowledge_base

    async def update_rag(self, execution_results: List[Dict[str, Any]]):
        """
        Update RAG system with new patterns from execution

        Args:
            execution_results: Test execution results
        """
        if not self.rag_system:
            logger.warning("RAG system not set, skipping update")
            return

        logger.info("Updating RAG system with execution results")

        # Extract successful patterns
        successful_tests = [r for r in execution_results if r.get('passed', False)]

        if successful_tests:
            # Index successful test patterns
            await self._index_successful_patterns(successful_tests)

        # Extract failure patterns
        failed_tests = [r for r in execution_results if not r.get('passed', False)]

        if failed_tests:
            # Index bug patterns
            await self._index_bug_patterns(failed_tests)

        # Extract new edge cases discovered
        edge_cases = self._extract_edge_cases(execution_results)
        if edge_cases:
            await self._index_edge_cases(edge_cases)

    async def _index_successful_patterns(self, successful_tests: List[Dict[str, Any]]):
        """Index successful test patterns"""
        try:
            documents = []
            for test in successful_tests:
                doc = {
                    'id': f"success_{test.get('name', 'unknown')}",
                    'type': 'test_case',
                    'content': self._format_test_for_indexing(test),
                    'metadata': {
                        'test_type': test.get('test_type'),
                        'endpoint': test.get('endpoint'),
                        'method': test.get('method'),
                        'passed': True,
                        'execution_time': test.get('execution_time')
                    }
                }
                documents.append(doc)

            # Index in RAG system
            if hasattr(self.rag_system, 'index_test_cases'):
                await self.rag_system.index_test_cases(documents)

            logger.info(f"Indexed {len(documents)} successful test patterns")

        except Exception as e:
            logger.error(f"Failed to index successful patterns: {str(e)}")

    async def _index_bug_patterns(self, failed_tests: List[Dict[str, Any]]):
        """Index bug patterns from failed tests"""
        try:
            if not self.knowledge_base:
                return

            for test in failed_tests:
                bug_pattern = {
                    'test_name': test.get('name'),
                    'endpoint': test.get('endpoint'),
                    'method': test.get('method'),
                    'test_type': test.get('test_type'),
                    'error': test.get('error'),
                    'expected': test.get('expected'),
                    'actual': test.get('actual'),
                    'test_data': test.get('request_data')
                }

                self.knowledge_base.add_knowledge('bug_patterns', bug_pattern)

            logger.info(f"Indexed {len(failed_tests)} bug patterns")

        except Exception as e:
            logger.error(f"Failed to index bug patterns: {str(e)}")

    async def _index_edge_cases(self, edge_cases: List[Dict[str, Any]]):
        """Index discovered edge cases"""
        try:
            if not self.knowledge_base:
                return

            for edge_case in edge_cases:
                self.knowledge_base.add_knowledge('edge_cases', edge_case)

            logger.info(f"Indexed {len(edge_cases)} edge cases")

        except Exception as e:
            logger.error(f"Failed to index edge cases: {str(e)}")

    def _extract_edge_cases(self, execution_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract edge cases from execution results"""
        edge_cases = []

        for result in execution_results:
            if result.get('test_type') == 'edge_case':
                edge_case = {
                    'description': result.get('name'),
                    'endpoint': result.get('endpoint'),
                    'method': result.get('method'),
                    'input_data': result.get('request_data'),
                    'expected_behavior': result.get('expected'),
                    'actual_behavior': result.get('actual'),
                    'discovered': True
                }
                edge_cases.append(edge_case)

        return edge_cases

    def _format_test_for_indexing(self, test: Dict[str, Any]) -> str:
        """Format test case for indexing"""
        parts = [
            f"Test: {test.get('name')}",
            f"Type: {test.get('test_type')}",
            f"Endpoint: {test.get('method')} {test.get('endpoint')}",
        ]

        if test.get('request_data'):
            parts.append(f"Input: {test['request_data']}")

        if test.get('assertions'):
            parts.append("Assertions:")
            for assertion in test['assertions']:
                parts.append(f"  - {assertion}")

        return '\n'.join(parts)

    async def update_rl_model(self, execution_results: List[Dict[str, Any]]):
        """
        Update RL model with rewards from execution

        Args:
            execution_results: Test execution results
        """
        if not self.rl_optimizer:
            logger.warning("RL optimizer not set, skipping update")
            return

        logger.info("Updating RL model with execution feedback")

        try:
            # Calculate metrics for reward calculation
            metrics = self._calculate_metrics(execution_results)

            # Calculate reward
            if hasattr(self.rl_optimizer, 'reward_calculator'):
                reward = self.rl_optimizer.reward_calculator.calculate_reward(
                    execution_results, metrics
                )

                logger.info(f"Calculated reward: {reward}")

                # Store experience and trigger training
                # (In a real scenario, we'd have state/action from test selection)
                # This is a simplified version

        except Exception as e:
            logger.error(f"Failed to update RL model: {str(e)}")

    def _calculate_metrics(self, execution_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate metrics from execution results"""
        total = len(execution_results)
        passed = sum(1 for r in execution_results if r.get('passed', False))
        failed = total - passed

        # Count bugs found (failed security/validation tests)
        bugs_found = sum(
            1 for r in execution_results
            if not r.get('passed', False) and
            r.get('test_type') in ['security', 'validation']
        )

        # Count edge cases covered
        edge_cases = sum(
            1 for r in execution_results
            if r.get('test_type') in ['edge_case', 'boundary']
        )

        # Count unique scenarios
        unique_scenarios = len(set(
            (r.get('endpoint'), r.get('method'), r.get('test_type'))
            for r in execution_results
        ))

        return {
            'total_tests': total,
            'passed_tests': passed,
            'failed_tests': failed,
            'bugs_found': bugs_found,
            'edge_cases_covered': edge_cases,
            'unique_scenarios': unique_scenarios,
            'code_coverage': 0,  # Would need actual coverage data
            'false_positives': 0,  # Would need manual verification
            'api_errors': sum(1 for r in execution_results if r.get('error'))
        }

    async def detect_drift(self, execution_results: List[Dict[str, Any]]) -> bool:
        """
        Detect API drift (changes in API behavior)

        Args:
            execution_results: Recent test execution results

        Returns:
            True if drift detected
        """
        logger.info("Checking for API drift")

        try:
            # Simple drift detection: compare with historical data
            # In a real implementation, this would use more sophisticated methods

            current_pass_rate = sum(
                1 for r in execution_results if r.get('passed', False)
            ) / len(execution_results) if execution_results else 0

            # Get historical pass rate from knowledge base
            historical_pass_rate = self._get_historical_pass_rate()

            if historical_pass_rate is None:
                # No historical data
                return False

            # Check for significant deviation (>20%)
            drift_threshold = 0.20
            deviation = abs(current_pass_rate - historical_pass_rate)

            if deviation > drift_threshold:
                logger.warning(
                    f"API drift detected: pass rate changed from "
                    f"{historical_pass_rate:.2%} to {current_pass_rate:.2%}"
                )
                return True

            return False

        except Exception as e:
            logger.error(f"Drift detection failed: {str(e)}")
            return False

    def _get_historical_pass_rate(self) -> float:
        """Get historical pass rate from knowledge base"""
        # This would retrieve historical data
        # For now, return None (no historical data)
        return None

    async def process_feedback(self, execution_results: List[Dict[str, Any]]):
        """
        Process all feedback from execution

        Args:
            execution_results: Test execution results
        """
        logger.info("Processing feedback from test execution")

        # Update RAG system
        await self.update_rag(execution_results)

        # Update RL model
        await self.update_rl_model(execution_results)

        # Detect drift
        drift_detected = await self.detect_drift(execution_results)

        if drift_detected:
            logger.warning("API drift detected - consider updating test suite")

        logger.info("Feedback processing complete")