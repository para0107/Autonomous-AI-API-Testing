"""
Report Writer Agent
"""

import logging
from typing import Dict, Any, List
from datetime import datetime
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ReportWriterAgent(BaseAgent):
    """Agent for generating test reports"""

    def __init__(self, llama_client):
        super().__init__(llama_client, 'report_writer')

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test report"""
        execution_results = input_data.get('execution_results', [])
        session = input_data.get('session', {})

        return await self.generate_report(execution_results, session)

    async def generate_report(self, execution_results: List[Dict[str, Any]],
                              session: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate QASE-style test report

        Args:
            execution_results: Test execution results
            session: Test session information

        Returns:
            Formatted test report
        """
        summary = self._generate_summary(execution_results)

        test_cases = []
        for result in execution_results:
            test_case = await self._generate_test_case_report(result)
            test_cases.append(test_case)

        recommendations = await self._generate_recommendations(execution_results)

        report = {
            'title': f"API Test Report - {session.get('id', 'Unknown Session')}",
            'generated_at': datetime.now().isoformat(),
            'summary': summary,
            'test_cases': test_cases,
            'recommendations': recommendations,
            'metadata': {
                'session_id': session.get('id'),
                'api_endpoint': session.get('request', {}).get('endpoint_url'),
                'total_duration': self._calculate_total_duration(execution_results)
            }
        }

        return report

    def _generate_summary(self, execution_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate test execution summary"""
        total = len(execution_results)
        passed = sum(1 for r in execution_results if r.get('passed', False))
        failed = total - passed

        by_type = {}
        for result in execution_results:
            test_type = result.get('test_type', 'unknown')
            if test_type not in by_type:
                by_type[test_type] = {'total': 0, 'passed': 0, 'failed': 0}
            by_type[test_type]['total'] += 1
            if result.get('passed', False):
                by_type[test_type]['passed'] += 1
            else:
                by_type[test_type]['failed'] += 1

        critical_failures = [
            r for r in execution_results
            if not r.get('passed', False) and
               r.get('test_type') in ['authentication', 'validation', 'security_edge_case']
        ]

        return {
            'total_tests': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': round(passed / total * 100, 2) if total > 0 else 0,
            'by_test_type': by_type,
            'critical_failures': len(critical_failures),
            'execution_time': self._calculate_total_duration(execution_results)
        }

    async def _generate_test_case_report(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate report for individual test case"""
        test_report = {
            'title': result.get('name', 'Unnamed Test'),
            'status': 'PASSED' if result.get('passed', False) else 'FAILED',
            'severity': self._determine_severity(result),
            'priority': self._determine_priority(result),
            'test_type': result.get('test_type', 'unknown'),
            'preconditions': await self._generate_preconditions(result),
            'steps': await self._generate_steps(result),
            'expected_result': result.get('expected', 'N/A'),
            'actual_result': result.get('actual', 'N/A'),
            'execution_time': result.get('execution_time', 0),
            'error': result.get('error') if not result.get('passed', False) else None,
            'attachments': {
                'request': result.get('request_data'),
                'response': result.get('response_data'),
                'logs': result.get('logs', [])
            }
        }

        if not result.get('passed', False):
            test_report['failure_analysis'] = await self._analyze_failure(result)

        return test_report

    async def _generate_preconditions(self, result: Dict[str, Any]) -> List[str]:
        """Generate test preconditions"""
        preconditions = []

        preconditions.append(f"API endpoint {result.get('endpoint', 'N/A')} is available")

        if result.get('test_type') != 'authentication':
            preconditions.append("Valid authentication token is available")

        if result.get('test_data'):
            preconditions.append("Test data is prepared and valid")

        if result.get('dependencies'):
            for dep in result['dependencies']:
                preconditions.append(f"Service {dep} is available")

        return preconditions

    async def _generate_steps(self, result: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate test steps"""
        steps = []

        steps.append({
            'action': 'Setup test environment',
            'expected': 'Environment is ready'
        })

        if result.get('test_data'):
            steps.append({
                'action': f"Prepare test data: {self._format_test_data(result['test_data'])}",
                'expected': 'Test data is valid'
            })

        steps.append({
            'action': f"Send {result.get('method', 'HTTP')} request to {result.get('endpoint', 'endpoint')}",
            'expected': f"Receive response with status {result.get('expected_status', '200')}"
        })

        if result.get('assertions'):
            for assertion in result['assertions']:
                steps.append({
                    'action': f"Validate: {assertion}",
                    'expected': 'Assertion passes'
                })

        # BUG FIX: cleanup step was inside the assertions `if` block — moved to always execute
        steps.append({
            'action': 'Clean up test data',
            'expected': 'Test environment is reset'
        })

        return steps

    async def _analyze_failure(self, result: Dict[str, Any]) -> str:
        """Analyze test failure and generate description"""
        prompt = f"""Analyze the following test failure and provide a brief explanation:

        Test: {result.get('name')}
        Type: {result.get('test_type')}
        Expected: {result.get('expected')}
        Actual: {result.get('actual')}
        Error: {result.get('error')}

        Provide a concise analysis of:
        1. Why the test failed
        2. Potential root cause
        3. Severity of the issue
        4. Recommended action

        Keep the response under 100 words."""

        analysis = await self.generate_with_retry(prompt)
        return analysis

    async def _generate_recommendations(self, execution_results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on test results"""
        failures = [r for r in execution_results if not r.get('passed', False)]

        if not failures:
            return ["All tests passed. Continue monitoring for regressions."]

        failure_types = {}
        for failure in failures:
            test_type = failure.get('test_type', 'unknown')
            if test_type not in failure_types:
                failure_types[test_type] = []
            failure_types[test_type].append(failure)

        prompt = f"""Based on the following test failures, provide actionable recommendations:

        Failure Summary:
        {self._format_failure_summary(failure_types)}

        Generate 3-5 specific recommendations for:
        1. Critical issues that need immediate attention
        2. Patterns indicating systematic problems
        3. Areas needing additional test coverage
        4. Performance or security concerns

        Return as a JSON array of strings."""

        recommendations = await self.generate_json_with_retry(prompt)

        if isinstance(recommendations, dict):
            recommendations = recommendations.get('recommendations', [])

        return recommendations

    def _determine_severity(self, result: Dict[str, Any]) -> str:
        """Determine test severity"""
        test_type = result.get('test_type', '')

        if test_type in ['authentication', 'security_edge_case']:
            return 'critical'
        elif test_type in ['validation', 'error_handling']:
            return 'major'
        elif test_type in ['boundary', 'edge_case']:
            return 'normal'
        else:
            return 'minor'

    def _determine_priority(self, result: Dict[str, Any]) -> str:
        """Determine test priority"""
        if not result.get('passed', False):
            severity = self._determine_severity(result)
            if severity == 'critical':
                return 'high'
            elif severity == 'major':
                return 'medium'
            else:
                return 'low'
        return 'low'

    def _format_test_data(self, test_data: Any) -> str:
        """Format test data for display"""
        if isinstance(test_data, dict):
            return ', '.join([f"{k}={v}" for k, v in list(test_data.items())[:3]])
        return str(test_data)[:100]

    def _format_failure_summary(self, failure_types: Dict[str, List]) -> str:
        """Format failure summary for prompt"""
        summary = []
        for test_type, failures in failure_types.items():
            summary.append(f"- {test_type}: {len(failures)} failures")
            for failure in failures[:2]:
                summary.append(f"  * {failure.get('name', 'Test')}: {failure.get('error', 'Unknown error')}")
        return '\n'.join(summary)

    def _calculate_total_duration(self, results: List[Dict[str, Any]]) -> float:
        """Calculate total test duration"""
        total = 0
        for result in results:
            total += result.get('execution_time', 0)
        return round(total, 2)