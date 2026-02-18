"""
Analyzes test execution results
"""

import logging
from typing import Dict, List, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class ResultAnalyzer:
    """Analyzes test execution results for insights"""

    def __init__(self):
        pass

    def analyze(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze test results

        Args:
            results: List of test execution results

        Returns:
            Analysis report
        """
        analysis = {
            'summary': self._generate_summary(results),
            'by_test_type': self._analyze_by_type(results),
            'by_endpoint': self._analyze_by_endpoint(results),
            'failures': self._analyze_failures(results),
            'performance': self._analyze_performance(results),
            'coverage': self._analyze_coverage(results)
        }

        return analysis

    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics"""
        total = len(results)
        passed = sum(1 for r in results if r.get('passed', False))
        failed = total - passed

        total_time = sum(r.get('execution_time', 0) for r in results)
        avg_time = total_time / total if total > 0 else 0

        return {
            'total_tests': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': (passed / total * 100) if total > 0 else 0,
            'total_execution_time': round(total_time, 2),
            'average_execution_time': round(avg_time, 2)
        }

    def _analyze_by_type(self, results: List[Dict[str, Any]]) -> Dict[str, Dict]:
        """Analyze results grouped by test type"""
        by_type = defaultdict(lambda: {'total': 0, 'passed': 0, 'failed': 0})

        for result in results:
            test_type = result.get('test_type', 'unknown')
            by_type[test_type]['total'] += 1

            if result.get('passed', False):
                by_type[test_type]['passed'] += 1
            else:
                by_type[test_type]['failed'] += 1

        # Calculate pass rates
        for test_type in by_type:
            stats = by_type[test_type]
            if stats['total'] > 0:
                stats['pass_rate'] = round(stats['passed'] / stats['total'] * 100, 2)
            else:
                stats['pass_rate'] = 0

        return dict(by_type)

    def _analyze_by_endpoint(self, results: List[Dict[str, Any]]) -> Dict[str, Dict]:
        """Analyze results grouped by endpoint"""
        by_endpoint = defaultdict(lambda: {'total': 0, 'passed': 0, 'failed': 0})

        for result in results:
            endpoint = result.get('endpoint', 'unknown')
            by_endpoint[endpoint]['total'] += 1

            if result.get('passed', False):
                by_endpoint[endpoint]['passed'] += 1
            else:
                by_endpoint[endpoint]['failed'] += 1

        # Calculate pass rates
        for endpoint in by_endpoint:
            stats = by_endpoint[endpoint]
            if stats['total'] > 0:
                stats['pass_rate'] = round(stats['passed'] / stats['total'] * 100, 2)
            else:
                stats['pass_rate'] = 0

        return dict(by_endpoint)

    def _analyze_failures(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze failure patterns"""
        failures = [r for r in results if not r.get('passed', False)]

        if not failures:
            return {
                'count': 0,
                'common_errors': [],
                'critical_failures': []
            }

        # Group by error type
        error_counts = defaultdict(int)
        for failure in failures:
            error = failure.get('error', 'Unknown error')
            # Simplify error message
            error_type = error.split(':')[0] if ':' in error else error
            error_counts[error_type] += 1

        # Sort by frequency
        common_errors = sorted(
            [{'error': k, 'count': v} for k, v in error_counts.items()],
            key=lambda x: x['count'],
            reverse=True
        )

        # Identify critical failures
        critical = [
            f for f in failures
            if f.get('test_type') in ['authentication', 'security', 'validation']
        ]

        return {
            'count': len(failures),
            'common_errors': common_errors[:5],  # Top 5
            'critical_failures': len(critical),
            'failure_rate_by_type': self._calculate_failure_rates(failures)
        }

    def _calculate_failure_rates(self, failures: List[Dict]) -> Dict[str, float]:
        """Calculate failure rates by test type"""
        type_counts = defaultdict(int)
        for failure in failures:
            test_type = failure.get('test_type', 'unknown')
            type_counts[test_type] += 1

        return dict(type_counts)

    def _analyze_performance(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance metrics"""
        execution_times = [r.get('execution_time', 0) for r in results]

        if not execution_times:
            return {
                'average': 0,
                'min': 0,
                'max': 0,
                'slow_tests': []
            }

        avg_time = sum(execution_times) / len(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)

        # Identify slow tests (> 2x average)
        threshold = avg_time * 2
        slow_tests = [
            {
                'name': r.get('name'),
                'endpoint': r.get('endpoint'),
                'execution_time': r.get('execution_time')
            }
            for r in results
            if r.get('execution_time', 0) > threshold
        ]

        return {
            'average_time': round(avg_time, 2),
            'min_time': round(min_time, 2),
            'max_time': round(max_time, 2),
            'slow_tests_count': len(slow_tests),
            'slow_tests': sorted(slow_tests,
                                 key=lambda x: x['execution_time'],
                                 reverse=True)[:5]  # Top 5 slowest
        }

    def _analyze_coverage(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze test coverage"""
        unique_endpoints = set(r.get('endpoint') for r in results)
        unique_methods = set(r.get('method') for r in results)
        unique_test_types = set(r.get('test_type') for r in results)

        # Count assertions
        total_assertions = sum(
            len(r.get('assertions', [])) for r in results
        )

        passed_assertions = sum(
            sum(1 for a in r.get('assertions', []) if a.get('passed', False))
            for r in results
        )

        return {
            'endpoints_tested': len(unique_endpoints),
            'http_methods_tested': len(unique_methods),
            'test_types_used': len(unique_test_types),
            'total_assertions': total_assertions,
            'passed_assertions': passed_assertions,
            'assertion_pass_rate': round(
                (passed_assertions / total_assertions * 100)
                if total_assertions > 0 else 0,
                2
            )
        }

    def generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []

        summary = analysis.get('summary', {})
        failures = analysis.get('failures', {})
        performance = analysis.get('performance', {})

        # Pass rate recommendations
        pass_rate = summary.get('pass_rate', 0)
        if pass_rate < 70:
            recommendations.append(
                "Critical: Pass rate is below 70%. Immediate investigation required."
            )
        elif pass_rate < 90:
            recommendations.append(
                "Warning: Pass rate is below 90%. Review failed tests and fix issues."
            )

        # Critical failures
        critical_count = failures.get('critical_failures', 0)
        if critical_count > 0:
            recommendations.append(
                f"Critical: {critical_count} critical test failures detected "
                "(authentication/security/validation). Address immediately."
            )

        # Performance issues
        slow_tests = performance.get('slow_tests_count', 0)
        if slow_tests > 5:
            recommendations.append(
                f"Performance: {slow_tests} slow tests detected. "
                "Consider optimizing API or test implementation."
            )

        # Common errors
        common_errors = failures.get('common_errors', [])
        if common_errors:
            top_error = common_errors[0]
            recommendations.append(
                f"Most common error: '{top_error['error']}' "
                f"({top_error['count']} occurrences). Focus debugging efforts here."
            )

        # Coverage recommendations
        coverage = analysis.get('coverage', {})
        test_types = coverage.get('test_types_used', 0)
        if test_types < 4:
            recommendations.append(
                "Coverage: Limited test type diversity. "
                "Consider adding more edge case and security tests."
            )

        return recommendations