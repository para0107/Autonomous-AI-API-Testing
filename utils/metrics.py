"""
Metrics Collection and Tracking Module

This module provides comprehensive metrics collection, tracking, and analysis
for the API testing agent system.
"""

import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import json


@dataclass
class TestMetrics:
    """Metrics for individual test execution"""
    test_id: str
    test_type: str
    endpoint: str
    method: str
    execution_time: float
    status: str  # passed, failed, error, skipped
    response_time: float
    status_code: Optional[int] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'test_id': self.test_id,
            'test_type': self.test_type,
            'endpoint': self.endpoint,
            'method': self.method,
            'execution_time': self.execution_time,
            'status': self.status,
            'response_time': self.response_time,
            'status_code': self.status_code,
            'error_message': self.error_message,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class AgentMetrics:
    """Metrics for agent execution"""
    agent_type: str
    execution_time: float
    tokens_used: int = 0
    api_calls: int = 0
    success: bool = True
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'agent_type': self.agent_type,
            'execution_time': self.execution_time,
            'tokens_used': self.tokens_used,
            'api_calls': self.api_calls,
            'success': self.success,
            'error_message': self.error_message,
            'timestamp': self.timestamp.isoformat()
        }


class MetricsCollector:
    """
    Centralized metrics collection and analysis system

    Tracks:
    - Test execution metrics
    - Agent performance metrics
    - API endpoint performance
    - Success/failure rates
    - Time-based analytics
    """

    def __init__(self):
        """Initialize metrics collector"""
        self.test_metrics: List[TestMetrics] = []
        self.agent_metrics: List[AgentMetrics] = []
        self.endpoint_stats: Dict[str, Dict] = defaultdict(lambda: {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_response_time': 0.0,
            'min_response_time': float('inf'),
            'max_response_time': 0.0,
            'status_codes': defaultdict(int)
        })
        self.start_time = time.time()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def record_test_metric(self, metric: TestMetrics) -> None:
        """
        Record a test execution metric

        Args:
            metric: TestMetrics object containing test execution data
        """
        self.test_metrics.append(metric)

        # Update endpoint statistics
        endpoint_key = f"{metric.method}:{metric.endpoint}"
        stats = self.endpoint_stats[endpoint_key]

        stats['total_requests'] += 1
        if metric.status == 'passed':
            stats['successful_requests'] += 1
        else:
            stats['failed_requests'] += 1

        stats['total_response_time'] += metric.response_time
        stats['min_response_time'] = min(stats['min_response_time'], metric.response_time)
        stats['max_response_time'] = max(stats['max_response_time'], metric.response_time)

        if metric.status_code:
            stats['status_codes'][metric.status_code] += 1

    def record_agent_metric(self, metric: AgentMetrics) -> None:
        """
        Record an agent execution metric

        Args:
            metric: AgentMetrics object containing agent execution data
        """
        self.agent_metrics.append(metric)

    def get_test_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for all tests

        Returns:
            Dictionary containing test summary metrics
        """
        if not self.test_metrics:
            return {
                'total_tests': 0,
                'passed': 0,
                'failed': 0,
                'errors': 0,
                'skipped': 0,
                'pass_rate': 0.0,
                'avg_execution_time': 0.0,
                'avg_response_time': 0.0
            }

        total = len(self.test_metrics)
        passed = sum(1 for m in self.test_metrics if m.status == 'passed')
        failed = sum(1 for m in self.test_metrics if m.status == 'failed')
        errors = sum(1 for m in self.test_metrics if m.status == 'error')
        skipped = sum(1 for m in self.test_metrics if m.status == 'skipped')

        total_exec_time = sum(m.execution_time for m in self.test_metrics)
        total_resp_time = sum(m.response_time for m in self.test_metrics)

        return {
            'total_tests': total,
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'skipped': skipped,
            'pass_rate': (passed / total * 100) if total > 0 else 0.0,
            'avg_execution_time': total_exec_time / total,
            'avg_response_time': total_resp_time / total,
            'total_execution_time': total_exec_time
        }

    def get_agent_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for all agents

        Returns:
            Dictionary containing agent summary metrics
        """
        if not self.agent_metrics:
            return {
                'total_executions': 0,
                'successful_executions': 0,
                'failed_executions': 0,
                'total_tokens': 0,
                'total_api_calls': 0,
                'avg_execution_time': 0.0
            }

        total = len(self.agent_metrics)
        successful = sum(1 for m in self.agent_metrics if m.success)
        total_tokens = sum(m.tokens_used for m in self.agent_metrics)
        total_api_calls = sum(m.api_calls for m in self.agent_metrics)
        total_exec_time = sum(m.execution_time for m in self.agent_metrics)

        return {
            'total_executions': total,
            'successful_executions': successful,
            'failed_executions': total - successful,
            'success_rate': (successful / total * 100) if total > 0 else 0.0,
            'total_tokens': total_tokens,
            'total_api_calls': total_api_calls,
            'avg_execution_time': total_exec_time / total if total > 0 else 0.0,
            'total_execution_time': total_exec_time
        }

    def get_endpoint_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed statistics for each endpoint

        Returns:
            Dictionary mapping endpoints to their statistics
        """
        result = {}

        for endpoint, stats in self.endpoint_stats.items():
            total = stats['total_requests']
            if total > 0:
                result[endpoint] = {
                    'total_requests': total,
                    'successful_requests': stats['successful_requests'],
                    'failed_requests': stats['failed_requests'],
                    'success_rate': (stats['successful_requests'] / total * 100),
                    'avg_response_time': stats['total_response_time'] / total,
                    'min_response_time': stats['min_response_time'],
                    'max_response_time': stats['max_response_time'],
                    'status_codes': dict(stats['status_codes'])
                }

        return result

    def get_test_type_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """
        Get breakdown of tests by type

        Returns:
            Dictionary mapping test types to their statistics
        """
        type_stats = defaultdict(lambda: {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'avg_execution_time': 0.0
        })

        for metric in self.test_metrics:
            stats = type_stats[metric.test_type]
            stats['total'] += 1

            if metric.status == 'passed':
                stats['passed'] += 1
            elif metric.status == 'failed':
                stats['failed'] += 1
            elif metric.status == 'error':
                stats['errors'] += 1

            stats['avg_execution_time'] += metric.execution_time

        # Calculate averages and pass rates
        result = {}
        for test_type, stats in type_stats.items():
            total = stats['total']
            result[test_type] = {
                'total': total,
                'passed': stats['passed'],
                'failed': stats['failed'],
                'errors': stats['errors'],
                'pass_rate': (stats['passed'] / total * 100) if total > 0 else 0.0,
                'avg_execution_time': stats['avg_execution_time'] / total if total > 0 else 0.0
            }

        return result

    def get_agent_type_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """
        Get breakdown of agent executions by type

        Returns:
            Dictionary mapping agent types to their statistics
        """
        agent_stats = defaultdict(lambda: {
            'executions': 0,
            'successful': 0,
            'failed': 0,
            'total_tokens': 0,
            'total_api_calls': 0,
            'total_execution_time': 0.0
        })

        for metric in self.agent_metrics:
            stats = agent_stats[metric.agent_type]
            stats['executions'] += 1

            if metric.success:
                stats['successful'] += 1
            else:
                stats['failed'] += 1

            stats['total_tokens'] += metric.tokens_used
            stats['total_api_calls'] += metric.api_calls
            stats['total_execution_time'] += metric.execution_time

        # Calculate averages and success rates
        result = {}
        for agent_type, stats in agent_stats.items():
            total = stats['executions']
            result[agent_type] = {
                'executions': total,
                'successful': stats['successful'],
                'failed': stats['failed'],
                'success_rate': (stats['successful'] / total * 100) if total > 0 else 0.0,
                'avg_tokens': stats['total_tokens'] / total if total > 0 else 0,
                'avg_api_calls': stats['total_api_calls'] / total if total > 0 else 0,
                'avg_execution_time': stats['total_execution_time'] / total if total > 0 else 0.0
            }

        return result

    def get_session_duration(self) -> float:
        """
        Get total session duration in seconds

        Returns:
            Session duration in seconds
        """
        return time.time() - self.start_time

    def get_comprehensive_report(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics report

        Returns:
            Dictionary containing all metrics and statistics
        """
        return {
            'session_id': self.session_id,
            'session_duration': self.get_session_duration(),
            'test_summary': self.get_test_summary(),
            'agent_summary': self.get_agent_summary(),
            'endpoint_statistics': self.get_endpoint_statistics(),
            'test_type_breakdown': self.get_test_type_breakdown(),
            'agent_type_breakdown': self.get_agent_type_breakdown(),
            'timestamp': datetime.now().isoformat()
        }

    def export_to_json(self, filepath: str) -> None:
        """
        Export all metrics to JSON file

        Args:
            filepath: Path to output JSON file
        """
        report = self.get_comprehensive_report()

        # Add raw metrics data
        report['raw_test_metrics'] = [m.to_dict() for m in self.test_metrics]
        report['raw_agent_metrics'] = [m.to_dict() for m in self.agent_metrics]

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

    def reset(self) -> None:
        """Reset all metrics"""
        self.test_metrics.clear()
        self.agent_metrics.clear()
        self.endpoint_stats.clear()
        self.start_time = time.time()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")


# Global metrics collector instance
_global_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """
    Get the global metrics collector instance

    Returns:
        Global MetricsCollector instance
    """
    return _global_collector


def reset_metrics() -> None:
    """Reset the global metrics collector"""
    _global_collector.reset()