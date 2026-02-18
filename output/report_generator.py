"""
Generates test reports in various formats
"""

import logging
import json
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path

from config import paths

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates comprehensive test reports"""

    def __init__(self):
        self.reports_dir = paths.REPORTS_DIR
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    async def generate_qase_report(self, execution_results: List[Dict[str, Any]],
                                   session: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate QASE-style test report

        Args:
            execution_results: Test execution results
            session: Test session information

        Returns:
            Formatted report
        """
        report = {
            'title': f"API Test Report - {session.get('id', 'Unknown')}",
            'generated_at': datetime.now().isoformat(),
            'session_id': session.get('id'),
            'summary': self._generate_summary(execution_results),
            'test_cases': self._format_test_cases(execution_results),
            'metadata': {
                'api_endpoint': session.get('request', {}).get('endpoint_url'),
                'total_duration': self._calculate_total_duration(execution_results),
                'environment': 'test'
            }
        }

        # Save report
        report_path = self._save_report(report, session.get('id'))
        report['report_path'] = str(report_path)

        return report

    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary section"""
        total = len(results)
        passed = sum(1 for r in results if r.get('passed', False))
        failed = total - passed

        # Group by test type
        by_type = {}
        for result in results:
            test_type = result.get('test_type', 'unknown')
            if test_type not in by_type:
                by_type[test_type] = {'total': 0, 'passed': 0, 'failed': 0}

            by_type[test_type]['total'] += 1
            if result.get('passed', False):
                by_type[test_type]['passed'] += 1
            else:
                by_type[test_type]['failed'] += 1

        # Critical failures
        critical_failures = [
            r for r in results
            if not r.get('passed', False) and
               r.get('test_type') in ['authentication', 'security', 'validation']
        ]

        return {
            'total_tests': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': round((passed / total * 100) if total > 0 else 0, 2),
            'by_test_type': by_type,
            'critical_failures': len(critical_failures),
            'execution_time': self._calculate_total_duration(results)
        }

    def _format_test_cases(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format test cases in QASE style"""
        formatted_tests = []

        for result in results:
            test_case = {
                'title': result.get('name', 'Unnamed Test'),
                'status': 'PASSED' if result.get('passed', False) else 'FAILED',
                'severity': self._determine_severity(result),
                'priority': self._determine_priority(result),
                'test_type': result.get('test_type', 'unknown'),
                'endpoint': result.get('endpoint'),
                'method': result.get('method'),
                'execution_time': result.get('execution_time', 0),
                'expected_status': result.get('expected_status'),
                'actual_status': result.get('actual_status'),
                'assertions': result.get('assertions', []),
                'error': result.get('error'),
                'timestamp': result.get('timestamp')
            }

            # Add attachments if available
            if 'request_data' in result or 'response_data' in result:
                test_case['attachments'] = {
                    'request': result.get('request_data'),
                    'response': result.get('response_data')
                }

            formatted_tests.append(test_case)

        return formatted_tests

    def _determine_severity(self, result: Dict[str, Any]) -> str:
        """Determine test severity"""
        test_type = result.get('test_type', '')

        if test_type in ['authentication', 'security']:
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
        return 'low'

    def _calculate_total_duration(self, results: List[Dict[str, Any]]) -> float:
        """Calculate total execution duration"""
        total = sum(r.get('execution_time', 0) for r in results)
        return round(total, 2)

    def _save_report(self, report: Dict[str, Any], session_id: str) -> Path:
        """Save report to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"report_{session_id}_{timestamp}.json"
        filepath = self.reports_dir / filename

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to {filepath}")
        return filepath

    def generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML report"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{report['title']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background: #f0f0f0; padding: 20px; margin-bottom: 20px; }}
        .test-case {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; }}
        .passed {{ border-left: 5px solid green; }}
        .failed {{ border-left: 5px solid red; }}
        .critical {{ background: #ffe0e0; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #4CAF50; color: white; }}
    </style>
</head>
<body>
    <h1>{report['title']}</h1>
    <p>Generated: {report['generated_at']}</p>

    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Tests:</strong> {report['summary']['total_tests']}</p>
        <p><strong>Passed:</strong> {report['summary']['passed']}</p>
        <p><strong>Failed:</strong> {report['summary']['failed']}</p>
        <p><strong>Pass Rate:</strong> {report['summary']['pass_rate']}%</p>
        <p><strong>Execution Time:</strong> {report['summary']['execution_time']}s</p>
    </div>

    <h2>Test Cases</h2>
"""

        for test in report['test_cases']:
            status_class = 'passed' if test['status'] == 'PASSED' else 'failed'
            severity_class = 'critical' if test['severity'] == 'critical' else ''

            html += f"""
    <div class="test-case {status_class} {severity_class}">
        <h3>{test['title']} - {test['status']}</h3>
        <p><strong>Type:</strong> {test['test_type']}</p>
        <p><strong>Endpoint:</strong> {test['method']} {test['endpoint']}</p>
        <p><strong>Severity:</strong> {test['severity']}</p>
        <p><strong>Priority:</strong> {test['priority']}</p>
        <p><strong>Execution Time:</strong> {test['execution_time']}s</p>
"""

            if test['error']:
                html += f"<p><strong>Error:</strong> {test['error']}</p>"

            html += "    </div>\n"

        html += """
</body>
</html>
"""
        return html

    def export_to_csv(self, report: Dict[str, Any]) -> str:
        """Export report to CSV format"""
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            'Test Name', 'Status', 'Type', 'Endpoint', 'Method',
            'Severity', 'Priority', 'Execution Time', 'Error'
        ])

        # Data
        for test in report['test_cases']:
            writer.writerow([
                test['title'],
                test['status'],
                test['test_type'],
                test['endpoint'],
                test['method'],
                test['severity'],
                test['priority'],
                test['execution_time'],
                test.get('error', '')
            ])

        return output.getvalue()


    async def generate(self, execution_results, api_spec=None):
        """
        Wrapper called by CoreEngine and TestGenerationPipeline.

        Bridges the interface gap: the pipeline calls generate(results, api_spec)
        but the actual implementation is generate_qase_report(results, session).

        Creates a synthetic session object from the api_spec context.
        """
        session = {
            'id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'request': {},
        }

        # If api_spec is provided, enrich the session with context
        if api_spec:
            session['request'] = {
                'api_spec': api_spec,
                'timestamp': datetime.now().isoformat(),
            }

        return await self.generate_qase_report(execution_results, session)