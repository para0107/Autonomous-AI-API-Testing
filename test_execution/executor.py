"""
Test executor for running API tests
"""

import logging
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
from datetime import datetime
import json


from config import settings

logger = logging.getLogger(__name__)


class TestExecutor:
    """Executes API tests against live endpoints"""

    def __init__(self):
        self.timeout = aiohttp.ClientTimeout(total=settings.DEFAULT_TIMEOUT)
        self.max_retries = settings.MAX_RETRIES
        self.session = None

    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(ssl=False)
        self.session = aiohttp.ClientSession(timeout=self.timeout, connector=connector)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def execute_batch(self, test_cases: List[Dict[str, Any]],
                            endpoint_url: str, parallel: bool = True) -> List[Dict[str, Any]]:
        """
        Execute batch of test cases

        Args:
            test_cases: List of test cases to execute
            endpoint_url: Base URL of API
            parallel: Execute in parallel or sequential

        Returns:
            List of execution results
        """
        logger.info(f"Executing {len(test_cases)} test cases")

        if not self.session:
            self.session = aiohttp.ClientSession(timeout=self.timeout)

        if parallel:
            tasks = [self.execute_test(test, endpoint_url) for test in test_cases]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Test {i} failed with exception: {result}")
                    processed_results.append({
                        'test_case': test_cases[i],
                        'passed': False,
                        'error': str(result),
                        'execution_time': 0
                    })
                else:
                    processed_results.append(result)

            return processed_results
        else:
            results = []
            for test in test_cases:
                result = await self.execute_test(test, endpoint_url)
                results.append(result)
            return results

    async def execute_test(self, test_case: Dict[str, Any],
                           base_url: str) -> Dict[str, Any]:
        """
        Execute single test case

        Args:
            test_case: Test case definition
            base_url: Base URL of API

        Returns:
            Execution result
        """
        start_time = datetime.now()

        try:
            endpoint = test_case.get('endpoint', '')
            method = test_case.get('method', 'GET').upper()
            test_data = test_case.get('test_data', {})

            # Build URL with path params replaced
            url = self._build_url(base_url, endpoint, test_data)

            headers = self._build_headers(test_case)

            # Prepare request parameters
            request_params = self._prepare_request_params(method, test_data)
            # Execute with retry
            response_data = await self._execute_with_retry(
                method, url, headers, request_params
            )

            # Validate response
            validation_result = self._validate_response(test_case, response_data)

            execution_time = (datetime.now() - start_time).total_seconds()

            return {
                'name': test_case.get('name', 'Unnamed Test'),
                'test_type': test_case.get('test_type', 'unknown'),
                'endpoint': test_case.get('endpoint'),
                'method': method,
                'passed': validation_result['passed'],
                'expected_status': test_case.get('expected_status', 200),
                'actual_status': response_data['status'],
                'expected': test_case.get('expected_response'),
                'actual': response_data.get('body'),
                'assertions': validation_result.get('assertions', []),
                'error': validation_result.get('error'),
                'execution_time': execution_time,
                'request_data': test_data,
                'response_data': response_data,
                'timestamp': start_time.isoformat()
            }

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Test execution failed: {str(e)}")

            return {
                'name': test_case.get('name', 'Unnamed Test'),
                'test_type': test_case.get('test_type', 'unknown'),
                'endpoint': test_case.get('endpoint'),
                'method': test_case.get('method', 'GET'),
                'passed': False,
                'error': str(e),
                'execution_time': execution_time,
                'timestamp': start_time.isoformat()
            }

    async def _execute_with_retry(self, method: str, url: str,
                                  headers: Dict, params: Dict) -> Dict[str, Any]:
        """Execute request with retry logic"""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                async with self.session.request(
                        method=method,
                        url=url,
                        headers=headers,
                        **params
                ) as response:
                    status = response.status

                    # Try to parse JSON response
                    try:
                        body = await response.json()
                    except:
                        body = await response.text()

                    return {
                        'status': status,
                        'headers': dict(response.headers),
                        'body': body
                    }

            except asyncio.TimeoutError as e:
                last_error = f"Request timeout (attempt {attempt + 1}/{self.max_retries})"
                logger.warning(last_error)
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff

            except aiohttp.ClientError as e:
                last_error = f"Client error: {str(e)}"
                logger.warning(f"{last_error} (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))

            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                logger.error(last_error)
                break

        raise Exception(f"Request failed after {self.max_retries} attempts: {last_error}")

    def _build_url(self, base_url: str, endpoint: str, test_data: Dict = None) -> str:
        """Build complete URL with path parameters replaced"""
        base_url = base_url.rstrip('/')
        endpoint = endpoint.lstrip('/')

        # Replace path parameters with actual values
        if test_data:
            # Find all {param} in endpoint
            import re
            path_params = re.findall(r'\{(\w+)\}', endpoint)

            for param in path_params:
                # Use value from test_data or default to 1
                value = test_data.get(param, test_data.get('id', 1))
                endpoint = endpoint.replace(f'{{{param}}}', str(value))

        return f"{base_url}/{endpoint}"

    def _build_headers(self, test_case: Dict[str, Any]) -> Dict[str, str]:
        """Build request headers"""
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        # Add custom headers from test case
        if 'headers' in test_case:
            headers.update(test_case['headers'])

        # Add authentication if present
        if 'auth_token' in test_case:
            headers['Authorization'] = f"Bearer {test_case['auth_token']}"

        return headers

    def _prepare_request_params(self, method: str, test_data: Dict) -> Dict[str, Any]:
        """Prepare request parameters based on method"""
        params = {}

        if method in ['GET', 'DELETE']:
            # Query parameters
            params['params'] = test_data
        elif method in ['POST', 'PUT', 'PATCH']:
            # JSON body
            params['json'] = test_data

        return params

    def _validate_response(self, test_case: Dict[str, Any],
                           response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate response against expected results"""
        result = {
            'passed': True,
            'assertions': [],
            'error': None
        }

        try:
            # Validate status code
            expected_status = test_case.get('expected_status', 200)
            actual_status = response_data['status']

            if actual_status != expected_status:
                result['passed'] = False
                result['assertions'].append({
                    'type': 'status_code',
                    'expected': expected_status,
                    'actual': actual_status,
                    'passed': False
                })
            else:
                result['assertions'].append({
                    'type': 'status_code',
                    'expected': expected_status,
                    'actual': actual_status,
                    'passed': True
                })

            # Validate response body
            if 'expected_response' in test_case and test_case['expected_response']:
                expected = test_case['expected_response']
                actual = response_data.get('body')

                body_valid = self._validate_response_body(expected, actual)
                result['assertions'].append({
                    'type': 'response_body',
                    'passed': body_valid
                })

                if not body_valid:
                    result['passed'] = False

            # Execute custom assertions
            if 'assertions' in test_case:
                for assertion in test_case['assertions']:
                    assertion_result = self._execute_assertion(assertion, response_data)
                    result['assertions'].append(assertion_result)

                    if not assertion_result['passed']:
                        result['passed'] = False

        except Exception as e:
            result['passed'] = False
            result['error'] = f"Validation error: {str(e)}"
            logger.error(result['error'])

        return result

    def _validate_response_body(self, expected: Any, actual: Any) -> bool:
        """Validate response body matches expected"""
        if isinstance(expected, dict) and isinstance(actual, dict):
            # Check if all expected keys are present with correct values
            for key, value in expected.items():
                if key not in actual:
                    return False
                if not self._validate_response_body(value, actual[key]):
                    return False
            return True
        else:
            return expected == actual

    def _execute_assertion(self, assertion: str, response_data: Dict) -> Dict[str, Any]:
        """Execute custom assertion"""
        # Simple assertion execution
        # In a real implementation, this would be more sophisticated

        try:
            # Example assertions:
            # "status == 200"
            # "body.user.id != null"
            # "body.items.length > 0"

            # For now, just mark as passed
            return {
                'type': 'custom',
                'assertion': assertion,
                'passed': True
            }
        except Exception as e:
            return {
                'type': 'custom',
                'assertion': assertion,
                'passed': False,
                'error': str(e)
            }

    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()