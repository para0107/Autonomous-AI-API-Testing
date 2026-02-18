"""
Test executor for running API tests
"""

import logging
import asyncio
import re
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
        self.auth_token = None
        self.ssl_verify = True

    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(ssl=self.ssl_verify)
        self.session = aiohttp.ClientSession(timeout=self.timeout, connector=connector)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def execute_batch(self, test_cases: List[Dict[str, Any]],
                            endpoint_url: str, parallel: bool = True) -> List[Dict[str, Any]]:
        """Execute batch of test cases"""
        logger.info(f"Executing {len(test_cases)} test cases")

        if not self.session:
            connector = aiohttp.TCPConnector(ssl=self.ssl_verify)
            self.session = aiohttp.ClientSession(timeout=self.timeout, connector=connector)

        if parallel:
            tasks = [self.execute_test(test, endpoint_url) for test in test_cases]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Test {i} failed with exception: {result}")
                    processed_results.append({
                        'test_case': test_cases[i],
                        'name': test_cases[i].get('name', 'unknown'),
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
        """Execute single test case"""
        start_time = datetime.now()

        try:
            endpoint = test_case.get('endpoint', '')
            method = test_case.get('method', 'GET').upper()
            test_data = test_case.get('test_data', {})

            url = self._build_url(base_url, endpoint, test_data)
            headers = self._build_headers(test_case)
            request_params = self._prepare_request_params(method, test_data)

            # Ensure session exists
            if not self.session:
                connector = aiohttp.TCPConnector(ssl=self.ssl_verify)
                self.session = aiohttp.ClientSession(timeout=self.timeout, connector=connector)

            response_data = await self._execute_with_retry(
                method, url, headers, request_params
            )

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
                    try:
                        body = await response.json()
                    except Exception:
                        body = await response.text()

                    return {
                        'status': status,
                        'headers': dict(response.headers),
                        'body': body
                    }

            except asyncio.TimeoutError:
                last_error = f"Request timeout (attempt {attempt + 1}/{self.max_retries})"
                logger.warning(last_error)
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))

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

        if test_data:
            path_params = re.findall(r'\{(\w+)\}', endpoint)
            for param in path_params:
                value = test_data.get(param, test_data.get('id', 1))
                endpoint = endpoint.replace(f'{{{param}}}', str(value))

        return f"{base_url}/{endpoint}"

    def _build_headers(self, test_case: Dict[str, Any]) -> Dict[str, str]:
        """Build request headers"""
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        if 'headers' in test_case:
            headers.update(test_case['headers'])

        # Use test-case-level auth, then fall back to executor-level auth
        auth_token = test_case.get('auth_token') or self.auth_token
        if auth_token:
            headers['Authorization'] = f"Bearer {auth_token}"

        return headers

    def _prepare_request_params(self, method: str, test_data: Dict) -> Dict[str, Any]:
        """Prepare request parameters based on method"""
        params = {}

        if method in ['GET', 'DELETE']:
            # Filter out path params from query params
            params['params'] = {
                k: v for k, v in test_data.items()
                if not isinstance(v, (dict, list))
            }
        elif method in ['POST', 'PUT', 'PATCH']:
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
            # 1. Validate status code
            expected_status = test_case.get('expected_status', 200)
            actual_status = response_data['status']

            status_passed = actual_status == expected_status
            result['assertions'].append({
                'type': 'status_code',
                'expected': expected_status,
                'actual': actual_status,
                'passed': status_passed
            })
            if not status_passed:
                result['passed'] = False

            # 2. Validate response body if expected
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

            # 3. FIX: Execute custom assertions properly
            if 'assertions' in test_case:
                for assertion_str in test_case['assertions']:
                    assertion_result = self._execute_assertion(assertion_str, response_data)
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
            for key, value in expected.items():
                if key not in actual:
                    return False
                if not self._validate_response_body(value, actual[key]):
                    return False
            return True
        else:
            return expected == actual

    def _execute_assertion(self, assertion: str, response_data: Dict) -> Dict[str, Any]:
        """
        Execute a custom assertion string against response data.

        FIX: Actually evaluates assertions instead of always returning True.

        Supported formats:
            "status == 200"
            "body.field == value"
            "body.field != null"
            "body.items.length > 0"
            "body.field contains value"
        """
        assertion = assertion.strip()
        body = response_data.get('body', {})
        status = response_data.get('status')

        try:
            # status == N
            m = re.match(r'status\s*==\s*(\d+)', assertion)
            if m:
                expected = int(m.group(1))
                passed = status == expected
                return {
                    'type': 'status_check', 'assertion': assertion,
                    'expected': expected, 'actual': status, 'passed': passed
                }

            # body.path operator value
            m = re.match(
                r'body\.(\S+)\s+(==|!=|>|<|>=|<=|contains|is not null|is null)\s*(.*)',
                assertion
            )
            if m:
                path = m.group(1)
                operator = m.group(2).strip()
                expected_str = m.group(3).strip().strip('"\'') if m.group(3) else None

                actual_value = self._get_nested_value(body, path)

                if operator == 'is not null':
                    passed = actual_value is not None
                elif operator == 'is null':
                    passed = actual_value is None
                elif operator == '==':
                    passed = str(actual_value) == expected_str
                elif operator == '!=':
                    if expected_str == 'null':
                        passed = actual_value is not None
                    else:
                        passed = str(actual_value) != expected_str
                elif operator == 'contains':
                    if isinstance(actual_value, (str, list)):
                        passed = expected_str in str(actual_value)
                    else:
                        passed = False
                elif operator in ('>', '<', '>=', '<='):
                    try:
                        actual_num = float(actual_value) if actual_value is not None else 0
                        expected_num = float(expected_str)
                        if operator == '>':
                            passed = actual_num > expected_num
                        elif operator == '<':
                            passed = actual_num < expected_num
                        elif operator == '>=':
                            passed = actual_num >= expected_num
                        elif operator == '<=':
                            passed = actual_num <= expected_num
                        else:
                            passed = False
                    except (ValueError, TypeError):
                        passed = False
                else:
                    passed = False

                return {
                    'type': 'body_assertion', 'assertion': assertion,
                    'path': path, 'operator': operator,
                    'expected': expected_str, 'actual': actual_value,
                    'passed': passed
                }

            # Handle .length specially: body.items.length > 0
            m = re.match(r'body\.(.+)\.length\s*(==|>|<|>=|<=)\s*(\d+)', assertion)
            if m:
                path = m.group(1)
                operator = m.group(2)
                expected_len = int(m.group(3))
                actual_value = self._get_nested_value(body, path)

                try:
                    actual_len = len(actual_value) if actual_value is not None else 0
                except TypeError:
                    actual_len = 0

                if operator == '==':
                    passed = actual_len == expected_len
                elif operator == '>':
                    passed = actual_len > expected_len
                elif operator == '<':
                    passed = actual_len < expected_len
                elif operator == '>=':
                    passed = actual_len >= expected_len
                elif operator == '<=':
                    passed = actual_len <= expected_len
                else:
                    passed = False

                return {
                    'type': 'length_assertion', 'assertion': assertion,
                    'expected_length': expected_len, 'actual_length': actual_len,
                    'passed': passed
                }

            # Unrecognized assertion format
            logger.warning(f"Unrecognized assertion format: {assertion}")
            return {
                'type': 'unknown', 'assertion': assertion,
                'passed': False, 'error': 'Unrecognized assertion format'
            }

        except Exception as e:
            return {
                'type': 'error', 'assertion': assertion,
                'passed': False, 'error': str(e)
            }

    def _get_nested_value(self, data: Any, path: str) -> Any:
        """Get value from nested dict/list using dot notation."""
        keys = path.split('.')
        value = data

        for key in keys:
            if value is None:
                return None

            if isinstance(value, dict):
                value = value.get(key)
            elif isinstance(value, list):
                try:
                    idx = int(key)
                    value = value[idx]
                except (ValueError, IndexError):
                    return None
            else:
                return None

        return value

    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()