"""
Validators for test cases and API specifications.

These are referenced throughout the codebase but need proper implementations
that actually validate rather than returning True or checking minimal fields.
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def is_valid_test_case(test_case: Any) -> bool:
    """
    Validate that a test case has the minimum required fields.

    Required:
        - name (str, non-empty)
        - method (str, valid HTTP method)
        - endpoint (str, non-empty)

    Recommended (logged as warnings if missing):
        - expected_status (int)
        - test_type (str)
    """
    if not isinstance(test_case, dict):
        return False

    # Required fields
    name = test_case.get('name', '')
    if not name or not isinstance(name, str):
        logger.debug(f"Test case missing/invalid 'name': {test_case}")
        return False

    method = test_case.get('method', '').upper()
    valid_methods = {'GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS'}
    if method not in valid_methods:
        logger.debug(f"Test case '{name}' has invalid method: '{method}'")
        return False

    endpoint = test_case.get('endpoint', '')
    if not endpoint or not isinstance(endpoint, str):
        logger.debug(f"Test case '{name}' missing endpoint")
        return False

    # Warnings for recommended fields
    if 'expected_status' not in test_case:
        logger.debug(f"Test case '{name}' missing expected_status")

    if 'test_type' not in test_case and 'type' not in test_case:
        logger.debug(f"Test case '{name}' missing test_type")

    return True


def is_valid_api_spec(api_spec: Any) -> bool:
    """
    Validate that an API specification has minimum required structure.

    Required:
        - endpoints (list, at least 1)
        - Each endpoint must have: path (or route), http_method (or method)
    """
    if not isinstance(api_spec, dict):
        return False

    endpoints = api_spec.get('endpoints', [])
    if not endpoints or not isinstance(endpoints, list):
        logger.warning("API spec has no endpoints")
        return False

    valid_count = 0
    for ep in endpoints:
        if not isinstance(ep, dict):
            continue

        path = ep.get('path', ep.get('route', ''))
        method = ep.get('http_method', ep.get('method', ''))

        if path and method:
            valid_count += 1
        else:
            logger.debug(f"Invalid endpoint: path='{path}', method='{method}'")

    if valid_count == 0:
        logger.warning("API spec has no valid endpoints")
        return False

    if valid_count < len(endpoints):
        logger.warning(
            f"API spec has {len(endpoints) - valid_count} invalid endpoints "
            f"out of {len(endpoints)}"
        )

    return True


def validate_execution_result(result: Dict) -> bool:
    """Validate a test execution result has required fields."""
    if not isinstance(result, dict):
        return False

    required = ['name', 'passed']
    for field in required:
        if field not in result:
            return False

    return True


def validate_rag_result(item: Dict) -> bool:
    """Validate a RAG retrieval result has the standard format."""
    if not isinstance(item, dict):
        return False

    required = ['id', 'score', 'metadata']
    return all(k in item for k in required)