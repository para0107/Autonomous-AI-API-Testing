"""
Validation utilities for test cases and API specifications
"""

import logging
import re
import json
from typing import Dict, List, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom validation error"""
    pass


class APISpecValidator:
    """Validates API specifications"""

    @staticmethod
    def validate_http_method(method: Optional[str]) -> bool:
        """Validate HTTP method"""
        if method is None:
            return False

        valid_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']
        return method.upper() in valid_methods

    @staticmethod
    def validate_endpoint_path(path: Optional[str]) -> bool:
        """Validate endpoint path"""
        if not path:
            return False
        return isinstance(path, str) and len(path) > 0

    @staticmethod
    def validate_parameter(param: Dict[str, Any]) -> bool:
        """Validate parameter definition"""
        required_fields = ['name']
        return all(field in param for field in required_fields)


class TestCaseValidator:
    """Validates test case definitions"""

    @staticmethod
    def validate_test_case(test_case: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate a test case

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(test_case, dict):
            return False, "Test case must be a dictionary"

        # Check required fields
        required_fields = ['name']
        missing_fields = [field for field in required_fields if field not in test_case or not test_case[field]]

        if missing_fields:
            return False, f"Missing required fields: {missing_fields}"

        # Validate method if present
        if 'method' in test_case and test_case['method'] is not None:
            if not APISpecValidator.validate_http_method(test_case['method']):
                return False, f"Invalid HTTP method: {test_case.get('method')}"

        # Validate endpoint if present
        if 'endpoint' in test_case and test_case['endpoint'] is not None:
            if not APISpecValidator.validate_endpoint_path(test_case['endpoint']):
                return False, f"Invalid endpoint path: {test_case.get('endpoint')}"

        return True, None

    @staticmethod
    def validate_test_suite(test_suite: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate a test suite

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        if not isinstance(test_suite, list):
            return False, ["Test suite must be a list"]

        for idx, test_case in enumerate(test_suite):
            is_valid, error = TestCaseValidator.validate_test_case(test_case)
            if not is_valid:
                errors.append(f"Test case {idx}: {error}")

        return len(errors) == 0, errors


class EmailValidator:
    """Email validation"""

    @staticmethod
    def validate(email: str) -> bool:
        """Validate email address"""
        if not email or not isinstance(email, str):
            return False

        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None


class URLValidator:
    """URL validation"""

    @staticmethod
    def validate(url: str) -> bool:
        """Validate URL"""
        if not url or not isinstance(url, str):
            return False

        pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        return re.match(pattern, url) is not None


class JSONSchemaValidator:
    """JSON schema validation"""

    @staticmethod
    def validate(data: Any, schema: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate data against JSON schema"""
        # Basic validation - can be extended with jsonschema library
        try:
            if not isinstance(data, dict):
                return False, "Data must be a dictionary"
            return True, None
        except Exception as e:
            return False, str(e)


class DataValidator:
    """General data validation"""

    @staticmethod
    def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> Tuple[bool, List[str]]:
        """Validate required fields are present"""
        missing = [field for field in required_fields if field not in data or data[field] is None]
        return len(missing) == 0, missing

    @staticmethod
    def validate_field_types(data: Dict[str, Any], field_types: Dict[str, type]) -> Tuple[bool, List[str]]:
        """Validate field types"""
        errors = []
        for field, expected_type in field_types.items():
            if field in data and not isinstance(data[field], expected_type):
                errors.append(f"{field} must be {expected_type.__name__}")
        return len(errors) == 0, errors


def is_valid_test_case(test_case: Dict[str, Any]) -> bool:
    """
    Check if a test case is valid

    Args:
        test_case: Test case dictionary

    Returns:
        True if valid, False otherwise
    """
    try:
        valid, error = TestCaseValidator.validate_test_case(test_case)
        if not valid:
            logger.debug(f"Invalid test case: {error}")
        return valid
    except Exception as e:
        logger.error(f"Error validating test case: {e}")
        return False


def is_valid_api_spec(api_spec: Dict[str, Any]) -> bool:
    """
    Check if an API specification is valid

    Args:
        api_spec: API specification dictionary

    Returns:
        True if valid, False otherwise
    """
    try:
        # Check if endpoints exist and is a list
        if 'endpoints' not in api_spec:
            logger.debug("API spec missing 'endpoints' field")
            return False

        if not isinstance(api_spec.get('endpoints'), list):
            logger.debug("API spec 'endpoints' must be a list")
            return False

        # Check if there's at least one endpoint
        if len(api_spec['endpoints']) == 0:
            logger.debug("API spec has no endpoints")
            return False

        return True

    except Exception as e:
        logger.error(f"Error validating API spec: {e}")
        return False


def is_valid_email(email: str) -> bool:
    """Validate email address"""
    return EmailValidator.validate(email)


def is_valid_url(url: str) -> bool:
    """Validate URL"""
    return URLValidator.validate(url)


def is_valid_json(data: str) -> bool:
    """Validate JSON string"""
    try:
        json.loads(data)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def validate_test_data(test_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate test data/payload

    Args:
        test_data: Test data dictionary

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(test_data, dict):
        return False, "Test data must be a dictionary"

    return True, None