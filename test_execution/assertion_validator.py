"""
Validates test assertions
"""

import logging
import re
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class AssertionValidator:
    """Validates test assertions against responses"""

    def __init__(self):
        self.assertion_patterns = {
            'equals': r'(\w+(?:\.\w+)*)\s*==\s*(.+)',
            'not_equals': r'(\w+(?:\.\w+)*)\s*!=\s*(.+)',
            'greater_than': r'(\w+(?:\.\w+)*)\s*>\s*(.+)',
            'less_than': r'(\w+(?:\.\w+)*)\s*<\s*(.+)',
            'contains': r'(\w+(?:\.\w+)*)\s+contains\s+(.+)',
            'not_null': r'(\w+(?:\.\w+)*)\s+is\s+not\s+null',
            'null': r'(\w+(?:\.\w+)*)\s+is\s+null',
            'length': r'(\w+(?:\.\w+)*)\.length\s*==\s*(\d+)',
        }

    def validate_assertions(self, assertions: List[str],
                            response_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Validate list of assertions

        Args:
            assertions: List of assertion strings
            response_data: Response data to validate against

        Returns:
            List of validation results
        """
        results = []

        for assertion in assertions:
            result = self.validate_assertion(assertion, response_data)
            results.append(result)

        return results

    def validate_assertion(self, assertion: str,
                           response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate single assertion"""
        assertion = assertion.strip()

        try:
            # Try different assertion patterns
            for assertion_type, pattern in self.assertion_patterns.items():
                match = re.match(pattern, assertion)
                if match:
                    return self._validate_by_type(
                        assertion_type, match, response_data, assertion
                    )

            # If no pattern matched, try eval (dangerous but flexible)
            logger.warning(f"No pattern matched for assertion: {assertion}")
            return {
                'assertion': assertion,
                'type': 'unknown',
                'passed': False,
                'error': 'Unknown assertion format'
            }

        except Exception as e:
            logger.error(f"Assertion validation failed: {str(e)}")
            return {
                'assertion': assertion,
                'type': 'error',
                'passed': False,
                'error': str(e)
            }

    def _validate_by_type(self, assertion_type: str, match,
                          response_data: Dict, assertion: str) -> Dict[str, Any]:
        """Validate assertion by type"""

        if assertion_type == 'equals':
            path = match.group(1)
            expected = match.group(2).strip().strip('"\'')
            actual = self._get_nested_value(response_data, path)

            passed = str(actual) == expected
            return {
                'assertion': assertion,
                'type': 'equals',
                'expected': expected,
                'actual': actual,
                'passed': passed
            }

        elif assertion_type == 'not_equals':
            path = match.group(1)
            expected = match.group(2).strip().strip('"\'')
            actual = self._get_nested_value(response_data, path)

            passed = str(actual) != expected
            return {
                'assertion': assertion,
                'type': 'not_equals',
                'expected': f"not {expected}",
                'actual': actual,
                'passed': passed
            }

        elif assertion_type == 'greater_than':
            path = match.group(1)
            threshold = float(match.group(2))
            actual = self._get_nested_value(response_data, path)

            try:
                actual_num = float(actual)
                passed = actual_num > threshold
            except:
                passed = False

            return {
                'assertion': assertion,
                'type': 'greater_than',
                'threshold': threshold,
                'actual': actual,
                'passed': passed
            }

        elif assertion_type == 'less_than':
            path = match.group(1)
            threshold = float(match.group(2))
            actual = self._get_nested_value(response_data, path)

            try:
                actual_num = float(actual)
                passed = actual_num < threshold
            except:
                passed = False

            return {
                'assertion': assertion,
                'type': 'less_than',
                'threshold': threshold,
                'actual': actual,
                'passed': passed
            }

        elif assertion_type == 'contains':
            path = match.group(1)
            expected = match.group(2).strip().strip('"\'')
            actual = self._get_nested_value(response_data, path)

            if isinstance(actual, (list, str)):
                passed = expected in actual
            else:
                passed = False

            return {
                'assertion': assertion,
                'type': 'contains',
                'expected': expected,
                'actual': actual,
                'passed': passed
            }

        elif assertion_type == 'not_null':
            path = match.group(1)
            actual = self._get_nested_value(response_data, path)

            passed = actual is not None
            return {
                'assertion': assertion,
                'type': 'not_null',
                'actual': actual,
                'passed': passed
            }

        elif assertion_type == 'null':
            path = match.group(1)
            actual = self._get_nested_value(response_data, path)

            passed = actual is None
            return {
                'assertion': assertion,
                'type': 'null',
                'actual': actual,
                'passed': passed
            }

        elif assertion_type == 'length':
            path = match.group(1)
            expected_length = int(match.group(2))
            actual = self._get_nested_value(response_data, path)

            try:
                actual_length = len(actual)
                passed = actual_length == expected_length
            except:
                passed = False
                actual_length = None

            return {
                'assertion': assertion,
                'type': 'length',
                'expected': expected_length,
                'actual': actual_length,
                'passed': passed
            }

        return {
            'assertion': assertion,
            'type': assertion_type,
            'passed': False,
            'error': 'Not implemented'
        }

    def _get_nested_value(self, data: Dict, path: str) -> Any:
        """Get value from nested dict using dot notation"""
        keys = path.split('.')
        value = data

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None

            if value is None:
                return None

        return value