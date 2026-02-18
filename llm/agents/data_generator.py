"""
Test Data Generator Agent
"""

import logging
import json
import random
import string
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class DataGeneratorAgent(BaseAgent):
    """Agent for generating test data"""

    def __init__(self, llama_client):
        super().__init__(llama_client, 'data_generator')
        self.faker_patterns = self._init_faker_patterns()

    def _init_faker_patterns(self) -> Dict[str, Any]:
        """Initialize common data patterns"""
        return {
            'email': lambda: f"test_{random.randint(1000, 9999)}@example.com",
            'phone': lambda: f"+1{random.randint(1000000000, 9999999999)}",
            'uuid': lambda: f"{'-'.join([''.join(random.choices(string.hexdigits.lower(), k=l)) for l in [8, 4, 4, 4, 12]])}",
            'date': lambda: (datetime.now() + timedelta(days=random.randint(-365, 365))).isoformat(),
            'url': lambda: f"https://example.com/{random.choice(['api', 'test', 'demo'])}/{random.randint(1, 100)}",
            'ip': lambda: f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 255)}"
        }

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test data"""
        test_cases = input_data.get('test_cases', [])
        api_spec = input_data.get('api_spec', {})

        return await self.generate_data(test_cases, api_spec)

    async def generate_data(self, test_cases: List[Dict[str, Any]],
                            api_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate test data for test cases

        Args:
            test_cases: List of test cases needing data
            api_spec: API specification

        Returns:
            Test data for each test case
        """
        test_data = {}

        for test_case in test_cases:
            test_id = test_case.get('name', str(hash(str(test_case))))

            # Check if test already has data
            if 'input' in test_case and test_case['input']:
                test_data[test_id] = test_case['input']
                continue

            # Generate data based on test type
            test_type = test_case.get('test_type', 'general')

            if test_type == 'happy_path':
                data = await self._generate_valid_data(test_case, api_spec)
            elif test_type == 'validation':
                data = await self._generate_validation_data(test_case, api_spec)
            elif test_type == 'boundary':
                data = await self._generate_boundary_data(test_case, api_spec)
            elif test_type == 'edge_case':
                data = await self._generate_edge_data(test_case, api_spec)
            else:
                data = await self._generate_generic_data(test_case, api_spec)

            test_data[test_id] = data

            # Update test case with generated data
            test_case['test_data'] = data

        return test_data

    async def _generate_valid_data(self, test_case: Dict[str, Any],
                                   api_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Generate valid test data"""
        parameters = api_spec.get('parameters', [])

        if not parameters:
            return {}

        # Build prompt for data generation
        param_info = []
        for param in parameters:
            param_info.append({
                'name': param.get('name'),
                'type': param.get('type'),
                'required': param.get('required'),
                'constraints': param.get('constraints', {})
            })

        prompt = f"""Generate valid test data for the following API parameters:

Parameters: {json.dumps(param_info, indent=2)}
Test Case: {test_case.get('description', 'Valid data test')}

Generate realistic, valid data that satisfies all constraints.
Include all required parameters and some optional ones.

Return as JSON object with parameter names as keys."""

        data = await self.generate_json_with_retry(prompt)

        # Apply faker patterns for common fields
        data = self._apply_faker_patterns(data)

        return data

    async def _generate_validation_data(self, test_case: Dict[str, Any],
                                        api_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data for validation testing"""
        # Check test description for what to validate
        description = test_case.get('description', '').lower()

        parameters = api_spec.get('parameters', [])
        data = {}

        for param in parameters:
            param_name = param.get('name')
            param_type = param.get('type')

            # Generate invalid data based on description
            if 'missing' in description or 'required' in description:
                if param.get('required') and random.random() > 0.5:
                    continue  # Skip required field
            elif 'type' in description:
                data[param_name] = self._generate_wrong_type(param_type)
            elif 'length' in description or 'size' in description:
                data[param_name] = self._generate_invalid_length(param)
            elif 'format' in description:
                data[param_name] = self._generate_invalid_format(param_type)
            else:
                # Generate valid data
                data[param_name] = self._generate_valid_value(param)

        return data

    async def _generate_boundary_data(self, test_case: Dict[str, Any],
                                      api_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Generate boundary test data"""
        parameters = api_spec.get('parameters', [])
        data = {}

        for param in parameters:
            param_name = param.get('name')
            constraints = param.get('constraints', {})

            # Generate boundary values
            if 'min' in constraints or 'max' in constraints:
                data[param_name] = self._generate_boundary_value(constraints)
            elif 'minLength' in constraints or 'maxLength' in constraints:
                data[param_name] = self._generate_boundary_string(constraints)
            else:
                data[param_name] = self._generate_valid_value(param)

        return data

    async def _generate_edge_data(self, test_case: Dict[str, Any],
                                  api_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Generate edge case data"""
        # Check if test case already specifies the edge data
        if 'input' in test_case:
            return test_case['input']

        parameters = api_spec.get('parameters', [])
        data = {}

        for param in parameters:
            param_name = param.get('name')
            param_type = param.get('type')

            # Generate edge values
            data[param_name] = self._generate_edge_value(param_type)

        return data

    async def _generate_generic_data(self, test_case: Dict[str, Any],
                                     api_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Generate generic test data"""
        parameters = api_spec.get('parameters', [])
        data = {}

        for param in parameters:
            if param.get('required') or random.random() > 0.3:
                data[param['name']] = self._generate_valid_value(param)

        return data

    def _generate_valid_value(self, param: Dict[str, Any]) -> Any:
        """Generate a valid value for a parameter"""
        param_name = param.get('name', '').lower()
        param_type = param.get('type', 'string')
        constraints = param.get('constraints', {})

        # Check for enum values
        if 'enum' in constraints:
            return random.choice(constraints['enum'])

        # Check for pattern-based generation
        if 'email' in param_name:
            return self.faker_patterns['email']()
        elif 'phone' in param_name or 'tel' in param_name:
            return self.faker_patterns['phone']()
        elif 'uuid' in param_name or 'guid' in param_name:
            return self.faker_patterns['uuid']()
        elif 'date' in param_name or 'time' in param_name:
            return self.faker_patterns['date']()
        elif 'url' in param_name or 'link' in param_name:
            return self.faker_patterns['url']()
        elif 'ip' in param_name:
            return self.faker_patterns['ip']()

        # Generate by type
        if param_type == 'integer':
            min_val = constraints.get('min', 1)
            max_val = constraints.get('max', 1000)
            return random.randint(min_val, max_val)
        elif param_type == 'number':
            min_val = constraints.get('min', 0.0)
            max_val = constraints.get('max', 1000.0)
            return round(random.uniform(min_val, max_val), 2)
        elif param_type == 'boolean':
            return random.choice([True, False])
        elif param_type == 'array':
            return [self._generate_valid_value({'type': 'string'}) for _ in range(random.randint(1, 5))]
        elif param_type == 'object':
            return {'key': 'value', 'nested': {'data': 'test'}}
        else:  # string
            min_len = constraints.get('minLength', 1)
            max_len = constraints.get('maxLength', 20)
            length = random.randint(min_len, max_len)
            return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    def _generate_wrong_type(self, correct_type: str) -> Any:
        """Generate wrong type data"""
        type_map = {
            'integer': 'not_a_number',
            'number': 'not_a_number',
            'boolean': 'not_a_boolean',
            'array': 'not_an_array',
            'object': 'not_an_object',
            'string': 12345
        }
        return type_map.get(correct_type, None)

    def _generate_invalid_length(self, param: Dict[str, Any]) -> str:
        """Generate string with invalid length"""
        constraints = param.get('constraints', {})

        if 'minLength' in constraints:
            # Generate shorter than minimum
            return 'x' * (constraints['minLength'] - 1) if constraints['minLength'] > 0 else ''
        elif 'maxLength' in constraints:
            # Generate longer than maximum
            return 'x' * (constraints['maxLength'] + 1)
        else:
            # Generate very long string
            return 'x' * 10000

    def _generate_invalid_format(self, param_type: str) -> str:
        """Generate invalid format data"""
        format_map = {
            'email': 'invalid-email',
            'date': 'not-a-date',
            'url': 'not://a.url',
            'uuid': 'not-a-uuid'
        }
        return format_map.get(param_type, 'invalid_format')

    def _generate_boundary_value(self, constraints: Dict[str, Any]) -> int:
        """Generate boundary value"""
        if 'min' in constraints:
            return constraints['min']
        elif 'max' in constraints:
            return constraints['max']
        else:
            return 0

    def _generate_boundary_string(self, constraints: Dict[str, Any]) -> str:
        """Generate boundary length string"""
        if 'minLength' in constraints:
            return 'x' * constraints['minLength']
        elif 'maxLength' in constraints:
            return 'x' * constraints['maxLength']
        else:
            return ''

    def _generate_edge_value(self, param_type: str) -> Any:
        """Generate edge case value"""
        edge_values = {
            'integer': random.choice([0, -1, 2147483647, -2147483648]),
            'number': random.choice([0.0, -0.0, float('inf'), float('-inf')]),
            'string': random.choice(['', ' ', '\n', '\t', '\\', '"', "'", '<script>alert(1)</script>']),
            'boolean': None,
            'array': random.choice([[], [None], [[[[[]]]]]]),
            'object': random.choice([{}, None, {'': ''}, {'null': None}])
        }
        return edge_values.get(param_type, None)

    def _apply_faker_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply faker patterns to common fields"""
        for key, value in data.items():
            key_lower = key.lower()

            # Check if value needs faker pattern
            if isinstance(value, str) and value.startswith('test_'):
                for pattern_key, pattern_func in self.faker_patterns.items():
                    if pattern_key in key_lower:
                        data[key] = pattern_func()
                        break

        return data