"""
Edge Case Generator Agent
"""

import logging
from typing import Dict, Any, List
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class EdgeCaseAgent(BaseAgent):
    """Agent for generating edge case tests"""

    def __init__(self, llama_client):
        super().__init__(llama_client, 'edge_case')

    async def execute(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate edge cases"""
        api_spec = input_data.get('api_spec', {})
        analysis = input_data.get('analyzer_results', {})

        return await self.generate_edge_cases(api_spec, analysis)

    async def generate_edge_cases(self, api_spec: Dict[str, Any],
                                  analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate edge case test scenarios

        Args:
            api_spec: API specification
            analysis: API analysis results

        Returns:
            List of edge case tests
        """
        edge_cases = []

        # Generate edge cases for each parameter
        parameters = api_spec.get('parameters', [])
        for param in parameters:
            param_edge_cases = await self._generate_parameter_edge_cases(param, api_spec)
            edge_cases.extend(param_edge_cases)

        # Generate combination edge cases
        if len(parameters) > 1:
            combo_edge_cases = await self._generate_combination_edge_cases(parameters, api_spec)
            edge_cases.extend(combo_edge_cases)

        # Generate security edge cases
        security_edge_cases = await self._generate_security_edge_cases(api_spec, analysis)
        edge_cases.extend(security_edge_cases)

        # Generate state-based edge cases
        if analysis.get('dependencies'):
            state_edge_cases = await self._generate_state_edge_cases(api_spec, analysis)
            edge_cases.extend(state_edge_cases)

        return edge_cases

    async def _generate_parameter_edge_cases(self, param: Dict[str, Any],
                                             api_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate edge cases for a single parameter"""
        param_name = param.get('name', 'unknown')
        param_type = param.get('type', 'string')
        constraints = param.get('constraints', {})

        prompt = f"""Generate edge case test scenarios for the following parameter:

Parameter: {param_name}
Type: {param_type}
Required: {param.get('required', False)}
Constraints: {constraints}
Endpoint: {api_spec.get('path')}
Method: {api_spec.get('method')}

Generate edge cases including:
1. Boundary values (min, max, just outside bounds)
2. Type mismatches
3. Special characters and encoding issues
4. Null/empty/undefined values
5. Extreme values (very large, very small, infinity, NaN)
6. Format violations
7. Injection attempts (SQL, XSS, command injection)

Return as JSON array with structure:
[{{
    "name": "edge case test name",
    "description": "what this tests",
    "parameter": "{param_name}",
    "value": "edge case value",
    "expected_behavior": "expected system response",
    "risk_level": "high|medium|low"
}}]"""

        response = await self.generate_json_with_retry(prompt)

        if isinstance(response, dict):
            response = [response]

        # Convert to test case format
        test_cases = []
        for edge_case in response:
            test_case = {
                'name': edge_case.get('name', f'Edge case for {param_name}'),
                'description': edge_case.get('description', ''),
                'test_type': 'edge_case',
                'endpoint': api_spec.get('path'),
                'method': api_spec.get('method'),
                'input': {param_name: edge_case.get('value')},
                'expected_behavior': edge_case.get('expected_behavior'),
                'risk_level': edge_case.get('risk_level', 'medium')
            }
            test_cases.append(test_case)

        return test_cases

    async def _generate_combination_edge_cases(self, parameters: List[Dict[str, Any]],
                                               api_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate edge cases for parameter combinations"""
        if len(parameters) < 2:
            return []

        # Select top parameters for combination
        params_to_combine = parameters[:3]  # Limit to avoid explosion

        param_info = [f"{p['name']}: {p['type']}" for p in params_to_combine]

        prompt = f"""Generate edge case combinations for multiple parameters:

Parameters: {param_info}
Endpoint: {api_spec.get('path')}
Method: {api_spec.get('method')}

Generate test cases that combine edge values across multiple parameters:
1. All parameters at minimum values
2. All parameters at maximum values
3. Mix of min and max values
4. One valid, others invalid
5. Conflicting values that violate business logic

Return as JSON array with structure:
[{{
    "name": "combination edge case name",
    "description": "what this tests",
    "inputs": {{"param1": "value1", "param2": "value2"}},
    "expected_behavior": "expected response",
    "test_rationale": "why this combination is important"
}}]"""

        response = await self.generate_json_with_retry(prompt)

        if isinstance(response, dict):
            response = [response]

        # Convert to test cases
        test_cases = []
        for combo in response:
            test_case = {
                'name': combo.get('name', 'Combination edge case'),
                'description': combo.get('description', ''),
                'test_type': 'edge_case_combination',
                'endpoint': api_spec.get('path'),
                'method': api_spec.get('method'),
                'input': combo.get('inputs', {}),
                'expected_behavior': combo.get('expected_behavior'),
                'test_rationale': combo.get('test_rationale')
            }
            test_cases.append(test_case)

        return test_cases

    async def _generate_security_edge_cases(self, api_spec: Dict[str, Any],
                                            analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate security-focused edge cases"""
        prompt = f"""Generate security edge cases for the following API:

Endpoint: {api_spec.get('path')}
Method: {api_spec.get('method')}
Auth Required: {analysis.get('auth_requirements', {}).get('required', False)}

Generate security test cases for:
1. SQL Injection attempts
2. XSS (Cross-Site Scripting) attempts
3. Command injection
4. Path traversal attacks
5. Authentication bypass attempts
6. Authorization elevation attempts
7. CSRF attacks
8. XXE injection (for XML endpoints)
9. Buffer overflow attempts
10. Rate limiting bypass

Return as JSON array with security-specific test cases."""

        response = await self.generate_json_with_retry(prompt)

        if isinstance(response, dict):
            response = [response]

        # Add security metadata
        test_cases = []
        for security_test in response:
            test_case = {
                'name': security_test.get('name', 'Security edge case'),
                'test_type': 'security_edge_case',
                'endpoint': api_spec.get('path'),
                'method': api_spec.get('method'),
                'risk_level': 'high',
                **security_test
            }
            test_cases.append(test_case)

        return test_cases

    async def _generate_state_edge_cases(self, api_spec: Dict[str, Any],
                                         analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate state-based edge cases"""
        dependencies = analysis.get('dependencies', [])

        prompt = f"""Generate state-based edge cases for the following API:

Endpoint: {api_spec.get('path')}
Method: {api_spec.get('method')}
Dependencies: {dependencies}

Generate test cases for:
1. Resource doesn't exist
2. Resource already exists (for creation)
3. Concurrent modifications
4. Stale data scenarios
5. Transaction rollback scenarios
6. Partial state updates
7. Race conditions

Return as JSON array."""

        response = await self.generate_json_with_retry(prompt)

        if isinstance(response, dict):
            response = [response]

        test_cases = []
        for state_test in response:
            test_case = {
                'name': state_test.get('name', 'State edge case'),
                'test_type': 'state_edge_case',
                'endpoint': api_spec.get('path'),
                'method': api_spec.get('method'),
                **state_test
            }
            test_cases.append(test_case)

        return test_cases