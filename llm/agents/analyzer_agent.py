"""
API Analyzer Agent
"""
import json
import logging
from typing import Dict, Any, List
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class AnalyzerAgent(BaseAgent):
    """Agent for analyzing API specifications"""

    def __init__(self, llama_client):
        super().__init__(llama_client, 'analyzer')

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze API specification"""
        api_spec = input_data.get('api_spec', {})
        context = input_data.get('context', {})

        return await self.analyze(api_spec, context)

#FIRST TRY OF ANALYZE METHOD
    # async def analyze(self, api_spec: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    #     """Analyze API endpoint"""
    #
    #     # ULTRA-STRICT prompt
    #     prompt = f"""CRITICAL: RESPOND WITH JSON ONLY. NO TEXT BEFORE OR AFTER.
    #
    # API Spec:
    # {json.dumps(api_spec, indent=2)}
    #
    # OUTPUT FORMAT (JSON OBJECT):
    # {{
    #   "endpoint": "/api/path",
    #   "method": "GET",
    #   "critical_parameters": ["param1"],
    #   "auth_requirements": {{"required": true, "type": "bearer"}},
    #   "business_logic": ["rule1"],
    #   "failure_points": ["point1"],
    #   "dependencies": ["service1"],
    #   "validation_rules": ["rule1"],
    #   "error_scenarios": ["scenario1"]
    # }}
    #
    # START YOUR RESPONSE WITH {{ IMMEDIATELY. NOTHING ELSE."""
    #
    #     schema = {
    #         "type": "object",
    #         "properties": {
    #             "endpoint": {"type": "string"},
    #             "method": {"type": "string"},
    #             "critical_parameters": {"type": "array"},
    #             "auth_requirements": {"type": "object"},
    #             "business_logic": {"type": "array"},
    #             "failure_points": {"type": "array"},
    #             "dependencies": {"type": "array"},
    #             "validation_rules": {"type": "array"},
    #             "error_scenarios": {"type": "array"}
    #         }
    #     }
    #
    #     response = await self.generate_json_with_retry(
    #         prompt,
    #         schema=schema,
    #         max_retries=3
    #     )
    #
    #     return response

    async def analyze(self, api_spec: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze API endpoint"""

        endpoint_info = api_spec.get('endpoints', [{}])[0] if api_spec.get('endpoints') else {}

        # ULTRA-SIMPLE prompt - just ask for endpoint and method
        prompt = f"""What is the endpoint path and HTTP method?

    Endpoint: {endpoint_info.get('path', 'unknown')}
    Method: {endpoint_info.get('method', 'unknown')}

    Respond ONLY with JSON:
    {{"endpoint": "/api/path", "method": "GET"}}"""

        try:
            response = await self.generate_json_with_retry(prompt, max_retries=2)

            if not isinstance(response, dict):
                raise ValueError("Response is not a dict")

            # Always provide defaults
            defaults = {
                'endpoint': endpoint_info.get('path', '/unknown'),
                'method': endpoint_info.get('method', 'GET'),
                'critical_parameters': [],
                'auth_requirements': {'required': False, 'type': 'none'},
                'validation_rules': [],
                'business_logic': [],
                'failure_points': [],
                'dependencies': [],
                'error_scenarios': []
            }

            # Merge response with defaults
            for key, default_value in defaults.items():
                if key not in response:
                    response[key] = default_value

            return response

        except Exception as e:
            logger.error(f"Analysis failed: {e}, using defaults")
            # ✅ Always return valid defaults
            return {
                'endpoint': endpoint_info.get('path', '/unknown'),
                'method': endpoint_info.get('method', 'GET'),
                'critical_parameters': [],
                'auth_requirements': {'required': False, 'type': 'none'},
                'validation_rules': [],
                'business_logic': [],
                'failure_points': [],
                'dependencies': [],
                'error_scenarios': []
            }

    def _build_analysis_prompt(self, api_spec: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Build prompt for API analysis"""

        # Format API specification
        api_info = self._format_api_spec(api_spec)

        # Format context
        context_info = self.format_context(context)

        prompt = f"""Analyze this API endpoint:

        {api_info}

        Return JSON with this structure:
        {{
            "endpoint": "path",
            "method": "HTTP_METHOD",
            "critical_parameters": ["param1", "param2"],
            "auth_requirements": {{"required": true, "type": "bearer"}},
            "business_logic": ["rule1", "rule2"],
            "failure_points": ["point1", "point2"],
            "validation_rules": ["rule1", "rule2"],
            "error_scenarios": ["scenario1", "scenario2"]
        }}

        Respond with ONLY the JSON object.
}}"""

        return prompt

    def _format_api_spec(self, api_spec: Dict[str, Any]) -> str:
        """Format API specification for prompt"""
        parts = []

        # Basic info
        parts.append(f"Endpoint: {api_spec.get('path', 'unknown')}")
        parts.append(f"Method: {api_spec.get('method', 'unknown')}")

        # Parameters
        if 'parameters' in api_spec:
            parts.append("\nParameters:")
            for param in api_spec['parameters']:
                param_str = f"  - {param.get('name')}: {param.get('type')} "
                param_str += f"({'required' if param.get('required') else 'optional'})"
                if 'constraints' in param:
                    param_str += f" - Constraints: {param['constraints']}"
                parts.append(param_str)

        # Request body
        if 'requestBody' in api_spec:
            parts.append(f"\nRequest Body: {api_spec['requestBody']}")

        # Responses
        if 'responses' in api_spec:
            parts.append("\nResponses:")
            for code, response in api_spec['responses'].items():
                parts.append(f"  {code}: {response.get('description', '')}")

        # Business logic
        if 'x-test-metadata' in api_spec:
            metadata = api_spec['x-test-metadata']
            if 'business_logic' in metadata:
                parts.append(f"\nBusiness Logic: {metadata['business_logic']}")
            if 'validators' in metadata:
                parts.append(f"\nValidators: {metadata['validators']}")

        return "\n".join(parts)

    def _get_analysis_schema(self) -> Dict[str, Any]:
        """Get JSON schema for analysis"""
        return {
            "type": "object",
            "properties": {
                "endpoint": {"type": "string"},
                "method": {"type": "string"},
                "critical_parameters": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "auth_requirements": {
                    "type": "object",
                    "properties": {
                        "required": {"type": "boolean"},
                        "type": {"type": "string"},
                        "scopes": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }
                },
                "business_logic": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "failure_points": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "dependencies": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "validation_rules": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "error_scenarios": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "performance": {
                    "type": "object",
                    "properties": {
                        "expected_latency": {"type": "string"},
                        "throughput": {"type": "string"}
                    }
                }
            }
        }

    def _assess_risks(self, api_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Assess testing risks"""
        risks = {
            'high': [],
            'medium': [],
            'low': []
        }

        # Check for authentication
        if not api_spec.get('security'):
            risks['high'].append('No authentication specified')

        # Check for data validation
        validators = api_spec.get('x-test-metadata', {}).get('validators', [])
        if not validators:
            risks['medium'].append('No explicit validation rules')

        # Check method type
        method = api_spec.get('method', '').upper()
        if method in ['DELETE', 'PUT']:
            risks['high'].append(f'{method} operation - data modification risk')

        # Check for file uploads
        if 'multipart/form-data' in str(api_spec):
            risks['high'].append('File upload - security risk')

        return risks

    def _assess_complexity(self, api_spec: Dict[str, Any]) -> str:
        """Assess API complexity"""
        complexity_score = 0

        # Parameter count
        params = api_spec.get('parameters', [])
        complexity_score += len(params)

        # Nested structures
        if 'requestBody' in api_spec:
            complexity_score += 5

        # Multiple response codes
        responses = api_spec.get('responses', {})
        complexity_score += len(responses)

        # Dependencies
        deps = api_spec.get('x-test-metadata', {}).get('dependencies', [])
        complexity_score += len(deps) * 2

        # Determine complexity level
        if complexity_score < 5:
            return 'low'
        elif complexity_score < 15:
            return 'medium'
        else:
            return 'high'