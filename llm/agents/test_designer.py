"""
Test Designer Agent - Generates test cases from API spec + RAG context.

Fixes:
- _format_rag_examples now handles the standardized Dict format (not tuples)
- Generates tests for ALL endpoints, not just first
- Consistent error handling
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class TestDesignerAgent:
    """Designs test cases using LLM + RAG context."""

    def __init__(self, llama_client):
        self.client = llama_client
        self.name = "test_designer"

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute test design task."""
        api_spec = input_data.get('api_spec', {})
        context = input_data.get('context', {})
        analyzer_results = input_data.get('analyzer_results', {})

        return await self.design_tests(api_spec, context, analyzer_results)

    async def design_tests(self, api_spec: Dict[str, Any],
                           context: Optional[Dict] = None,
                           analysis: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Design comprehensive test cases.

        Returns:
            Dict with 'happy_path_tests', 'negative_tests', 'validation_tests'
        """
        endpoints = api_spec.get('endpoints', [])
        models = api_spec.get('models', [])
        rules = api_spec.get('validation_rules', [])

        if not endpoints:
            logger.warning("No endpoints to design tests for")
            return {'happy_path_tests': [], 'negative_tests': [], 'validation_tests': []}

        # Format RAG examples for context
        rag_examples = self._format_rag_examples(context) if context else ""

        # Generate tests for each endpoint
        all_happy_path = []
        all_negative = []
        all_validation = []

        for endpoint in endpoints:
            try:
                tests = await self._design_endpoint_tests(
                    endpoint, models, rules, rag_examples, analysis
                )
                all_happy_path.extend(tests.get('happy_path', []))
                all_negative.extend(tests.get('negative', []))
                all_validation.extend(tests.get('validation', []))
            except Exception as e:
                logger.error(f"Failed to design tests for {endpoint.get('path')}: {e}")

        return {
            'happy_path_tests': all_happy_path,
            'negative_tests': all_negative,
            'validation_tests': all_validation,
        }

    async def _design_endpoint_tests(self, endpoint: Dict, models: List[Dict],
                                     rules: List, rag_examples: str,
                                     analysis: Optional[Dict]) -> Dict[str, List]:
        """Design tests for a single endpoint."""
        method = endpoint.get('http_method', endpoint.get('method', 'GET'))
        path = endpoint.get('path', endpoint.get('route', ''))
        params = endpoint.get('parameters', [])

        prompt = self._build_test_prompt(endpoint, models, rules, rag_examples)

        try:
            response = await self.client.generate(
                prompt=prompt,
                system_prompt=(
                    "You are an expert API test designer. Generate test cases in JSON format. "
                    "Return a JSON object with keys: happy_path (list), negative (list), validation (list). "
                    "Each test case should have: name, method, endpoint, test_data (dict), "
                    "expected_status (int), expected_response (dict or null), assertions (list of strings), "
                    "test_type, priority (high/medium/low)."
                ),
                max_tokens=3000,
                temperature=0.4
            )

            tests = self._parse_test_response(response, method, path)
            return tests

        except Exception as e:
            logger.warning(f"LLM test generation failed for {method} {path}: {e}")
            # Return basic auto-generated tests
            return self._generate_basic_tests(endpoint, rules)

    def _build_test_prompt(self, endpoint: Dict, models: List[Dict],
                           rules: List, rag_examples: str) -> str:
        """Build prompt for test generation."""
        method = endpoint.get('http_method', endpoint.get('method', 'GET'))
        path = endpoint.get('path', endpoint.get('route', ''))
        params = endpoint.get('parameters', [])

        lines = [
            f"Generate test cases for this API endpoint:",
            f"",
            f"Method: {method}",
            f"Path: {path}",
        ]

        if params:
            lines.append("Parameters:")
            for p in params:
                name = p.get('name', '')
                ptype = p.get('type', p.get('data_type', 'string'))
                required = p.get('required', False)
                location = p.get('location', p.get('in', ''))
                lines.append(f"  - {name} ({ptype}, {location}, required={required})")

        if endpoint.get('request_body') or endpoint.get('body_model'):
            lines.append(f"Request Body Model: {endpoint.get('request_body', endpoint.get('body_model'))}")

        if endpoint.get('response_model'):
            lines.append(f"Response Model: {endpoint.get('response_model')}")

        # Add relevant model details
        for model in models[:5]:
            model_name = model.get('name', '')
            if model_name in str(endpoint):
                fields = model.get('properties', model.get('fields', []))
                lines.append(f"\nModel {model_name}:")
                for f in fields[:10]:
                    if isinstance(f, dict):
                        lines.append(f"  - {f.get('name')}: {f.get('type', 'string')}")

        # Add relevant validation rules
        applicable_rules = [r for r in rules if r.get('endpoint') == path]
        if applicable_rules:
            lines.append("\nValidation Rules:")
            for r in applicable_rules[:10]:
                lines.append(f"  - {r.get('field', '')}: {r.get('rule', r.get('type', ''))}")

        # Add RAG examples
        if rag_examples:
            lines.append(f"\nSimilar test examples from knowledge base:\n{rag_examples}")

        lines.append("\nGenerate comprehensive tests covering happy path, negative cases, and validation.")

        return '\n'.join(lines)

    def _format_rag_examples(self, context: Dict) -> str:
        """
        Format RAG context for inclusion in prompts.

        FIX: Handles standardized Dict format from Retriever.
        Old code tried to unpack tuples — now we use dict keys consistently.
        """
        if not context:
            return ""

        lines = []

        similar = context.get('similar_tests', [])
        if similar:
            lines.append("Similar Tests:")
            for item in similar[:5]:
                # FIX: Standardized format is always Dict with id, score, metadata
                if isinstance(item, dict):
                    metadata = item.get('metadata', {})
                    score = item.get('score', 0)

                    # metadata might be a nested dict or a flat dict
                    if isinstance(metadata, dict):
                        title = metadata.get('title', metadata.get('name', item.get('id', 'unknown')))
                        description = metadata.get('description', '')
                        test_type = metadata.get('type', metadata.get('test_type', ''))
                    else:
                        title = str(metadata)
                        description = ''
                        test_type = ''

                    lines.append(f"  [{score:.2f}] {title}")
                    if description:
                        lines.append(f"    {description[:200]}")
                    if test_type:
                        lines.append(f"    Type: {test_type}")

        edge_cases = context.get('edge_cases', [])
        if edge_cases:
            lines.append("\nEdge Case Patterns:")
            for item in edge_cases[:3]:
                if isinstance(item, dict):
                    metadata = item.get('metadata', {})
                    title = (metadata.get('title', '') if isinstance(metadata, dict)
                             else str(metadata))
                    lines.append(f"  - {title}")

        return '\n'.join(lines) if lines else ""

    def _parse_test_response(self, response: str, method: str, path: str) -> Dict[str, List]:
        """Parse LLM response into test cases."""
        result = {'happy_path': [], 'negative': [], 'validation': []}

        if not response:
            return result

        text = response.strip()

        # Remove markdown fences
        if '```' in text:
            lines = text.split('\n')
            clean_lines = []
            in_code = False
            for line in lines:
                if line.strip().startswith('```'):
                    in_code = not in_code
                    continue
                if in_code or not line.strip().startswith('```'):
                    clean_lines.append(line)
            text = '\n'.join(clean_lines)

        try:
            parsed = json.loads(text.strip())
        except json.JSONDecodeError:
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.warning("Could not parse LLM test response as JSON")
                    return result
            else:
                return result

        if not isinstance(parsed, dict):
            return result

        # Normalize and validate each test case
        for category in ('happy_path', 'negative', 'validation'):
            tests = parsed.get(category, [])
            if not isinstance(tests, list):
                continue
            for test in tests:
                if isinstance(test, dict):
                    normalized = self._normalize_test_case(test, method, path, category)
                    if normalized:
                        result[category].append(normalized)

        return result

    def _normalize_test_case(self, test: Dict, default_method: str,
                             default_path: str, category: str) -> Optional[Dict]:
        """Ensure test case has all required fields."""
        return {
            'name': test.get('name', f'{category}_test'),
            'method': test.get('method', default_method).upper(),
            'endpoint': test.get('endpoint', default_path),
            'test_data': test.get('test_data', {}),
            'expected_status': int(test.get('expected_status', 200)),
            'expected_response': test.get('expected_response'),
            'assertions': test.get('assertions', []),
            'test_type': test.get('test_type', category),
            'priority': test.get('priority', 'medium'),
        }

    def _generate_basic_tests(self, endpoint: Dict, rules: List) -> Dict[str, List]:
        """Generate basic tests when LLM fails."""
        method = endpoint.get('http_method', endpoint.get('method', 'GET')).upper()
        path = endpoint.get('path', endpoint.get('route', ''))
        params = endpoint.get('parameters', [])

        happy_path = [{
            'name': f'Happy path: {method} {path}',
            'method': method,
            'endpoint': path,
            'test_data': self._generate_sample_data(params),
            'expected_status': 200 if method == 'GET' else 201 if method == 'POST' else 200,
            'expected_response': None,
            'assertions': [f'status == {200 if method == "GET" else 201 if method == "POST" else 200}'],
            'test_type': 'happy_path',
            'priority': 'high',
        }]

        negative = []
        if method in ('POST', 'PUT', 'PATCH'):
            negative.append({
                'name': f'Empty body: {method} {path}',
                'method': method,
                'endpoint': path,
                'test_data': {},
                'expected_status': 400,
                'expected_response': None,
                'assertions': ['status == 400'],
                'test_type': 'negative',
                'priority': 'medium',
            })

        return {'happy_path': happy_path, 'negative': negative, 'validation': []}

    def _generate_sample_data(self, params: List[Dict]) -> Dict:
        """Generate sample data from parameter definitions."""
        data = {}
        for p in params:
            name = p.get('name', '')
            ptype = p.get('type', p.get('data_type', 'string')).lower()
            if 'id' in name.lower():
                data[name] = 1
            elif ptype in ('int', 'integer', 'number'):
                data[name] = 1
            elif ptype == 'boolean':
                data[name] = True
            elif ptype == 'array':
                data[name] = []
            else:
                data[name] = f"test_{name}"
        return data