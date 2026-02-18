"""
Edge Case Agent - Generates edge case and boundary tests.

Fixes:
- Consistent interface with other agents
- Returns properly structured test cases (not just descriptions)
- Handles standardized RAG format
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class EdgeCaseAgent:
    """Generates edge case test scenarios."""

    def __init__(self, llama_client):
        self.client = llama_client
        self.name = "edge_case"

    async def execute(self, input_data: Dict[str, Any]) -> List[Dict]:
        """Execute edge case generation."""
        api_spec = input_data.get('api_spec', {})
        context = input_data.get('context', {})
        analyzer_results = input_data.get('analyzer_results', {})

        return await self.generate_edge_cases(api_spec, context, analyzer_results)

    async def generate_edge_cases(self, api_spec: Dict,
                                  context: Optional[Dict] = None,
                                  analysis: Optional[Dict] = None) -> List[Dict]:
        """Generate edge case tests for all endpoints."""
        endpoints = api_spec.get('endpoints', [])
        if not endpoints:
            return []

        all_edge_cases = []

        for endpoint in endpoints:
            try:
                cases = await self._generate_for_endpoint(endpoint, context)
                all_edge_cases.extend(cases)
            except Exception as e:
                logger.warning(f"Edge case generation failed for {endpoint.get('path')}: {e}")
                # Fall back to template-based generation
                all_edge_cases.extend(self._template_edge_cases(endpoint))

        return all_edge_cases

    async def _generate_for_endpoint(self, endpoint: Dict,
                                     context: Optional[Dict]) -> List[Dict]:
        """Generate edge cases for a single endpoint via LLM."""
        method = endpoint.get('http_method', endpoint.get('method', 'GET')).upper()
        path = endpoint.get('path', endpoint.get('route', ''))
        params = endpoint.get('parameters', [])

        # Build context from RAG
        rag_hint = ""
        if context and context.get('edge_cases'):
            examples = []
            for item in context['edge_cases'][:3]:
                if isinstance(item, dict):
                    meta = item.get('metadata', {})
                    title = meta.get('title', '') if isinstance(meta, dict) else str(meta)
                    if title:
                        examples.append(title)
            if examples:
                rag_hint = f"\nRelevant edge case patterns: {', '.join(examples)}"

        prompt = (
            f"Generate edge case tests for: {method} {path}\n"
            f"Parameters: {json.dumps([p.get('name', '') for p in params])}\n"
            f"{rag_hint}\n\n"
            f"Return a JSON array of test objects. Each must have: "
            f"name, method, endpoint, test_data, expected_status, "
            f"test_type, priority, assertions (list of strings).\n"
            f"Focus on: boundary values, special characters, null/empty, "
            f"unicode, very long strings, negative numbers, zero, max int, "
            f"concurrent access patterns, type coercion."
        )

        try:
            response = await self.client.generate(
                prompt=prompt,
                system_prompt=(
                    "You are an expert at finding edge cases in APIs. "
                    "Return ONLY a JSON array of test case objects."
                ),
                max_tokens=2000,
                temperature=0.5
            )
            return self._parse_response(response, method, path)
        except Exception as e:
            logger.warning(f"LLM edge case generation failed: {e}")
            return self._template_edge_cases(endpoint)

    def _parse_response(self, response: str, method: str, path: str) -> List[Dict]:
        """Parse LLM response into test case dicts."""
        if not response:
            return []

        text = response.strip()
        # Remove markdown
        if '```' in text:
            text = re.sub(r'```(?:json)?\n?', '', text).strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\[[\s\S]*\]', text)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    return []
            else:
                return []

        if not isinstance(parsed, list):
            parsed = [parsed]

        results = []
        for item in parsed:
            if isinstance(item, dict) and item.get('name'):
                # Ensure required fields
                item.setdefault('method', method)
                item.setdefault('endpoint', path)
                item.setdefault('test_data', {})
                item.setdefault('expected_status', 400)
                item.setdefault('test_type', 'edge_case')
                item.setdefault('priority', 'medium')
                item.setdefault('assertions', [])
                results.append(item)

        return results

    def _template_edge_cases(self, endpoint: Dict) -> List[Dict]:
        """Generate template-based edge cases when LLM fails."""
        method = endpoint.get('http_method', endpoint.get('method', 'GET')).upper()
        path = endpoint.get('path', endpoint.get('route', ''))
        params = endpoint.get('parameters', [])

        cases = []

        # Boundary: very large ID
        if '{id}' in path or any(p.get('name') == 'id' for p in params):
            cases.append({
                'name': f'Edge: Max integer ID - {method} {path}',
                'method': method,
                'endpoint': path,
                'test_data': {'id': 2147483647},
                'expected_status': 404,
                'test_type': 'edge_case',
                'priority': 'medium',
                'assertions': [],
            })
            cases.append({
                'name': f'Edge: Zero ID - {method} {path}',
                'method': method,
                'endpoint': path,
                'test_data': {'id': 0},
                'expected_status': 400,
                'test_type': 'edge_case',
                'priority': 'medium',
                'assertions': [],
            })
            cases.append({
                'name': f'Edge: Negative ID - {method} {path}',
                'method': method,
                'endpoint': path,
                'test_data': {'id': -1},
                'expected_status': 400,
                'test_type': 'edge_case',
                'priority': 'medium',
                'assertions': [],
            })

        # String params: special chars, empty, very long
        string_params = [
            p for p in params
            if p.get('type', 'string').lower() in ('string', 'str')
            and p.get('location', p.get('in', '')) != 'path'
        ]

        for param in string_params[:3]:
            name = param.get('name', 'field')

            cases.append({
                'name': f'Edge: Empty string {name} - {method} {path}',
                'method': method,
                'endpoint': path,
                'test_data': {name: ''},
                'expected_status': 400,
                'test_type': 'edge_case',
                'priority': 'medium',
                'assertions': [],
            })
            cases.append({
                'name': f'Edge: Unicode {name} - {method} {path}',
                'method': method,
                'endpoint': path,
                'test_data': {name: '你好世界 🎉 مرحبا'},
                'expected_status': 200,
                'test_type': 'edge_case',
                'priority': 'low',
                'assertions': [],
            })
            cases.append({
                'name': f'Edge: Very long {name} - {method} {path}',
                'method': method,
                'endpoint': path,
                'test_data': {name: 'x' * 10000},
                'expected_status': 400,
                'test_type': 'edge_case',
                'priority': 'medium',
                'assertions': [],
            })

        # POST/PUT with null body
        if method in ('POST', 'PUT', 'PATCH'):
            cases.append({
                'name': f'Edge: Null body - {method} {path}',
                'method': method,
                'endpoint': path,
                'test_data': None,
                'expected_status': 400,
                'test_type': 'edge_case',
                'priority': 'high',
                'assertions': [],
            })

        return cases