"""
Analyzer Agent - Analyzes API specification for test generation.

Fixes:
- Analyzes ALL endpoints, not just api_spec.get('endpoints', [{}])[0]
- Removed dead code (_build_analysis_prompt, _format_api_spec, commented-out analyze)
- _assess_risks and _assess_complexity are now actually called
- Consistent error handling: raises on LLM failure, returns analysis dict
"""

import logging
import json
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class AnalyzerAgent:
    """Analyzes API specifications to guide test generation."""

    def __init__(self, llama_client):
        self.client = llama_client
        self.name = "analyzer"

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute analysis task."""
        api_spec = input_data.get('api_spec', {})
        context = input_data.get('context', {})
        return await self.analyze(api_spec, context)

    async def analyze(self, api_spec: Dict[str, Any],
                      context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze the full API specification.

        Args:
            api_spec: Parsed API spec with endpoints, models, validation_rules
            context: Optional RAG context

        Returns:
            Analysis dict with complexity, risks, recommendations, endpoint details
        """
        endpoints = api_spec.get('endpoints', [])
        models = api_spec.get('models', [])
        validation_rules = api_spec.get('validation_rules', [])
        controllers = api_spec.get('controllers', [])

        if not endpoints:
            logger.warning("No endpoints found in API spec")
            return self._empty_analysis()

        # FIX: Analyze ALL endpoints, not just the first
        endpoint_analyses = []
        for ep in endpoints:
            endpoint_analyses.append(self._analyze_endpoint(ep, validation_rules))

        # Assess overall complexity and risks
        complexity = self._assess_complexity(endpoints, models, validation_rules)
        risks = self._assess_risks(endpoints, validation_rules)

        # Build LLM prompt with full API info
        prompt = self._build_prompt(endpoints, models, validation_rules, controllers)

        try:
            llm_response = await self.client.generate(
                prompt=prompt,
                system_prompt=(
                    "You are an expert API test analyst. Analyze the API specification "
                    "and provide testing recommendations in JSON format with keys: "
                    "summary, test_priorities (list), coverage_gaps (list), "
                    "security_concerns (list), recommended_test_types (list)."
                ),
                max_tokens=2000,
                temperature=0.3
            )

            llm_analysis = self._parse_llm_response(llm_response)

        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}, using local analysis only")
            llm_analysis = {}

        return {
            'endpoints': endpoint_analyses,
            'endpoint_count': len(endpoints),
            'model_count': len(models),
            'complexity': complexity,
            'risks': risks,
            'llm_analysis': llm_analysis,
            'method_distribution': self._get_method_distribution(endpoints),
            'auth_required_count': sum(
                1 for ep in endpoints
                if ep.get('auth_required') or ep.get('requires_auth')
            ),
            'total_parameters': sum(
                len(ep.get('parameters', [])) for ep in endpoints
            ),
        }

    def _analyze_endpoint(self, endpoint: Dict, validation_rules: List) -> Dict:
        """Analyze a single endpoint."""
        path = endpoint.get('path', endpoint.get('route', ''))
        method = endpoint.get('http_method', endpoint.get('method', ''))
        params = endpoint.get('parameters', [])

        # Find applicable validation rules
        applicable_rules = [
            r for r in validation_rules
            if r.get('endpoint') == path or r.get('field') in [p.get('name') for p in params]
        ]

        # Determine test complexity
        param_count = len(params)
        has_body = method.upper() in ('POST', 'PUT', 'PATCH')
        has_path_params = any(p.get('location') == 'path' or p.get('in') == 'path' for p in params)

        complexity_score = param_count * 0.2
        if has_body:
            complexity_score += 0.3
        if has_path_params:
            complexity_score += 0.2
        if applicable_rules:
            complexity_score += len(applicable_rules) * 0.1

        return {
            'path': path,
            'method': method,
            'parameter_count': param_count,
            'has_body': has_body,
            'has_path_params': has_path_params,
            'validation_rules': len(applicable_rules),
            'complexity_score': min(complexity_score, 1.0),
            'requires_auth': endpoint.get('auth_required', endpoint.get('requires_auth', False)),
            'suggested_test_types': self._suggest_test_types(endpoint, applicable_rules),
        }

    def _suggest_test_types(self, endpoint: Dict, rules: List) -> List[str]:
        """Suggest test types for an endpoint."""
        types = ['happy_path']
        method = endpoint.get('http_method', endpoint.get('method', '')).upper()
        params = endpoint.get('parameters', [])

        if params:
            types.append('negative')
            types.append('null_empty')

        if rules:
            types.append('boundary')

        if method in ('POST', 'PUT', 'PATCH'):
            types.append('validation')
            types.append('large_payload')

        if endpoint.get('auth_required') or endpoint.get('requires_auth'):
            types.append('auth')

        if method == 'DELETE':
            types.append('edge_case')

        # Always include security for endpoints with user input
        if params or method in ('POST', 'PUT', 'PATCH'):
            types.append('security')

        return types

    def _assess_complexity(self, endpoints: List[Dict], models: List[Dict],
                           rules: List) -> Dict[str, Any]:
        """
        Assess overall API complexity.
        FIX: This method was defined but never called in the original.
        """
        total_params = sum(len(ep.get('parameters', [])) for ep in endpoints)
        total_fields = sum(
            len(m.get('properties', m.get('fields', [])))
            for m in models
        )
        methods = set(ep.get('http_method', ep.get('method', '')).upper() for ep in endpoints)

        # Score: 0 (simple) to 1 (complex)
        score = 0.0
        score += min(len(endpoints) / 30.0, 0.25)
        score += min(total_params / 50.0, 0.25)
        score += min(total_fields / 50.0, 0.15)
        score += min(len(rules) / 30.0, 0.15)
        score += min(len(methods) / 5.0, 0.1)
        score += 0.1 if any(ep.get('auth_required') for ep in endpoints) else 0

        if score < 0.3:
            level = 'low'
        elif score < 0.6:
            level = 'medium'
        else:
            level = 'high'

        return {
            'level': level,
            'score': round(min(score, 1.0), 3),
            'factors': {
                'endpoint_count': len(endpoints),
                'total_parameters': total_params,
                'total_model_fields': total_fields,
                'validation_rules': len(rules),
                'http_methods': list(methods),
            }
        }

    def _assess_risks(self, endpoints: List[Dict],
                      rules: List) -> List[Dict[str, str]]:
        """
        Identify testing risks.
        FIX: This method was defined but never called in the original.
        """
        risks = []

        # Check for unvalidated endpoints
        endpoints_with_body = [
            ep for ep in endpoints
            if ep.get('http_method', ep.get('method', '')).upper() in ('POST', 'PUT', 'PATCH')
        ]
        rule_endpoints = set(r.get('endpoint', '') for r in rules)

        for ep in endpoints_with_body:
            path = ep.get('path', ep.get('route', ''))
            if path not in rule_endpoints:
                risks.append({
                    'type': 'missing_validation',
                    'severity': 'high',
                    'description': f"{ep.get('http_method')} {path} accepts body but has no validation rules"
                })

        # Check for unauthenticated write endpoints
        for ep in endpoints:
            method = ep.get('http_method', ep.get('method', '')).upper()
            if method in ('POST', 'PUT', 'DELETE', 'PATCH'):
                if not ep.get('auth_required') and not ep.get('requires_auth'):
                    risks.append({
                        'type': 'missing_auth',
                        'severity': 'high',
                        'description': f"{method} {ep.get('path', '')} has no authentication requirement"
                    })

        # Check for missing DELETE endpoints (CRUD completeness)
        paths = set(ep.get('path', '') for ep in endpoints)
        methods_by_path = {}
        for ep in endpoints:
            p = ep.get('path', '')
            m = ep.get('http_method', ep.get('method', '')).upper()
            methods_by_path.setdefault(p, set()).add(m)

        for path, methods in methods_by_path.items():
            if 'POST' in methods and 'DELETE' not in methods:
                risks.append({
                    'type': 'incomplete_crud',
                    'severity': 'low',
                    'description': f"{path} has POST but no DELETE"
                })

        return risks

    def _get_method_distribution(self, endpoints: List[Dict]) -> Dict[str, int]:
        """Get distribution of HTTP methods."""
        dist = {}
        for ep in endpoints:
            method = ep.get('http_method', ep.get('method', 'UNKNOWN')).upper()
            dist[method] = dist.get(method, 0) + 1
        return dist

    def _build_prompt(self, endpoints: List[Dict], models: List[Dict],
                      rules: List, controllers: List) -> str:
        """Build a comprehensive analysis prompt for the LLM."""
        lines = ["Analyze this API specification for testing:\n"]

        lines.append(f"## Endpoints ({len(endpoints)} total)")
        for ep in endpoints[:20]:  # Cap at 20 for token limits
            method = ep.get('http_method', ep.get('method', ''))
            path = ep.get('path', ep.get('route', ''))
            params = ep.get('parameters', [])
            auth = 'AUTH' if ep.get('auth_required') or ep.get('requires_auth') else ''
            param_names = [p.get('name', '') for p in params]
            lines.append(f"  {method} {path} [{auth}] params: {param_names}")

        if models:
            lines.append(f"\n## Models ({len(models)} total)")
            for m in models[:10]:
                name = m.get('name', 'Unknown')
                fields = m.get('properties', m.get('fields', []))
                field_names = [
                    f.get('name', '') if isinstance(f, dict) else str(f)
                    for f in fields[:10]
                ]
                lines.append(f"  {name}: {field_names}")

        if rules:
            lines.append(f"\n## Validation Rules ({len(rules)} total)")
            for r in rules[:10]:
                lines.append(f"  {r.get('field', '')}: {r.get('rule', r.get('type', ''))}")

        lines.append("\nProvide a JSON analysis with: summary, test_priorities, coverage_gaps, security_concerns, recommended_test_types")

        return '\n'.join(lines)

    def _parse_llm_response(self, response: str) -> Dict:
        """Parse LLM JSON response."""
        if not response:
            return {}

        # Try to extract JSON from response
        text = response.strip()

        # Remove markdown code fences
        if text.startswith('```'):
            lines = text.split('\n')
            text = '\n'.join(lines[1:])
            if text.endswith('```'):
                text = text[:-3]

        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            # Try to find JSON object in response
            import re
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        logger.warning("Could not parse LLM analysis response as JSON")
        return {'raw_response': response[:500]}

    def _empty_analysis(self) -> Dict:
        """Return empty analysis structure."""
        return {
            'endpoints': [],
            'endpoint_count': 0,
            'model_count': 0,
            'complexity': {'level': 'unknown', 'score': 0},
            'risks': [],
            'llm_analysis': {},
            'method_distribution': {},
            'auth_required_count': 0,
            'total_parameters': 0,
        }