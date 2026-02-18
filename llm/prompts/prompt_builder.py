"""
Dynamic prompt construction
"""

import logging
from typing import Dict, Any, List, Optional
import json

from llm.prompts.prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Builds dynamic prompts for LLM agents"""

    def __init__(self):
        self.templates = PromptTemplates()

    def build_analysis_prompt(self, api_spec: Dict[str, Any],
                              context: Dict[str, Any] = None) -> str:
        """Build prompt for API analysis"""

        # Format API specification
        api_str = self._format_api_spec(api_spec)

        prompt_parts = [
            "You are an expert API tester. Analyze the following API endpoint:",
            "",
            api_str,
            ""
        ]

        # Add context if available
        if context:
            prompt_parts.extend([
                "Consider the following context from similar APIs:",
                self._format_context(context),
                ""
            ])

        # Add specific instructions
        prompt_parts.extend([
            "Provide a comprehensive analysis including:",
            "1. All parameters that need testing",
            "2. Business logic constraints",
            "3. Security considerations",
            "4. Edge cases and boundary conditions",
            "5. Performance requirements",
            "",
            "Format your response as structured JSON."
        ])

        return "\n".join(prompt_parts)

    def build_test_generation_prompt(self, test_type: str,
                                     api_spec: Dict[str, Any],
                                     context: Dict[str, Any] = None,
                                     examples: List[Dict[str, Any]] = None) -> str:
        """Build prompt for test generation"""

        prompt_parts = [
            f"Generate {test_type} test cases for the following API:",
            "",
            self._format_api_spec(api_spec),
            ""
        ]

        # Add examples if provided
        if examples:
            prompt_parts.extend([
                "Here are some example test cases for reference:",
                self._format_examples(examples),
                ""
            ])

        # Add context
        if context:
            prompt_parts.extend([
                "Context from similar tests:",
                self._format_context(context),
                ""
            ])

        # Add test-type specific instructions
        instructions = self._get_test_type_instructions(test_type)
        prompt_parts.extend([
            "Generate test cases that:",
            instructions,
            "",
            "Return as JSON array with structure:",
            self._get_test_case_schema(test_type)
        ])

        return "\n".join(prompt_parts)

    def build_data_generation_prompt(self, parameters: List[Dict[str, Any]],
                                     test_type: str = "valid",
                                     constraints: Dict[str, Any] = None) -> str:
        """Build prompt for test data generation"""

        prompt_parts = [
            f"Generate {test_type} test data for the following parameters:",
            "",
            json.dumps(parameters, indent=2),
            ""
        ]

        if constraints:
            prompt_parts.extend([
                "Apply the following constraints:",
                json.dumps(constraints, indent=2),
                ""
            ])

        if test_type == "valid":
            prompt_parts.append("Generate realistic, valid data that satisfies all constraints.")
        elif test_type == "invalid":
            prompt_parts.append("Generate invalid data that violates constraints for testing validation.")
        elif test_type == "edge":
            prompt_parts.append("Generate edge case data including boundary values and extreme cases.")

        prompt_parts.extend([
            "",
            "Return as JSON object with parameter names as keys."
        ])

        return "\n".join(prompt_parts)

    def build_report_prompt(self, results: List[Dict[str, Any]],
                            report_type: str = "summary") -> str:
        """Build prompt for report generation"""

        if report_type == "summary":
            return self._build_summary_prompt(results)
        elif report_type == "detailed":
            return self._build_detailed_report_prompt(results)
        elif report_type == "recommendations":
            return self._build_recommendations_prompt(results)
        else:
            return self._build_generic_report_prompt(results)

    def _format_api_spec(self, api_spec: Dict[str, Any]) -> str:
        """Format API specification for prompt"""
        parts = []

        parts.append(f"Endpoint: {api_spec.get('path', 'N/A')}")
        parts.append(f"Method: {api_spec.get('method', 'N/A')}")

        if 'parameters' in api_spec:
            parts.append("\nParameters:")
            for param in api_spec['parameters']:
                param_str = f"  - {param.get('name')}: {param.get('type')} "
                param_str += f"({'required' if param.get('required') else 'optional'})"

                if 'constraints' in param:
                    param_str += f"\n    Constraints: {param['constraints']}"

                parts.append(param_str)

        if 'requestBody' in api_spec:
            parts.append(f"\nRequest Body: {json.dumps(api_spec['requestBody'], indent=2)}")

        if 'responses' in api_spec:
            parts.append("\nResponses:")
            for code, response in api_spec['responses'].items():
                parts.append(f"  {code}: {response.get('description', 'N/A')}")

        return "\n".join(parts)

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context for prompt"""
        parts = []

        if 'similar_tests' in context:
            parts.append("Similar Tests:")
            for test in context['similar_tests'][:3]:
                parts.append(f"  - {test.get('name', 'Test')}")

        if 'edge_cases' in context:
            parts.append("\nEdge Cases:")
            for edge in context['edge_cases'][:3]:
                parts.append(f"  - {edge.get('description', 'Edge case')}")

        return "\n".join(parts)

    def _format_examples(self, examples: List[Dict[str, Any]]) -> str:
        """Format examples for prompt"""
        formatted = []
        for example in examples[:3]:  # Limit to 3 examples
            formatted.append(json.dumps(example, indent=2))
        return "\n---\n".join(formatted)

    def _get_test_type_instructions(self, test_type: str) -> str:
        """Get instructions for specific test type"""
        instructions = {
            'happy_path': """- Cover basic successful operations
- Use valid, realistic data
- Verify expected responses
- Test different valid parameter combinations""",

            'validation': """- Test required field validation
- Verify data type constraints
- Check format requirements
- Test length and size limits
- Validate pattern matching""",

            'authentication': """- Test with no credentials
- Use invalid tokens
- Test expired tokens
- Verify permission requirements
- Check authorization levels""",

            'edge_case': """- Test boundary values
- Use extreme inputs
- Test null/empty values
- Try special characters
- Test type mismatches""",

            'security': """- Test SQL injection
- Check XSS vulnerabilities
- Test command injection
- Verify path traversal protection
- Test authentication bypass""",

            'performance': """- Test response times
- Verify throughput
- Test concurrent access
- Check resource usage
- Test with large payloads"""
        }

        return instructions.get(test_type, "- Generate comprehensive test cases")

    def _get_test_case_schema(self, test_type: str) -> str:
        """Get JSON schema for test case"""
        base_schema = {
            "name": "descriptive test name",
            "description": "what this test validates",
            "test_type": test_type,
            "input": {"parameter": "value"},
            "expected_status": 200,
            "assertions": ["assertion1", "assertion2"]
        }

        # Add type-specific fields
        if test_type == "performance":
            base_schema["performance_criteria"] = {
                "max_response_time": 1000,
                "min_throughput": 100
            }
        elif test_type == "security":
            base_schema["attack_vector"] = "injection type"
            base_schema["expected_protection"] = "how API should respond"

        return json.dumps(base_schema, indent=2)

    def _build_summary_prompt(self, results: List[Dict[str, Any]]) -> str:
        """Build summary report prompt"""
        stats = {
            'total': len(results),
            'passed': sum(1 for r in results if r.get('passed')),
            'failed': sum(1 for r in results if not r.get('passed'))
        }

        return f"""Generate a concise test execution summary:

Statistics:
{json.dumps(stats, indent=2)}

Failed Tests:
{self._format_failures(results)}

Provide:
1. Overall assessment
2. Critical issues
3. Recommendations"""

    def _build_detailed_report_prompt(self, results: List[Dict[str, Any]]) -> str:
        """Build detailed report prompt"""
        return f"""Generate a detailed test report for:

{json.dumps(results[:5], indent=2)}  

Include for each test:
1. Test objective
2. Execution steps
3. Results
4. Failure analysis (if applicable)"""

    def _build_recommendations_prompt(self, results: List[Dict[str, Any]]) -> str:
        """Build recommendations prompt"""
        failures = [r for r in results if not r.get('passed')]

        return f"""Based on these test failures:

{json.dumps(failures[:5], indent=2)}

Provide:
1. Root cause analysis
2. Priority fixes
3. Additional test coverage needed
4. Process improvements"""

    def _build_generic_report_prompt(self, results: List[Dict[str, Any]]) -> str:
        """Build generic report prompt"""
        return f"""Generate a test report for the following results:

{json.dumps(results[:10], indent=2)}

Format as a professional test report."""

    def _format_failures(self, results: List[Dict[str, Any]]) -> str:
        """Format test failures for prompt"""
        failures = [r for r in results if not r.get('passed')][:5]

        formatted = []
        for failure in failures:
            formatted.append(f"- {failure.get('name', 'Test')}: {failure.get('error', 'Failed')}")

        return "\n".join(formatted) if formatted else "No failures"