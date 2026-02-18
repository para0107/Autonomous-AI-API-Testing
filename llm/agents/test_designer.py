"""
Test Designer Agent - WITH RAG CONTEXT
"""
import asyncio
import json
import logging
from typing import Dict, Any, List

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class TestDesignerAgent(BaseAgent):
    """Agent for designing test cases"""

    def __init__(self, llama_client):
        super().__init__(llama_client, 'test_designer')

    async def execute(self, input_data: Dict[str, Any]) -> dict[str, Any]:
        """Design test cases"""
        analysis = input_data.get('analyzer_results', {})
        context = input_data.get('context', {})
        config = input_data.get('config', {})

        return await self.design_tests(analysis, context)

    async def design_tests(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Design tests with RAG context"""

        # Log RAG availability
        similar_count = len(context.get('similar_tests', []))
        logger.info(f"Designing tests with {similar_count} RAG examples available")

        happy_path = []
        try:
            happy_path = await self._generate_happy_path_tests_with_rag(analysis, context)
            logger.info(f"✅ Generated {len(happy_path)} happy path tests")
        except Exception as e:
            logger.error(f"Happy path generation failed: {e}")

        return {
            'happy_path_tests': happy_path,
            'edge_case_tests': [],
            'validation_tests': [],
            'total_tests': len(happy_path)
        }

    async def _generate_happy_path_tests_with_rag(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate tests using RAG examples"""

        # Format RAG examples for prompt
        rag_section = self._format_rag_examples(context)

        endpoint = analysis.get('endpoint', '/api/endpoint')
        method = analysis.get('method', 'GET')

        prompt = f"""You are an expert API test designer. Generate 5 comprehensive test cases.

ENDPOINT TO TEST:
- Path: {endpoint}
- Method: {method}
- Auth Required: {analysis.get('auth_requirements', {}).get('required', False)}

{rag_section}

Generate 5 test cases in this EXACT JSON format:
[
  {{
    "name": "test_get_reservation_valid_id",
    "test_type": "happy_path",
    "method": "{method}",
    "endpoint": "{endpoint}",
    "test_data": {{"id": 1}},
    "expected_status": 200,
    "description": "Retrieve reservation with valid ID"
  }}
]

CRITICAL RULES:
1. Use the actual endpoint: {endpoint}
2. Each test MUST have: name, test_type, method, endpoint, test_data, expected_status
3. Generate realistic test data based on the endpoint
4. Return ONLY the JSON array, no other text
5. Learn from the examples above to create similar high-quality tests

Generate JSON array now:"""

        try:
            tests = await self.generate_json_with_retry(prompt, max_retries=2)
            if isinstance(tests, list) and len(tests) > 0:
                logger.info(f"Generated {len(tests)} tests using RAG context")
                return tests[:10]
            return []
        except Exception as e:
            logger.error(f"Test generation failed: {e}")
            return []

    def _format_rag_examples(self, context: Dict[str, Any]) -> str:
        """Format RAG examples for the prompt"""
        similar_tests = context.get('similar_tests', [])

        if not similar_tests or len(similar_tests) == 0:
            logger.warning("No RAG examples to format")
            return "⚠️ No similar test examples found in knowledge base."

        logger.info(f"Formatting {len(similar_tests)} RAG examples")

        examples_text = "📚 EXAMPLES FROM KNOWLEDGE BASE (476 test cases):\n"
        examples_text += "Learn from these real test cases:\n\n"

        # Handle different data structures
        for idx, item in enumerate(similar_tests[:3], 1):
            try:
                # Try unpacking as tuple
                if isinstance(item, tuple) and len(item) == 2:
                    distance, metadata = item
                elif isinstance(item, tuple) and len(item) == 3:
                    # If it's (id, distance, metadata)
                    _, distance, metadata = item
                elif isinstance(item, dict):
                    # If it's already a dict
                    distance = item.get('score', 0.5)
                    metadata = item
                else:
                    logger.warning(f"Unknown item type: {type(item)}")
                    continue

                if not metadata:
                    continue

                title = metadata.get('title', 'Unknown')
                description = metadata.get('description', '')
                steps = metadata.get('steps', [])

                similarity = max(0, 1 - distance) if distance < 10 else distance
                examples_text += f"Example {idx} (relevance: {similarity:.1%}): {title}\n"

                if description:
                    examples_text += f"  Description: {description}\n"

                if steps:
                    for step_idx, step in enumerate(steps[:2], 1):
                        action = step.get('action', '')
                        expected = step.get('expected_result', '')
                        if action:
                            examples_text += f"  Step: {action}\n"
                        if expected:
                            examples_text += f"    Expected: {expected}\n"

                examples_text += "\n"

            except Exception as e:
                logger.error(f"Error formatting RAG example {idx}: {e}")
                continue

        return examples_text