"""
Parse and validate LLM responses
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)


class ResponseParser:
    """Parses and validates LLM responses"""

    def __init__(self):
        self.parsers = {
            'json': self.parse_json,
            'list': self.parse_list,
            'code': self.parse_code,
            'text': self.parse_text,
            'structured': self.parse_structured
        }

    def parse(self, response: str, expected_format: str = 'json') -> Any:
        """
        Parse LLM response based on expected format

        Args:
            response: Raw LLM response
            expected_format: Expected response format

        Returns:
            Parsed response
        """
        parser = self.parsers.get(expected_format, self.parse_text)
        return parser(response)

    def parse_json(self, response: str) -> Union[Dict, List]:
        """Parse JSON response"""
        # Clean response
        cleaned = self._clean_json_response(response)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")

            # Try to extract JSON from response
            json_match = re.search(r'(\{.*\}|\[.*\])', cleaned, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass

            # Return as structured text
            return self.parse_structured(response)

    def parse_list(self, response: str) -> List[str]:
        """Parse list response"""
        # Try JSON first
        try:
            result = self.parse_json(response)
            if isinstance(result, list):
                return result
        except:
            pass

        # Parse as text list
        lines = response.strip().split('\n')
        items = []

        for line in lines:
            # Remove list markers
            line = re.sub(r'^[\-\*\d\.]+\s*', '', line.strip())
            if line:
                items.append(line)

        return items

    def parse_code(self, response: str) -> Dict[str, str]:
        """Parse code response"""
        # Extract code blocks
        code_blocks = re.findall(r'```(\w*)\n(.*?)```', response, re.DOTALL)

        if code_blocks:
            return {
                'language': code_blocks[0][0] or 'unknown',
                'code': code_blocks[0][1].strip(),
                'description': self._extract_description(response, code_blocks[0][1])
            }

        # Try to extract inline code
        inline_code = re.findall(r'`([^`]+)`', response)
        if inline_code:
            return {
                'language': 'unknown',
                'code': '\n'.join(inline_code),
                'description': response
            }

        return {
            'language': 'unknown',
            'code': response,
            'description': ''
        }

    def parse_text(self, response: str) -> str:
        """Parse plain text response"""
        return response.strip()

    def parse_structured(self, response: str) -> Dict[str, Any]:
        """Parse structured text response"""
        result = {}

        # Look for sections
        sections = re.split(r'\n(?=\d+\.|#{1,3}\s|\*\*)', response)

        current_section = 'content'
        for section in sections:
            section = section.strip()
            if not section:
                continue

            # Check if it's a header
            header_match = re.match(r'^(?:\d+\.|#{1,3}\s|\*\*)(.+)', section)
            if header_match:
                current_section = header_match.group(1).strip().lower().replace(' ', '_')
                # Remove header from content
                section = re.sub(r'^(?:\d+\.|#{1,3}\s|\*\*).*\n?', '', section)

            if current_section not in result:
                result[current_section] = []

            result[current_section].append(section.strip())

        # Convert single-item lists to strings
        for key, value in result.items():
            if isinstance(value, list) and len(value) == 1:
                result[key] = value[0]

        return result

    def validate_test_case(self, test_case: Dict[str, Any]) -> bool:
        """Validate test case structure"""
        required_fields = ['name', 'endpoint', 'method']

        for field in required_fields:
            if field not in test_case:
                logger.warning(f"Missing required field: {field}")
                return False

        return True

    def validate_analysis(self, analysis: Dict[str, Any]) -> bool:
        """Validate analysis structure"""
        required_fields = ['endpoint', 'method', 'critical_parameters']

        for field in required_fields:
            if field not in analysis:
                logger.warning(f"Missing required analysis field: {field}")
                return False

        return True

    def _clean_json_response(self, response: str) -> str:
        """Clean JSON response"""
        # Remove markdown code blocks
        response = re.sub(r'```json?\n?', '', response)
        response = re.sub(r'```', '', response)

        # Remove leading/trailing whitespace
        response = response.strip()

        # Remove any text before first { or [
        json_start = min(
            response.find('{') if '{' in response else len(response),
            response.find('[') if '[' in response else len(response)
        )

        if json_start < len(response):
            response = response[json_start:]

        # Remove any text after last } or ]
        json_end = max(
            response.rfind('}') if '}' in response else -1,
            response.rfind(']') if ']' in response else -1
        )

        if json_end > -1:
            response = response[:json_end + 1]

        return response

    def _extract_description(self, full_response: str, code: str) -> str:
        """Extract description from response"""
        # Remove code from response
        description = full_response.replace(code, '')
        description = re.sub(r'```\w*\n?```', '', description)

        return description.strip()