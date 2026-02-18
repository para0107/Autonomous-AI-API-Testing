"""
Endpoint extraction from parsed API code
"""

import logging
from typing import Dict, List, Any
import re

logger = logging.getLogger(__name__)


class EndpointExtractor:
    """Extracts and normalizes API endpoints from parsed code"""

    def __init__(self):
        self.endpoint_patterns = {
            'path_param': r'{(\w+)}|:(\w+)',
            'query_param': r'\?.*',
            'version': r'/v\d+/',
        }

    def extract(self, parsed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract endpoints from parsed data

        Args:
            parsed_data: Parsed code data from language parsers

        Returns:
            List of normalized endpoint definitions
        """
        endpoints = parsed_data.get('endpoints', [])
        normalized_endpoints = []

        for endpoint in endpoints:
            normalized = self.normalize_endpoint(endpoint)
            if normalized:
                normalized_endpoints.append(normalized)

        return normalized_endpoints

    def normalize_endpoint(self, endpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize endpoint data across different languages"""
        path = endpoint.get('path') or endpoint.get('route') or endpoint.get('url', '')

        # Extract common fields
        normalized = {
            'path': self.normalize_path(path),
            'method': self.normalize_method(endpoint.get('method') or
                                            endpoint.get('http_method', 'GET')),
            'name': endpoint.get('function_name') or
                    endpoint.get('method_name') or
                    endpoint.get('name', 'unknown'),
            'parameters': self.normalize_parameters(endpoint.get('parameters', [])),
            'authentication': self.extract_auth_requirements(endpoint),
            'return_type': endpoint.get('return_type'),
            'description': endpoint.get('description', ''),
            'tags': self.extract_tags(endpoint)
        }

        return normalized

    def normalize_path(self, path: str) -> str:
        """Normalize API path"""
        if not path:
            return ''

        # Ensure path starts with /
        if not path.startswith('/'):
            path = '/' + path

        # Remove trailing slash
        path = path.rstrip('/')

        # Normalize path parameters
        # Convert different formats to OpenAPI style {param}
        path = re.sub(r':(\w+)', r'{\1}', path)  # Express style
        path = re.sub(r'<(\w+)>', r'{\1}', path)  # Flask style

        return path

    def normalize_method(self, method: str) -> str:
        """Normalize HTTP method"""
        if isinstance(method, list):
            return method[0].upper() if method else 'GET'
        return method.upper()

    def normalize_parameters(self, parameters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize parameter definitions"""
        normalized_params = []

        for param in parameters:
            normalized = {
                'name': param.get('name', ''),
                'type': self.normalize_type(param.get('type', 'string')),
                'in': self.normalize_param_location(param.get('source') or
                                                    param.get('in', 'query')),
                'required': param.get('required', False),
                'description': param.get('description', ''),
                'default': param.get('default'),
                'constraints': self.extract_constraints(param)
            }

            normalized_params.append(normalized)

        return normalized_params

    def normalize_type(self, type_str: str) -> str:
        """Normalize data types across languages"""
        type_mapping = {
            # C# to OpenAPI
            'string': 'string',
            'String': 'string',
            'int': 'integer',
            'Int32': 'integer',
            'long': 'integer',
            'Int64': 'integer',
            'float': 'number',
            'double': 'number',
            'Double': 'number',
            'bool': 'boolean',
            'Boolean': 'boolean',
            'DateTime': 'string',
            'DateOnly': 'string',
            'Guid': 'string',

            # Python to OpenAPI
            'str': 'string',
            'int': 'integer',
            'float': 'number',
            'bool': 'boolean',
            'list': 'array',
            'dict': 'object',

            # Java to OpenAPI
            'String': 'string',
            'Integer': 'integer',
            'Long': 'integer',
            'Float': 'number',
            'Double': 'number',
            'Boolean': 'boolean',
            'Date': 'string',
            'List': 'array',
            'Map': 'object',
        }

        # Handle generic types
        if '<' in type_str:
            base_type = type_str.split('<')[0]
            return type_mapping.get(base_type, 'object')

        return type_mapping.get(type_str, type_str.lower())

    def normalize_param_location(self, location: str) -> str:
        """Normalize parameter location"""
        location_mapping = {
            'body': 'body',
            'query': 'query',
            'path': 'path',
            'route': 'path',
            'header': 'header',
            'form': 'formData'
        }

        return location_mapping.get(location.lower(), 'query')

    def extract_constraints(self, param: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameter constraints"""
        constraints = {}

        # Common constraint fields
        constraint_fields = [
            'min', 'max', 'minLength', 'maxLength',
            'pattern', 'enum', 'format', 'multipleOf'
        ]

        for field in constraint_fields:
            if field in param:
                constraints[field] = param[field]

        # Extract from validations
        if 'validations' in param:
            for validation in param['validations']:
                if validation.get('type') == 'min_length':
                    constraints['minLength'] = validation.get('value')
                elif validation.get('type') == 'max_length':
                    constraints['maxLength'] = validation.get('value')
                elif validation.get('type') == 'pattern':
                    constraints['pattern'] = validation.get('value')

        return constraints

    def extract_auth_requirements(self, endpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Extract authentication requirements"""
        auth = endpoint.get('authorization', {})

        if isinstance(auth, dict):
            return auth

        # Check for common auth patterns
        auth_info = {
            'required': False,
            'type': None,
            'scopes': []
        }

        # Check various auth indicators
        if endpoint.get('auth_required'):
            auth_info['required'] = True

        if endpoint.get('auth_type'):
            auth_info['type'] = endpoint['auth_type']

        return auth_info

    def extract_tags(self, endpoint: Dict[str, Any]) -> List[str]:
        """Extract tags for endpoint categorization"""
        tags = []

        # Add controller/class name as tag
        if endpoint.get('controller'):
            tags.append(endpoint['controller'])

        # Add explicit tags
        if endpoint.get('tags'):
            if isinstance(endpoint['tags'], list):
                tags.extend(endpoint['tags'])
            else:
                tags.append(endpoint['tags'])

        # Add method-based tags
        method = endpoint.get('method', '').upper()
        if method in ['POST', 'PUT', 'PATCH']:
            tags.append('mutation')
        elif method == 'GET':
            tags.append('query')
        elif method == 'DELETE':
            tags.append('deletion')

        return list(set(tags))

    def group_endpoints(self, endpoints: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group endpoints by resource"""
        grouped = {}

        for endpoint in endpoints:
            # Group by resource name (first path segment after base)
            path = endpoint.get('path', '')
            if path:
                # Extract resource from path
                parts = path.strip('/').split('/')
                if parts:
                    # Skip version numbers
                    resource = parts[0]
                    if resource.startswith('v') and resource[1:].isdigit():
                        resource = parts[1] if len(parts) > 1 else 'root'

                    if resource not in grouped:
                        grouped[resource] = []
                    grouped[resource].append(endpoint)

        return grouped