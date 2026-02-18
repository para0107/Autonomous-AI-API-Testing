"""
Builds OpenAPI-like specification from parsed code
"""

import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class SpecificationBuilder:
    """Builds API specification from parsed data"""

    def build(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build API specification"""
        logger.info("Building API specification")

        # Extract all data from parsed results
        all_endpoints = []
        all_models = []
        all_controllers = []

        # Handle both single file and multiple files
        if isinstance(parsed_data, dict):
            if 'endpoints' in parsed_data:
                # Single file format
                all_endpoints = parsed_data.get('endpoints', [])
                all_models = parsed_data.get('models', [])

                # Extract controller info
                for endpoint in all_endpoints:
                    controller_name = endpoint.get('controller', 'Unknown')
                    if controller_name not in [c['name'] for c in all_controllers]:
                        all_controllers.append({'name': controller_name})

            elif 'results' in parsed_data:
                # Multiple files format
                for result in parsed_data.get('results', []):
                    all_endpoints.extend(result.get('endpoints', []))
                    all_models.extend(result.get('models', []))

                    # Extract controller info
                    for endpoint in result.get('endpoints', []):
                        controller_name = endpoint.get('controller', 'Unknown')
                        if controller_name not in [c['name'] for c in all_controllers]:
                            all_controllers.append({'name': controller_name})

        logger.info(f"Building spec with {len(all_endpoints)} endpoints, {len(all_models)} models")

        specification = {
            'openapi': '3.0.0',
            'info': {
                'title': 'Generated API Specification',
                'version': '1.0.0'
            },
            'paths': self._build_paths(all_endpoints),
            'endpoints': all_endpoints,  # CRITICAL: Keep raw endpoints
            'controllers': all_controllers,
            'models': all_models,
            'components': {
                'schemas': self._build_schemas(all_models)
            }
        }

        logger.info(f"Specification built with {len(specification['endpoints'])} endpoints")
        return specification

    def _build_paths(self, endpoints: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build OpenAPI paths from endpoints"""
        paths = {}

        for endpoint in endpoints:
            path = endpoint.get('path') or endpoint.get('route', '/unknown')
            method = endpoint.get('http_method', 'GET').lower()

            if path not in paths:
                paths[path] = {}

            paths[path][method] = {
                'operationId': endpoint.get('method_name', 'unknown'),
                'parameters': self._build_parameters(endpoint.get('parameters', [])),
                'responses': {
                    '200': {'description': 'Successful response'},
                    '400': {'description': 'Bad request'},
                    '401': {'description': 'Unauthorized'},
                    '404': {'description': 'Not found'},
                    '500': {'description': 'Internal server error'}
                }
            }

            # Add authorization info if present
            if endpoint.get('authorization', {}).get('required'):
                paths[path][method]['security'] = [{'BearerAuth': []}]

        return paths

    def _build_parameters(self, parameters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build OpenAPI parameters"""
        openapi_params = []

        for param in parameters:
            param_location = param.get('source', 'query')

            # Map source to OpenAPI 'in' field
            in_mapping = {
                'query': 'query',
                'route': 'path',
                'path': 'path',
                'body': 'body',
                'header': 'header'
            }

            openapi_param = {
                'name': param.get('name', 'unknown'),
                'in': in_mapping.get(param_location, 'query'),
                'required': param.get('required', False),
                'schema': {
                    'type': self._map_type(param.get('type', 'string'))
                }
            }

            openapi_params.append(openapi_param)

        return openapi_params

    def _map_type(self, csharp_type: str) -> str:
        """Map C# types to OpenAPI types"""
        type_mapping = {
            'int': 'integer',
            'long': 'integer',
            'string': 'string',
            'bool': 'boolean',
            'double': 'number',
            'float': 'number',
            'decimal': 'number',
            'DateTime': 'string',
            'DateOnly': 'string',
            'Guid': 'string'
        }

        # Remove nullable marker
        clean_type = csharp_type.replace('?', '')

        # Check for generic types
        if '<' in clean_type:
            return 'object'

        return type_mapping.get(clean_type, 'string')

    def _build_schemas(self, models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build OpenAPI schemas from models"""
        schemas = {}

        for model in models:
            model_name = model.get('name', 'UnknownModel')
            properties = {}

            for prop in model.get('properties', []):
                properties[prop['name']] = {
                    'type': self._map_type(prop['type']),
                    'nullable': prop.get('nullable', False)
                }

            schemas[model_name] = {
                'type': 'object',
                'properties': properties
            }

        return schemas