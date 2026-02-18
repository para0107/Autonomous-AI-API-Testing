"""
C# specific parser for API code
"""

import re
import logging
from typing import Dict, List, Any
from .base_parser import BaseParser

logger = logging.getLogger(__name__)


class CSharpParser(BaseParser):
    """Parser for C# API code"""

    def __init__(self):
        super().__init__()
        self.patterns = {
            'class': r'public\s+class\s+(\w+)',
            'route': r'\[Route\("([^"]+)"\)\]',
            'http_method': r'\[Http(Get|Post|Put|Delete|Patch)(?:\("([^"]*)"\))?\]',
            'authorize': r'\[Authorize(?:\(([^)]*)\))?\]',
            'from_body': r'\[FromBody\]\s*(\w+(?:<.*?>)?)\s+(\w+)',
            'from_query': r'\[FromQuery\]\s*(\w+\??)\s+(\w+)',
            'from_route': r'\[FromRoute\]\s*(\w+\??)\s+(\w+)',
            'method': r'public\s+(?:async\s+)?(?:Task<)?(\w+)(?:>)?\s+(\w+)\s*\([^)]*\)',
            'service_injection': r'private\s+readonly\s+(\w+)\s+_(\w+);',
            'validation': r'RuleFor\(.*?\)\..*?;',
            'exception': r'throw\s+new\s+(\w+Exception)\((.*?)\)',
        }

    def parse(self, code_files: List[str]) -> Dict[str, Any]:
        """Parse C# code files"""
        logger.info(f"Parsing {len(code_files)} C# files")

        results = []
        for file_path in code_files:
            code = self.read_file(file_path)

            file_result = {
                'file': file_path,
                'endpoints': self.extract_endpoints(code),
                'services': self.extract_services(code),
                'validators': self.extract_validation_rules(code),
                'methods': self.extract_methods(code),
                'models': self.extract_models(code),
                'dependencies': self.extract_dependencies(code)
            }

            logger.info(f"Extracted {len(file_result['endpoints'])} endpoints from {file_path}")

            results.append(file_result)

        return self.combine_results(results)

    def extract_endpoints(self, code: str) -> List[Dict[str, Any]]:
        """Extract API endpoints from C# controller"""
        endpoints = []

        # Find controller class
        class_match = re.search(self.patterns['class'], code)
        if not class_match:
            logger.warning("No controller class found")
            return endpoints

        controller_name = class_match.group(1)
        logger.info(f"Found controller: {controller_name}")

        # Find base route
        base_route_match = re.search(r'\[Route\("([^"]+)"\)\]', code)
        base_route = ""
        if base_route_match:
            base_route = base_route_match.group(1)
            if not base_route.startswith("api/"):
                base_route = f"api/{base_route}"
            logger.info(f"Base route: {base_route}")
        else:
            logger.warning("No base route found")

        # Check for controller-level authorization
        controller_auth = re.search(r'class\s+' + controller_name + r'.*?\[Authorize(?:\(([^)]*)\))?\]', code,
                                    re.DOTALL)

        # Split code into method blocks - look for HTTP method attributes
        method_blocks = re.split(r'(?=\s*\[Http(?:Get|Post|Put|Delete|Patch))', code)

        logger.info(f"Found {len(method_blocks)} potential method blocks")

        for block in method_blocks:
            # Find HTTP method attribute
            http_match = re.search(r'\[Http(Get|Post|Put|Delete|Patch)(?:\("([^"]*)"\))?\]', block)
            if not http_match:
                continue

            http_method = http_match.group(1).upper()
            route_template = http_match.group(2) if http_match.group(2) else ""

            # Find method signature - FIXED PATTERN
            method_match = re.search(
                r'public\s+(?:async\s+)?(?:Task<)?(ActionResult<?[^>]*>?|IActionResult)>?\s+(\w+)\s*\(',
                block
            )

            if not method_match:
                logger.warning(f"Could not parse method signature in block starting with [{http_method}]")
                continue

            method_name = method_match.group(2)
            logger.info(f"Found method: {method_name} [{http_method}]")

            # Extract parameters
            parameters = self.extract_parameters(block)

            # Check for method-level authorization
            method_auth = re.search(r'\[Authorize(?:\(([^)]*)\))?\]', block)
            has_auth = bool(controller_auth or method_auth)
            policy = None
            if method_auth:
                policy = self._extract_policy(method_auth)
            elif controller_auth:
                policy = self._extract_policy(controller_auth)

            # Build complete route
            full_route = self._build_route(base_route, route_template, parameters)

            endpoint = {
                'controller': controller_name,
                'method_name': method_name,
                'http_method': http_method,
                'route': full_route,
                'path': full_route,
                'parameters': parameters,
                'authorization': {
                    'required': has_auth,
                    'policy': policy
                }
            }

            endpoints.append(endpoint)
            logger.info(f"Added endpoint: {http_method} {full_route}")

        logger.info(f"Total endpoints extracted: {len(endpoints)}")
        return endpoints

    def _build_route(self, base_route: str, route_template: str, parameters: List[Dict]) -> str:
        """Build complete route path"""
        # Start with base route
        if not base_route.startswith('/'):
            base_route = f"/{base_route}"

        # Add route template if exists
        if route_template:
            if not route_template.startswith('/'):
                route_template = f"/{route_template}"
            full_route = f"{base_route}{route_template}"
        else:
            full_route = base_route

        # Ensure single slashes
        full_route = re.sub(r'/+', '/', full_route)

        return full_route

    def _extract_policy(self, auth_match):
        """Extract policy from Authorize attribute"""
        if not auth_match:
            return None

        auth_content = auth_match.group(1) if auth_match.lastindex and auth_match.group(1) else ""
        policy_match = re.search(r'Policy\s*=\s*"([^"]+)"', auth_content)
        return policy_match.group(1) if policy_match else None

    def extract_methods(self, code: str) -> List[Dict[str, Any]]:
        """Extract methods from C# code"""
        methods = []

        method_matches = re.finditer(self.patterns['method'], code)

        for match in method_matches:
            return_type = match.group(1)
            method_name = match.group(2)

            # Extract method body
            start_pos = match.end()
            brace_count = 0
            end_pos = start_pos

            for i, char in enumerate(code[start_pos:], start_pos):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i
                        break

            method_body = code[start_pos:end_pos + 1] if end_pos > start_pos else ""

            methods.append({
                'name': method_name,
                'return_type': return_type,
                'body': method_body,
                'parameters': self.extract_parameters(code[match.start():end_pos + 1])
            })

        return methods

    def extract_parameters(self, method_code: str) -> List[Dict[str, Any]]:
        """Extract parameters from C# method"""
        parameters = []

        # FromBody parameters
        from_body_matches = re.findall(r'\[FromBody\]\s*(\w+(?:<.*?>)?)\s+(\w+)', method_code)
        for match in from_body_matches:
            parameters.append({
                'name': match[1],
                'type': match[0],
                'source': 'body',
                'required': True
            })

        # FromQuery parameters
        from_query_matches = re.findall(r'\[FromQuery\]\s*(\w+\??(?:<.*?>)?)\s+(\w+)', method_code)
        for match in from_query_matches:
            param_type = match[0]
            parameters.append({
                'name': match[1],
                'type': param_type,
                'source': 'query',
                'required': '?' not in param_type
            })

        # FromRoute parameters
        from_route_matches = re.findall(r'\[FromRoute\]\s*(\w+\??)\s+(\w+)', method_code)
        for match in from_route_matches:
            parameters.append({
                'name': match[1],
                'type': match[0],
                'source': 'route',
                'required': '?' not in match[0]
            })

        # Method signature parameters (route parameters from method signature)
        # Extract parameters from method signature like: MethodName(int userId, int id)
        sig_match = re.search(r'\(([^)]*)\)', method_code)
        if sig_match:
            params_str = sig_match.group(1)
            # Pattern to match: type name (ignoring attributes)
            param_pattern = r'(?:\[.*?\]\s*)?(\w+\??(?:<.*?>)?)\s+(\w+)(?:\s*=\s*[^,)]+)?'

            for match in re.finditer(param_pattern, params_str):
                param_type = match.group(1)
                param_name = match.group(2)

                # Check if already added with attribute
                if not any(p['name'] == param_name for p in parameters):
                    # Determine source based on parameter name and type
                    source = 'route'  # Default for path parameters

                    parameters.append({
                        'name': param_name,
                        'type': param_type,
                        'source': source,
                        'required': '?' not in param_type and '=' not in match.group(0)
                    })

        return parameters

    def extract_validation_rules(self, code: str) -> List[Dict[str, Any]]:
        """Extract FluentValidation rules from C# code"""
        validators = []

        # Find validator classes
        validator_classes = re.findall(r'public\s+class\s+(\w+Validator)\s*:\s*AbstractValidator<(\w+)>', code)

        for validator_name, model_type in validator_classes:
            # Extract validation rules
            rules = re.findall(self.patterns['validation'], code)

            parsed_rules = []
            for rule in rules:
                # Parse rule details
                rule_parts = self.parse_validation_rule(rule)
                if rule_parts:
                    parsed_rules.append(rule_parts)

            validators.append({
                'validator_name': validator_name,
                'model_type': model_type,
                'rules': parsed_rules
            })

        return validators

    def parse_validation_rule(self, rule: str) -> Dict[str, Any]:
        """Parse a single FluentValidation rule"""

        # Extract property name
        prop_match = re.search(r'RuleFor\(.*?=>\s*.*?\.(\w+)\)', rule)
        if not prop_match:
            return None

        property_name = prop_match.group(1)

        # Extract validation methods
        validations = []

        # Common validation patterns
        patterns = {
            'NotEmpty': r'\.NotEmpty\(\)',
            'NotNull': r'\.NotNull\(\)',
            'Length': r'\.Length\((\d+)(?:,\s*(\d+))?\)',
            'MinimumLength': r'\.MinimumLength\((\d+)\)',
            'MaximumLength': r'\.MaximumLength\((\d+)\)',
            'EmailAddress': r'\.EmailAddress\(\)',
            'Matches': r'\.Matches\("([^"]+)"\)',
            'GreaterThan': r'\.GreaterThan\(([^)]+)\)',
            'LessThan': r'\.LessThan\(([^)]+)\)',
            'Must': r'\.Must\(([^)]+)\)',
            'WithMessage': r'\.WithMessage\("([^"]+)"\)'
        }

        for validation_type, pattern in patterns.items():
            match = re.search(pattern, rule)
            if match:
                validation = {'type': validation_type}
                if match.groups():
                    validation['params'] = match.groups()
                validations.append(validation)

        return {
            'property': property_name,
            'validations': validations
        }

    def extract_services(self, code: str) -> List[Dict[str, Any]]:
        """Extract service dependencies from C# code"""
        services = []

        # Find service injections
        service_matches = re.findall(self.patterns['service_injection'], code)

        for match in service_matches:
            services.append({
                'type': match[0],
                'name': match[1]
            })

        return services

    def extract_dependencies(self, code: str) -> List[str]:
        """Extract using statements from C# code"""
        dependencies = re.findall(r'using\s+([\w.]+);', code)
        return list(set(dependencies))

    def extract_models(self, code: str) -> List[Dict[str, Any]]:
        """Extract DTOs and models from C# code"""
        models = []

        # Find DTO/model classes
        dto_pattern = r'public\s+class\s+(\w+(?:DTO|Model|Request|Response))'
        dto_matches = re.finditer(dto_pattern, code)

        for match in dto_matches:
            model_name = match.group(1)

            # Extract properties
            properties = self.extract_properties(code, match.end())

            models.append({
                'name': model_name,
                'properties': properties
            })

        return models

    def extract_properties(self, code: str, start_pos: int) -> List[Dict[str, Any]]:
        """Extract properties from a C# class"""
        properties = []

        # Find the class body
        brace_count = 0
        class_start = code.find('{', start_pos)
        class_end = class_start

        for i, char in enumerate(code[class_start:], class_start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    class_end = i
                    break

        class_body = code[class_start:class_end]

        # Extract properties
        prop_pattern = r'public\s+(\w+(?:<.*?>)?)\s+(\w+)\s*{\s*get;\s*(?:set;)?'
        prop_matches = re.finditer(prop_pattern, class_body)

        for match in prop_matches:
            properties.append({
                'type': match.group(1),
                'name': match.group(2),
                'nullable': '?' in match.group(1)
            })

        return properties