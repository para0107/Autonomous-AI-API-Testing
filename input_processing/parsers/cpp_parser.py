"""
C++ specific parser for API code
"""

import re
import logging
from typing import Dict, List, Any
from .base_parser import BaseParser

logger = logging.getLogger(__name__)


class CppParser(BaseParser):
    """Parser for C++ API code (using various REST frameworks)"""

    def __init__(self):
        super().__init__()
        self.patterns = {
            'class': r'class\s+(\w+)',
            'struct': r'struct\s+(\w+)',
            'method': r'(?:virtual\s+)?(?:static\s+)?(\w+(?:\s*<[^>]+>)?)\s+(\w+)\s*\((.*?)\)',
            'crow_route': r'CROW_ROUTE\(.*?,\s*"([^"]+)"\)',
            'pistache_route': r'Routes::(?:Get|Post|Put|Delete)\(.*?"([^"]+)"',
            'restbed_route': r'resource->set_path\(\s*"([^"]+)"\s*\)',
            'include': r'#include\s*[<"]([^>"]+)[>"]',
            'namespace': r'namespace\s+(\w+)',
            'template': r'template\s*<([^>]+)>',
            'validation': r'if\s*\(.*?(validate|check|verify).*?\)',
        }

    def parse(self, code_files: List[str]) -> Dict[str, Any]:
        """Parse C++ code files"""
        logger.info(f"Parsing {len(code_files)} C++ files")

        results = []
        for file_path in code_files:
            code = self.read_file(file_path)

            # Detect framework
            framework = self.detect_framework(code)

            file_result = {
                'file': file_path,
                'endpoints': self.extract_endpoints(code, framework),
                'methods': self.extract_methods(code),
                'classes': self.extract_classes(code),
                'structs': self.extract_structs(code),
                'validators': self.extract_validation_rules(code),
                'dependencies': self.extract_dependencies(code),
                'framework': framework
            }

            results.append(file_result)

        return self.combine_results(results)

    def detect_framework(self, code: str) -> str:
        """Detect which C++ REST framework is being used"""
        if 'crow.h' in code or 'CROW_ROUTE' in code:
            return 'crow'
        elif 'pistache' in code:
            return 'pistache'
        elif 'restbed' in code:
            return 'restbed'
        elif 'cpprest' in code or 'http_listener' in code:
            return 'cpprest'
        else:
            return 'unknown'

    def extract_endpoints(self, code: str, framework: str) -> List[Dict[str, Any]]:
        """Extract API endpoints based on framework"""
        if framework == 'crow':
            return self.extract_crow_endpoints(code)
        elif framework == 'pistache':
            return self.extract_pistache_endpoints(code)
        elif framework == 'restbed':
            return self.extract_restbed_endpoints(code)
        elif framework == 'cpprest':
            return self.extract_cpprest_endpoints(code)
        else:
            return []

    def extract_crow_endpoints(self, code: str) -> List[Dict[str, Any]]:
        """Extract Crow framework endpoints"""
        endpoints = []

        # Crow route pattern
        route_pattern = r'CROW_ROUTE\((\w+),\s*"([^"]+)"\)(?:\.methods\((.*?)\))?\s*\(\[.*?\]\((.*?)\)'

        for match in re.finditer(route_pattern, code, re.DOTALL):
            app_name = match.group(1)
            route = match.group(2)
            methods = match.group(3)
            params = match.group(4)

            # Parse HTTP methods
            if methods:
                http_methods = re.findall(r'"(\w+)"', methods)
            else:
                http_methods = ['GET']

            # Parse parameters
            parameters = self.parse_crow_parameters(params)

            for method in http_methods:
                endpoints.append({
                    'route': route,
                    'method': method,
                    'parameters': parameters,
                    'framework': 'crow'
                })

        return endpoints

    def parse_crow_parameters(self, params_str: str) -> List[Dict[str, Any]]:
        """Parse Crow route parameters"""
        parameters = []

        # Parse function parameters
        param_pattern = r'(?:const\s+)?(\w+(?:\s*<[^>]+>)?)\s*(?:const\s*)?\&?\s+(\w+)'

        for match in re.finditer(param_pattern, params_str):
            param_type = match.group(1)
            param_name = match.group(2)

            # Determine parameter source
            if 'request' in param_name.lower():
                continue  # Skip request object
            elif 'response' in param_name.lower():
                continue  # Skip response object
            else:
                parameters.append({
                    'name': param_name,
                    'type': param_type,
                    'source': 'query'  # Default to query
                })

        return parameters

    def extract_methods(self, code: str) -> List[Dict[str, Any]]:
        """Extract C++ methods"""
        methods = []

        method_matches = re.finditer(self.patterns['method'], code)

        for match in method_matches:
            return_type = match.group(1)
            method_name = match.group(2)
            params_str = match.group(3)

            # Skip constructors and destructors
            if method_name.startswith('~') or return_type == method_name:
                continue

            methods.append({
                'name': method_name,
                'return_type': return_type,
                'parameters': self.parse_cpp_parameters(params_str)
            })

        return methods

    def parse_cpp_parameters(self, params_str: str) -> List[Dict[str, Any]]:
        """Parse C++ method parameters"""
        if not params_str.strip() or params_str.strip() == 'void':
            return []

        parameters = []

        # Split by comma, handling nested templates
        params = self.split_parameters(params_str)

        for param in params:
            param = param.strip()
            if not param:
                continue

            # Parse parameter
            param_match = re.match(
                r'(?:const\s+)?(\w+(?:\s*<[^>]+>)?)\s*(?:const\s*)?\&?\*?\s+(\w+)(?:\s*=\s*(.+))?',
                param
            )

            if param_match:
                parameters.append({
                    'type': param_match.group(1),
                    'name': param_match.group(2),
                    'default': param_match.group(3)
                })

        return parameters

    def split_parameters(self, params_str: str) -> List[str]:
        """Split parameters handling nested templates"""
        params = []
        current = ''
        depth = 0

        for char in params_str:
            if char == '<':
                depth += 1
            elif char == '>':
                depth -= 1
            elif char == ',' and depth == 0:
                params.append(current.strip())
                current = ''
                continue
            current += char

        if current.strip():
            params.append(current.strip())

        return params

    def extract_classes(self, code: str) -> List[Dict[str, Any]]:
        """Extract C++ classes"""
        classes = []

        class_pattern = r'class\s+(\w+)(?:\s*:\s*(?:public|private|protected)\s+(\w+))?'

        for match in re.finditer(class_pattern, code):
            class_name = match.group(1)
            base_class = match.group(2)

            # Extract class members
            members = self.extract_class_members(code, class_name)

            classes.append({
                'name': class_name,
                'base': base_class,
                'members': members
            })

        return classes

    def extract_class_members(self, code: str, class_name: str) -> Dict[str, Any]:
        """Extract members from a C++ class"""
        members = {
            'public': [],
            'private': [],
            'protected': []
        }

        # Find class body
        class_pattern = rf'class\s+{class_name}.*?{{(.*?)}};'
        class_match = re.search(class_pattern, code, re.DOTALL)

        if class_match:
            class_body = class_match.group(1)

            # Current visibility
            visibility = 'private'  # Default for class

            lines = class_body.split('\n')
            for line in lines:
                line = line.strip()

                # Check for visibility specifier
                if line.startswith('public:'):
                    visibility = 'public'
                elif line.startswith('private:'):
                    visibility = 'private'
                elif line.startswith('protected:'):
                    visibility = 'protected'
                else:
                    # Parse member
                    member_match = re.match(r'(\w+(?:\s*<[^>]+>)?)\s+(\w+);', line)
                    if member_match:
                        members[visibility].append({
                            'type': member_match.group(1),
                            'name': member_match.group(2)
                        })

        return members

    def extract_structs(self, code: str) -> List[Dict[str, Any]]:
        """Extract C++ structs"""
        structs = []

        struct_matches = re.finditer(self.patterns['struct'], code)

        for match in struct_matches:
            struct_name = match.group(1)

            # Extract struct members
            members = self.extract_struct_members(code, struct_name)

            structs.append({
                'name': struct_name,
                'members': members
            })

        return structs

    def extract_struct_members(self, code: str, struct_name: str) -> List[Dict[str, Any]]:
        """Extract members from a C++ struct"""
        members = []

        # Find struct body
        struct_pattern = rf'struct\s+{struct_name}.*?{{(.*?)}};'
        struct_match = re.search(struct_pattern, code, re.DOTALL)

        if struct_match:
            struct_body = struct_match.group(1)

            # Extract members
            member_pattern = r'(\w+(?:\s*<[^>]+>)?)\s+(\w+);'

            for match in re.finditer(member_pattern, struct_body):
                members.append({
                    'type': match.group(1),
                    'name': match.group(2)
                })

        return members

    def extract_validation_rules(self, code: str) -> List[Dict[str, Any]]:
        """Extract validation logic from C++ code"""
        validators = []

        # Find validation functions
        validation_functions = re.findall(
            r'(?:bool|int)\s+(validate\w+)\s*\((.*?)\)',
            code
        )

        for func_name, params in validation_functions:
            validators.append({
                'function': func_name,
                'parameters': params
            })

        # Find inline validations
        if_validations = re.findall(self.patterns['validation'], code)
        for validation in if_validations:
            validators.append({
                'type': 'inline',
                'content': validation
            })

        return validators

    def extract_dependencies(self, code: str) -> List[str]:
        """Extract include statements"""
        dependencies = re.findall(self.patterns['include'], code)
        return list(set(dependencies))

    def extract_pistache_endpoints(self, code: str) -> List[Dict[str, Any]]:
        """Extract Pistache framework endpoints"""
        endpoints = []

        # Pistache route patterns
        route_patterns = [
            (r'Routes::Get\(router,\s*"([^"]+)"', 'GET'),
            (r'Routes::Post\(router,\s*"([^"]+)"', 'POST'),
            (r'Routes::Put\(router,\s*"([^"]+)"', 'PUT'),
            (r'Routes::Delete\(router,\s*"([^"]+)"', 'DELETE'),
        ]

        for pattern, method in route_patterns:
            for match in re.finditer(pattern, code):
                route = match.group(1)
                endpoints.append({
                    'route': route,
                    'method': method,
                    'framework': 'pistache'
                })

        return endpoints

    def extract_restbed_endpoints(self, code: str) -> List[Dict[str, Any]]:
        """Extract Restbed framework endpoints"""
        endpoints = []

        # Restbed resource pattern
        resource_pattern = r'resource->set_path\(\s*"([^"]+)"\s*\)'
        method_pattern = r'resource->set_method_handler\(\s*"(\w+)"'

        routes = re.findall(resource_pattern, code)
        methods = re.findall(method_pattern, code)

        for route in routes:
            for method in methods:
                endpoints.append({
                    'route': route,
                    'method': method,
                    'framework': 'restbed'
                })

        return endpoints

    def extract_cpprest_endpoints(self, code: str) -> List[Dict[str, Any]]:
        """Extract C++ REST SDK endpoints"""
        endpoints = []

        # cpprest patterns
        listener_pattern = r'listener\.support\(methods::(\w+),\s*"([^"]+)"'

        for match in re.finditer(listener_pattern, code):
            method = match.group(1).upper()
            route = match.group(2)

            endpoints.append({
                'route': route,
                'method': method,
                'framework': 'cpprest'
            })

        return endpoints