"""
Java specific parser for API code
"""

import re
import logging
from typing import Dict, List, Any
from .base_parser import BaseParser

logger = logging.getLogger(__name__)


class JavaParser(BaseParser):
    """Parser for Java API code (Spring Boot, JAX-RS)"""

    def __init__(self):
        super().__init__()
        self.patterns = {
            'class': r'(?:@\w+\s*)*(?:public\s+)?class\s+(\w+)',
            'rest_controller': r'@RestController',
            'controller': r'@Controller',
            'request_mapping': r'@RequestMapping\([\'"]([^\'"]+)[\'"]\)',
            'get_mapping': r'@GetMapping(?:\([\'"]([^\'"]*)[\'"].*?\))?',
            'post_mapping': r'@PostMapping(?:\([\'"]([^\'"]*)[\'"].*?\))?',
            'put_mapping': r'@PutMapping(?:\([\'"]([^\'"]*)[\'"].*?\))?',
            'delete_mapping': r'@DeleteMapping(?:\([\'"]([^\'"]*)[\'"].*?\))?',
            'patch_mapping': r'@PatchMapping(?:\([\'"]([^\'"]*)[\'"].*?\))?',
            'path_variable': r'@PathVariable(?:\([\'"](\w+)[\'"\)])?\s+(\w+)\s+(\w+)',
            'request_param': r'@RequestParam(?:\([\'"](\w+)[\'"].*?\))?\s+(\w+)\s+(\w+)',
            'request_body': r'@RequestBody\s+(\w+)\s+(\w+)',
            'method': r'(?:public|private|protected)\s+(?:static\s+)?(?:final\s+)?(\S+)\s+(\w+)\s*\((.*?)\)',
            'validation': r'@(NotNull|NotEmpty|NotBlank|Size|Min|Max|Pattern|Email)',
            'service': r'@(Service|Component|Repository)',
            'autowired': r'@Autowired\s+(?:private\s+)?(\w+)\s+(\w+)',
            'exception': r'throw\s+new\s+(\w+)\((.*?)\)',
        }

    def parse(self, code_files: List[str]) -> Dict[str, Any]:
        """Parse Java code files"""
        logger.info(f"Parsing {len(code_files)} Java files")

        results = []
        for file_path in code_files:
            code = self.read_file(file_path)

            file_result = {
                'file': file_path,
                'endpoints': self.extract_endpoints(code),
                'methods': self.extract_methods(code),
                'models': self.extract_models(code),
                'validators': self.extract_validation_rules(code),
                'services': self.extract_services(code),
                'dependencies': self.extract_dependencies(code)
            }

            results.append(file_result)

        return self.combine_results(results)

    def extract_endpoints(self, code: str) -> List[Dict[str, Any]]:
        """Extract Spring Boot endpoints"""
        endpoints = []

        # Check if it's a REST controller
        if not (re.search(self.patterns['rest_controller'], code) or
                re.search(self.patterns['controller'], code)):
            return endpoints

        # Find class name
        class_match = re.search(self.patterns['class'], code)
        controller_name = class_match.group(1) if class_match else 'Unknown'

        # Find base mapping
        base_mapping = ''
        base_match = re.search(self.patterns['request_mapping'], code)
        if base_match:
            base_mapping = base_match.group(1)

        # Find all method mappings
        mapping_patterns = [
            ('GET', self.patterns['get_mapping']),
            ('POST', self.patterns['post_mapping']),
            ('PUT', self.patterns['put_mapping']),
            ('DELETE', self.patterns['delete_mapping']),
            ('PATCH', self.patterns['patch_mapping'])
        ]

        # Split code into method blocks
        method_blocks = re.split(r'(?=(?:public|private|protected)\s+)', code)

        for block in method_blocks:
            for http_method, pattern in mapping_patterns:
                mapping_match = re.search(pattern, block)
                if mapping_match:
                    sub_path = mapping_match.group(1) if mapping_match.group(1) else ""

                    # Extract method signature
                    method_match = re.search(self.patterns['method'], block)
                    if method_match:
                        return_type = method_match.group(1)
                        method_name = method_match.group(2)
                        params_str = method_match.group(3)

                        # Build full path
                        full_path = f"{base_mapping}/{sub_path}".replace('//', '/').rstrip('/')

                        endpoint = {
                            'controller': controller_name,
                            'method_name': method_name,
                            'http_method': http_method,
                            'path': full_path,
                            'return_type': return_type,
                            'parameters': self.extract_parameters(block)
                        }
                        endpoints.append(endpoint)

        return endpoints

    def extract_parameters(self, method_code: str) -> List[Dict[str, Any]]:
        """Extract parameters from Java method"""
        parameters = []

        # PathVariable
        path_var_matches = re.findall(self.patterns['path_variable'], method_code)
        for match in path_var_matches:
            param_name = match[0] if match[0] else match[2]
            parameters.append({
                'name': param_name,
                'type': match[1],
                'source': 'path',
                'required': True
            })

        # RequestParam
        request_param_matches = re.findall(self.patterns['request_param'], method_code)
        for match in request_param_matches:
            param_name = match[0] if match[0] else match[2]
            parameters.append({
                'name': param_name,
                'type': match[1],
                'source': 'query',
                'required': 'required=false' not in method_code
            })

        # RequestBody
        request_body_matches = re.findall(self.patterns['request_body'], method_code)
        for match in request_body_matches:
            parameters.append({
                'name': match[1],
                'type': match[0],
                'source': 'body',
                'required': True
            })

        return parameters

    def extract_methods(self, code: str) -> List[Dict[str, Any]]:
        """Extract all methods from Java code"""
        methods = []

        method_matches = re.finditer(self.patterns['method'], code)

        for match in method_matches:
            return_type = match.group(1)
            method_name = match.group(2)
            params_str = match.group(3)

            methods.append({
                'name': method_name,
                'return_type': return_type,
                'parameters': self.parse_method_parameters(params_str)
            })

        return methods

    def parse_method_parameters(self, params_str: str) -> List[Dict[str, Any]]:
        """Parse method parameters from string"""
        if not params_str.strip():
            return []

        parameters = []
        # Match type and name pairs
        param_pattern = r'(?:@\w+(?:\([^)]*\))?\s*)*(\w+(?:<[^>]+>)?)\s+(\w+)'

        for match in re.finditer(param_pattern, params_str):
            parameters.append({
                'type': match.group(1),
                'name': match.group(2)
            })

        return parameters

    def extract_validation_rules(self, code: str) -> List[Dict[str, Any]]:
        """Extract Bean Validation annotations"""
        validators = []

        # Find model classes
        model_classes = re.findall(r'class\s+(\w+)(?:DTO|Request|Model)', code)

        for model_name in model_classes:
            # Extract class body
            class_pattern = rf'class\s+{model_name}.*?{{(.*?)}}'
            class_match = re.search(class_pattern, code, re.DOTALL)

            if class_match:
                class_body = class_match.group(1)

                # Find fields with validation
                field_pattern = r'(?:@(\w+)(?:\(([^)]*)\))?\s*)*private\s+(\w+)\s+(\w+);'

                for match in re.finditer(field_pattern, class_body):
                    field_type = match.group(3)
                    field_name = match.group(4)

                    # Find all validation annotations
                    validations = []
                    annotation_pattern = r'@(\w+)(?:\(([^)]*)\))?'

                    field_block = code[match.start():match.end()]
                    for ann_match in re.finditer(annotation_pattern, field_block):
                        validation_type = ann_match.group(1)
                        params = ann_match.group(2)

                        if validation_type in ['NotNull', 'NotEmpty', 'NotBlank', 'Size',
                                               'Min', 'Max', 'Pattern', 'Email']:
                            validations.append({
                                'type': validation_type,
                                'params': params
                            })

                    if validations:
                        validators.append({
                            'model': model_name,
                            'field': field_name,
                            'type': field_type,
                            'validations': validations
                        })

        return validators

    def extract_models(self, code: str) -> List[Dict[str, Any]]:
        """Extract model classes (DTOs, Entities)"""
        models = []

        # Find model classes
        model_pattern = r'(?:@\w+\s*)*class\s+(\w+)(?:DTO|Model|Entity|Request|Response)'
        model_matches = re.finditer(model_pattern, code)

        for match in model_matches:
            model_name = match.group(1)

            # Extract fields
            fields = self.extract_model_fields(code, model_name)

            models.append({
                'name': model_name,
                'fields': fields
            })

        return models

    def extract_model_fields(self, code: str, class_name: str) -> List[Dict[str, Any]]:
        """Extract fields from a Java class"""
        fields = []

        # Find class body
        class_pattern = rf'class\s+{class_name}.*?{{(.*?)}}'
        class_match = re.search(class_pattern, code, re.DOTALL)

        if class_match:
            class_body = class_match.group(1)

            # Extract fields
            field_pattern = r'private\s+(\w+(?:<[^>]+>)?)\s+(\w+);'

            for match in re.finditer(field_pattern, class_body):
                fields.append({
                    'type': match.group(1),
                    'name': match.group(2)
                })

        return fields

    def extract_services(self, code: str) -> List[Dict[str, Any]]:
        """Extract service dependencies"""
        services = []

        # Find autowired services
        autowired_matches = re.findall(self.patterns['autowired'], code)

        for match in autowired_matches:
            services.append({
                'type': match[0],
                'name': match[1]
            })

        return services

    def extract_dependencies(self, code: str) -> List[str]:
        """Extract import statements"""
        dependencies = re.findall(r'import\s+([\w\.]+);', code)
        return list(set(dependencies))