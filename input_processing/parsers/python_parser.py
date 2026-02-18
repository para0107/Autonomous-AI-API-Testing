"""
Python specific parser for API code
"""

import ast
import re
import logging
from typing import Dict, List, Any, Optional
from .base_parser import BaseParser

logger = logging.getLogger(__name__)


class PythonParser(BaseParser):
    """Parser for Python API code (Flask, FastAPI, Django)"""

    def __init__(self):
        super().__init__()
        self.framework = None
        self.patterns = {
            'flask_route': r'@app\.route\([\'"]([^\'"]+)[\'"].*?\)',
            'fastapi_route': r'@(?:app|router)\.(get|post|put|delete|patch)\([\'"]([^\'"]+)[\'"].*?\)',
            'django_path': r'path\([\'"]([^\'"]+)[\'"].*?\)',
            'methods': r'methods=\[(.*?)\]',
            'decorator': r'@(\w+)(?:\((.*?)\))?',
            'function': r'(?:async\s+)?def\s+(\w+)\s*\((.*?)\)',
            'type_hint': r'(\w+)\s*:\s*([\w\[\]]+)(?:\s*=\s*(.+?))?(?:,|\))',
            'pydantic_model': r'class\s+(\w+)\((?:BaseModel|BaseSettings|.*Model)\)',
            'validation': r'Field\((.*?)\)',
            'docstring': r'"""(.*?)"""',
        }

    def parse(self, code_files: List[str]) -> Dict[str, Any]:
        """Parse Python code files"""
        logger.info(f"Parsing {len(code_files)} Python files")

        results = []
        for file_path in code_files:
            code = self.read_file(file_path)

            # Detect framework
            self.detect_framework(code)

            # Parse with AST for better accuracy
            try:
                tree = ast.parse(code)
                ast_result = self.parse_ast(tree, code)
            except SyntaxError as e:
                logger.warning(f"AST parsing failed for {file_path}: {e}")
                ast_result = {}

            # Combine with regex parsing for framework-specific patterns
            file_result = {
                'file': file_path,
                'endpoints': self.extract_endpoints(code),
                'methods': ast_result.get('functions', []),
                'models': self.extract_models(code),
                'validators': self.extract_validation_rules(code),
                'dependencies': self.extract_dependencies(code),
                'framework': self.framework
            }

            results.append(file_result)

        return self.combine_results(results)

    def detect_framework(self, code: str):
        """Detect which Python framework is being used"""
        if 'from flask import' in code or 'import flask' in code:
            self.framework = 'flask'
        elif 'from fastapi import' in code or 'import fastapi' in code:
            self.framework = 'fastapi'
        elif 'from django' in code or 'import django' in code:
            self.framework = 'django'
        else:
            self.framework = 'unknown'

    def parse_ast(self, tree: ast.AST, source_code: str) -> Dict[str, Any]:
        """Parse Python AST to extract detailed information"""
        result = {
            'classes': [],
            'functions': [],
            'imports': []
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = self.parse_class(node, source_code)
                result['classes'].append(class_info)

            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                func_info = self.parse_function(node, source_code)
                result['functions'].append(func_info)

            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                import_info = self.parse_import(node)
                result['imports'].extend(import_info)

        return result

    def parse_class(self, node: ast.ClassDef, source_code: str) -> Dict[str, Any]:
        """Parse a class definition"""
        class_info = {
            'name': node.name,
            'bases': [self.get_name(base) for base in node.bases],
            'methods': [],
            'attributes': [],
            'decorators': [self.get_name(dec) for dec in node.decorator_list]
        }

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_info = self.parse_function(item, source_code)
                class_info['methods'].append(method_info)

            elif isinstance(item, ast.AnnAssign):
                # Type-annotated attribute
                if isinstance(item.target, ast.Name):
                    attr_info = {
                        'name': item.target.id,
                        'type': self.get_annotation(item.annotation),
                        'default': self.get_value(item.value) if item.value else None
                    }
                    class_info['attributes'].append(attr_info)

        return class_info

    def parse_function(self, node, source_code: str) -> Dict[str, Any]:
        """Parse a function definition"""
        func_info = {
            'name': node.name,
            'async': isinstance(node, ast.AsyncFunctionDef),
            'parameters': self.parse_parameters(node.args),
            'return_type': self.get_annotation(node.returns) if node.returns else None,
            'decorators': [self.get_decorator_info(dec) for dec in node.decorator_list],
            'docstring': ast.get_docstring(node)
        }

        # Extract function body
        if hasattr(node, 'lineno'):
            func_info['line_number'] = node.lineno

        return func_info

    def parse_parameters(self, args: ast.arguments) -> List[Dict[str, Any]]:
        """Parse function parameters"""
        params = []

        # Regular arguments
        for i, arg in enumerate(args.args):
            param = {
                'name': arg.arg,
                'type': self.get_annotation(arg.annotation) if arg.annotation else None,
                'default': None
            }

            # Check for defaults
            default_offset = len(args.args) - len(args.defaults)
            if i >= default_offset:
                default_index = i - default_offset
                param['default'] = self.get_value(args.defaults[default_index])

            params.append(param)

        # Keyword-only arguments
        for i, arg in enumerate(args.kwonlyargs):
            param = {
                'name': arg.arg,
                'type': self.get_annotation(arg.annotation) if arg.annotation else None,
                'default': self.get_value(args.kw_defaults[i]) if i < len(args.kw_defaults) else None,
                'keyword_only': True
            }
            params.append(param)

        return params

    def extract_endpoints(self, code: str) -> List[Dict[str, Any]]:
        """Extract API endpoints based on framework"""
        endpoints = []

        if self.framework == 'flask':
            endpoints = self.extract_flask_endpoints(code)
        elif self.framework == 'fastapi':
            endpoints = self.extract_fastapi_endpoints(code)
        elif self.framework == 'django':
            endpoints = self.extract_django_endpoints(code)

        return endpoints

    def extract_flask_endpoints(self, code: str) -> List[Dict[str, Any]]:
        """Extract Flask endpoints"""
        endpoints = []

        # Find route decorators
        route_pattern = r'@app\.route\([\'"]([^\'"]+)[\'"](?:.*?methods=\[(.*?)\])?\)'

        # Split code into function blocks
        function_blocks = re.split(r'(?=(?:async\s+)?def\s+)', code)

        for block in function_blocks:
            route_match = re.search(route_pattern, block)
            if not route_match:
                continue

            route = route_match.group(1)
            methods_str = route_match.group(2)

            if methods_str:
                methods = [m.strip().strip('"\'') for m in methods_str.split(',')]
            else:
                methods = ['GET']

            # Extract function signature
            func_match = re.search(self.patterns['function'], block)
            if func_match:
                func_name = func_match.group(1)
                params_str = func_match.group(2)

                endpoint = {
                    'route': route,
                    'methods': methods,
                    'function_name': func_name,
                    'parameters': self.parse_flask_parameters(params_str),
                    'framework': 'flask'
                }
                endpoints.append(endpoint)

        return endpoints

    def extract_fastapi_endpoints(self, code: str) -> List[Dict[str, Any]]:
        """Extract FastAPI endpoints"""
        endpoints = []

        # FastAPI route pattern
        route_pattern = r'@(?:app|router)\.(get|post|put|delete|patch)\([\'"]([^\'"]+)[\'"].*?\)'

        function_blocks = re.split(r'(?=(?:async\s+)?def\s+)', code)

        for block in function_blocks:
            route_match = re.search(route_pattern, block)
            if not route_match:
                continue

            method = route_match.group(1).upper()
            route = route_match.group(2)

            # Extract function and parameters
            func_match = re.search(self.patterns['function'], block)
            if func_match:
                func_name = func_match.group(1)
                params_str = func_match.group(2)

                # Parse FastAPI parameters with type hints
                parameters = self.parse_fastapi_parameters(params_str, block)

                endpoint = {
                    'route': route,
                    'method': method,
                    'function_name': func_name,
                    'parameters': parameters,
                    'framework': 'fastapi',
                    'async': 'async def' in block
                }
                endpoints.append(endpoint)

        return endpoints

    def parse_fastapi_parameters(self, params_str: str, function_block: str) -> List[Dict[str, Any]]:
        """Parse FastAPI parameters with type hints and dependencies"""
        parameters = []

        # Parse parameter string
        param_pattern = r'(\w+)\s*:\s*([\w\[\]\.]+)(?:\s*=\s*([^,\)]+))?'

        for match in re.finditer(param_pattern, params_str):
            param_name = match.group(1)
            param_type = match.group(2)
            default_value = match.group(3)

            # Determine parameter source
            source = 'body'  # Default for Pydantic models
            if default_value:
                if 'Query' in default_value:
                    source = 'query'
                elif 'Path' in default_value:
                    source = 'path'
                elif 'Body' in default_value:
                    source = 'body'
                elif 'Header' in default_value:
                    source = 'header'

            parameters.append({
                'name': param_name,
                'type': param_type,
                'source': source,
                'required': default_value is None or '...' in str(default_value),
                'default': self.parse_default_value(default_value) if default_value else None
            })

        return parameters

    def extract_models(self, code: str) -> List[Dict[str, Any]]:
        """Extract Pydantic models or dataclasses"""
        models = []

        # Pydantic models
        model_pattern = r'class\s+(\w+)\((?:BaseModel|BaseSettings|.*Model)\):'
        model_matches = re.finditer(model_pattern, code)

        for match in model_matches:
            model_name = match.group(1)
            model_body = self.extract_class_body(code, match.end())

            fields = self.extract_model_fields(model_body)
            validators = self.extract_pydantic_validators(model_body)

            models.append({
                'name': model_name,
                'type': 'pydantic',
                'fields': fields,
                'validators': validators
            })

        # Dataclasses
        dataclass_pattern = r'@dataclass(?:\(.*?\))?\s*class\s+(\w+)'
        dataclass_matches = re.finditer(dataclass_pattern, code)

        for match in dataclass_matches:
            model_name = match.group(1)
            model_body = self.extract_class_body(code, match.end())

            fields = self.extract_dataclass_fields(model_body)

            models.append({
                'name': model_name,
                'type': 'dataclass',
                'fields': fields
            })

        return models

    def extract_model_fields(self, class_body: str) -> List[Dict[str, Any]]:
        """Extract fields from a Pydantic model"""
        fields = []

        # Field with type annotation
        field_pattern = r'(\w+)\s*:\s*([\w\[\]\.]+)(?:\s*=\s*(.+?))?(?:\n|$)'

        for match in re.finditer(field_pattern, class_body):
            field_name = match.group(1)
            field_type = match.group(2)
            default_value = match.group(3)

            field = {
                'name': field_name,
                'type': field_type,
                'required': True
            }

            # Parse Field() definitions
            if default_value and 'Field' in default_value:
                field_params = self.parse_field_params(default_value)
                field.update(field_params)
            elif default_value:
                field['default'] = default_value.strip()
                field['required'] = False

            fields.append(field)

        return fields

    def parse_field_params(self, field_str: str) -> Dict[str, Any]:
        """Parse Pydantic Field() parameters"""
        params = {}

        # Extract Field parameters
        patterns = {
            'default': r'default=([^,\)]+)',
            'min_length': r'min_length=(\d+)',
            'max_length': r'max_length=(\d+)',
            'ge': r'ge=([^,\)]+)',
            'le': r'le=([^,\)]+)',
            'regex': r'regex=[\'"]([^\'"]+)[\'"]',
            'description': r'description=[\'"]([^\'"]+)[\'"]'
        }

        for param_name, pattern in patterns.items():
            match = re.search(pattern, field_str)
            if match:
                params[param_name] = match.group(1)

        # Check if required
        if '...' in field_str:
            params['required'] = True
        elif 'default' in params:
            params['required'] = False

        return params

    def extract_validation_rules(self, code: str) -> List[Dict[str, Any]]:
        """Extract validation rules from Python code"""
        validators = []

        # Pydantic validators
        validator_pattern = r'@validator\([\'"](\w+)[\'"].*?\)\s*(?:async\s+)?def\s+(\w+)'

        for match in re.finditer(validator_pattern, code):
            field_name = match.group(1)
            validator_name = match.group(2)

            validators.append({
                'type': 'pydantic_validator',
                'field': field_name,
                'function': validator_name
            })

        # Field validations
        field_validation_pattern = r'(\w+)\s*:\s*[\w\[\]]+\s*=\s*Field\((.*?)\)'

        for match in re.finditer(field_validation_pattern, code):
            field_name = match.group(1)
            field_params = match.group(2)

            validations = []
            if 'min_length' in field_params:
                validations.append(
                    {'type': 'min_length', 'value': re.search(r'min_length=(\d+)', field_params).group(1)})
            if 'max_length' in field_params:
                validations.append(
                    {'type': 'max_length', 'value': re.search(r'max_length=(\d+)', field_params).group(1)})
            if 'regex' in field_params:
                validations.append(
                    {'type': 'regex', 'value': re.search(r'regex=[\'"]([^\'"]+)[\'"]', field_params).group(1)})

            if validations:
                validators.append({
                    'type': 'field_validation',
                    'field': field_name,
                    'validations': validations
                })

        return validators

    def extract_dependencies(self, code: str) -> List[str]:
        """Extract import statements"""
        dependencies = []

        # Import statements
        import_pattern = r'(?:from\s+([\w\.]+)\s+)?import\s+([\w\s,\*]+)'

        for match in re.finditer(import_pattern, code):
            module = match.group(1) if match.group(1) else ''
            imports = match.group(2)

            if module:
                dependencies.append(module)
            else:
                # Direct imports
                for imp in imports.split(','):
                    dependencies.append(imp.strip())

        return list(set(dependencies))

    def extract_methods(self, code: str) -> List[Dict[str, Any]]:
        """Extract all methods/functions from code"""
        methods = []

        function_pattern = r'(?:async\s+)?def\s+(\w+)\s*\((.*?)\)(?:\s*->\s*([\w\[\]\.]+))?:'

        for match in re.finditer(function_pattern, code):
            func_name = match.group(1)
            params = match.group(2)
            return_type = match.group(3)

            methods.append({
                'name': func_name,
                'parameters': self.parse_function_params(params),
                'return_type': return_type,
                'async': 'async def' in code[max(0, match.start() - 10):match.start() + 10]
            })

        return methods

    # Helper methods
    def get_name(self, node):
        """Get name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self.get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            return self.get_name(node.func)
        return str(node)

    def get_annotation(self, node):
        """Get type annotation as string"""
        if node is None:
            return None
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Subscript):
            return f"{self.get_name(node.value)}[{self.get_annotation(node.slice)}]"
        elif isinstance(node, ast.Attribute):
            return f"{self.get_name(node.value)}.{node.attr}"
        return ast.unparse(node) if hasattr(ast, 'unparse') else str(node)

    def get_value(self, node):
        """Get value from AST node"""
        if node is None:
            return None
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.List):
            return [self.get_value(item) for item in node.elts]
        elif isinstance(node, ast.Dict):
            return {self.get_value(k): self.get_value(v) for k, v in zip(node.keys, node.values)}
        return ast.unparse(node) if hasattr(ast, 'unparse') else str(node)

    def get_decorator_info(self, node):
        """Get decorator information"""
        if isinstance(node, ast.Name):
            return {'name': node.id}
        elif isinstance(node, ast.Call):
            return {
                'name': self.get_name(node.func),
                'args': [self.get_value(arg) for arg in node.args]
            }
        elif isinstance(node, ast.Attribute):
            return {'name': f"{self.get_name(node.value)}.{node.attr}"}
        return {'name': str(node)}

    def extract_class_body(self, code: str, start_pos: int) -> str:
        """Extract class body from code"""
        # Find the class body by indentation
        lines = code[start_pos:].split('\n')
        class_lines = []
        base_indent = None

        for line in lines:
            if line.strip():
                if base_indent is None:
                    base_indent = len(line) - len(line.lstrip())
                    class_lines.append(line)
                elif line.startswith(' ' * base_indent):
                    class_lines.append(line)
                else:
                    break
            else:
                class_lines.append(line)

        return '\n'.join(class_lines)

    def parse_function_params(self, params_str: str) -> List[Dict[str, Any]]:
        """Parse function parameters from string"""
        if not params_str.strip():
            return []

        parameters = []
        # Simple parameter parsing - can be enhanced
        params = params_str.split(',')

        for param in params:
            param = param.strip()
            if not param or param == 'self':
                continue

            # Check for type hint
            if ':' in param:
                name, type_hint = param.split(':', 1)
                name = name.strip()
                type_hint = type_hint.split('=')[0].strip() if '=' in type_hint else type_hint.strip()
                default = param.split('=')[1].strip() if '=' in param else None
            else:
                name = param.split('=')[0].strip()
                type_hint = None
                default = param.split('=')[1].strip() if '=' in param else None

            parameters.append({
                'name': name,
                'type': type_hint,
                'default': default,
                'required': default is None
            })

        return parameters

    def parse_default_value(self, default_str: str):
        """Parse default value from string"""
        if not default_str:
            return None

        default_str = default_str.strip()

        # Remove function calls like Query(), Path(), etc.
        if '(' in default_str:
            # Extract the first argument if it exists
            match = re.search(r'\((.*?)\)', default_str)
            if match:
                inner = match.group(1).strip()
                if inner and inner != '...':
                    return inner

        return default_str if default_str != '...' else None

    def parse_flask_parameters(self, params_str: str) -> List[Dict[str, Any]]:
        """Parse Flask function parameters"""
        parameters = []

        if not params_str.strip():
            return parameters

        # Simple parameter parsing
        params = params_str.split(',')

        for param in params:
            param = param.strip()
            if param:
                parameters.append({
                    'name': param,
                    'source': 'route',  # Typically route parameters in Flask
                    'required': True
                })

        return parameters

    def extract_django_endpoints(self, code: str) -> List[Dict[str, Any]]:
        """Extract Django URL patterns"""
        endpoints = []

        # Django URL patterns
        url_pattern = r'path\([\'"]([^\'"]+)[\'"],\s*(\w+)(?:\.as_view\(\))?,.*?\)'

        for match in re.finditer(url_pattern, code):
            route = match.group(1)
            view_name = match.group(2)

            endpoints.append({
                'route': route,
                'view': view_name,
                'framework': 'django'
            })

        return endpoints

    def extract_dataclass_fields(self, class_body: str) -> List[Dict[str, Any]]:
        """Extract fields from a dataclass"""
        fields = []

        # Field with type annotation
        field_pattern = r'(\w+)\s*:\s*([\w\[\]\.]+)(?:\s*=\s*field\((.*?)\))?'

        for match in re.finditer(field_pattern, class_body):
            field_name = match.group(1)
            field_type = match.group(2)
            field_params = match.group(3)

            field = {
                'name': field_name,
                'type': field_type
            }

            if field_params:
                # Parse field() parameters
                if 'default=' in field_params:
                    default_match = re.search(r'default=([^,\)]+)', field_params)
                    if default_match:
                        field['default'] = default_match.group(1)

            fields.append(field)

        return fields

    def extract_pydantic_validators(self, class_body: str) -> List[Dict[str, Any]]:
        """Extract Pydantic validators from class body"""
        validators = []

        # Root validators
        root_validator_pattern = r'@root_validator(?:\((.*?)\))?\s*(?:async\s+)?def\s+(\w+)'

        for match in re.finditer(root_validator_pattern, class_body):
            params = match.group(1)
            func_name = match.group(2)

            validators.append({
                'type': 'root_validator',
                'function': func_name,
                'pre': 'pre=True' in params if params else False
            })

        return validators

    def parse_import(self, node):
        """Parse import statements from AST"""
        imports = []

        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    'module': alias.name,
                    'alias': alias.asname
                })
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for alias in node.names:
                imports.append({
                    'module': module,
                    'name': alias.name,
                    'alias': alias.asname
                })

        return imports