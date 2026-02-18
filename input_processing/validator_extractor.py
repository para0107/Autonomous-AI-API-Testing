"""
Extract validation rules from code
"""

import logging
import re
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class ValidatorExtractor:
    """Extracts validation rules from different validation frameworks"""

    def __init__(self):
        self.validation_patterns = {
            'fluent_validation': {
                'not_empty': r'\.NotEmpty\(\)',
                'not_null': r'\.NotNull\(\)',
                'length': r'\.Length\((\d+)(?:,\s*(\d+))?\)',
                'email': r'\.EmailAddress\(\)',
                'matches': r'\.Matches\("([^"]+)"\)',
                'range': r'\.InclusiveBetween\(([^,]+),\s*([^)]+)\)'
            },
            'bean_validation': {
                'not_null': r'@NotNull',
                'not_empty': r'@NotEmpty',
                'size': r'@Size\((?:min=(\d+))?(?:,?\s*max=(\d+))?\)',
                'email': r'@Email',
                'pattern': r'@Pattern\(regexp="([^"]+)"'
            },
            'pydantic': {
                'field': r'Field\(([^)]+)\)',
                'validator': r'@validator\([\'"](\w+)[\'"]',
                'root_validator': r'@root_validator'
            }
        }

    def extract(self, parsed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract validation rules from parsed data

        Args:
            parsed_data: Parsed code data

        Returns:
            List of validation rules
        """
        validators = parsed_data.get('validators', [])

        # Enhance with additional extraction
        enhanced_validators = []

        for validator in validators:
            enhanced = self.enhance_validator(validator)
            enhanced_validators.append(enhanced)

        # Extract inline validations from methods
        methods = parsed_data.get('methods', [])
        inline_validations = self.extract_inline_validations(methods)
        enhanced_validators.extend(inline_validations)

        return enhanced_validators

    def enhance_validator(self, validator: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance validator with additional information"""
        enhanced = validator.copy()

        # Categorize validation type
        enhanced['category'] = self.categorize_validation(validator)

        # Extract validation strength
        enhanced['strength'] = self.determine_validation_strength(validator)

        # Generate test cases for validation
        enhanced['test_cases'] = self.generate_validation_test_cases(validator)

        return enhanced

    def categorize_validation(self, validator: Dict[str, Any]) -> str:
        """Categorize validation type"""
        validations = validator.get('validations', [])
        rules = validator.get('rules', [])

        all_rules = validations + rules

        # Check validation types
        for rule in all_rules:
            rule_type = rule.get('type', '').lower() if isinstance(rule, dict) else str(rule).lower()

            if any(x in rule_type for x in ['email', 'url', 'phone']):
                return 'format'
            elif any(x in rule_type for x in ['length', 'size', 'min', 'max']):
                return 'size'
            elif any(x in rule_type for x in ['required', 'notnull', 'notempty']):
                return 'presence'
            elif any(x in rule_type for x in ['pattern', 'regex', 'matches']):
                return 'pattern'
            elif any(x in rule_type for x in ['range', 'between']):
                return 'range'

        return 'custom'

    def determine_validation_strength(self, validator: Dict[str, Any]) -> str:
        """Determine how strict the validation is"""
        validations = validator.get('validations', [])
        rules = validator.get('rules', [])

        all_rules = validations + rules

        # Count validation rules
        rule_count = len(all_rules)

        # Check for strict validations
        has_required = any('required' in str(r).lower() or
                           'notnull' in str(r).lower()
                           for r in all_rules)

        has_pattern = any('pattern' in str(r).lower() or
                          'regex' in str(r).lower()
                          for r in all_rules)

        if rule_count > 3 or (has_required and has_pattern):
            return 'strict'
        elif rule_count > 1 or has_required:
            return 'moderate'
        else:
            return 'lenient'

    def generate_validation_test_cases(self, validator: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate test cases for validation rules"""
        test_cases = []

        field_name = validator.get('field') or validator.get('property', 'field')
        validations = validator.get('validations', [])

        for validation in validations:
            if isinstance(validation, dict):
                val_type = validation.get('type', '')

                # Generate test cases based on validation type
                if val_type == 'NotEmpty':
                    test_cases.extend([
                        {'input': {field_name: ''}, 'expected': 'fail', 'reason': 'empty value'},
                        {'input': {field_name: 'value'}, 'expected': 'pass', 'reason': 'non-empty value'}
                    ])
                elif val_type == 'Length':
                    params = validation.get('params', ())
                    if len(params) >= 2:
                        min_len, max_len = int(params[0]), int(params[1])
                        test_cases.extend([
                            {'input': {field_name: 'a' * (min_len - 1)}, 'expected': 'fail',
                             'reason': 'below min length'},
                            {'input': {field_name: 'a' * min_len}, 'expected': 'pass', 'reason': 'at min length'},
                            {'input': {field_name: 'a' * max_len}, 'expected': 'pass', 'reason': 'at max length'},
                            {'input': {field_name: 'a' * (max_len + 1)}, 'expected': 'fail',
                             'reason': 'above max length'}
                        ])
                elif val_type == 'EmailAddress':
                    test_cases.extend([
                        {'input': {field_name: 'invalid'}, 'expected': 'fail', 'reason': 'invalid email'},
                        {'input': {field_name: 'test@example.com'}, 'expected': 'pass', 'reason': 'valid email'},
                        {'input': {field_name: 'test@'}, 'expected': 'fail', 'reason': 'incomplete email'}
                    ])
                elif val_type == 'Pattern' or val_type == 'Matches':
                    pattern = validation.get('params', [''])[0] if validation.get('params') else ''
                    test_cases.append({
                        'input': {field_name: 'test_pattern'},
                        'expected': 'depends',
                        'reason': f'must match pattern: {pattern}'
                    })

        return test_cases

    def extract_inline_validations(self, methods: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract validation logic from method bodies"""
        inline_validations = []

        for method in methods:
            method_name = method.get('name', '')
            body = method.get('body', '')

            if not body:
                continue

            # Look for validation patterns in method body
            validations = self.extract_validation_patterns(body)

            if validations:
                inline_validations.append({
                    'type': 'inline',
                    'method': method_name,
                    'validations': validations
                })

        return inline_validations

    def extract_validation_patterns(self, code: str) -> List[Dict[str, Any]]:
        """Extract validation patterns from code"""
        validations = []

        # Common validation patterns
        patterns = {
            'null_check': r'if\s*\(.*?(?:==|!=)\s*null',
            'empty_check': r'if\s*\(.*?\.(?:isEmpty|empty|Length\s*==\s*0)',
            'range_check': r'if\s*\(.*?(?:<|>|<=|>=)\s*\d+',
            'regex_check': r'if\s*\(.*?\.(?:matches|test|match)\(',
            'type_check': r'if\s*\(.*?instanceof',
            'custom_validation': r'if\s*\(.*?(?:isValid|validate|check)',
        }

        for val_type, pattern in patterns.items():
            if re.search(pattern, code, re.IGNORECASE):
                validations.append({
                    'type': val_type,
                    'pattern': pattern
                })

        return validations

    def merge_validators(self, validators_list: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Merge validators from multiple sources"""
        merged = {}

        for validators in validators_list:
            for validator in validators:
                key = f"{validator.get('model', '')}_{validator.get('field', '')}"

                if key not in merged:
                    merged[key] = validator
                else:
                    # Merge validations
                    existing = merged[key]
                    existing['validations'].extend(validator.get('validations', []))
                    existing['rules'].extend(validator.get('rules', []))

        return list(merged.values())