"""
Input processing module initialization
"""

from .parser_factory import ParserFactory
from .endpoint_extractor import EndpointExtractor
from .specification_builder import SpecificationBuilder
from .validator_extractor import ValidatorExtractor

class InputProcessor:
    """Main input processor that coordinates parsing and extraction"""

    def __init__(self):
        self.parser_factory = ParserFactory()
        self.endpoint_extractor = EndpointExtractor()
        self.spec_builder = SpecificationBuilder()
        self.validator_extractor = ValidatorExtractor()

    def parse_code(self, code_files, language):
        """Parse code files"""
        parser = self.parser_factory.get_parser(language)
        return parser.parse(code_files)

    def build_specification(self, parsed_data):
        """Build API specification from parsed data"""
        return self.spec_builder.build(parsed_data)

    def extract_validation_rules(self, parsed_data):
        """Extract validation rules"""
        return self.validator_extractor.extract(parsed_data)

    def extract_business_logic(self, parsed_data):
        """Extract business logic patterns"""
        # For now, return empty list
        # TODO: Implement business logic extraction
        return []


__all__ = [
    'InputProcessor',
    'ParserFactory',
    'EndpointExtractor',
    'SpecificationBuilder',
    'ValidatorExtractor'
]