"""
Factory for creating language-specific parsers
"""

import logging
from typing import Optional

from .parsers import (
    BaseParser,
    CSharpParser,
    PythonParser,
    JavaParser,
    CppParser
)

logger = logging.getLogger(__name__)


class ParserFactory:
    """Factory class for creating language-specific parsers"""

    def __init__(self):
        self.parsers = {
            'csharp': CSharpParser,
            'python': PythonParser,
            'java': JavaParser,
            'cpp': CppParser
        }

        logger.info(f"Parser factory initialized with support for: {list(self.parsers.keys())}")

    def get_parser(self, language: str) -> Optional[BaseParser]:
        """
        Get parser for specific language

        Args:
            language: Programming language name

        Returns:
            Language-specific parser instance
        """
        language = language.lower()

        if language not in self.parsers:
            raise ValueError(f"Unsupported language: {language}. Supported languages: {list(self.parsers.keys())}")

        parser_class = self.parsers[language]
        logger.info(f"Creating {language} parser")

        return parser_class()




    def register_parser(self, language: str, parser_class: type):
        """
        Register a new parser for a language

        Args:
            language: Language name
            parser_class: Parser class
        """
        if not issubclass(parser_class, BaseParser):
            raise TypeError("Parser class must inherit from BaseParser")

        self.parsers[language.lower()] = parser_class
        logger.info(f"Registered new parser for {language}")

    def get_supported_languages(self) -> list:
        """Get list of supported languages"""
        return list(self.parsers.keys())