"""
Knowledge base management for RAG system

FIX: Added add_entry() convenience method that auto-detects knowledge type.
     This is called by FeedbackLoop and RAGSystem.
"""

import logging
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

from config import paths

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """Manages the knowledge base for API testing"""

    def __init__(self):
        self.knowledge_dir = paths.DATA_DIR / "knowledge_base"
        self.knowledge_dir.mkdir(exist_ok=True)

        self.knowledge_types = {
            'test_patterns': 'Common test patterns and strategies',
            'edge_cases': 'Edge cases and boundary conditions',
            'validation_rules': 'Validation patterns and rules',
            'api_patterns': 'Common API design patterns',
            'bug_patterns': 'Common bugs and issues',
            'best_practices': 'Testing best practices'
        }

        self._initialize_knowledge_base()

    def _initialize_knowledge_base(self):
        """Initialize knowledge base with default knowledge"""
        self.knowledge = {}
        for knowledge_type in self.knowledge_types:
            self.knowledge[knowledge_type] = self._load_knowledge(knowledge_type)

        if not self.knowledge['test_patterns']:
            self._add_default_test_patterns()

        if not self.knowledge['edge_cases']:
            self._add_default_edge_cases()

    def add_entry(self, entry: Dict[str, Any]):
        """
        Add entry to knowledge base, auto-detecting the knowledge type.

        FIX: This method was called by FeedbackLoop and RAGSystem but didn't exist.

        Args:
            entry: Dict with at least 'type' key to determine category.
                   Falls back to 'test_patterns' if type is unrecognized.
        """
        entry_type = entry.get('type', '')

        # Map entry types to knowledge base categories
        type_mapping = {
            'successful_test_pattern': 'test_patterns',
            'failed_test_pattern': 'bug_patterns',
            'test_pattern': 'test_patterns',
            'edge_case': 'edge_cases',
            'validation': 'validation_rules',
            'validation_rule': 'validation_rules',
            'api_pattern': 'api_patterns',
            'bug_pattern': 'bug_patterns',
            'best_practice': 'best_practices',
        }

        knowledge_type = type_mapping.get(entry_type, 'test_patterns')

        # Ensure the knowledge type exists
        if knowledge_type not in self.knowledge_types:
            knowledge_type = 'test_patterns'

        self.add_knowledge(knowledge_type, entry)

    def _add_default_test_patterns(self):
        """Add default test patterns"""
        patterns = [
            {
                'name': 'Happy Path Testing',
                'description': 'Test with valid inputs and expected behavior',
                'applicable_to': ['GET', 'POST', 'PUT', 'DELETE'],
                'test_data': {
                    'strategy': 'Use realistic, valid data',
                    'examples': ['Valid IDs', 'Complete objects', 'Proper formats']
                }
            },
            {
                'name': 'Null/Empty Testing',
                'description': 'Test with null, empty, or missing values',
                'applicable_to': ['POST', 'PUT'],
                'test_data': {
                    'strategy': 'Test each field with null/empty',
                    'examples': ['null', '""', '{}', '[]', 'undefined']
                }
            },
            {
                'name': 'Boundary Testing',
                'description': 'Test at limits of acceptable values',
                'applicable_to': ['All'],
                'test_data': {
                    'strategy': 'Test min, max, and edge values',
                    'examples': ['MIN_INT', 'MAX_INT', '0', '-1', 'MAX_LENGTH']
                }
            },
            {
                'name': 'Authentication Testing',
                'description': 'Test authentication and authorization',
                'applicable_to': ['All'],
                'test_data': {
                    'strategy': 'Test with various auth states',
                    'examples': ['No token', 'Invalid token', 'Expired token', 'Wrong permissions']
                }
            },
            {
                'name': 'Concurrent Access Testing',
                'description': 'Test concurrent requests to same resource',
                'applicable_to': ['PUT', 'DELETE'],
                'test_data': {
                    'strategy': 'Send multiple simultaneous requests',
                    'examples': ['Race conditions', 'Deadlocks', 'Data consistency']
                }
            }
        ]

        self.knowledge['test_patterns'] = patterns
        self._save_knowledge('test_patterns')

    def _add_default_edge_cases(self):
        """Add default edge cases"""
        edge_cases = [
            {
                'type': 'string',
                'cases': [
                    {'value': '', 'description': 'Empty string'},
                    {'value': ' ', 'description': 'Single space'},
                    {'value': 'A' * 10000, 'description': 'Very long string'},
                    {'value': '<script>alert("xss")</script>', 'description': 'XSS attempt'},
                    {'value': "'; DROP TABLE users; --", 'description': 'SQL injection'},
                    {'value': '../../etc/passwd', 'description': 'Path traversal'},
                    {'value': 'emojis unicode', 'description': 'Unicode and emojis'}
                ]
            },
            {
                'type': 'integer',
                'cases': [
                    {'value': 0, 'description': 'Zero'},
                    {'value': -1, 'description': 'Negative one'},
                    {'value': 2147483647, 'description': 'MAX_INT'},
                    {'value': -2147483648, 'description': 'MIN_INT'},
                    {'value': None, 'description': 'Null'},
                ]
            },
            {
                'type': 'array',
                'cases': [
                    {'value': [], 'description': 'Empty array'},
                    {'value': [None], 'description': 'Array with null'},
                    {'value': [[[[[]]]]], 'description': 'Deeply nested array'},
                ]
            },
        ]

        self.knowledge['edge_cases'] = edge_cases
        self._save_knowledge('edge_cases')

    def add_knowledge(self, knowledge_type: str, item: Dict[str, Any]):
        """Add new knowledge item"""
        if knowledge_type not in self.knowledge_types:
            logger.warning(f"Unknown knowledge type: {knowledge_type}, using 'test_patterns'")
            knowledge_type = 'test_patterns'

        if knowledge_type not in self.knowledge:
            self.knowledge[knowledge_type] = []

        # Add metadata
        item['added_at'] = datetime.now().isoformat()
        if 'id' not in item:
            item['id'] = str(hash(json.dumps(str(item), sort_keys=True)))

        self.knowledge[knowledge_type].append(item)
        self._save_knowledge(knowledge_type)

    def get_knowledge(self, knowledge_type: str, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get knowledge items with optional filtering"""
        if knowledge_type not in self.knowledge_types:
            return []

        items = self.knowledge.get(knowledge_type, [])

        if filters:
            filtered_items = []
            for item in items:
                match = all(item.get(key) == value for key, value in filters.items())
                if match:
                    filtered_items.append(item)
            return filtered_items

        return items

    def search_knowledge(self, query: str, knowledge_types: List[str] = None) -> List[Dict[str, Any]]:
        """Search across knowledge base"""
        if knowledge_types is None:
            knowledge_types = list(self.knowledge_types.keys())

        results = []
        query_lower = query.lower()

        for knowledge_type in knowledge_types:
            if knowledge_type in self.knowledge:
                for item in self.knowledge[knowledge_type]:
                    item_str = json.dumps(item).lower()
                    if query_lower in item_str:
                        results.append({
                            'knowledge_type': knowledge_type,
                            'item': item
                        })

        return results

    def get_test_pattern_for_endpoint(self, method: str, endpoint_type: str = None) -> List[Dict[str, Any]]:
        """Get relevant test patterns for an endpoint"""
        patterns = self.get_knowledge('test_patterns')
        return [
            p for p in patterns
            if 'All' in p.get('applicable_to', []) or method in p.get('applicable_to', [])
        ]

    def get_edge_cases_for_type(self, data_type: str) -> List[Dict[str, Any]]:
        """Get edge cases for a specific data type"""
        for group in self.get_knowledge('edge_cases'):
            if group.get('type') == data_type:
                return group.get('cases', [])
        return []

    def add_test_result(self, test_case: Dict[str, Any], result: Dict[str, Any]):
        """Add test execution result to knowledge base"""
        if result.get('passed'):
            self.add_knowledge('test_patterns', {
                'endpoint': test_case.get('endpoint'),
                'method': test_case.get('method'),
                'test_type': test_case.get('test_type'),
                'result': 'success'
            })
        else:
            self.add_knowledge('bug_patterns', {
                'endpoint': test_case.get('endpoint'),
                'method': test_case.get('method'),
                'test_type': test_case.get('test_type'),
                'error': result.get('error'),
            })

    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        stats = {'total_items': 0, 'by_type': {}, 'last_updated': None}

        for knowledge_type in self.knowledge_types:
            items = self.knowledge.get(knowledge_type, [])
            count = len(items)
            stats['total_items'] += count
            stats['by_type'][knowledge_type] = count

            for item in items:
                updated = item.get('updated_at') or item.get('added_at')
                if updated and (stats['last_updated'] is None or updated > stats['last_updated']):
                    stats['last_updated'] = updated

        return stats

    def export_knowledge(self, output_path: Path = None) -> Path:
        """Export all knowledge to a file"""
        if output_path is None:
            output_path = self.knowledge_dir / f"knowledge_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        export_data = {
            'exported_at': datetime.now().isoformat(),
            'knowledge_types': self.knowledge_types,
            'knowledge': self.knowledge
        }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported knowledge to {output_path}")
        return output_path

    def import_knowledge(self, import_path: Path, merge: bool = True):
        """Import knowledge from a file"""
        with open(import_path, 'r') as f:
            import_data = json.load(f)

        if merge:
            for knowledge_type, items in import_data.get('knowledge', {}).items():
                if knowledge_type in self.knowledge_types:
                    if knowledge_type not in self.knowledge:
                        self.knowledge[knowledge_type] = []

                    existing_ids = {item.get('id') for item in self.knowledge[knowledge_type]}
                    for item in items:
                        if item.get('id') not in existing_ids:
                            self.knowledge[knowledge_type].append(item)

                    self._save_knowledge(knowledge_type)
        else:
            self.knowledge = import_data.get('knowledge', {})
            for knowledge_type in self.knowledge:
                self._save_knowledge(knowledge_type)

        logger.info(f"Imported knowledge from {import_path}")

    def _load_knowledge(self, knowledge_type: str) -> List[Dict[str, Any]]:
        """Load knowledge from file"""
        file_path = self.knowledge_dir / f"{knowledge_type}.json"
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load {knowledge_type}: {e}")
        return []

    def _save_knowledge(self, knowledge_type: str):
        """Save knowledge to file"""
        file_path = self.knowledge_dir / f"{knowledge_type}.json"
        try:
            with open(file_path, 'w') as f:
                json.dump(self.knowledge.get(knowledge_type, []), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save {knowledge_type}: {e}")