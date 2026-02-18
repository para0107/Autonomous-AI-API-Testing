"""
Knowledge base management for RAG system
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
        # Load existing knowledge
        self.knowledge = {}
        for knowledge_type in self.knowledge_types:
            self.knowledge[knowledge_type] = self._load_knowledge(knowledge_type)

        # Add default knowledge if empty
        if not self.knowledge['test_patterns']:
            self._add_default_test_patterns()

        if not self.knowledge['edge_cases']:
            self._add_default_edge_cases()

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
                    {'value': '  ', 'description': 'Multiple spaces'},
                    {'value': '\n\t', 'description': 'Whitespace characters'},
                    {'value': 'A' * 10000, 'description': 'Very long string'},
                    {'value': '<script>alert("xss")</script>', 'description': 'XSS attempt'},
                    {'value': "'; DROP TABLE users; --", 'description': 'SQL injection'},
                    {'value': '../../etc/passwd', 'description': 'Path traversal'},
                    {'value': 'émojis 👍 unicode', 'description': 'Unicode and emojis'}
                ]
            },
            {
                'type': 'integer',
                'cases': [
                    {'value': 0, 'description': 'Zero'},
                    {'value': -1, 'description': 'Negative one'},
                    {'value': 2147483647, 'description': 'MAX_INT'},
                    {'value': -2147483648, 'description': 'MIN_INT'},
                    {'value': 1.5, 'description': 'Decimal (for int field)'},
                    {'value': 'abc', 'description': 'String (for int field)'},
                    {'value': None, 'description': 'Null'},
                    {'value': float('inf'), 'description': 'Infinity'},
                    {'value': float('nan'), 'description': 'NaN'}
                ]
            },
            {
                'type': 'array',
                'cases': [
                    {'value': [], 'description': 'Empty array'},
                    {'value': [None], 'description': 'Array with null'},
                    {'value': list(range(10000)), 'description': 'Very large array'},
                    {'value': [[[[[]]]]], 'description': 'Deeply nested array'},
                    {'value': [1, 'string', None, {}], 'description': 'Mixed types'}
                ]
            },
            {
                'type': 'date',
                'cases': [
                    {'value': '0000-00-00', 'description': 'Invalid date'},
                    {'value': '2024-02-30', 'description': 'Non-existent date'},
                    {'value': '1970-01-01', 'description': 'Unix epoch'},
                    {'value': '9999-12-31', 'description': 'Far future'},
                    {'value': 'today', 'description': 'String instead of date'},
                    {'value': '2024/01/01', 'description': 'Wrong format'}
                ]
            }
        ]

        self.knowledge['edge_cases'] = edge_cases
        self._save_knowledge('edge_cases')

    def add_knowledge(self, knowledge_type: str, item: Dict[str, Any]):
        """Add new knowledge item"""
        if knowledge_type not in self.knowledge_types:
            raise ValueError(f"Unknown knowledge type: {knowledge_type}")

        if knowledge_type not in self.knowledge:
            self.knowledge[knowledge_type] = []

        # Add metadata
        item['added_at'] = datetime.now().isoformat()
        item['id'] = str(hash(json.dumps(item, sort_keys=True)))

        self.knowledge[knowledge_type].append(item)
        self._save_knowledge(knowledge_type)

        logger.info(f"Added knowledge item to {knowledge_type}")

    def get_knowledge(self, knowledge_type: str, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get knowledge items with optional filtering"""
        if knowledge_type not in self.knowledge_types:
            raise ValueError(f"Unknown knowledge type: {knowledge_type}")

        items = self.knowledge.get(knowledge_type, [])

        if filters:
            filtered_items = []
            for item in items:
                match = True
                for key, value in filters.items():
                    if key not in item or item[key] != value:
                        match = False
                        break
                if match:
                    filtered_items.append(item)
            return filtered_items

        return items

    def update_knowledge(self, knowledge_type: str, item_id: str, updates: Dict[str, Any]):
        """Update a knowledge item"""
        if knowledge_type not in self.knowledge_types:
            raise ValueError(f"Unknown knowledge type: {knowledge_type}")

        items = self.knowledge.get(knowledge_type, [])

        for i, item in enumerate(items):
            if item.get('id') == item_id:
                items[i].update(updates)
                items[i]['updated_at'] = datetime.now().isoformat()
                self._save_knowledge(knowledge_type)
                logger.info(f"Updated knowledge item {item_id} in {knowledge_type}")
                return True

        return False

    def remove_knowledge(self, knowledge_type: str, item_id: str):
        """Remove a knowledge item"""
        if knowledge_type not in self.knowledge_types:
            raise ValueError(f"Unknown knowledge type: {knowledge_type}")

        items = self.knowledge.get(knowledge_type, [])

        self.knowledge[knowledge_type] = [
            item for item in items if item.get('id') != item_id
        ]

        self._save_knowledge(knowledge_type)
        logger.info(f"Removed knowledge item {item_id} from {knowledge_type}")

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
            # Merge with existing knowledge
            for knowledge_type, items in import_data.get('knowledge', {}).items():
                if knowledge_type in self.knowledge_types:
                    if knowledge_type not in self.knowledge:
                        self.knowledge[knowledge_type] = []

                    # Add items that don't exist
                    existing_ids = {item.get('id') for item in self.knowledge[knowledge_type]}
                    for item in items:
                        if item.get('id') not in existing_ids:
                            self.knowledge[knowledge_type].append(item)

                    self._save_knowledge(knowledge_type)
        else:
            # Replace existing knowledge
            self.knowledge = import_data.get('knowledge', {})
            for knowledge_type in self.knowledge:
                self._save_knowledge(knowledge_type)

        logger.info(f"Imported knowledge from {import_path}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        stats = {
            'total_items': 0,
            'by_type': {},
            'last_updated': None
        }

        for knowledge_type in self.knowledge_types:
            items = self.knowledge.get(knowledge_type, [])
            count = len(items)
            stats['total_items'] += count
            stats['by_type'][knowledge_type] = count

            # Find last updated
            for item in items:
                updated = item.get('updated_at') or item.get('added_at')
                if updated and (stats['last_updated'] is None or updated > stats['last_updated']):
                    stats['last_updated'] = updated

        return stats

    def search_knowledge(self, query: str, knowledge_types: List[str] = None) -> List[Dict[str, Any]]:
        """Search across knowledge base"""
        if knowledge_types is None:
            knowledge_types = list(self.knowledge_types.keys())

        results = []
        query_lower = query.lower()

        for knowledge_type in knowledge_types:
            if knowledge_type in self.knowledge:
                for item in self.knowledge[knowledge_type]:
                    # Search in string representation
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

        relevant_patterns = []
        for pattern in patterns:
            applicable_to = pattern.get('applicable_to', [])
            if 'All' in applicable_to or method in applicable_to:
                relevant_patterns.append(pattern)

        return relevant_patterns

    def get_edge_cases_for_type(self, data_type: str) -> List[Dict[str, Any]]:
        """Get edge cases for a specific data type"""
        edge_cases = self.get_knowledge('edge_cases')

        for edge_case_group in edge_cases:
            if edge_case_group.get('type') == data_type:
                return edge_case_group.get('cases', [])

        return []

    def add_test_result(self, test_case: Dict[str, Any], result: Dict[str, Any]):
        """Add test execution result to knowledge base"""
        # Store successful test patterns
        if result.get('passed'):
            pattern = {
                'endpoint': test_case.get('endpoint'),
                'method': test_case.get('method'),
                'test_type': test_case.get('test_type'),
                'test_data': test_case.get('test_data'),
                'assertions': test_case.get('assertions'),
                'execution_time': result.get('execution_time'),
                'result': 'success'
            }
            self.add_knowledge('test_patterns', pattern)
        else:
            # Store bug patterns
            bug = {
                'endpoint': test_case.get('endpoint'),
                'method': test_case.get('method'),
                'test_type': test_case.get('test_type'),
                'test_data': test_case.get('test_data'),
                'error': result.get('error'),
                'expected': test_case.get('expected'),
                'actual': result.get('actual')
            }
            self.add_knowledge('bug_patterns', bug)