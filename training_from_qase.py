"""
Training Script for API Testing Agent using QASE Test Files

Integrates with existing project structure
"""

import json
import asyncio
from pathlib import Path
from typing import List, Dict
import sys
import re
import numpy as np
import torch
from dotenv import load_dotenv
import os

from reinforcement_learning import ExperienceBuffer

load_dotenv()


# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag.knowledge_base import KnowledgeBase
from rag.vector_store import VectorStore
from rag.embeddings import EmbeddingManager
from rag.chunking import ChunkingStrategy
from rag.indexer import Indexer
from reinforcement_learning.rl_optimizer import RLOptimizer, rl_config
from test_execution.executor import TestExecutor
from utils.logger import setup_logger
from config import paths

logger = setup_logger(__name__)


class QASETestParser:
    """Parses QASE test files into system format"""

    def __init__(self):
        self.test_patterns = []
        self.edge_cases = []
        self.validation_rules = []

    def _to_text(self, value) -> str:
        """Safely convert value to string (return empty string for None/invalid)."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            return str(value)
        except Exception:
            return ""

    def parse_qase_files(self, file_paths: List[Path]) -> Dict[str, List]:
        """Parse all QASE JSON files"""
        logger.info(f"Parsing {len(file_paths)} QASE files...")

        all_data: Dict[str, List] = {
            'test_patterns': [],
            'edge_cases': [],
            'validation_rules': [],
            'api_patterns': []
        }

        for file_path in file_paths:
            logger.info(f"Parsing {file_path.name}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                parsed = self._parse_suite_recursive(data.get('suites', []))

                all_data['test_patterns'].extend(parsed['test_patterns'])
                all_data['edge_cases'].extend(parsed['edge_cases'])
                all_data['validation_rules'].extend(parsed['validation_rules'])
                all_data['api_patterns'].extend(parsed['api_patterns'])

        logger.info(f"Parsed {len(all_data['test_patterns'])} test patterns")
        logger.info(f"Parsed {len(all_data['edge_cases'])} edge cases")
        logger.info(f"Parsed {len(all_data['validation_rules'])} validation rules")
        logger.info(f"Parsed {len(all_data['api_patterns'])} API patterns")

        return all_data

    def _parse_suite_recursive(self, suites: List[Dict]) -> Dict[str, List]:
        """Recursively parse test suites"""
        data: Dict[str, List] = {
            'test_patterns': [],
            'edge_cases': [],
            'validation_rules': [],
            'api_patterns': []
        }

        for suite in suites:
            for case in suite.get('cases', []) or []:
                parsed_case = self._parse_test_case(case)

                if self._is_edge_case(case):
                    data['edge_cases'].append(parsed_case)
                else:
                    data['test_patterns'].append(parsed_case)

                validation = self._extract_validation_rules(case)
                if validation:
                    data['validation_rules'].append(validation)

                api_pattern = self._extract_api_pattern(case)
                if api_pattern:
                    data['api_patterns'].append(api_pattern)

            nested = self._parse_suite_recursive(suite.get('suites', []) or [])
            for key in data:
                data[key].extend(nested[key])

        return data

    def _parse_test_case(self, case: Dict) -> Dict:
        """Parse individual test case"""
        # Safely extract fields
        title = self._to_text(case.get('title', ''))
        description = self._to_text(case.get('description', ''))
        endpoint_info = self._extract_endpoint_info(case)

        return {
            'id': case.get('id'),
            'name': title,
            'description': description,
            'test_type': self._determine_test_type(case),
            'priority': case.get('priority', 'medium'),
            'severity': case.get('severity', 'normal'),
            'endpoint': endpoint_info.get('endpoint', ''),
            'method': endpoint_info.get('method', ''),
            'preconditions': self._to_text(case.get('preconditions', '')),
            'postconditions': self._to_text(case.get('postconditions', '')),
            'steps': self._parse_steps(case.get('steps', []) or []),
            'expected_status': endpoint_info.get('expected_status', 200),
            'parameters': self._parse_parameters(case.get('params', []) or []),
            'tags': case.get('tags', []) or []
        }

    def _extract_endpoint_info(self, case: Dict) -> Dict:
        """Extract endpoint, method, and expected status"""
        info: Dict = {'endpoint': '', 'method': '', 'expected_status': 200}

        for step in (case.get('steps', []) or []):
            action = self._to_text(step.get('action', ''))
            expected = self._to_text(step.get('expected_result', ''))

            method_match = re.search(r'\b(GET|POST|PUT|DELETE|PATCH)\s+([/\w{}-]+)', action, re.IGNORECASE)
            if method_match:
                info['method'] = method_match.group(1).upper()
                info['endpoint'] = method_match.group(2)

            status_match = re.search(r'\b(\d{3})\b', expected)
            if status_match:
                info['expected_status'] = int(status_match.group(1))

        return info

    def _parse_steps(self, steps: List[Dict]) -> List[Dict]:
        """Parse test steps"""
        return [{
            'position': step.get('position', 0),
            'action': self._to_text(step.get('action', '')),
            'expected_result': self._to_text(step.get('expected_result', '')),
            'data': step.get('data', '')
        } for step in steps]

    def _parse_parameters(self, params: List[Dict]) -> List[Dict]:
        """Parse test parameters"""
        return [{
            'name': self._to_text(param.get('title', '')),
            'values': param.get('values', []) or []
        } for param in params]

    def _determine_test_type(self, case: Dict) -> str:
        """Determine test type from title/description"""
        title = self._to_text(case.get('title', '')).lower()
        desc = self._to_text(case.get('description', '')).lower()
        text = f"{title} {desc}"

        if any(word in text for word in ['duplicate', 'invalid', 'negative', 'fail', 'error']):
            return 'validation'
        elif any(word in text for word in ['auth', 'login', 'token', 'permission']):
            return 'authentication'
        elif any(word in text for word in ['security', 'injection', 'xss', 'csrf']):
            return 'security'
        elif any(word in text for word in ['edge', 'boundary', 'limit', 'null', 'empty']):
            return 'edge_case'
        elif any(word in text for word in ['successful', 'positive', 'happy']):
            return 'happy_path'
        else:
            return 'functional'

    def _is_edge_case(self, case: Dict) -> bool:
        """Determine if test is an edge case"""
        title = self._to_text(case.get('title', '')).lower()
        desc = self._to_text(case.get('description', '')).lower()

        edge_keywords = [
            'duplicate', 'invalid', 'null', 'empty', 'boundary',
            'limit', 'maximum', 'minimum', 'negative', 'special'
        ]

        return any(keyword in f"{title} {desc}" for keyword in edge_keywords)

    def _extract_validation_rules(self, case: Dict) -> Dict:
        """Extract validation rules"""
        title = self._to_text(case.get('title', ''))
        desc = self._to_text(case.get('description', ''))

        if 'duplicate' in title.lower() or 'unique' in desc.lower():
            endpoint_info = self._extract_endpoint_info(case)
            return {
                'rule_type': 'uniqueness',
                'field': self._extract_field_name(title, desc),
                'endpoint': endpoint_info.get('endpoint', ''),
                'description': desc
            }

        return {}

    def _extract_field_name(self, title: str, desc: str) -> str:
        """Extract field name from text"""
        text = f"{self._to_text(title)} {self._to_text(desc)}".lower()
        fields = ['email', 'username', 'id', 'password', 'license', 'plate']

        for field in fields:
            if field in text:
                return field

        return 'unknown'

    def _extract_api_pattern(self, case: Dict) -> Dict:
        """Extract API pattern for RAG"""
        endpoint_info = self._extract_endpoint_info(case)

        if not endpoint_info['endpoint']:
            return {}

        return {
            'endpoint': endpoint_info['endpoint'],
            'method': endpoint_info['method'],
            'test_type': self._determine_test_type(case),
            'expected_status': endpoint_info['expected_status'],
            'description': self._to_text(case.get('description', '')),
            'preconditions': self._to_text(case.get('preconditions', ''))
        }


class TrainingPipeline:
    """Complete training pipeline using existing project components"""

    def __init__(self, api_base_url: str, hf_token: str = None):
        self.api_base_url = api_base_url
        self.parser = QASETestParser()

        self.knowledge_base = KnowledgeBase()
        self.vector_store = VectorStore()
        self.embedding_manager = EmbeddingManager(hf_token=hf_token)
        self.chunking_strategy = ChunkingStrategy()
        self.indexer = Indexer(
            self.vector_store,
            self.embedding_manager,
            self.chunking_strategy
        )
        self.rl_optimizer = RLOptimizer()
        self.executor = TestExecutor()

    async def train(self, qase_files: List[Path], execute_tests: bool = True):
        """Complete training workflow"""

        logger.info("=" * 80)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("=" * 80)

        logger.info("\nSTEP 1: Parsing QASE Test Files...")
        parsed_data = self.parser.parse_qase_files(qase_files)

        logger.info("\nSTEP 2: Populating Knowledge Base...")
        await self._populate_knowledge_base(parsed_data)

        logger.info("\nSTEP 3: Indexing Knowledge with FAISS...")
        await self._index_knowledge(parsed_data)

        execution_results: List[Dict] = []
        if execute_tests:
            logger.info("\nSTEP 4: Executing Tests Against API...")
            execution_results = await self._execute_tests(parsed_data['test_patterns'])

            logger.info("\nSTEP 5: Training RL Model...")
            await self._train_rl_model(parsed_data, execution_results)
        else:
            logger.info("\nSTEP 4 & 5: Skipped (execute_tests=False)")

        logger.info("\nSTEP 6: Saving Training Results...")
        self._save_results(parsed_data, execution_results)

        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

        self._print_summary(parsed_data, execution_results)

    async def _populate_knowledge_base(self, parsed_data: Dict):
        """Add parsed data to knowledge base"""

        # Clear existing knowledge base to avoid duplicates
        logger.info("Clearing existing knowledge base...")
        self.knowledge_base = KnowledgeBase()

        for pattern in parsed_data['test_patterns']:
            self.knowledge_base.add_knowledge('test_patterns', pattern)
        logger.info(f"Added {len(parsed_data['test_patterns'])} test patterns")

        for edge_case in parsed_data['edge_cases']:
            self.knowledge_base.add_knowledge('edge_cases', edge_case)
        logger.info(f"Added {len(parsed_data['edge_cases'])} edge cases")

        for rule in parsed_data['validation_rules']:
            if rule:
                self.knowledge_base.add_knowledge('validation_rules', rule)
        logger.info(f"Added {len([r for r in parsed_data['validation_rules'] if r])} validation rules")

        for api_pattern in parsed_data['api_patterns']:
            if api_pattern:
                self.knowledge_base.add_knowledge('api_patterns', api_pattern)
        logger.info(f"Added {len([a for a in parsed_data['api_patterns'] if a])} API patterns")

    async def _index_knowledge(self, parsed_data: Dict):
        """Index knowledge for RAG retrieval using existing vector store"""

        await self._index_items(
            'test_patterns',
            parsed_data['test_patterns'],
            "Test Pattern"
        )

        await self._index_items(
            'edge_cases',
            parsed_data['edge_cases'],
            "Edge Case"
        )

        filtered_rules = [r for r in parsed_data['validation_rules'] if r]
        await self._index_items(
            'validation_rules',
            filtered_rules,
            "Validation Rule"
        )

        filtered_patterns = [a for a in parsed_data['api_patterns'] if a]
        await self._index_items(
            'api_specifications',
            filtered_patterns,
            "API Pattern"
        )

        logger.info("All knowledge indexed successfully")

    async def _index_items(self, index_name: str, items: List[Dict], item_type: str):
        """Index a list of items into vector store"""

        if not items:
            logger.info(f"No {item_type}s to index")
            return

        logger.info(f"Indexing {len(items)} {item_type}s into '{index_name}'...")

        # Clear existing index to avoid duplicates
        logger.info(f"Clearing existing index '{index_name}'...")
        self.vector_store.clear_index(index_name)

        texts = []
        for item in items:
            text_parts = [
                item.get('name', ''),
                item.get('description', ''),
                item.get('endpoint', ''),
                item.get('method', ''),
                str(item.get('test_type', ''))
            ]
            text = ' '.join([p for p in text_parts if p])
            texts.append(text)

        embeddings = await self.embedding_manager.generate_embeddings(texts)
        embeddings_array = np.array(embeddings)

        self.vector_store.add(
            index_name=index_name,
            embeddings=embeddings_array,
            metadata=items,
            ids=None
        )

        self.vector_store.save_index(index_name)

        logger.info(f"Indexed {len(items)} {item_type}s to data/vectors/{index_name}/")

    async def _execute_tests(self, test_patterns: List[Dict]) -> List[Dict]:
        """Execute tests against local API"""

        logger.info(f"Executing {len(test_patterns)} tests against {self.api_base_url}...")

        results: List[Dict] = []
        executed_count = 0

        async with self.executor as executor:  # This initializes the session
            for i, test in enumerate(test_patterns, 1):
                if not test.get('endpoint') or not test.get('method'):
                    continue

                try:
                    result = await executor.execute_test(test, self.api_base_url)
                    results.append(result)
                    executed_count += 1

                    status = "PASS" if result.get('passed') else "FAIL"
                    if i % 10 == 0 or result.get('passed') == False:
                        logger.info(
                            f"[{executed_count}/{len(test_patterns)}] {status} - {test.get('name', 'Unnamed')[:50]}")

                except Exception as e:
                    logger.error(f"Error executing test {test.get('name')}: {e}")

        pass_count = sum(1 for r in results if r.get('passed'))
        pass_rate = (pass_count / len(results) * 100) if results else 0

        logger.info(f"\nExecution Results:")
        logger.info(f"   Total Tests: {len(results)}")
        logger.info(f"   Passed: {pass_count} ({pass_rate:.1f}%)")
        logger.info(f"   Failed: {len(results) - pass_count}")

        return results

    async def _train_rl_model(self, parsed_data: Dict, execution_results: List[Dict]):
        """Train RL model from execution results"""

        if not execution_results:
            logger.warning("No execution results to train on")
            return

        logger.info("Creating training experiences...")

        # Clear existing experience buffer to avoid duplicates
        logger.info("Clearing existing experience buffer...")
        self.rl_optimizer.experience_buffer = ExperienceBuffer(rl_config.buffer_size)

        for result in execution_results:
            state_vector = self._create_state_vector(result)
            state_tensor = torch.FloatTensor(state_vector)

            action_idx = self._test_type_to_action(result.get('test_type', 'functional'))
            action_tensor = torch.LongTensor([action_idx])

            reward = 10.0 if result.get('passed') else -5.0

            next_state_tensor = torch.FloatTensor(state_vector)

            self.rl_optimizer.experience_buffer.add(
                state_tensor,
                action_tensor,
                reward,
                next_state_tensor,
                done=True
            )

        logger.info(f"Created {len(self.rl_optimizer.experience_buffer)} training experiences")

        logger.info("Training RL model...")
        try:
            if len(self.rl_optimizer.experience_buffer) >= 32:
                for _ in range(50):
                    self.rl_optimizer.train()
                logger.info("RL training completed")
            else:
                logger.warning("Not enough experiences for training")
        except Exception as e:
            logger.error(f"RL training error: {e}")
            logger.info("Continuing without RL training...")

    def _create_state_vector(self, result: Dict) -> np.ndarray:
        """Create simple state vector from test result"""
        features = [
            1.0 if result.get('test_type') == 'happy_path' else 0.0,
            1.0 if result.get('test_type') == 'validation' else 0.0,
            1.0 if result.get('test_type') == 'edge_case' else 0.0,
            1.0 if result.get('test_type') == 'security' else 0.0,
            1.0 if result.get('method') == 'GET' else 0.0,
            1.0 if result.get('method') == 'POST' else 0.0,
            1.0 if result.get('passed') else 0.0,
        ]
        return np.array(features + [0.0] * (640 - len(features)))

    def _test_type_to_action(self, test_type: str) -> int:
        """Convert test type to action index"""
        test_types = ['happy_path', 'validation', 'authentication', 'security',
                     'edge_case', 'boundary', 'functional', 'integration',
                     'performance', 'other']
        try:
            return test_types.index(test_type)
        except ValueError:
            return len(test_types) - 1

    def _save_results(self, parsed_data: Dict, execution_results: List[Dict]):
        """Save all training results"""

        kb_path = self.knowledge_base.export_knowledge()
        logger.info(f"Knowledge base saved to: {kb_path}")

        logger.info(f"Vector indices saved to: {paths.VECTOR_STORE_DIR}")

        training_dir = paths.BASE_DIR / 'data' / 'training'
        training_dir.mkdir(parents=True, exist_ok=True)

        training_data_path = training_dir / 'qase_training_data.json'
        with open(training_data_path, 'w') as f:
            json.dump({
                'parsed_data': parsed_data,
                'execution_results': execution_results[:100]
            }, f, indent=2)
        logger.info(f"Training data saved to: {training_data_path}")

        if execution_results:
            rl_model_path = paths.BASE_DIR / 'data' / 'models' / 'rl_qase_trained.pth'
            rl_model_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                self.rl_optimizer.save_checkpoint(str(rl_model_path))
                logger.info(f"RL model saved to: {rl_model_path}")
            except Exception as e:
                logger.warning(f"Could not save RL model: {e}")

    def _print_summary(self, parsed_data: Dict, execution_results: List[Dict]):
        """Print training summary"""

        stats = self.knowledge_base.get_statistics()

        print("\n" + "=" * 80)
        print("TRAINING SUMMARY")
        print("=" * 80)
        print(f"Parsed Data:")
        print(f"  Test Patterns:     {len(parsed_data['test_patterns'])}")
        print(f"  Edge Cases:        {len(parsed_data['edge_cases'])}")
        print(f"  Validation Rules:  {len([r for r in parsed_data['validation_rules'] if r])}")
        print(f"  API Patterns:      {len([a for a in parsed_data['api_patterns'] if a])}")
        print(f"\nKnowledge Base:")
        print(f"  Total Items:       {stats['total_items']}")
        print(f"  By Type:           {stats['by_type']}")
        print(f"\nVector Store:")
        vector_stats = self.vector_store.get_all_stats()
        for index_name, index_stats in vector_stats.items():
            print(f"  {index_name}: {index_stats['total_embeddings']} embeddings")

        if execution_results:
            pass_count = sum(1 for r in execution_results if r.get('passed'))
            pass_rate = (pass_count / len(execution_results) * 100) if execution_results else 0
            print(f"\nTest Execution:")
            print(f"  Total Tests:       {len(execution_results)}")
            print(f"  Passed:            {pass_count} ({pass_rate:.1f}%)")
            print(f"  Failed:            {len(execution_results) - pass_count}")

        print("=" * 80)
        print("\nYour system is now trained and ready to use!")
        print("\nFiles created:")
        print(f"   - data/knowledge_base/*.json")
        print(f"   - data/vectors/*/")
        print(f"   - data/training/qase_training_data.json")
        if execution_results:
            print(f"   - data/models/rl_qase_trained.pth")
        print("=" * 80)


async def main():
    """Main training entry point"""

    logger.info("API Testing Agent - QASE Training Pipeline")
    logger.info("Starting automated training process...")

    # Configuration
    QASE_FILES = [
        Path('data/training/QA-Backend-Data.json')
    ]
    API_BASE_URL = "https://localhost:7063"
    EXECUTE_TESTS = True

    # Verify files exist
    missing_files = [f for f in QASE_FILES if not f.exists()]
    if missing_files:
        logger.error(f"Missing required files: {[f.name for f in missing_files]}")
        return

    logger.info(f"Configuration:")
    logger.info(f"  QASE Files: {len(QASE_FILES)} files")
    logger.info(f"  API URL: {API_BASE_URL}")
    logger.info(f"  Execute Tests: {'Yes' if EXECUTE_TESTS else 'No (knowledge base only)'}")

    # Get HuggingFace token from environment
    HF_TOKEN = os.getenv("HG_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

    # Initialize and run training
    pipeline = TrainingPipeline(api_base_url=API_BASE_URL, hf_token=HF_TOKEN)
    await pipeline.train(QASE_FILES, execute_tests=EXECUTE_TESTS)

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("   1. Verify training: Check data/vectors/ and data/knowledge_base/")
    logger.info("   2. Test retrieval: Search for similar patterns in knowledge base")
    logger.info("   3. Generate tests: Run main.py with your API code")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())