"""
Load QA-Backend-Data.json test cases into RAG system
Handles nested suite structure from test management system
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag.embeddings import EmbeddingManager
from rag.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_test_cases(suite_data, parent_path=""):
    """Recursively extract all test cases from nested suite structure"""
    test_cases = []

    suite_title = suite_data.get('title', 'Unknown Suite')
    current_path = f"{parent_path} > {suite_title}" if parent_path else suite_title

    # Extract cases from current suite
    for case in suite_data.get('cases', []):
        case['suite_path'] = current_path
        test_cases.append(case)

    # Recursively process nested suites
    for nested_suite in suite_data.get('suites', []):
        test_cases.extend(extract_test_cases(nested_suite, current_path))

    return test_cases


def create_searchable_text(test_case):
    """Create searchable text from test case"""
    parts = []

    if test_case.get('title'):
        parts.append(f"Title: {test_case['title']}")

    if test_case.get('description'):
        parts.append(f"Description: {test_case['description']}")

    if test_case.get('suite_path'):
        parts.append(f"Suite: {test_case['suite_path']}")

    # Extract HTTP method and endpoint from steps
    for step in test_case.get('steps', []):
        action = step.get('action', '')
        if action:
            parts.append(f"Action: {action}")

        expected = step.get('expected_result', '')
        if expected:
            parts.append(f"Expected: {expected}")

        data = step.get('data', '')
        if data:
            parts.append(f"Data: {data}")

    if test_case.get('preconditions'):
        parts.append(f"Preconditions: {test_case['preconditions']}")

    parts.append(f"Priority: {test_case.get('priority', 'medium')}")
    parts.append(f"Type: {test_case.get('type', 'other')}")

    return " | ".join(parts)


def categorize_test_case(test_case):
    """Determine which RAG index to use for this test case"""
    test_type = test_case.get('type', 'other')
    priority = test_case.get('priority', 'medium')
    title = test_case.get('title', '').lower()

    if priority == 'high' or test_type in ['smoke', 'critical']:
        return 'test_patterns'
    elif 'negative' in title or 'fail' in title or 'invalid' in title:
        return 'edge_cases'
    elif test_case.get('preconditions'):
        return 'validation_rules'
    else:
        return 'test_patterns'


async def load_test_cases_to_rag():
    """Load test cases from QA-Backend-Data.json into RAG"""

    # Load JSON file
    kb_path = Path("data/training/QA-Backend-Data.json")

    if not kb_path.exists():
        logger.error(f"❌ Knowledge base not found: {kb_path}")
        logger.info(f"Current directory: {Path.cwd()}")
        logger.info(f"Looking for: {kb_path.absolute()}")
        return False

    logger.info(f"📖 Loading test cases from {kb_path}")

    with open(kb_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract all test cases
    logger.info("Extracting test cases from nested suites...")
    all_test_cases = []

    for top_suite in data.get('suites', []):
        all_test_cases.extend(extract_test_cases(top_suite))

    logger.info(f"Found {len(all_test_cases)} test cases")

    if len(all_test_cases) == 0:
        logger.error("No test cases found in knowledge base!")
        return False

    # Initialize RAG components
    logger.info("Initializing RAG components...")
    em = EmbeddingManager()
    vs = VectorStore()

    # Group test cases by index
    index_groups = {
        'test_patterns': [],
        'edge_cases': [],
        'validation_rules': []
    }

    logger.info("Categorizing and generating embeddings...")
    for idx, test_case in enumerate(all_test_cases):
        try:
            # Categorize
            index_name = categorize_test_case(test_case)

            # Create searchable text
            search_text = create_searchable_text(test_case)

            # Generate embedding
            embedding = await em.embed_text(search_text)

            # Store metadata
            metadata = {
                'id': test_case.get('id'),
                'title': test_case.get('title'),
                'suite_path': test_case.get('suite_path'),
                'description': test_case.get('description'),
                'priority': test_case.get('priority'),
                'type': test_case.get('type'),
                'steps': test_case.get('steps', []),
                'preconditions': test_case.get('preconditions'),
                'search_text': search_text[:200]  # Truncate for storage
            }

            # Add to group
            index_groups[index_name].append({
                'id': test_case.get('id', idx),
                'embedding': embedding,
                'metadata': metadata
            })

            if (idx + 1) % 20 == 0:
                logger.info(f"  Processed {idx + 1}/{len(all_test_cases)} test cases...")

        except Exception as e:
            logger.error(f"  Error processing test case {test_case.get('id', idx)}: {e}")
            continue

    # Add to vector store in batches
    logger.info("\nAdding embeddings to vector store...")
    total_added = 0

    for index_name, items in index_groups.items():
        if not items:
            logger.info(f"  {index_name}: 0 items (skipping)")
            continue

        try:
            # Prepare batch data
            embeddings_array = np.vstack([item['embedding'] for item in items])
            metadata_list = [item['metadata'] for item in items]
            ids_list = [int(item['id']) for item in items]

            # Add to vector store
            vs.add(
                index_name=index_name,
                embeddings=embeddings_array,
                metadata=metadata_list,
                ids=ids_list
            )

            total_added += len(items)
            logger.info(f"  ✅ {index_name}: Added {len(items)} items")

        except Exception as e:
            logger.error(f"  ❌ {index_name}: Failed to add items: {e}")
            continue

    # Save vector store
    logger.info("\nSaving vector store...")
    try:
        vs.save_all()
        logger.info("✅ Vector store saved successfully")
    except Exception as e:
        logger.error(f"❌ Failed to save vector store: {e}")
        return False

    # Print statistics
    logger.info("\n" + "="*60)
    logger.info("LOADING COMPLETE")
    logger.info("="*60)
    logger.info(f"Total test cases processed: {len(all_test_cases)}")
    logger.info(f"Successfully added to RAG: {total_added}")

    for index_name, items in index_groups.items():
        logger.info(f"  - {index_name}: {len(items)} items")

    # Verify by searching
    logger.info("\n🔍 Verifying RAG retrieval...")
    test_queries = [
        "GET user by ID",
        "POST create user",
        "reservation endpoint"
    ]

    for query in test_queries:
        try:
            test_embedding = await em.embed_text(query)
            ids, distances, metadata = vs.search(
                index_name='test_patterns',
                query_embedding=test_embedding,
                k=3
            )

            logger.info(f"\nQuery: '{query}'")
            logger.info(f"  Found {len([m for m in metadata if m])} results:")

            for i, (dist, meta) in enumerate(zip(distances, metadata), 1):
                if meta:
                    title = meta.get('title', 'Unknown')
                    logger.info(f"    {i}. [distance: {dist:.3f}] {title}")
        except Exception as e:
            logger.error(f"  Search failed: {e}")

    logger.info("="*60)
    return total_added > 0


if __name__ == "__main__":
    try:
        result = asyncio.run(load_test_cases_to_rag())
        sys.exit(0 if result else 1)
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)