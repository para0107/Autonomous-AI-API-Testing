"""
Load QA-Backend_Data.json into RAG system
"""

import asyncio
import json
import logging
import sys
import numpy as np
from pathlib import Path

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


async def load_knowledge_base():
    """Load knowledge base from JSON file into RAG"""

    kb_path = Path("data/knowledge_base/QA-Backend_Data.json")

    if not kb_path.exists():
        logger.error(f"Knowledge base not found: {kb_path}")
        logger.info("Please ensure QA-Backend_Data.json exists in data/knowledge_base/")
        return False

    logger.info(f"Loading knowledge base from {kb_path}")

    with open(kb_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"Found {len(data)} items in knowledge base")

    logger.info("Initializing RAG components...")
    em = EmbeddingManager()
    vs = VectorStore()

    # Batch items by index so we call vs.add() once per index
    batches = {
        'test_patterns': {'embeddings': [], 'metadata': []},
        'edge_cases': {'embeddings': [], 'metadata': []},
        'validation_rules': {'embeddings': [], 'metadata': []},
    }

    success_count = 0
    error_count = 0

    for idx, item in enumerate(data):
        try:
            if 'edge_case' in item:
                index_name = 'edge_cases'
            elif 'validation' in item or 'rule' in item:
                index_name = 'validation_rules'
            else:
                index_name = 'test_patterns'

            text_parts = []
            for key, value in item.items():
                if isinstance(value, str) and len(value) < 500:
                    text_parts.append(f"{key}: {value}")
                elif isinstance(value, dict):
                    text_parts.append(f"{key}: {json.dumps(value)}")
            search_text = " ".join(text_parts)

            # FIXED: embed_text() is the correct method name
            embedding = await em.embed_text(search_text)

            batches[index_name]['embeddings'].append(embedding)
            batches[index_name]['metadata'].append(item)

            success_count += 1

            if (idx + 1) % 50 == 0:
                logger.info(f"  Processed {idx + 1}/{len(data)} items...")

        except Exception as e:
            error_count += 1
            logger.error(f"  Error processing item {idx}: {e}")
            continue

    # FIXED: vs.add() takes (index_name, embeddings_ndarray, metadata_list)
    # The old code called vs.add_item() one item at a time which doesn't exist
    for index_name, batch in batches.items():
        if not batch['embeddings']:
            logger.info(f"  No items for index '{index_name}', skipping")
            continue
        embeddings_array = np.vstack(batch['embeddings'])
        vs.add(index_name, embeddings_array, batch['metadata'])
        logger.info(f"  Added {len(batch['metadata'])} items to '{index_name}'")

    # FIXED: save_all() is the correct method (not save())
    logger.info("Saving vector store...")
    vs.save_all()

    logger.info(f"Successfully loaded {success_count} items")
    if error_count > 0:
        logger.warning(f"{error_count} items failed")

    # Verify — vs.search() is synchronous, returns (ids, distances, metadata)
    logger.info("Verifying RAG retrieval...")
    test_query = "GET endpoint test"
    test_embedding = await em.embed_text(test_query)

    # FIXED: no await, unpack as tuple
    ids, distances, metadata = vs.search('test_patterns', test_embedding, k=3)

    valid_results = [(d, m) for d, m in zip(distances, metadata) if m]
    logger.info(f"Test search returned {len(valid_results)} results")
    for i, (dist, item) in enumerate(valid_results[:3], 1):
        score = 1 / (1 + dist)
        logger.info(f"  {i}. Score: {score:.3f}")

    return success_count > 0


if __name__ == "__main__":
    try:
        result = asyncio.run(load_knowledge_base())
        sys.exit(0 if result else 1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)