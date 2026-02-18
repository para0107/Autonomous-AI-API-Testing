"""
RAG Retriever - Fixed version

Fixes:
- All retrieval methods pass index_name to vector_store.search()
- Consistent return format: List[Dict] with id, score, metadata, rank, source
- Handles VectorStore returning (ids, distances, metadata) tuple
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieves relevant test cases and patterns from the vector store."""

    def __init__(self, vector_store, knowledge_base=None):
        self.vector_store = vector_store
        self.knowledge_base = knowledge_base
        logger.info("Retriever initialized")

    async def retrieve_similar_tests(self, query_embedding: np.ndarray,
                                     k: int = 10) -> List[Dict]:
        """Retrieve similar test cases from test_patterns index."""
        return await self._search_index('test_patterns', query_embedding, k, 'similar_tests')

    async def retrieve_edge_cases(self, query_embedding: np.ndarray,
                                  k: int = 10) -> List[Dict]:
        """Retrieve edge case patterns from edge_cases index."""
        return await self._search_index('edge_cases', query_embedding, k, 'edge_cases')

    async def retrieve_validation_patterns(self, query_embedding: np.ndarray,
                                           k: int = 10) -> List[Dict]:
        """Retrieve validation patterns from validation_rules index."""
        return await self._search_index('validation_rules', query_embedding, k, 'validation_patterns')

    async def _search_index(self, index_name: str, query_embedding: np.ndarray,
                            k: int, source_tag: str) -> List[Dict]:
        """
        Search a specific vector store index and return standardized results.

        Args:
            index_name: Which FAISS index to search
            query_embedding: Query vector
            k: Number of results
            source_tag: Tag for identifying result source

        Returns:
            List[Dict] with keys: id, score, metadata, rank, source
        """
        try:
            # FIX: VectorStore.search() requires (index_name, query_embedding, k)
            ids, distances, metadata = self.vector_store.search(
                index_name, query_embedding, k
            )

            results = []
            for rank, (id_, dist, meta) in enumerate(zip(ids, distances, metadata), 1):
                if id_ == -1 or not meta:
                    continue

                # Convert L2 distance to similarity score (0-1 range)
                # L2 distance: 0 = identical, higher = more different
                score = max(0.0, 1.0 / (1.0 + dist))

                results.append({
                    'id': str(id_),
                    'score': score,
                    'metadata': meta,
                    'rank': rank,
                    'source': source_tag,
                })

            return results

        except Exception as e:
            logger.warning(f"Search failed on index '{index_name}': {e}")
            return []

    async def retrieve(self, index_name: str, query_embedding: np.ndarray,
                       k: int = 10, source_tag: str = 'general') -> List[Dict]:
        """
        Generic retrieval from any index.

        Args:
            index_name: Which index to search
            query_embedding: Query vector
            k: Number of results
            source_tag: Tag identifying the retrieval source

        Returns:
            List[Dict] with standardized keys
        """
        return await self._search_index(index_name, query_embedding, k, source_tag)