"""
RAG Retriever - Standardized version

Fix: All retrieval methods now return a consistent format:
    List[Dict] where each Dict has:
        - id: str
        - score: float (similarity score, 0.0 to 1.0)
        - metadata: Dict (the actual content/data)
        - rank: int (1-based position)
        - source: str (which retrieval method produced this)
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


# Canonical result type
class RetrievalResult:
    """Standard result from any retrieval operation."""

    __slots__ = ('id', 'score', 'metadata', 'rank', 'source')

    def __init__(self, id: str, score: float, metadata: Dict, rank: int, source: str = ''):
        self.id = id
        self.score = score
        self.metadata = metadata
        self.rank = rank
        self.source = source

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'score': self.score,
            'metadata': self.metadata,
            'rank': self.rank,
            'source': self.source,
        }


class Retriever:
    """Retrieves relevant test cases and patterns from the vector store."""

    def __init__(self, vector_store, knowledge_base=None):
        self.vector_store = vector_store
        self.knowledge_base = knowledge_base
        logger.info("Retriever initialized")

    async def retrieve(self, query_embedding: np.ndarray,
                       k: int = 10,
                       filter_dict: Optional[Dict] = None,
                       source_tag: str = 'general') -> List[Dict]:
        """
        Retrieve top-k similar items from the vector store.

        Args:
            query_embedding: Query vector
            k: Number of results
            filter_dict: Optional metadata filters
            source_tag: Tag identifying the retrieval source

        Returns:
            List[Dict] with standardized keys: id, score, metadata, rank, source
        """
        try:
            raw_results = self.vector_store.search(query_embedding, k=k)
            return self._standardize_results(raw_results, source_tag)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []

    async def retrieve_similar_tests(self, query_embedding: np.ndarray,
                                     k: int = 10) -> List[Dict]:
        """Retrieve similar test cases."""
        return await self.retrieve(
            query_embedding, k=k, source_tag='similar_tests'
        )

    async def retrieve_edge_cases(self, query_embedding: np.ndarray,
                                  k: int = 10) -> List[Dict]:
        """Retrieve edge case patterns."""
        return await self.retrieve(
            query_embedding, k=k,
            filter_dict={'type': 'edge_case'},
            source_tag='edge_cases'
        )

    async def retrieve_validation_patterns(self, query_embedding: np.ndarray,
                                           k: int = 10) -> List[Dict]:
        """Retrieve validation patterns."""
        return await self.retrieve(
            query_embedding, k=k,
            filter_dict={'type': 'validation'},
            source_tag='validation_patterns'
        )

    def _standardize_results(self, raw_results: Any,
                             source_tag: str) -> List[Dict]:
        """
        Convert any result format from VectorStore into standard List[Dict].

        Handles these known formats:
            - List[Dict] with id, score, metadata
            - List[Tuple] of (id, score, metadata)
            - List[Tuple] of (id, score)
            - Single dict with 'results' key
            - numpy arrays with metadata
        """
        if raw_results is None:
            return []

        # If it's a dict with a 'results' key, unwrap
        if isinstance(raw_results, dict):
            if 'results' in raw_results:
                raw_results = raw_results['results']
            else:
                raw_results = [raw_results]

        standardized = []

        for rank, item in enumerate(raw_results, 1):
            try:
                result = self._parse_single_result(item, rank, source_tag)
                if result:
                    standardized.append(result)
            except Exception as e:
                logger.warning(f"Failed to parse result at rank {rank}: {e}")
                continue

        return standardized

    def _parse_single_result(self, item: Any, rank: int,
                             source_tag: str) -> Optional[Dict]:
        """Parse a single result item into standard format."""

        # Already a dict with expected keys
        if isinstance(item, dict):
            return {
                'id': str(item.get('id', f'result_{rank}')),
                'score': float(item.get('score', item.get('similarity', 0.0))),
                'metadata': item.get('metadata', item.get('data', item)),
                'rank': rank,
                'source': source_tag,
            }

        # Tuple: (id, score, metadata) or (id, score)
        if isinstance(item, (tuple, list)):
            if len(item) >= 3:
                return {
                    'id': str(item[0]),
                    'score': float(item[1]),
                    'metadata': item[2] if isinstance(item[2], dict) else {'data': item[2]},
                    'rank': rank,
                    'source': source_tag,
                }
            elif len(item) == 2:
                return {
                    'id': str(item[0]),
                    'score': float(item[1]) if not isinstance(item[1], str) else 0.0,
                    'metadata': {},
                    'rank': rank,
                    'source': source_tag,
                }

        # Fallback: wrap as metadata
        return {
            'id': f'result_{rank}',
            'score': 0.0,
            'metadata': {'data': str(item)},
            'rank': rank,
            'source': source_tag,
        }