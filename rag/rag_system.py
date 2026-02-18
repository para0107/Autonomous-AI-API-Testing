"""
RAG System - main interface for retrieval-augmented generation.

Provides a clean facade over embeddings, vector store, knowledge base, and retriever.

FIX: Auto-initializes all components in __init__ instead of requiring
     a separate initialize() call that was never made.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union

from rag.embeddings import EmbeddingManager
from rag.vector_store import VectorStore
from rag.knowledge_base import KnowledgeBase
from rag.retriever import Retriever

logger = logging.getLogger(__name__)


class RAGSystem:
    """
    Facade for the RAG subsystem.

    Exposes:
        - generate_embeddings(data) -> np.ndarray
        - retrieve_similar_tests(embedding, k) -> List[Dict]
        - retrieve_edge_cases(embedding, k) -> List[Dict]
        - retrieve_validation_patterns(embedding, k) -> List[Dict]
        - knowledge_base (for direct access / feedback updates)
    """

    def __init__(self):
        logger.info("Initializing RAG System")

        self.embedding_manager = EmbeddingManager()

        # FIX: Auto-initialize all components instead of leaving them None
        self._vector_store = VectorStore()
        self._knowledge_base = KnowledgeBase()
        self._retriever = Retriever(self._vector_store, self._knowledge_base)
        self._initialized = True

        logger.info("RAG System fully initialized with vector store, knowledge base, and retriever")

    @property
    def knowledge_base(self):
        return self._knowledge_base

    @property
    def vector_store(self):
        return self._vector_store

    @property
    def retriever(self):
        return self._retriever

    async def generate_embeddings(self, data: Union[str, Dict, List]) -> np.ndarray:
        """Generate embeddings for input data."""
        return await self.embedding_manager.generate_embeddings(data)

    async def retrieve_similar_tests(self, query_embedding: np.ndarray,
                                     k: int = 10) -> List[Dict]:
        """Retrieve similar test cases."""
        if not self._retriever:
            logger.warning("RAG retriever not initialized, returning empty results")
            return []
        return await self._retriever.retrieve_similar_tests(query_embedding, k)

    async def retrieve_edge_cases(self, query_embedding: np.ndarray,
                                  k: int = 10) -> List[Dict]:
        """Retrieve edge case patterns."""
        if not self._retriever:
            logger.warning("RAG retriever not initialized, returning empty results")
            return []
        return await self._retriever.retrieve_edge_cases(query_embedding, k)

    async def retrieve_validation_patterns(self, query_embedding: np.ndarray,
                                           k: int = 10) -> List[Dict]:
        """Retrieve validation patterns."""
        if not self._retriever:
            logger.warning("RAG retriever not initialized, returning empty results")
            return []
        return await self._retriever.retrieve_validation_patterns(query_embedding, k)

    async def add_to_knowledge_base(self, entry: Dict[str, Any]):
        """Add an entry to the knowledge base and index it."""
        if self._knowledge_base is None:
            logger.warning("Knowledge base not initialized")
            return

        self._knowledge_base.add_entry(entry)

        # Also index in vector store for retrieval
        if self._vector_store and entry.get('searchable_text'):
            try:
                embedding = await self.generate_embeddings(entry['searchable_text'])
                # Determine index from entry type
                entry_type = entry.get('type', 'test_patterns')
                index_name = {
                    'successful_test_pattern': 'test_patterns',
                    'failed_test_pattern': 'bug_patterns',
                    'edge_case': 'edge_cases',
                    'validation': 'validation_rules',
                }.get(entry_type, 'test_patterns')

                self._vector_store.add(
                    index_name=index_name,
                    embeddings=embedding.reshape(1, -1),
                    metadata=[entry],
                )
            except Exception as e:
                logger.warning(f"Failed to index entry in vector store: {e}")