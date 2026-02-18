"""
Retrieval logic for RAG system
"""

import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from sentence_transformers import CrossEncoder
import heapq

from config import rag_config

logger = logging.getLogger(__name__)


class Retriever:
    """Handles retrieval of relevant documents from vector store"""

    def __init__(self, vector_store, embedding_manager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

        # Load reranking model if enabled - token comes from rag_config
        self.reranker = None
        if rag_config.rerank:
            try:
                logger.info(f"Loading reranking model: {rag_config.rerank_model}")

                # Use token from config
                if rag_config.hf_token:
                    logger.info("Using HuggingFace token from config")
                    # Try both parameter names for compatibility
                    try:
                        self.reranker = CrossEncoder(
                            rag_config.rerank_model,
                            token=rag_config.hf_token  # Newer versions use 'token'
                        )
                    except TypeError:
                        # Fallback for older versions
                        self.reranker = CrossEncoder(
                            rag_config.rerank_model,
                            use_auth_token=rag_config.hf_token
                        )
                else:
                    logger.warning("No HuggingFace token found in config, attempting anonymous access")
                    logger.warning("Add HG_TOKEN to your .env file for authenticated model access")
                    self.reranker = CrossEncoder(rag_config.rerank_model)

                logger.info(f"✅ Successfully loaded reranking model: {rag_config.rerank_model}")

            except Exception as e:
                logger.error(f"❌ Failed to load reranking model: {e}")
                logger.warning("⚠️  Continuing without reranking - results may be less accurate")
                logger.warning("To fix: Add HG_TOKEN to your .env file or set rerank=False in config")
                self.reranker = None
        else:
            logger.info("Reranking disabled in config")

    async def retrieve(self, query: Union[str, np.ndarray],
                       index_name: str, k: int = 10,
                       rerank: bool = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents

        Args:
            query: Query text or embedding
            index_name: Index to search
            k: Number of results
            rerank: Whether to rerank results

        Returns:
            List of retrieved documents
        """
        # Generate embedding if query is text
        if isinstance(query, str):
            query_embedding = await self.embedding_manager.embed_text(query)
        else:
            query_embedding = query

        # Search vector store
        fetch_k = k * 2 if (rerank and self.reranker) else k
        ids, distances, metadata = self.vector_store.search(
            index_name, query_embedding, fetch_k
        )

        # Prepare results
        results = []
        for i, (id_, dist, meta) in enumerate(zip(ids, distances, metadata)):
            if id_ != -1:  # Valid result
                result = {
                    'id': id_,
                    'score': 1 / (1 + dist),  # Convert distance to similarity score
                    'metadata': meta,
                    'rank': i
                }
                results.append(result)

        # Rerank if enabled AND reranker is available
        should_rerank = (rerank if rerank is not None else rag_config.rerank)
        if should_rerank and self.reranker and isinstance(query, str):
            try:
                results = self._rerank_results(query, results)
                logger.debug(f"Reranked {len(results)} results")
            except Exception as e:
                logger.warning(f"Reranking failed: {e}, using original ranking")

        return results[:k]

    async def retrieve_similar_tests(self, query_embedding: np.ndarray,
                                     k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve similar test cases"""
        results = await self.retrieve(
            query_embedding,
            'test_patterns',
            k
        )

        # Enhance with test-specific information
        for result in results:
            if 'test_code' in result.get('metadata', {}):
                result['test_type'] = self._classify_test_type(result['metadata']['test_code'])

        return results

    async def retrieve_edge_cases(self, query_embedding: np.ndarray,
                                  k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant edge cases"""
        results = await self.retrieve(
            query_embedding,
            'edge_cases',
            k
        )

        # Filter by relevance threshold
        threshold = rag_config.similarity_threshold
        filtered = [r for r in results if r['score'] >= threshold]

        return filtered

    async def retrieve_validation_patterns(self, query_embedding: np.ndarray,
                                           k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve validation patterns"""
        return await self.retrieve(
            query_embedding,
            'validation_rules',
            k
        )

    async def hybrid_search(self, query: str, indices: List[str] = None,
                            k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform hybrid search across multiple indices

        Args:
            query: Query text
            indices: Indices to search (None for all)
            k: Number of results per index

        Returns:
            Combined and ranked results
        """
        # Generate embedding
        query_embedding = await self.embedding_manager.embed_text(query)

        # Search across indices
        all_results = self.vector_store.search_multiple_indices(
            query_embedding, indices, k
        )

        # Combine and rank results
        combined_results = []
        for index_name, index_results in all_results.items():
            for i, (id_, dist, meta) in enumerate(zip(
                    index_results['ids'],
                    index_results['distances'],
                    index_results['metadata']
            )):
                if id_ != -1:
                    result = {
                        'id': id_,
                        'index': index_name,
                        'score': 1 / (1 + dist),
                        'metadata': meta,
                        'rank': i
                    }
                    combined_results.append(result)

        # Sort by score
        combined_results.sort(key=lambda x: x['score'], reverse=True)

        # Rerank if enabled
        if rag_config.rerank and self.reranker:
            try:
                combined_results = self._rerank_results(query, combined_results)
            except Exception as e:
                logger.warning(f"Reranking failed in hybrid search: {e}")

        # Apply MMR for diversity
        if len(combined_results) > k:
            combined_results = self._apply_mmr(query_embedding, combined_results, k)

        return combined_results[:k]

    def _rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank results using cross-encoder"""
        if not results:
            return results

        # Prepare pairs for reranking
        pairs = []
        for result in results:
            # Extract text from metadata
            text = self._extract_text_from_metadata(result['metadata'])
            pairs.append([query, text])

        # Get reranking scores
        scores = self.reranker.predict(pairs)

        # Update scores
        for result, score in zip(results, scores):
            result['rerank_score'] = float(score)
            result['original_score'] = result['score']
            result['score'] = float(score)  # Replace with rerank score

        # Sort by new scores
        results.sort(key=lambda x: x['score'], reverse=True)

        return results

    def _extract_text_from_metadata(self, metadata: Dict[str, Any]) -> str:
        """Extract searchable text from metadata"""
        text_fields = ['content', 'text', 'description', 'name', 'test_code', 'code']

        texts = []
        for field in text_fields:
            if field in metadata and metadata[field]:
                texts.append(str(metadata[field]))

        return ' '.join(texts) if texts else str(metadata)

    def _classify_test_type(self, test_code: str) -> str:
        """Classify test type from code"""
        test_code_lower = test_code.lower()

        if 'validation' in test_code_lower or 'invalid' in test_code_lower:
            return 'validation'
        elif 'edge' in test_code_lower or 'boundary' in test_code_lower:
            return 'edge_case'
        elif 'auth' in test_code_lower or 'unauthorized' in test_code_lower:
            return 'authentication'
        elif 'security' in test_code_lower:
            return 'security'
        else:
            return 'functional'

    def _apply_mmr(self, query_embedding: np.ndarray,
                   results: List[Dict[str, Any]], k: int,
                   lambda_param: float = 0.5) -> List[Dict[str, Any]]:
        """
        Apply Maximal Marginal Relevance for diversity

        Args:
            query_embedding: Query embedding
            results: Candidate results
            k: Number of results to select
            lambda_param: Trade-off between relevance and diversity (0-1)

        Returns:
            Diversified results
        """
        if len(results) <= k:
            return results

        selected = []
        remaining = results.copy()

        # Select first result (highest score)
        selected.append(remaining.pop(0))

        while len(selected) < k and remaining:
            mmr_scores = []

            for candidate in remaining:
                # Relevance score
                relevance = candidate['score']

                # Diversity score (simplified)
                diversity = 1.0

                # MMR score
                mmr = lambda_param * relevance - (1 - lambda_param) * (1 - diversity)
                mmr_scores.append(mmr)

            # Select best MMR score
            best_idx = np.argmax(mmr_scores)
            selected.append(remaining.pop(best_idx))

        return selected