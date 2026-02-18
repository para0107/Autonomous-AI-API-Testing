"""
RAG (Retrieval-Augmented Generation) configuration
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class RAGConfig:
    """Configuration for RAG system with FAISS"""

    # HuggingFace Token - Load from environment
    hf_token: str = field(default_factory=lambda:
        os.getenv("HF_TOKEN") or
        None
    )

    # Embedding models
    text_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    code_embedding_model: str = "microsoft/codebert-base"
    embedding_dimension: int = 384

    # FAISS settings
    index_type: str = "IVF"  # IVF for accuracy
    nlist: int = 10  # Number of clusters
    nprobe: int = 10  # Number of clusters to search
    metric: str = "cosine"  # Similarity metric

    # Chunking settings
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_chunks_per_document: int = 100

    # Retrieval settings
    top_k: int = 10  # Number of results to retrieve
    similarity_threshold: float = 0.3
    rerank: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"

    # Knowledge base structure
    knowledge_levels: Dict[str, str] = field(default_factory=lambda: {
        'global': 'General API patterns and REST principles',
        'domain': 'Business domain specific knowledge',
        'service': 'Service-specific patterns and rules',
        'endpoint': 'Endpoint-specific test cases',
        'edge_cases': 'Edge cases and bug patterns'
    })

    # Vector store indices
    indices: List[str] = field(default_factory=lambda: [
        'test_patterns',
        'edge_cases',
        'validation_rules',
        'api_specifications',
        'bug_patterns',
        'successful_tests'
    ])

    # Caching
    enable_cache: bool = True
    cache_ttl: int = 3600  # Cache time-to-live in seconds
    max_cache_size: int = 1000  # Maximum cached queries

    # Update settings
    incremental_indexing: bool = True
    batch_size: int = 32
    update_frequency: str = "after_execution"  # "realtime", "batch", "after_execution"

    def get_index_config(self, index_name: str) -> Dict[str, Any]:
        """Get configuration for specific index"""
        return {
            'dimension': self.embedding_dimension,
            'index_type': self.index_type,
            'nlist': self.nlist,
            'nprobe': self.nprobe,
            'metric': self.metric
        }

    def get_retrieval_config(self) -> Dict[str, Any]:
        """Get retrieval configuration"""
        return {
            'top_k': self.top_k,
            'threshold': self.similarity_threshold,
            'rerank': self.rerank,
            'rerank_model': self.rerank_model if self.rerank else None,
            'hf_token': self.hf_token  # Include token in retrieval config
        }


# Global instance - loads token from environment automatically
rag_config = RAGConfig()