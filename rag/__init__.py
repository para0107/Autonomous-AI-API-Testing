"""
Retrieval-Augmented Generation (RAG) system for test knowledge
"""

from .embeddings import EmbeddingManager
from .vector_store import VectorStore
from .retriever import Retriever
from .chunking import ChunkingStrategy
from .indexer import Indexer
from .knowledge_base import KnowledgeBase


class RAGSystem:
    """Main RAG system coordinating all components"""

    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStore()
        self.retriever = Retriever(self.vector_store, self.embedding_manager)
        self.chunking_strategy = ChunkingStrategy()
        self.indexer = Indexer(self.vector_store, self.embedding_manager, self.chunking_strategy)
        self.knowledge_base = KnowledgeBase()

    async def generate_embeddings(self, data):
        """Generate embeddings for data"""
        return await self.embedding_manager.generate_embeddings(data)

    async def retrieve_similar_tests(self, embeddings, k=10):
        """Retrieve similar test cases"""
        return await self.retriever.retrieve_similar_tests(embeddings, k)

    async def retrieve_edge_cases(self, embeddings, k=5):
        """Retrieve relevant edge cases"""
        return await self.retriever.retrieve_edge_cases(embeddings, k)

    async def retrieve_validation_patterns(self, embeddings, k=5):
        """Retrieve validation patterns"""
        return await self.retriever.retrieve_validation_patterns(embeddings, k)

    async def index_knowledge(self, documents):
        """Index new knowledge into vector store"""
        return await self.indexer.index_documents(documents)


__all__ = [
    'RAGSystem',
    'EmbeddingManager',
    'VectorStore',
    'Retriever',
    'ChunkingStrategy',
    'Indexer',
    'KnowledgeBase'
]