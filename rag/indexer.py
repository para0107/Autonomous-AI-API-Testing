"""
Document indexing for vector store
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

from config import rag_config, paths

logger = logging.getLogger(__name__)


class Indexer:
    """Manages indexing of documents into vector store"""

    def __init__(self, vector_store, embedding_manager, chunking_strategy):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.chunking_strategy = chunking_strategy

        self.indexed_documents = set()
        self.index_metadata_file = paths.VECTOR_STORE_DIR / "indexed_docs.json"

        self._load_indexed_documents()

    async def index_documents(self, documents: List[Dict[str, Any]],
                              index_name: str = None) -> Dict[str, Any]:
        """
        Index multiple documents

        Args:
            documents: List of documents to index
            index_name: Target index name

        Returns:
            Indexing statistics
        """
        stats = {
            'total_documents': len(documents),
            'indexed': 0,
            'skipped': 0,
            'failed': 0,
            'chunks_created': 0
        }

        for doc in documents:
            try:
                result = await self.index_document(doc, index_name)
                if result['status'] == 'indexed':
                    stats['indexed'] += 1
                    stats['chunks_created'] += result['chunks']
                elif result['status'] == 'skipped':
                    stats['skipped'] += 1
            except Exception as e:
                logger.error(f"Failed to index document: {e}")
                stats['failed'] += 1

        return stats

    async def index_document(self, document: Dict[str, Any],
                             index_name: str = None) -> Dict[str, Any]:
        """
        Index a single document

        Args:
            document: Document to index
            index_name: Target index name

        Returns:
            Indexing result
        """
        # Check if already indexed
        doc_id = document.get('id', str(hash(document.get('content', ''))))
        if doc_id in self.indexed_documents and not document.get('force_reindex'):
            return {'status': 'skipped', 'reason': 'already_indexed'}

        # Determine index
        if index_name is None:
            index_name = self._determine_index(document)

        # Extract content
        content = self._extract_content(document)
        if not content:
            return {'status': 'skipped', 'reason': 'no_content'}

        # Chunk document
        chunks = self.chunking_strategy.chunk_document(
            content,
            document.get('metadata', {}),
            strategy=document.get('chunking_strategy', 'sliding_window')
        )

        # Generate embeddings for chunks
        embeddings = []
        metadata_list = []

        for chunk in chunks:
            # Generate embedding
            embedding = await self.embedding_manager.embed_text(chunk.text)
            embeddings.append(embedding)

            # Prepare metadata
            chunk_metadata = {
                'document_id': doc_id,
                'chunk_id': chunk.chunk_id,
                'text': chunk.text,
                'start_idx': chunk.start_idx,
                'end_idx': chunk.end_idx,
                **chunk.metadata
            }
            metadata_list.append(chunk_metadata)

        # Add to vector store
        if embeddings:
            import numpy as np
            embeddings_array = np.vstack(embeddings)

            self.vector_store.add(
                index_name,
                embeddings_array,
                metadata_list
            )

            # Mark as indexed
            self.indexed_documents.add(doc_id)
            self._save_indexed_documents()

            logger.info(f"Indexed document {doc_id} with {len(chunks)} chunks")

            return {
                'status': 'indexed',
                'document_id': doc_id,
                'chunks': len(chunks),
                'index': index_name
            }

        return {'status': 'skipped', 'reason': 'no_chunks'}

    async def index_test_cases(self, test_cases: List[Dict[str, Any]]):
        """Index test cases specifically"""
        for test_case in test_cases:
            # Prepare document
            document = {
                'id': test_case.get('id', str(hash(test_case.get('name', '')))),
                'content': self._format_test_case(test_case),
                'metadata': {
                    'type': 'test_case',
                    'test_type': test_case.get('test_type'),
                    'endpoint': test_case.get('endpoint'),
                    'method': test_case.get('method'),
                    **test_case
                },
                'chunking_strategy': 'test'
            }

            await self.index_document(document, 'test_patterns')

    async def index_api_specifications(self, api_specs: List[Dict[str, Any]]):
        """Index API specifications"""
        for spec in api_specs:
            # Index each endpoint
            for path, methods in spec.get('paths', {}).items():
                for method, operation in methods.items():
                    document = {
                        'id': f"{path}_{method}",
                        'content': self._format_api_operation(path, method, operation),
                        'metadata': {
                            'type': 'api_specification',
                            'path': path,
                            'method': method,
                            'operationId': operation.get('operationId'),
                            'tags': operation.get('tags', [])
                        }
                    }

                    await self.index_document(document, 'api_specifications')

    def _determine_index(self, document: Dict[str, Any]) -> str:
        """Determine appropriate index for document"""
        doc_type = document.get('type') or document.get('metadata', {}).get('type')

        if doc_type == 'test_case':
            return 'test_patterns'
        elif doc_type == 'edge_case':
            return 'edge_cases'
        elif doc_type == 'validation':
            return 'validation_rules'
        elif doc_type == 'api_specification':
            return 'api_specifications'
        elif doc_type == 'bug':
            return 'bug_patterns'
        else:
            return 'test_patterns'  # Default

    def _extract_content(self, document: Dict[str, Any]) -> str:
        """Extract textual content from document"""
        if 'content' in document:
            return document['content']
        elif 'text' in document:
            return document['text']
        elif 'code' in document:
            return document['code']
        else:
            # Try to construct from other fields
            parts = []
            for key in ['name', 'description', 'summary', 'body']:
                if key in document:
                    parts.append(f"{key}: {document[key]}")
            return '\n'.join(parts)

    def _format_test_case(self, test_case: Dict[str, Any]) -> str:
        """Format test case for indexing"""
        parts = [
            f"Test: {test_case.get('name', 'Unnamed')}",
            f"Type: {test_case.get('test_type', 'unknown')}",
            f"Endpoint: {test_case.get('endpoint', '')}",
            f"Method: {test_case.get('method', '')}",
        ]

        if 'description' in test_case:
            parts.append(f"Description: {test_case['description']}")

        if 'steps' in test_case:
            parts.append("Steps:")
            for i, step in enumerate(test_case['steps'], 1):
                parts.append(f"  {i}. {step}")

        if 'assertions' in test_case:
            parts.append("Assertions:")
            for assertion in test_case['assertions']:
                parts.append(f"  - {assertion}")

        if 'test_data' in test_case:
            parts.append(f"Test Data: {json.dumps(test_case['test_data'])}")

        return '\n'.join(parts)

    def _format_api_operation(self, path: str, method: str,
                              operation: Dict[str, Any]) -> str:
        """Format API operation for indexing"""
        parts = [
            f"Endpoint: {method.upper()} {path}",
            f"Summary: {operation.get('summary', '')}",
            f"Description: {operation.get('description', '')}",
        ]

        if 'parameters' in operation:
            parts.append("Parameters:")
            for param in operation['parameters']:
                param_str = f"  - {param.get('name')}: {param.get('type', 'string')} " \
                            f"({'required' if param.get('required') else 'optional'})"
                parts.append(param_str)

        if 'requestBody' in operation:
            parts.append("Request Body: " + str(operation['requestBody']))

        if 'responses' in operation:
            parts.append("Responses:")
            for code, response in operation['responses'].items():
                parts.append(f"  {code}: {response.get('description', '')}")

        return '\n'.join(parts)

    def _load_indexed_documents(self):
        """Load list of indexed documents"""
        if self.index_metadata_file.exists():
            try:
                with open(self.index_metadata_file, 'r') as f:
                    data = json.load(f)
                    self.indexed_documents = set(data.get('indexed_documents', []))
            except Exception as e:
                logger.warning(f"Failed to load indexed documents: {e}")

    def _save_indexed_documents(self):
        """Save list of indexed documents"""
        try:
            with open(self.index_metadata_file, 'w') as f:
                json.dump({
                    'indexed_documents': list(self.indexed_documents)
                }, f)
        except Exception as e:
            logger.warning(f"Failed to save indexed documents: {e}")

    async def update_index(self, document: Dict[str, Any], index_name: str):
        """Update an existing document in the index"""
        doc_id = document.get('id')
        if doc_id:
            # Force reindex
            document['force_reindex'] = True
            return await self.index_document(document, index_name)

    def clear_index(self, index_name: str):
        """Clear an index and its metadata"""
        self.vector_store.clear_index(index_name)

        # Clear from indexed documents
        # Would need to track which docs are in which index
        logger.info(f"Cleared index {index_name}")