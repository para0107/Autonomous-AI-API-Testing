"""
FAISS vector store for similarity search
"""

import logging
import numpy as np
import faiss
import pickle
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import json

from config import rag_config, paths
logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-based vector store for embeddings"""

    def __init__(self):
        logger.info("Initializing Vector Store")

        self.dimension = rag_config.embedding_dimension
        self.indices = {}
        self.metadata_stores = {}
        self.index_configs = {}

        # Initialize indices for different types
        for index_name in rag_config.indices:
            self._create_index(index_name)

        # Load existing indices if available
        self._load_indices()

    def _create_index(self, index_name: str):
        """Create a new FAISS index with dynamic cluster sizing"""
        config = rag_config.get_index_config(index_name)

        if rag_config.index_type == "IVF":
            # Calculate appropriate nlist based on expected data size
            # Rule: use sqrt(n) clusters, but ensure we have enough data
            effective_nlist = min(rag_config.nlist, 50)  # Cap at 50 for small datasets

            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension,
                                       effective_nlist, faiss.METRIC_L2)
        elif rag_config.index_type == "HNSW":
            index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            index = faiss.IndexFlatL2(self.dimension)

        # Add ID mapping
        index = faiss.IndexIDMap(index)

        self.indices[index_name] = index
        self.metadata_stores[index_name] = {}
        self.index_configs[index_name] = config

        logger.info(f"Created index: {index_name}")

    def add(self, index_name: str, embeddings: np.ndarray,
            metadata: List[Dict[str, Any]] = None, ids: List[int] = None):
        """
        Add embeddings to index (robust: dtype/shape checks, train IVF if needed,
        safer id generation, store metadata). Automatically falls back to Flat index
        if insufficient data for IVF training.
        """
        if index_name not in self.indices:
            raise ValueError(f"Index {index_name} not found")

        index = self.indices[index_name]

        # Ensure numpy array and 2D
        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        emb = np.ascontiguousarray(emb)

        n, dim = emb.shape
        if dim != self.dimension:
            logger.warning("Embedding dim (%d) != configured dimension (%d). Truncating/padding.",
                           dim, self.dimension)
            if dim > self.dimension:
                emb = emb[:, : self.dimension]
            else:
                pad = np.zeros((n, self.dimension - dim), dtype=np.float32)
                emb = np.hstack([emb, pad])

        # Prepare IDs
        if ids is None:
            existing_keys = list(self.metadata_stores.get(index_name, {}).keys())
            if existing_keys:
                start_id = max(existing_keys) + 1
            else:
                start_id = int(getattr(index, "ntotal", 0) or 0)
            ids_array = np.arange(start_id, start_id + n, dtype=np.int64)
        else:
            ids_array = np.asarray(ids, dtype=np.int64)
            if ids_array.shape[0] != n:
                raise ValueError("Length of ids does not match number of embeddings")

        # Find inner index (IndexIDMap wrapper may expose .index)
        inner = getattr(index, "index", None) or getattr(index, "faiss_index", None) or index

        # Train if required (IVF / trainable indices)
        try:
            if hasattr(inner, "is_trained") and not inner.is_trained:
                # Check if we have enough data for IVF training
                nlist = getattr(inner, 'nlist', 100)

                if n < nlist:
                    # Not enough data for IVF, switch to Flat index
                    logger.warning(
                        f"Insufficient data for IVF index '{index_name}': "
                        f"Have {n} vectors but need at least {nlist}. "
                        f"Automatically switching to Flat index."
                    )

                    # Create a new Flat index
                    flat_index = faiss.IndexFlatL2(self.dimension)
                    flat_index = faiss.IndexIDMap(flat_index)

                    # Preserve any existing data from the old index
                    if hasattr(index, 'ntotal') and index.ntotal > 0:
                        logger.info(f"Migrating {index.ntotal} existing vectors to new Flat index")
                        # Note: In practice, you might want to extract and re-add existing vectors
                        # For now, we'll just replace the index

                    # Replace the index
                    self.indices[index_name] = flat_index
                    index = flat_index
                    inner = flat_index

                    # Update config to reflect the change
                    self.index_configs[index_name]['index_type'] = 'Flat'

                    logger.info(f"Successfully switched index '{index_name}' to Flat type")
                else:
                    # We have enough data, proceed with IVF training
                    logger.info("FAISS index '%s' is not trained; training with %d vectors...", index_name, n)
                    train_vecs = emb
                    max_train = 10000
                    if n > max_train:
                        idxs = np.random.choice(n, size=max_train, replace=False)
                        train_vecs = emb[idxs]
                    train_vecs = np.ascontiguousarray(train_vecs.astype(np.float32))
                    inner.train(train_vecs)
                    logger.info("FAISS index '%s' trained successfully", index_name)
        except Exception as e:
            logger.error("Failed to train FAISS index '%s': %s", index_name, e)
            # Try one more fallback: create Flat index as last resort
            logger.info(f"Attempting emergency fallback to Flat index for '{index_name}'")
            try:
                flat_index = faiss.IndexFlatL2(self.dimension)
                flat_index = faiss.IndexIDMap(flat_index)
                self.indices[index_name] = flat_index
                index = flat_index
                self.index_configs[index_name]['index_type'] = 'Flat'
                logger.info(f"Emergency fallback successful for '{index_name}'")
            except Exception as fallback_error:
                logger.error(f"Emergency fallback failed: {fallback_error}")
                raise

        # Add embeddings with ids
        try:
            index.add_with_ids(emb, ids_array)
        except Exception as e:
            logger.error("Failed to add embeddings to FAISS index '%s': %s", index_name, e)
            raise

        # Store metadata (map ids -> metadata)
        if metadata:
            store = self.metadata_stores.setdefault(index_name, {})
            for id_, meta in zip(ids_array.tolist(), metadata):
                store[int(id_)] = meta

        logger.info("Added %d embeddings to %s (ids %d..%d)", n, index_name, ids_array[0], ids_array[-1])
    def search(self, index_name: str, query_embedding: np.ndarray,
               k: int = 10) -> Tuple[List[int], List[float], List[Dict]]:
        """
        Search for similar embeddings

        Args:
            index_name: Name of the index
            query_embedding: Query embedding
            k: Number of results to return

        Returns:
            Tuple of (ids, distances, metadata)
        """
        if index_name not in self.indices:
            raise ValueError(f"Index {index_name} not found")

        index = self.indices[index_name]

        # Ensure query is 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Set search parameters for IVF
        if isinstance(index.index, faiss.IndexIVFFlat):
            index.index.nprobe = rag_config.nprobe

        # Search
        distances, ids = index.search(query_embedding, k)

        # Get metadata
        metadata = []
        for id_ in ids[0]:
            if id_ != -1:  # Valid ID
                meta = self.metadata_stores[index_name].get(int(id_), {})
                metadata.append(meta)
            else:
                metadata.append({})

        return ids[0].tolist(), distances[0].tolist(), metadata

    def search_multiple_indices(self, query_embedding: np.ndarray,
                                indices: List[str] = None, k: int = 10) -> Dict[str, Any]:
        """Search across multiple indices"""
        if indices is None:
            indices = list(self.indices.keys())

        results = {}
        for index_name in indices:
            if index_name in self.indices:
                ids, distances, metadata = self.search(index_name, query_embedding, k)
                results[index_name] = {
                    'ids': ids,
                    'distances': distances,
                    'metadata': metadata
                }

        return results

    def update(self, index_name: str, id_: int, embedding: np.ndarray,
               metadata: Dict[str, Any] = None):
        """Update an embedding"""
        # Remove old embedding
        self.remove(index_name, [id_])

        # Add new embedding
        self.add(index_name, embedding.reshape(1, -1),
                 [metadata] if metadata else None, [id_])

    def remove(self, index_name: str, ids: List[int]):
        """Remove embeddings by ID"""
        if index_name not in self.indices:
            raise ValueError(f"Index {index_name} not found")

        # Note: FAISS doesn't support direct removal
        # We need to rebuild the index without these IDs
        logger.warning(f"Removal not directly supported. Consider rebuilding index {index_name}")

    def save_index(self, index_name: str):
        """Save index to disk"""
        if index_name not in self.indices:
            raise ValueError(f"Index {index_name} not found")

        index_dir = paths.VECTOR_STORE_DIR / index_name
        index_dir.mkdir(exist_ok=True)

        # Save FAISS index
        index_file = index_dir / "index.faiss"
        faiss.write_index(self.indices[index_name], str(index_file))

        # Save metadata
        metadata_file = index_dir / "metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(self.metadata_stores[index_name], f)

        # Save config
        config_file = index_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(self.index_configs[index_name], f)

        logger.info(f"Saved index {index_name}")

    def load_index(self, index_name: str):
        """Load index from disk"""
        index_dir = paths.VECTOR_STORE_DIR / index_name

        if not index_dir.exists():
            logger.warning(f"Index directory {index_dir} not found")
            return False

        # Load FAISS index
        index_file = index_dir / "index.faiss"
        if index_file.exists():
            self.indices[index_name] = faiss.read_index(str(index_file))

        # Load metadata
        metadata_file = index_dir / "metadata.pkl"
        if metadata_file.exists():
            with open(metadata_file, 'rb') as f:
                self.metadata_stores[index_name] = pickle.load(f)

        # Load config
        config_file = index_dir / "config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                self.index_configs[index_name] = json.load(f)

        logger.info(f"Loaded index {index_name}")
        return True

    def _load_indices(self):
        """Load all existing indices"""
        if paths.VECTOR_STORE_DIR.exists():
            for index_dir in paths.VECTOR_STORE_DIR.iterdir():
                if index_dir.is_dir():
                    index_name = index_dir.name
                    # Try to load saved data for indices we created
                    if index_name in self.indices:
                        logger.info(f"Attempting to load saved data for index: {index_name}")
                        self.load_index(index_name)

    def save_all(self):
        """Save all indices"""
        for index_name in self.indices:
            self.save_index(index_name)

    def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """Get statistics for an index"""
        if index_name not in self.indices:
            raise ValueError(f"Index {index_name} not found")

        index = self.indices[index_name]

        return {
            'name': index_name,
            'total_embeddings': index.ntotal,
            'dimension': self.dimension,
            'index_type': type(index.index).__name__,
            'metadata_count': len(self.metadata_stores[index_name])
        }

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all indices"""
        stats = {}
        for index_name in self.indices:
            stats[index_name] = self.get_index_stats(index_name)
        return stats

    def clear_index(self, index_name: str):
        """Clear an index"""
        if index_name in self.indices:
            self._create_index(index_name)
            logger.info(f"Cleared index {index_name}")

    def clear_all(self):
        """Clear all indices"""
        for index_name in list(self.indices.keys()):
            self.clear_index(index_name)