"""
Embedding generation and management
"""

import logging
import os

import numpy as np
from typing import Dict, List, Any, Union
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel
import hashlib
import pickle
from pathlib import Path

from config import rag_config, paths


logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

class EmbeddingManager:
    """
    Embedding generation and management.

    Async methods are provided so the class can be awaited from an async pipeline.
    Synchronous model calls are executed with asyncio.to_thread to avoid blocking.
    """

    def __init__(self, hf_token: str = None):
        import asyncio
        import os
        import hashlib
        from pathlib import Path

        logger.info("Initializing Embedding Manager")

        # Resolve token: explicit arg -> standard env var -> project-specific HG_TOKEN -> HF_TOKEN -> None
        self.hf_token = hf_token or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HG_TOKEN") or os.getenv("HF_TOKEN")
        if self.hf_token:
            logger.info("Using Hugging Face token from environment/argument")
        else:
            logger.info("No Hugging Face token found; attempting anonymous access")

        # canonical default id; can be overridden by rag_config
        text_model_id = getattr(rag_config, "text_embedding_model", None) or "sentence-transformers/all-MiniLM-L6-v2"

        # try loading SentenceTransformer (may raise)
        try:
            # note: sentence-transformers historically accepts `use_auth_token`; newer versions accept `token`.
            # keep `use_auth_token` for compatibility; if you upgrade, replace with `token=self.hf_token`.
            self.text_model = SentenceTransformer(text_model_id, use_auth_token=self.hf_token)
            try:
                self.text_dim = int(self.text_model.get_sentence_embedding_dimension())
            except Exception:
                self.text_dim = int(getattr(rag_config, "embedding_dimension", 384))
            logger.info("Loaded text embedding model: %s (dim=%s)", text_model_id, self.text_dim)
        except Exception as e:
            logger.warning(
                "Failed to load text embedding model '%s': %s\n"
                " - Confirm the model id is correct (case-sensitive)\n"
                " - Ensure your token is valid and has access (HUGGINGFACE_HUB_TOKEN / HG_TOKEN / HF_TOKEN)",
                text_model_id, e
            )
            self.text_model = None
            self.text_dim = int(getattr(rag_config, "embedding_dimension", 384))

        # Load code model (token passed where supported)
        self.code_model = self._load_code_model()

        # Cache and dims
        self.cache_dir = paths.VECTOR_STORE_DIR / "embedding_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, np.ndarray] = {}
        self.code_dim = int(getattr(rag_config, "embedding_dimension", self.text_dim))

    def _load_code_model(self):
        """Attempt to load a code embedding model (AutoTokenizer + AutoModel)."""
        try:
            # Transformers new arg is `token=...`; older versions accept `use_auth_token`.
            # Using `token` which is the current recommended name.
            tokenizer = AutoTokenizer.from_pretrained(rag_config.code_embedding_model, token=self.hf_token)
            model = AutoModel.from_pretrained(rag_config.code_embedding_model, token=self.hf_token)
            logger.info("Loaded code embedding model: %s", rag_config.code_embedding_model)
            return {"tokenizer": tokenizer, "model": model}
        except Exception as e:
            logger.warning("Failed to load code model '%s': %s. Falling back to text model for code embeddings.",
                           getattr(rag_config, "code_embedding_model", "<unset>"), e)
            return None

    async def generate_embeddings(self, data: Union[str, Dict, List]) -> np.ndarray:
        """Dispatch to the appropriate embedding generator based on input type."""
        if isinstance(data, str):
            return await self.embed_text(data)
        if isinstance(data, dict):
            return await self.embed_structured(data)
        if isinstance(data, list):
            return await self.embed_batch(data)
        raise ValueError(f"Unsupported data type: {type(data)}")

    async def embed_text(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Generate embedding for a single text (async)."""
        import asyncio

        if text is None:
            text = ""

        cache_key = self._get_cache_key(text)
        if use_cache:
            if cache_key in self.cache:
                return self.cache[cache_key]
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                self.cache[cache_key] = cached
                return cached

        # If a SentenceTransformer model is available, encode via thread
        embedding = None
        if self.text_model is not None:
            try:
                embedding = await asyncio.to_thread(self.text_model.encode, text, convert_to_numpy=True)
            except Exception as e:
                logger.warning("Text model encode failed, using fallback embedding: %s", e)

        # Fallback deterministic embedding (uses sha256)
        if embedding is None:
            embedding = self._fallback_vector(text)

        embedding = self._normalize(np.asarray(embedding, dtype=np.float32))

        if use_cache:
            self.cache[cache_key] = embedding
            self._save_to_cache(cache_key, embedding)

        return embedding

    async def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for multiple texts (async). Returns (n, dim) numpy array."""
        import asyncio

        if not texts:
            return np.empty((0, self.text_dim), dtype=np.float32)

        # If model available, try batch encode once for performance
        if self.text_model is not None:
            try:
                # call model.encode in thread to avoid blocking
                batch_embeddings = await asyncio.to_thread(self.text_model.encode, texts, convert_to_numpy=True)
                batch_embeddings = np.asarray(batch_embeddings, dtype=np.float32)
                # If returned shape is (dim,) for single item, expand
                if batch_embeddings.ndim == 1:
                    batch_embeddings = batch_embeddings.reshape(1, -1)
                # Ensure column count matches text_dim; if not, truncate or pad with zeros
                if batch_embeddings.shape[1] != self.text_dim:
                    target = self.text_dim
                    cur = batch_embeddings.shape[1]
                    if cur > target:
                        batch_embeddings = batch_embeddings[:, :target]
                    else:
                        pad = np.zeros((batch_embeddings.shape[0], target - cur), dtype=np.float32)
                        batch_embeddings = np.hstack([batch_embeddings, pad])
                # Normalize rows
                norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                batch_embeddings = batch_embeddings / norms
                # Save per-item cache asynchronously (non-blocking)
                for t, emb in zip(texts, batch_embeddings):
                    key = self._get_cache_key(t)
                    self.cache[key] = emb
                    self._save_to_cache(key, emb)
                return batch_embeddings
            except Exception as e:
                logger.warning("Batch encoding with text model failed, falling back to per-item encoding: %s", e)

        # Fallback: encode items individually (uses embed_text which applies cache/fallback)
        tasks = [self.embed_text(t) for t in texts]
        results = await asyncio.gather(*tasks)
        stacked = np.vstack(results).astype(np.float32)
        # Ensure dimension
        if stacked.shape[1] != self.text_dim:
            if stacked.shape[1] > self.text_dim:
                stacked = stacked[:, : self.text_dim]
            else:
                pad = np.zeros((stacked.shape[0], self.text_dim - stacked.shape[1]), dtype=np.float32)
                stacked = np.hstack([stacked, pad])
        return stacked

    async def embed_code(self, code: str, language: str = None) -> np.ndarray:
        """Generate embedding for code. Prefer code model, else fallback to text model."""
        if self.code_model:
            try:
                return await self._embed_with_codebert(code)
            except Exception as e:
                logger.warning("Code model embedding failed, falling back to text embed: %s", e)

        processed = self._preprocess_code(code, language)
        return await self.embed_text(processed)

    async def _embed_with_codebert(self, code: str) -> np.ndarray:
        """Embed using loaded AutoModel for code (sync operations executed in thread)."""
        import asyncio
        tokenizer = self.code_model["tokenizer"]
        model = self.code_model["model"]

        def _encode():
            inputs = tokenizer(code, return_tensors="pt", max_length=512, truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                # mean pool over sequence dimension
                emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                return emb.squeeze()

        emb = await asyncio.to_thread(_encode)
        emb = np.asarray(emb, dtype=np.float32)
        emb = self._normalize(emb)
        return emb

    async def embed_structured(self, data: Dict[str, Any]) -> np.ndarray:
        """Convert structured object to text and embed it."""
        parts: List[str] = []

        # Common fields
        if data.get("endpoint"):
            parts.append(f"Endpoint: {data.get('endpoint')}")
        if data.get("path"):
            parts.append(f"Path: {data.get('path')}")
        if data.get("method"):
            parts.append(f"Method: {data.get('method')}")
        if data.get("name"):
            parts.append(f"Name: {data.get('name')}")
        if data.get("description"):
            parts.append(f"Description: {data.get('description')}")

        # parameters or parameters-like
        params = data.get("parameters") or data.get("params") or []
        for p in params:
            name = p.get("name") or p.get("title") or ""
            vals = p.get("values") or p.get("example") or ""
            parts.append(f"Param: {name} Values: {vals}")

        # any other fields appended
        for k, v in data.items():
            if k not in {"endpoint", "path", "method", "name", "description", "parameters", "params"}:
                parts.append(f"{k}: {v}")

        combined = " | ".join([str(p) for p in parts if p])
        return await self.embed_text(combined)

    def _fallback_vector(self, text: str) -> np.ndarray:
        """Deterministic fallback embedding from SHA256 digest expanded/padded to text_dim."""
        if text is None:
            text = ""
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # repeat digest bytes until we have enough bytes
        repeats = (self.text_dim // len(digest)) + 1
        data = (digest * repeats)[: self.text_dim]
        arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
        # map 0..255 -> -1..1
        arr = (arr / 127.5) - 1.0
        return self._normalize(arr)

    def _get_cache_key(self, text: str) -> str:
        """Short cache key for a text blob."""
        if text is None:
            text = ""
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def _save_to_cache(self, key: str, embedding: np.ndarray):
        """Persist a single embedding to disk (best-effort)."""
        try:
            filename = self.cache_dir / f"{key}.pkl"
            with open(filename, "wb") as f:
                pickle.dump(embedding, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.debug("Could not save embedding cache %s: %s", key, e)

    def _load_from_cache(self, key: str) -> Union[np.ndarray, None]:
        """Load embedding from disk if present."""
        try:
            filename = self.cache_dir / f"{key}.pkl"
            if not filename.exists():
                return None
            with open(filename, "rb") as f:
                emb = pickle.load(f)
                return np.asarray(emb, dtype=np.float32)
        except Exception as e:
            logger.debug("Could not load embedding cache %s: %s", key, e)
            return None

    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize an embedding to unit length."""
        emb = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            return emb / norm
        return emb

    def clear_cache(self):
        """Clear in-memory and on-disk cache."""
        try:
            for p in self.cache_dir.glob("*.pkl"):
                try:
                    p.unlink()
                except Exception:
                    pass
            self.cache.clear()
            logger.info("Embedding cache cleared")
        except Exception as e:
            logger.warning("Failed to clear embedding cache: %s", e)
