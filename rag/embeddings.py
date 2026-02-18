"""
Embedding generation and management
"""

import logging
import os
import re
import hashlib
import pickle
import asyncio

import numpy as np
from typing import Dict, List, Any, Union
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
        logger.info("Initializing Embedding Manager")

        # Resolve token
        self.hf_token = (
            hf_token
            or os.getenv("HUGGINGFACE_HUB_TOKEN")
            or os.getenv("HG_TOKEN")
            or os.getenv("HF_TOKEN")
        )
        if self.hf_token:
            logger.info("Using Hugging Face token from environment/argument")
        else:
            logger.info("No Hugging Face token found; attempting anonymous access")

        text_model_id = (
            getattr(rag_config, "text_embedding_model", None)
            or "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Load text embedding model
        try:
            from sentence_transformers import SentenceTransformer
            self.text_model = SentenceTransformer(text_model_id, use_auth_token=self.hf_token)
            try:
                self.text_dim = int(self.text_model.get_sentence_embedding_dimension())
            except Exception:
                self.text_dim = int(getattr(rag_config, "embedding_dimension", 384))
            logger.info("Loaded text embedding model: %s (dim=%s)", text_model_id, self.text_dim)
        except Exception as e:
            logger.warning("Failed to load text embedding model '%s': %s", text_model_id, e)
            self.text_model = None
            self.text_dim = int(getattr(rag_config, "embedding_dimension", 384))

        # Load code model
        self.code_model = self._load_code_model()

        # Cache
        self.cache_dir = paths.VECTOR_STORE_DIR / "embedding_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, np.ndarray] = {}
        self.code_dim = int(getattr(rag_config, "embedding_dimension", self.text_dim))

    def _load_code_model(self):
        """Attempt to load a code embedding model."""
        try:
            from transformers import AutoTokenizer, AutoModel
            tokenizer = AutoTokenizer.from_pretrained(
                rag_config.code_embedding_model, token=self.hf_token
            )
            model = AutoModel.from_pretrained(
                rag_config.code_embedding_model, token=self.hf_token
            )
            logger.info("Loaded code embedding model: %s", rag_config.code_embedding_model)
            return {"tokenizer": tokenizer, "model": model}
        except Exception as e:
            logger.warning(
                "Failed to load code model '%s': %s. Falling back to text model.",
                getattr(rag_config, "code_embedding_model", "<unset>"), e
            )
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

        embedding = None
        if self.text_model is not None:
            try:
                embedding = await asyncio.to_thread(
                    self.text_model.encode, text, convert_to_numpy=True
                )
            except Exception as e:
                logger.warning("Text model encode failed, using fallback: %s", e)

        if embedding is None:
            embedding = self._fallback_vector(text)

        embedding = self._normalize(np.asarray(embedding, dtype=np.float32))

        if use_cache:
            self.cache[cache_key] = embedding
            self._save_to_cache(cache_key, embedding)

        return embedding

    async def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for multiple texts. Returns (n, dim) array."""
        if not texts:
            return np.empty((0, self.text_dim), dtype=np.float32)

        if self.text_model is not None:
            try:
                batch_embeddings = await asyncio.to_thread(
                    self.text_model.encode, texts, convert_to_numpy=True
                )
                batch_embeddings = np.asarray(batch_embeddings, dtype=np.float32)
                if batch_embeddings.ndim == 1:
                    batch_embeddings = batch_embeddings.reshape(1, -1)

                # Ensure correct dimension
                if batch_embeddings.shape[1] != self.text_dim:
                    batch_embeddings = self._fix_dimension(batch_embeddings)

                # Normalize rows
                norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                batch_embeddings = batch_embeddings / norms

                # Cache individual items
                for t, emb in zip(texts, batch_embeddings):
                    key = self._get_cache_key(t)
                    self.cache[key] = emb
                    self._save_to_cache(key, emb)

                return batch_embeddings
            except Exception as e:
                logger.warning("Batch encoding failed, falling back to per-item: %s", e)

        # Fallback: encode items individually
        tasks = [self.embed_text(t) for t in texts]
        results = await asyncio.gather(*tasks)
        stacked = np.vstack(results).astype(np.float32)
        if stacked.shape[1] != self.text_dim:
            stacked = self._fix_dimension(stacked)
        return stacked

    async def embed_code(self, code: str, language: str = None) -> np.ndarray:
        """Generate embedding for code."""
        if self.code_model:
            try:
                return await self._embed_with_codebert(code)
            except Exception as e:
                logger.warning("Code model embedding failed, falling back: %s", e)

        # FIX: _preprocess_code was missing — now implemented below
        processed = self._preprocess_code(code, language)
        return await self.embed_text(processed)

    def _preprocess_code(self, code: str, language: str = None) -> str:
        """
        Preprocess code for text-based embedding.

        Strips comments, normalizes whitespace, extracts identifiers
        so the text embedding model gets meaningful tokens.
        """
        if not code:
            return ""

        lines = code.split('\n')
        processed_lines = []

        for line in lines:
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                continue

            # Skip single-line comments
            if language in ('python',) and stripped.startswith('#'):
                continue
            if language in ('csharp', 'java', 'cpp') and stripped.startswith('//'):
                continue

            # Remove inline comments
            if language in ('python',):
                line_clean = re.sub(r'#.*$', '', stripped)
            elif language in ('csharp', 'java', 'cpp'):
                line_clean = re.sub(r'//.*$', '', stripped)
            else:
                line_clean = stripped

            if line_clean.strip():
                processed_lines.append(line_clean.strip())

        # Join and collapse multiple spaces
        text = ' '.join(processed_lines)
        text = re.sub(r'\s+', ' ', text)

        # Extract meaningful identifiers (camelCase/PascalCase splitting)
        words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\b)', text)
        if words:
            text = text + ' ' + ' '.join(words[:50])  # Append top identifiers

        # Truncate to reasonable length for embedding model
        return text[:2000]

    async def _embed_with_codebert(self, code: str) -> np.ndarray:
        """Embed using CodeBERT model."""
        import torch

        tokenizer = self.code_model["tokenizer"]
        model = self.code_model["model"]

        def _encode():
            inputs = tokenizer(
                code, return_tensors="pt", max_length=512,
                truncation=True, padding=True
            )
            with torch.no_grad():
                outputs = model(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                return emb.squeeze()

        emb = await asyncio.to_thread(_encode)
        emb = np.asarray(emb, dtype=np.float32)
        return self._normalize(emb)

    async def embed_structured(self, data: Dict[str, Any]) -> np.ndarray:
        """Convert structured object to text and embed it."""
        parts: List[str] = []

        for key in ('endpoint', 'path', 'method', 'name', 'description'):
            if data.get(key):
                parts.append(f"{key.title()}: {data[key]}")

        params = data.get("parameters") or data.get("params") or []
        for p in params:
            name = p.get("name") or p.get("title") or ""
            vals = p.get("values") or p.get("example") or ""
            parts.append(f"Param: {name} Values: {vals}")

        skip_keys = {'endpoint', 'path', 'method', 'name', 'description', 'parameters', 'params'}
        for k, v in data.items():
            if k not in skip_keys:
                parts.append(f"{k}: {v}")

        combined = " | ".join([str(p) for p in parts if p])
        return await self.embed_text(combined)

    def _fix_dimension(self, embeddings: np.ndarray) -> np.ndarray:
        """Fix embedding dimension to match text_dim."""
        cur = embeddings.shape[1]
        target = self.text_dim
        if cur > target:
            return embeddings[:, :target]
        else:
            pad = np.zeros((embeddings.shape[0], target - cur), dtype=np.float32)
            return np.hstack([embeddings, pad])

    def _fallback_vector(self, text: str) -> np.ndarray:
        """Deterministic fallback embedding from SHA256."""
        if text is None:
            text = ""
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        repeats = (self.text_dim // len(digest)) + 1
        data = (digest * repeats)[:self.text_dim]
        arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
        arr = (arr / 127.5) - 1.0
        return self._normalize(arr)

    def _get_cache_key(self, text: str) -> str:
        if text is None:
            text = ""
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def _save_to_cache(self, key: str, embedding: np.ndarray):
        try:
            filename = self.cache_dir / f"{key}.pkl"
            with open(filename, "wb") as f:
                pickle.dump(embedding, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.debug("Could not save embedding cache %s: %s", key, e)

    def _load_from_cache(self, key: str) -> Union[np.ndarray, None]:
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
        emb = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            return emb / norm
        return emb

    def clear_cache(self):
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