# file: scripts/index_knowledge.py
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from rag.chunking import ChunkingStrategy
from rag.embeddings import EmbeddingManager
from rag.indexer import Indexer
from rag.vector_store import VectorStore


class KnowledgeIndexerRunner:
    """
    Indexes files into the RAG vector store using actual component classes.
    """

    def __init__(self, out_dir: Optional[Path] = None) -> None:
        self.out_dir = Path(out_dir or "data/vectors")
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.chunker = ChunkingStrategy()
        self.em = EmbeddingManager()
        self.vs = VectorStore()
        self.indexer = Indexer(self.vs, self.em, self.chunker)

    def index_paths(
        self,
        paths: Iterable[Path],
        namespace: str = "test_patterns",
    ) -> Path:
        """
        Build or update the vector index for the given files.

        Args:
            paths: File paths to index.
            namespace: Index name to use (must be one of the configured indices).

        Returns:
            Path to the vector store directory.
        """
        valid_paths = [Path(p) for p in paths if Path(p).is_file()]
        if not valid_paths:
            raise FileNotFoundError("No valid files to index")

        documents = []
        for p in valid_paths:
            content = p.read_text(encoding="utf-8")
            # Detect chunking strategy from file extension
            strategy = "code" if p.suffix in {".py", ".cs", ".java", ".cpp"} else "sliding_window"
            documents.append({
                "content": content,
                "metadata": {"source": str(p), "filename": p.name},
                "chunking_strategy": strategy,
            })

        # index_documents() is async — run it synchronously here
        stats = asyncio.run(self.indexer.index_documents(documents, index_name=namespace))

        # Persist to disk
        self.vs.save_all()

        print(
            f"Indexed {stats['indexed']} documents "
            f"({stats['chunks_created']} chunks) into '{namespace}'. "
            f"Skipped: {stats['skipped']}, Failed: {stats['failed']}."
        )

        return self.out_dir