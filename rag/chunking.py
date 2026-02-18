"""
Document chunking strategies for RAG
"""

import logging
import re
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a document chunk"""
    text: str
    metadata: Dict[str, Any]
    start_idx: int
    end_idx: int
    chunk_id: str


class ChunkingStrategy:
    """Manages different chunking strategies for documents"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document: str, metadata: Dict[str, Any] = None,
                       strategy: str = 'sliding_window') -> List[Chunk]:
        """
        Chunk a document using specified strategy

        Args:
            document: Document text
            metadata: Document metadata
            strategy: Chunking strategy

        Returns:
            List of chunks
        """
        if strategy == 'sliding_window':
            return self.sliding_window_chunk(document, metadata)
        elif strategy == 'semantic':
            return self.semantic_chunk(document, metadata)
        elif strategy == 'code':
            return self.code_chunk(document, metadata)
        elif strategy == 'test':
            return self.test_chunk(document, metadata)
        else:
            return self.sliding_window_chunk(document, metadata)

    def sliding_window_chunk(self, document: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """Sliding window chunking"""
        chunks = []

        # Split into sentences first for better boundaries
        sentences = self._split_sentences(document)

        current_chunk = []
        current_length = 0
        start_idx = 0

        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)

            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata['chunk_index'] = len(chunks)

                chunks.append(Chunk(
                    text=chunk_text,
                    metadata=chunk_metadata,
                    start_idx=start_idx,
                    end_idx=start_idx + len(chunk_text),
                    chunk_id=f"chunk_{len(chunks)}"
                ))

                # Overlap handling
                overlap_sentences = []
                overlap_length = 0
                for j in range(len(current_chunk) - 1, -1, -1):
                    overlap_length += len(current_chunk[j])
                    if overlap_length >= self.chunk_overlap:
                        overlap_sentences = current_chunk[j:]
                        break

                current_chunk = overlap_sentences
                current_length = sum(len(s) for s in current_chunk)
                start_idx = chunks[-1].end_idx - current_length

            current_chunk.append(sentence)
            current_length += sentence_length

        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata['chunk_index'] = len(chunks)

            chunks.append(Chunk(
                text=chunk_text,
                metadata=chunk_metadata,
                start_idx=start_idx,
                end_idx=start_idx + len(chunk_text),
                chunk_id=f"chunk_{len(chunks)}"
            ))

        return chunks

    def semantic_chunk(self, document: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """Semantic chunking based on topic boundaries"""
        chunks = []

        # Split by paragraphs
        paragraphs = document.split('\n\n')

        current_chunk = []
        current_length = 0
        start_idx = 0

        for para in paragraphs:
            para_length = len(para)

            if current_length + para_length > self.chunk_size and current_chunk:
                # Create chunk at paragraph boundary
                chunk_text = '\n\n'.join(current_chunk)
                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata['chunk_index'] = len(chunks)

                chunks.append(Chunk(
                    text=chunk_text,
                    metadata=chunk_metadata,
                    start_idx=start_idx,
                    end_idx=start_idx + len(chunk_text),
                    chunk_id=f"chunk_{len(chunks)}"
                ))

                current_chunk = []
                current_length = 0
                start_idx += len(chunk_text) + 2  # Account for \n\n

            current_chunk.append(para)
            current_length += para_length

        # Add remaining
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata['chunk_index'] = len(chunks)

            chunks.append(Chunk(
                text=chunk_text,
                metadata=chunk_metadata,
                start_idx=start_idx,
                end_idx=start_idx + len(chunk_text),
                chunk_id=f"chunk_{len(chunks)}"
            ))

        return chunks

    def code_chunk(self, document: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """Chunk code documents by functions/classes"""
        chunks = []

        # Detect language
        language = metadata.get('language', 'unknown') if metadata else 'unknown'

        # Split by function/class definitions
        if language in ['python', 'csharp', 'java']:
            blocks = self._split_code_blocks(document, language)
        else:
            # Fall back to line-based chunking
            blocks = self._split_by_lines(document, 50)  # 50 lines per chunk

        for i, block in enumerate(blocks):
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata['chunk_index'] = i
            chunk_metadata['block_type'] = self._detect_block_type(block)

            chunks.append(Chunk(
                text=block,
                metadata=chunk_metadata,
                start_idx=0,  # Would need to track actual positions
                end_idx=len(block),
                chunk_id=f"code_chunk_{i}"
            ))

        return chunks

    def test_chunk(self, document: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """Chunk test documents by test cases"""
        chunks = []

        # Split by test function patterns
        test_patterns = [
            r'def test_\w+',  # Python
            r'public void \w+Test',  # Java/C#
            r'it\([\'"].*?[\'"]',  # JavaScript
            r'TEST\(',  # C++ Google Test
        ]

        # Try to identify test boundaries
        test_blocks = []
        current_block = []
        in_test = False

        lines = document.split('\n')
        for line in lines:
            # Check if line starts a new test
            if any(re.search(pattern, line) for pattern in test_patterns):
                if current_block:
                    test_blocks.append('\n'.join(current_block))
                current_block = [line]
                in_test = True
            elif in_test:
                current_block.append(line)

        # Add last block
        if current_block:
            test_blocks.append('\n'.join(current_block))

        # Create chunks from test blocks
        for i, block in enumerate(test_blocks):
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata['chunk_index'] = i
            chunk_metadata['test_name'] = self._extract_test_name(block)

            chunks.append(Chunk(
                text=block,
                metadata=chunk_metadata,
                start_idx=0,
                end_idx=len(block),
                chunk_id=f"test_{i}"
            ))

        return chunks if chunks else self.sliding_window_chunk(document, metadata)

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _split_code_blocks(self, code: str, language: str) -> List[str]:
        """Split code into logical blocks"""
        blocks = []

        if language == 'python':
            # Split by class and function definitions
            pattern = r'^(class |def |async def )'
            parts = re.split(pattern, code, flags=re.MULTILINE)

            current_block = parts[0] if parts[0].strip() else ''
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    block = parts[i] + parts[i + 1]
                    if current_block:
                        blocks.append(current_block)
                    current_block = block

            if current_block:
                blocks.append(current_block)

        elif language in ['csharp', 'java']:
            # Split by class and method definitions
            # Simplified - would need proper parsing for accuracy
            lines = code.split('\n')
            current_block = []
            brace_count = 0

            for line in lines:
                current_block.append(line)
                brace_count += line.count('{') - line.count('}')

                if brace_count == 0 and current_block and '{' in ''.join(current_block):
                    blocks.append('\n'.join(current_block))
                    current_block = []

            if current_block:
                blocks.append('\n'.join(current_block))

        return blocks if blocks else [code]

    def _split_by_lines(self, text: str, lines_per_chunk: int) -> List[str]:
        """Split text by number of lines"""
        lines = text.split('\n')
        chunks = []

        for i in range(0, len(lines), lines_per_chunk):
            chunk = '\n'.join(lines[i:i + lines_per_chunk])
            chunks.append(chunk)

        return chunks

    def _detect_block_type(self, code_block: str) -> str:
        """Detect type of code block"""
        if 'class ' in code_block:
            return 'class'
        elif 'def ' in code_block or 'function ' in code_block:
            return 'function'
        elif 'test' in code_block.lower():
            return 'test'
        else:
            return 'unknown'

    def _extract_test_name(self, test_block: str) -> str:
        """Extract test name from test block"""
        # Try different patterns
        patterns = [
            r'def (test_\w+)',
            r'public void (\w+Test)',
            r'it\([\'"]([^\'"]*)[\'"',
            r'TEST\((\w+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, test_block)
            if match:
                return match.group(1)

        return 'unknown_test'