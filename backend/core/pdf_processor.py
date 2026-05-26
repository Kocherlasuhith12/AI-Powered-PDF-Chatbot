"""
backend/core/pdf_processor.py
Handles PDF parsing, metadata extraction, and smart recursive chunking.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
from config import MAX_CHUNK_SIZE, CHUNK_OVERLAP


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """A single text chunk with full provenance."""
    text: str
    page_number: int          # 1-indexed
    chunk_index: int          # position within the page
    doc_id: str               # SHA-256 of the original file
    source_filename: str
    total_pages: int
    char_start: int = 0       # character offset within the page text
    metadata: dict = field(default_factory=dict)


@dataclass
class DocumentMeta:
    """High-level metadata extracted from a PDF."""
    doc_id: str
    filename: str
    total_pages: int
    total_chunks: int
    title: Optional[str]
    author: Optional[str]
    file_size_kb: float
    word_count: int


# ── Helpers ──────────────────────────────────────────────────────────────────

def _compute_doc_id(file_bytes: bytes) -> str:
    """Stable SHA-256 hash used as the document's primary key."""
    return hashlib.sha256(file_bytes).hexdigest()[:16]


def _clean_text(raw: str) -> str:
    """Normalise whitespace and remove junk characters from extracted text."""
    # Collapse multiple newlines / spaces
    text = re.sub(r"\n{3,}", "\n\n", raw)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Drop non-printable characters
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)
    return text.strip()


# ── Main processor ───────────────────────────────────────────────────────────

class PDFProcessor:
    """
    Parses a PDF and produces a list of DocumentChunk objects ready for
    embedding.  Uses LangChain's RecursiveCharacterTextSplitter so chunks
    respect sentence/paragraph boundaries rather than cutting mid-word.
    """

    def __init__(
        self,
        chunk_size: int = MAX_CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            length_function=len,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def process(self, file_bytes: bytes, filename: str) -> tuple[List[DocumentChunk], DocumentMeta]:
        """
        Parse *file_bytes* as a PDF and return (chunks, metadata).
        Raises ValueError if the PDF has no extractable text.
        """
        doc_id = _compute_doc_id(file_bytes)
        reader = PdfReader(path_or_stream := __import__("io").BytesIO(file_bytes))

        total_pages = len(reader.pages)
        pdf_info = reader.metadata or {}

        all_chunks: List[DocumentChunk] = []
        total_words = 0
        global_chunk_idx = 0

        for page_num, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            page_text = _clean_text(raw_text)
            if not page_text:
                continue

            total_words += len(page_text.split())

            # Split this page's text into sub-chunks
            page_chunks = self.splitter.split_text(page_text)

            char_cursor = 0
            for chunk_text in page_chunks:
                if not chunk_text.strip():
                    continue

                # Approximate char_start within original page text
                char_start = page_text.find(chunk_text[:50], char_cursor)
                char_start = max(char_start, 0)
                char_cursor = char_start + len(chunk_text)

                all_chunks.append(
                    DocumentChunk(
                        text=chunk_text.strip(),
                        page_number=page_num,
                        chunk_index=global_chunk_idx,
                        doc_id=doc_id,
                        source_filename=filename,
                        total_pages=total_pages,
                        char_start=char_start,
                        metadata={
                            "page": page_num,
                            "chunk_index": global_chunk_idx,
                            "doc_id": doc_id,
                            "source": filename,
                        },
                    )
                )
                global_chunk_idx += 1

        if not all_chunks:
            raise ValueError(
                "No extractable text found in this PDF. "
                "It may be a scanned image — OCR support coming soon."
            )

        meta = DocumentMeta(
            doc_id=doc_id,
            filename=filename,
            total_pages=total_pages,
            total_chunks=len(all_chunks),
            title=str(pdf_info.get("/Title", "")) or None,
            author=str(pdf_info.get("/Author", "")) or None,
            file_size_kb=round(len(file_bytes) / 1024, 1),
            word_count=total_words,
        )

        return all_chunks, meta

    def extract_full_text(self, file_bytes: bytes) -> str:
        """Return the full concatenated text of the PDF (for summarisation)."""
        reader = PdfReader(__import__("io").BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(_clean_text(text))
        return "\n\n".join(p for p in pages if p)
