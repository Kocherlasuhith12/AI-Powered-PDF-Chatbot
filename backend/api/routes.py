"""
backend/api/routes.py
FastAPI route definitions for DocuBot.

Endpoints:
  POST /upload          Upload + ingest a PDF
  POST /chat            Ask a question about an ingested PDF
  GET  /summary/{doc_id}  Get the auto-generated document summary
  GET  /sessions/{session_id}/clear  Clear conversation history
  GET  /health          Health check
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

from backend.core.pdf_processor import PDFProcessor
from backend.core.rag_engine import rag_engine
from backend.core.memory_manager import memory_manager
from backend.utils.helpers import generate_session_id, confidence_label

router = APIRouter()
pdf_processor = PDFProcessor()


# ── Pydantic models ───────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    doc_id: str
    session_id: str
    filename: str
    total_pages: int
    total_chunks: int
    file_size_kb: float
    word_count: int
    title: str | None
    author: str | None
    already_indexed: bool
    summary: str


class ChatRequest(BaseModel):
    question: str
    doc_id: str
    session_id: str
    filename: str = "document"


class CitationOut(BaseModel):
    page_number: int
    chunk_index: int
    excerpt: str
    relevance_score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[CitationOut]
    confidence: float
    confidence_label: str
    confidence_emoji: str
    tokens_used: int
    fallback: bool
    response_time_ms: int


class HealthResponse(BaseModel):
    status: str
    version: str


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    return {"status": "ok", "version": "1.0.0"}


@router.post("/upload", response_model=UploadResponse, tags=["documents"])
async def upload_pdf(
    file: Annotated[UploadFile, File(description="PDF file to ingest")],
    session_id: Annotated[str, Form()] = "",
):
    """
    Upload a PDF, chunk it, embed it into ChromaDB, and return metadata.
    If *session_id* is empty a new one is generated.
    Re-uploading the same file (same SHA-256) skips re-embedding.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are accepted.",
        )

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # Parse + chunk
    try:
        chunks, meta = pdf_processor.process(file_bytes, file.filename)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    # Session management
    if not session_id:
        session_id = generate_session_id()
    memory_manager.set_active_doc(session_id, meta.doc_id, meta.filename)

    # Ingest (idempotent)
    already_indexed = rag_engine.doc_exists(meta.doc_id)
    rag_engine.ingest(chunks, meta.doc_id)

    # Auto-summarise (with caching)
    from config import UPLOAD_DIR
    summary_cache_path = Path(UPLOAD_DIR) / f"{meta.doc_id}.summary"
    if summary_cache_path.exists():
        summary = summary_cache_path.read_text(encoding="utf-8")
    else:
        full_text = pdf_processor.extract_full_text(file_bytes)
        summary = rag_engine.summarise(full_text, meta.filename)
        try:
            summary_cache_path.write_text(summary, encoding="utf-8")
        except Exception:
            pass

    return UploadResponse(
        doc_id=meta.doc_id,
        session_id=session_id,
        filename=meta.filename,
        total_pages=meta.total_pages,
        total_chunks=meta.total_chunks,
        file_size_kb=meta.file_size_kb,
        word_count=meta.word_count,
        title=meta.title,
        author=meta.author,
        already_indexed=already_indexed,
        summary=summary,
    )


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(req: ChatRequest):
    """
    Answer a question about the specified document using RAG + Claude.
    Returns the answer, source citations, and a confidence score.
    """
    if not req.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question must not be empty.",
        )
    if not req.doc_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="doc_id is required.",
        )
    if not rag_engine.doc_exists(req.doc_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{req.doc_id}' not found. Please upload it first.",
        )

    t0 = time.perf_counter()
    rag_resp = rag_engine.answer(
        question=req.question,
        doc_id=req.doc_id,
        session_id=req.session_id,
        filename=req.filename,
    )
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    label, emoji = confidence_label(rag_resp.confidence)

    return ChatResponse(
        answer=rag_resp.answer,
        sources=[
            CitationOut(
                page_number=s.page_number,
                chunk_index=s.chunk_index,
                excerpt=s.excerpt,
                relevance_score=s.relevance_score,
            )
            for s in rag_resp.sources
        ],
        confidence=rag_resp.confidence,
        confidence_label=label,
        confidence_emoji=emoji,
        tokens_used=rag_resp.tokens_used,
        fallback=rag_resp.fallback,
        response_time_ms=elapsed_ms,
    )


@router.get("/sessions/{session_id}/clear", tags=["sessions"])
async def clear_session(session_id: str):
    """Clear the conversation history for a session."""
    memory_manager.clear_session(session_id)
    return {"message": f"Session '{session_id}' cleared.", "session_id": session_id}


@router.delete("/documents/{doc_id}", tags=["documents"])
async def delete_document(doc_id: str):
    """
    Delete a document's vector store collection and its cached summary.
    """
    if not rag_engine.doc_exists(doc_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )
    
    # Delete from vector database
    db_deleted = rag_engine.delete_doc(doc_id)
    
    # Delete cached summary
    from config import UPLOAD_DIR
    summary_cache_path = Path(UPLOAD_DIR) / f"{doc_id}.summary"
    cache_deleted = False
    if summary_cache_path.exists():
        try:
            summary_cache_path.unlink()
            cache_deleted = True
        except Exception:
            pass
            
    return {
        "message": f"Document '{doc_id}' deleted successfully.",
        "doc_id": doc_id,
        "db_deleted": db_deleted,
        "cache_deleted": cache_deleted,
    }


@router.get("/documents/{doc_id}/chunks", tags=["documents"])
async def get_document_chunks(doc_id: str):
    """
    Get all stored document chunks for a document.
    """
    if not rag_engine.doc_exists(doc_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )
    return rag_engine.get_all_chunks(doc_id)
