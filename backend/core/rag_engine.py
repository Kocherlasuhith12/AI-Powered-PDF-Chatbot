"""
backend/core/rag_engine.py
The heart of DocuBot — handles embedding, vector storage, retrieval,
confidence scoring, and Claude-powered answer generation.
"""
from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
import anthropic

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
from config import (
    ANTHROPIC_API_KEY,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    TOP_K_RESULTS,
    MIN_CONFIDENCE,
    CLAUDE_MODEL,
    MAX_TOKENS,
    MAX_HISTORY_TURNS,
)
from backend.core.pdf_processor import DocumentChunk, DocumentMeta
from backend.core.memory_manager import memory_manager


# ── Response dataclass ────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    answer: str
    sources: List[SourceCitation]
    confidence: float          # 0.0 (low) – 1.0 (high)
    doc_id: str
    session_id: str
    tokens_used: int = 0
    fallback: bool = False     # True when no good chunks were found


@dataclass
class SourceCitation:
    page_number: int
    chunk_index: int
    text: str                  # Full text of the chunk
    excerpt: str               # First 200 chars of the chunk
    relevance_score: float     # Normalised cosine similarity


# ── RAG Engine ────────────────────────────────────────────────────────────────

class RAGEngine:
    """
    Encapsulates the full RAG pipeline:
      1. Embed chunks → store in ChromaDB
      2. Query embedding → cosine search → top-K chunks
      3. Build prompt with retrieved context + conversation history
      4. Call Claude → parse answer → attach citations
    """

    def __init__(self) -> None:
        # Local ONNX-based embedder (no API cost, no internet required at runtime)
        self._embed_fn = DefaultEmbeddingFunction()

        # Persistent ChromaDB client
        self._chroma = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

        # Anthropic client
        try:
            self._claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        except Exception:
            self._claude = None

    # ── Ingest ────────────────────────────────────────────────────────────

    def ingest(self, chunks: List[DocumentChunk], doc_id: str) -> int:
        """
        Embed all *chunks* and store them in a per-document ChromaDB collection.
        Returns the number of chunks stored.
        Skips ingestion if this doc_id was already indexed (idempotent).
        """
        collection_name = f"doc_{doc_id}"

        # Check if already indexed
        existing = [c.name for c in self._chroma.list_collections()]
        if collection_name in existing:
            col = self._chroma.get_collection(
                name=collection_name,
                embedding_function=self._embed_fn,
            )
            if col.count() > 0:
                return col.count()  # already done

        collection = self._chroma.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

        # Batch upsert for efficiency
        BATCH_SIZE = 100
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            collection.upsert(
                ids=[f"{doc_id}_{c.chunk_index}" for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.metadata for c in batch],
            )

        return collection.count()

    # ── Query ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        doc_id: str,
        top_k: int = TOP_K_RESULTS,
    ) -> tuple[List[SourceCitation], float]:
        """
        Embed *query*, search the collection for *doc_id*, and return
        (citations, confidence).  Confidence is 1 – mean_distance (cosine).
        """
        collection_name = f"doc_{doc_id}"
        try:
            collection = self._chroma.get_collection(
                name=collection_name,
                embedding_function=self._embed_fn,
            )
        except Exception:
            return [], 0.0

        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        docs      = results["documents"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0]   # cosine distance: 0=identical

        citations: List[SourceCitation] = []
        for doc_text, meta, dist in zip(docs, metas, distances):
            relevance = max(0.0, round(1.0 - dist, 3))
            citations.append(
                SourceCitation(
                    page_number=int(meta.get("page", 0)),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    text=doc_text,
                    excerpt=doc_text[:200].replace("\n", " "),
                    relevance_score=relevance,
                )
            )

        # Overall confidence = mean relevance of top-3 results
        top_scores = [c.relevance_score for c in citations[:3]]
        confidence = round(sum(top_scores) / len(top_scores), 3) if top_scores else 0.0

        return citations, confidence

    def retrieve_multiple(
        self,
        query: str,
        doc_ids: List[str],
        top_k: int = TOP_K_RESULTS,
    ) -> tuple[List[SourceCitation], float]:
        """Retrieve top-K chunks across multiple document collections."""
        all_citations = []
        for doc_id in doc_ids:
            citations, _ = self.retrieve(query, doc_id, top_k=top_k)
            all_citations.extend(citations)
        
        # Sort by relevance score descending
        all_citations.sort(key=lambda x: x.relevance_score, reverse=True)
        top_citations = all_citations[:top_k]
        
        # Recalculate confidence
        top_scores = [c.relevance_score for c in top_citations[:3]]
        confidence = round(sum(top_scores) / len(top_scores), 3) if top_scores else 0.0
        
        return top_citations, confidence

    # ── Generate ─────────────────────────────────────────────────────────

    def answer(
        self,
        question: str,
        doc_id: str,
        session_id: str,
        filename: str = "document",
    ) -> RAGResponse:
        """
        Full RAG pipeline: retrieve → build prompt → call Claude → return response.
        """
        if "," in doc_id:
            doc_ids = [d.strip() for d in doc_id.split(",") if d.strip()]
        else:
            doc_ids = [doc_id]

        citations, confidence = self.retrieve_multiple(question, doc_ids)

        # ── Low-confidence fallback ──────────────────────────────────────
        if confidence < MIN_CONFIDENCE or not citations:
            fallback_answer = (
                "I couldn't find a confident answer to that question in the document. "
                "Try rephrasing, or ask about a topic you've seen mentioned in the text."
            )
            memory_manager.add_exchange(session_id, question, fallback_answer)
            return RAGResponse(
                answer=fallback_answer,
                sources=[],
                confidence=confidence,
                doc_id=doc_id,
                session_id=session_id,
                fallback=True,
            )

        # ── Build context block ──────────────────────────────────────────
        context_parts = []
        for i, cit in enumerate(citations, 1):
            context_parts.append(
                f"[Source {i} — Page {cit.page_number}, relevance {cit.relevance_score:.0%}]\n"
                f"{cit.text}"
            )
        context_block = "\n\n".join(context_parts)

        # ── Build conversation history ───────────────────────────────────
        history = memory_manager.get(session_id).get_history_text(MAX_HISTORY_TURNS)

        # ── System prompt ────────────────────────────────────────────────
        system_prompt = textwrap.dedent(f"""
            You are DocuBot, an expert document analyst. You answer questions about
            the PDF document "{filename}" strictly using the provided context chunks.

            Rules:
            1. Base your answer ONLY on the context provided below.
            2. Cite sources inline using [Page X] notation.
            3. If the context doesn't contain enough info, say so honestly.
            4. Keep answers concise but complete. Use bullet points for lists.
            5. Maintain continuity with the prior conversation.
        """).strip()

        # ── User message ─────────────────────────────────────────────────
        user_message = textwrap.dedent(f"""
            CONTEXT FROM DOCUMENT:
            {context_block}

            {'PRIOR CONVERSATION:' + chr(10) + history if history else ''}

            QUESTION: {question}
        """).strip()

        # ── Call Claude ──────────────────────────────────────────────────
        if not self._claude:
            answer_text = (
                "⚠️ **LLM Generation Error**\n\n"
                "The Anthropic client could not be initialized. "
                "Please ensure you have configured a valid `ANTHROPIC_API_KEY` in your environment or `.env` file."
            )
            tokens_used = 0
            fallback = True
        else:
            try:
                response = self._claude.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=MAX_TOKENS,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                )
                answer_text = response.content[0].text
                tokens_used = response.usage.input_tokens + response.usage.output_tokens
                fallback = False
            except Exception as e:
                answer_text = (
                    f"⚠️ **LLM Generation Error**\n\n"
                    f"Failed to generate an answer because the Anthropic API call failed. "
                    f"Please ensure you have configured a valid `ANTHROPIC_API_KEY` in your environment or `.env` file.\n\n"
                    f"*Error details: {e}*"
                )
                tokens_used = 0
                fallback = True

        # ── Persist to memory ────────────────────────────────────────────
        memory_manager.add_exchange(session_id, question, answer_text)

        return RAGResponse(
            answer=answer_text,
            sources=citations,
            confidence=confidence,
            doc_id=doc_id,
            session_id=session_id,
            tokens_used=tokens_used,
            fallback=fallback,
        )

    # ── Summarise ────────────────────────────────────────────────────────

    def summarise(self, full_text: str, filename: str) -> str:
        """
        Generate a structured summary of the document.
        Truncates input to ~8000 chars to stay within token limits.
        """
        truncated = full_text[:8000]
        if len(full_text) > 8000:
            truncated += "\n\n[...document continues...]"

        prompt = textwrap.dedent(f"""
            Summarise this document called "{filename}". Structure your response as:

            **Document Overview**
            A 2–3 sentence high-level summary.

            **Key Topics**
            Bullet list of the main topics covered.

            **Key Takeaways**
            3–5 most important facts or conclusions.

            Document text:
            {truncated}
        """).strip()
        if not self._claude:
            return (
                "⚠️ **Summary generation unavailable**\n\n"
                "The Anthropic client could not be initialized. "
                "Please make sure a valid `ANTHROPIC_API_KEY` is configured in your environment or `.env` file."
            )

        try:
            response = self._claude.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            return (
                f"⚠️ **Summary generation unavailable**\n\n"
                f"Could not generate summary using Anthropic Claude. Please make sure a valid `ANTHROPIC_API_KEY` "
                f"is configured in your environment or `.env` file.\n\n"
                f"*(Error detail: {e})*"
            )

    # ── Collection management ─────────────────────────────────────────────

    def get_all_chunks(self, doc_id: str) -> List[dict]:
        """Retrieve all document chunks from ChromaDB for display/inspection."""
        if "," in doc_id:
            doc_ids = [d.strip() for d in doc_id.split(",") if d.strip()]
            all_chunks = []
            for d in doc_ids:
                all_chunks.extend(self.get_all_chunks(d))
            return all_chunks

        collection_name = f"doc_{doc_id}"
        try:
            collection = self._chroma.get_collection(
                name=collection_name,
                embedding_function=self._embed_fn,
            )
            results = collection.get(include=["documents", "metadatas"])
            chunks = []
            for doc_text, meta in zip(results["documents"], results["metadatas"]):
                chunks.append({
                    "text": doc_text,
                    "page_number": int(meta.get("page", 0)),
                    "chunk_index": int(meta.get("chunk_index", 0)),
                })
            # Sort chronologically by chunk index
            chunks.sort(key=lambda x: x["chunk_index"])
            return chunks
        except Exception:
            return []

    def delete_doc(self, doc_id: str) -> bool:
        """Remove a document's collection from ChromaDB."""
        if "," in doc_id:
            doc_ids = [d.strip() for d in doc_id.split(",") if d.strip()]
            return all(self.delete_doc(d) for d in doc_ids)

        collection_name = f"doc_{doc_id}"
        try:
            self._chroma.delete_collection(collection_name)
            return True
        except Exception:
            return False

    def doc_exists(self, doc_id: str) -> bool:
        """Check if a doc_id has already been indexed."""
        if "," in doc_id:
            doc_ids = [d.strip() for d in doc_id.split(",") if d.strip()]
            return all(self.doc_exists(d) for d in doc_ids)

        collection_name = f"doc_{doc_id}"
        try:
            existing = [c.name for c in self._chroma.list_collections()]
            if collection_name not in existing:
                return False
            col = self._chroma.get_collection(
                name=collection_name,
                embedding_function=self._embed_fn,
            )
            return col.count() > 0
        except Exception:
            return False


# Module-level singleton
rag_engine = RAGEngine()
