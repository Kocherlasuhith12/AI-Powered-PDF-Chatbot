"""
frontend/app.py — DocuBot Streamlit UI

Run with:
    streamlit run frontend/app.py
"""
from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).parents[1]))
from backend.utils.helpers import confidence_label

# Try importing backend core files for Direct (Serverless) execution mode
try:
    from backend.core.pdf_processor import PDFProcessor
    from backend.core.rag_engine import rag_engine
    from backend.core.memory_manager import memory_manager
    DIRECT_MODE = True
except ImportError:
    DIRECT_MODE = False

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000/api/v1"

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuBot — PDF Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* Gradient header */
.hero-title {
    background: linear-gradient(135deg, #6C63FF 0%, #3EC6E0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.6rem;
    font-weight: 800;
    margin-bottom: 0;
}
.hero-sub {
    color: #9CA3AF;
    font-size: 1rem;
    margin-top: 0;
}

/* Chat bubbles */
.user-bubble {
    background: #2D3748;
    border-left: 4px solid #6C63FF;
    border-radius: 0 12px 12px 0;
    padding: 12px 16px;
    margin: 8px 0;
}
.bot-bubble {
    background: #1A2035;
    border-left: 4px solid #3EC6E0;
    border-radius: 0 12px 12px 0;
    padding: 12px 16px;
    margin: 8px 0;
}
.citation-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.85rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}
.confidence-bar-wrap {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 6px;
    height: 8px;
    width: 100%;
    margin-top: 4px;
}
.stat-chip {
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.78rem;
    display: inline-block;
    color: #E2E8F0;
    margin: 2px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state initialisation ──────────────────────────────────────────────
def _init_state():
    defaults = {
        "session_id":    uuid.uuid4().hex[:12],
        "doc_id":        None,
        "filename":      None,
        "doc_meta":      None,      # full UploadResponse dict
        "summary":       None,
        "chat_history":  [],        # list of {role, content, meta}
        "uploading":     False,
        "api_error":     None,
        "uploaded_docs": [],        # list of dicts: {"doc_id", "filename", "doc_meta", "summary"}
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── API helpers ───────────────────────────────────────────────────────────────

def api_upload(file_bytes: bytes, filename: str) -> dict:
    if DIRECT_MODE:
        pdf_processor = PDFProcessor()
        chunks, meta = pdf_processor.process(file_bytes, filename)
        
        # Check if already indexed
        already_indexed = rag_engine.doc_exists(meta.doc_id)
        rag_engine.ingest(chunks, meta.doc_id)
        
        # Check cached summary first
        from config import UPLOAD_DIR
        summary_path = Path(UPLOAD_DIR) / f"{meta.doc_id}.summary"
        if summary_path.exists():
            summary = summary_path.read_text(encoding="utf-8")
        else:
            full_text = pdf_processor.extract_full_text(file_bytes)
            summary = rag_engine.summarise(full_text, filename)
            try:
                summary_path.write_text(summary, encoding="utf-8")
            except Exception:
                pass
        
        # Set active doc
        memory_manager.set_active_doc(st.session_state.session_id, meta.doc_id, filename)
        
        return {
            "doc_id": meta.doc_id,
            "session_id": st.session_state.session_id,
            "filename": filename,
            "total_pages": meta.total_pages,
            "total_chunks": meta.total_chunks,
            "file_size_kb": meta.file_size_kb,
            "word_count": meta.word_count,
            "title": meta.title,
            "author": meta.author,
            "already_indexed": already_indexed,
            "summary": summary,
        }
    else:
        resp = requests.post(
            f"{API_BASE}/upload",
            files={"file": (filename, file_bytes, "application/pdf")},
            data={"session_id": st.session_state.session_id},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()


def api_chat(question: str) -> dict:
    if DIRECT_MODE:
        t0 = time.perf_counter()
        rag_resp = rag_engine.answer(
            question=question,
            doc_id=st.session_state.doc_id,
            session_id=st.session_state.session_id,
            filename=st.session_state.filename or "document",
        )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        label, emoji = confidence_label(rag_resp.confidence)
        
        return {
            "answer": rag_resp.answer,
            "sources": [
                {
                    "page_number": s.page_number,
                    "chunk_index": s.chunk_index,
                    "excerpt": s.excerpt,
                    "relevance_score": s.relevance_score,
                }
                for s in rag_resp.sources
            ],
            "confidence": rag_resp.confidence,
            "confidence_label": label,
            "confidence_emoji": emoji,
            "tokens_used": rag_resp.tokens_used,
            "fallback": rag_resp.fallback,
            "response_time_ms": elapsed_ms,
        }
    else:
        payload = {
            "question":   question,
            "doc_id":     st.session_state.doc_id,
            "session_id": st.session_state.session_id,
            "filename":   st.session_state.filename or "document",
        }
        resp = requests.post(f"{API_BASE}/chat", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()


def api_clear_session():
    if DIRECT_MODE:
        memory_manager.clear_session(st.session_state.session_id)
    else:
        requests.get(
            f"{API_BASE}/sessions/{st.session_state.session_id}/clear",
            timeout=10,
        )


def api_health() -> bool:
    if DIRECT_MODE:
        return True
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def api_delete_document(doc_id: str) -> dict:
    if DIRECT_MODE:
        db_deleted = rag_engine.delete_doc(doc_id)
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
            "db_deleted": db_deleted,
            "cache_deleted": cache_deleted,
        }
    else:
        resp = requests.delete(f"{API_BASE}/documents/{doc_id}", timeout=20)
        resp.raise_for_status()
        return resp.json()


def api_get_chunks(doc_id: str) -> list[dict]:
    if DIRECT_MODE:
        return rag_engine.get_all_chunks(doc_id)
    else:
        try:
            resp = requests.get(f"{API_BASE}/documents/{doc_id}/chunks", timeout=20)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return []


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📄 DocuBot")
    st.caption("AI-Powered PDF Intelligence")

    st.divider()

    # API health indicator
    healthy = api_health()
    if healthy:
        st.success("🟢 API connected", icon=None)
    else:
        st.error("🔴 API offline — start backend first", icon=None)
        st.code("uvicorn main:app --reload --port 8000", language="bash")

    st.divider()

    # File uploader
    st.markdown("### Upload PDF")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        help="Max 500 MB per file",
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    if uploaded_files and healthy:
        if st.button("📥 Ingest Documents", use_container_width=True, type="primary"):
            with st.spinner("Parsing, chunking, embedding documents…"):
                try:
                    for f in uploaded_files:
                        # Check if already uploaded in this session to avoid redundancy
                        if any(d["filename"] == f.name for d in st.session_state.uploaded_docs):
                            continue
                        
                        file_bytes = f.read()
                        result = api_upload(file_bytes, f.name)
                        
                        st.session_state.uploaded_docs.append({
                            "doc_id": result["doc_id"],
                            "filename": result["filename"],
                            "doc_meta": result,
                            "summary": result["summary"],
                        })
                    
                    # Update active document variables (comma-separated for multi-doc)
                    if st.session_state.uploaded_docs:
                        st.session_state.doc_id = ",".join(d["doc_id"] for d in st.session_state.uploaded_docs)
                        st.session_state.filename = ", ".join(d["filename"] for d in st.session_state.uploaded_docs)
                        # Store metadata/summary of the last uploaded for metrics
                        st.session_state.doc_meta = st.session_state.uploaded_docs[-1]["doc_meta"]
                        st.session_state.summary = st.session_state.uploaded_docs[-1]["summary"]
                        
                    st.session_state.api_error = None
                    st.success("✅ Ready to chat with all documents!")
                    st.rerun()
                except Exception as e:
                    if type(e).__name__ in ("RerunException", "StopException"):
                        raise e
                    if hasattr(e, "response") and e.response is not None:
                        try:
                            detail = e.response.json().get("detail", str(e))
                        except Exception:
                            detail = str(e)
                    else:
                        detail = str(e)
                    st.session_state.api_error = detail
                    st.error(f"Ingestion failed: {detail}")

    # Document stats
    if st.session_state.uploaded_docs:
        st.divider()
        st.markdown("### 📊 Active Documents")
        for doc in st.session_state.uploaded_docs:
            st.caption(f"📄 **{doc['filename']}** ({doc['doc_meta']['total_pages']} pgs)")
        
        st.divider()
        st.markdown("### 📊 Total Document Stats")
        total_pages = sum(d["doc_meta"]["total_pages"] for d in st.session_state.uploaded_docs)
        total_chunks = sum(d["doc_meta"]["total_chunks"] for d in st.session_state.uploaded_docs)
        total_words = sum(d["doc_meta"]["word_count"] for d in st.session_state.uploaded_docs)
        total_kb = sum(d["doc_meta"]["file_size_kb"] for d in st.session_state.uploaded_docs)
        
        col1, col2 = st.columns(2)
        col1.metric("Pages",  total_pages)
        col2.metric("Chunks", total_chunks)
        col1.metric("Words",  f"{total_words:,}")
        col2.metric("Size",   f"{total_kb:.1f} KB")

    st.divider()

    # Clear conversation
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        api_clear_session()
        st.session_state.chat_history = []
        st.toast("Conversation cleared.")

    # Delete Document Embeddings
    if st.session_state.doc_id:
        if st.button("🚨 Delete Document Embeddings", use_container_width=True, type="secondary"):
            with st.spinner("Deleting document embeddings and cache..."):
                try:
                    api_delete_document(st.session_state.doc_id)
                    # Reset state
                    st.session_state.doc_id = None
                    st.session_state.filename = None
                    st.session_state.doc_meta = None
                    st.session_state.summary = None
                    st.session_state.uploaded_docs = []
                    st.session_state.chat_history = []
                    st.success("Document embeddings deleted!")
                    st.rerun()
                except Exception as e:
                    if type(e).__name__ in ("RerunException", "StopException"):
                        raise e
                    st.error(f"Deletion failed: {e}")

    # Session ID (for debugging)
    with st.expander("🔧 Session Info"):
        st.code(f"session: {st.session_state.session_id}\ndoc_id:  {st.session_state.doc_id or 'none'}")


# ── Main panel ────────────────────────────────────────────────────────────────

st.markdown('<p class="hero-title">DocuBot</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Upload any PDF → Ask anything → Get cited, confident answers</p>',
    unsafe_allow_html=True,
)
st.divider()

# ── Main Interface ────────────────────────────────────────────────────────────
if not st.session_state.doc_id:
    st.info(
        "👈 Upload a PDF in the sidebar to get started. "
        "You can ask questions like:\n"
        "- *What is the main argument of this paper?*\n"
        "- *Summarise section 3.*\n"
        "- *What methodology was used?*",
        icon="💡",
    )
else:
    # Set up the dashboard tabs
    tab_chat, tab_overview, tab_chunks = st.tabs([
        "💬 Q&A Chatbot Assistant", 
        "📋 Document Insights & Summary", 
        "🧩 Vector Index Chunk Explorer"
    ])

    # ── Tab 1: Q&A Chatbot ───────────────────────────────────────────────────
    with tab_chat:
        # Render existing chat history
        for msg in st.session_state.chat_history:
            role = msg["role"]
            with st.chat_message(role):
                st.markdown(msg["content"])
                
                if role == "assistant":
                    # Confidence + timing badge
                    meta = msg.get("meta", {})
                    if meta:
                        label, emoji = confidence_label(meta.get("confidence", 0))
                        badge_html = (
                            f'<div style="margin-top: 8px; display: flex; flex-wrap: wrap; gap: 6px;">'
                            f'<span class="stat-chip">{emoji} {label} confidence</span>'
                            f'<span class="stat-chip">⏱ {meta.get("response_time_ms", 0)}ms</span>'
                            f'<span class="stat-chip">🔤 {meta.get("tokens_used", 0)} tokens</span>'
                            f'</div>'
                        )
                        st.markdown(badge_html, unsafe_allow_html=True)

                    # Citations
                    sources = msg.get("sources", [])
                    if sources:
                        with st.expander(f"📎 {len(sources)} source(s) cited", expanded=False):
                            for i, src in enumerate(sources, 1):
                                rel_pct = int(src["relevance_score"] * 100)
                                st.markdown(
                                    f'<div class="citation-card">'
                                    f'<b>Source {i}</b> · Page {src["page_number"]} · '
                                    f'Relevance: {rel_pct}%<br>'
                                    f'<i>"{src["excerpt"][:180]}…"</i>'
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

        # Suggested questions (only when history is empty)
        if not st.session_state.chat_history:
            st.markdown("#### 💬 Try asking…")
            suggestions = [
                "What is this document about?",
                "What are the key findings or conclusions?",
                "List the main topics covered.",
                "What methodology or approach is used?",
            ]
            cols = st.columns(2)
            for i, suggestion in enumerate(suggestions):
                if cols[i % 2].button(suggestion, key=f"sug_{i}", use_container_width=True):
                    st.session_state["_prefill"] = suggestion
                    st.rerun()

        # Input box
        prefill = st.session_state.pop("_prefill", "")
        question = st.chat_input(
            placeholder=f"Ask anything about '{st.session_state.filename}'…",
        )
        if prefill:
            question = prefill

        if question:
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": question})

            with st.spinner("Retrieving context and generating answer…"):
                try:
                    result = api_chat(question)
                    st.session_state.chat_history.append(
                        {
                            "role":    "assistant",
                            "content": result["answer"],
                            "sources": result.get("sources", []),
                            "meta": {
                                "confidence":       result["confidence"],
                                "confidence_label": result["confidence_label"],
                                "tokens_used":      result["tokens_used"],
                                "response_time_ms": result["response_time_ms"],
                                "fallback":         result["fallback"],
                            },
                        }
                    )
                except Exception as e:
                    if type(e).__name__ in ("RerunException", "StopException"):
                        raise e
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": f"⚠️ Error: {str(e)}", "sources": [], "meta": {}}
                    )

            st.rerun()

    # ── Tab 2: Document Summary ──────────────────────────────────────────────
    with tab_overview:
        st.markdown("### 📄 Document Analysis Overview")
        
        # Inline stats grid
        if st.session_state.uploaded_docs:
            total_pages = sum(d["doc_meta"]["total_pages"] for d in st.session_state.uploaded_docs)
            total_chunks = sum(d["doc_meta"]["total_chunks"] for d in st.session_state.uploaded_docs)
            total_words = sum(d["doc_meta"]["word_count"] for d in st.session_state.uploaded_docs)
            total_kb = sum(d["doc_meta"]["file_size_kb"] for d in st.session_state.uploaded_docs)
            
            meta_grid = (
                f'<div style="display: flex; gap: 16px; margin: 12px 0 24px 0; flex-wrap: wrap;">'
                f'<div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 12px 20px; min-width: 120px; text-align: center;">'
                f'<div style="font-size: 0.8rem; color: #9CA3AF;">Total Pages</div><div style="font-size: 1.5rem; font-weight: bold; color: #3EC6E0;">{total_pages}</div>'
                f'</div>'
                f'<div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 12px 20px; min-width: 120px; text-align: center;">'
                f'<div style="font-size: 0.8rem; color: #9CA3AF;">Total Chunks</div><div style="font-size: 1.5rem; font-weight: bold; color: #6C63FF;">{total_chunks}</div>'
                f'</div>'
                f'<div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 12px 20px; min-width: 120px; text-align: center;">'
                f'<div style="font-size: 0.8rem; color: #9CA3AF;">Total Words</div><div style="font-size: 1.5rem; font-weight: bold; color: #E2E8F0;">{total_words:,}</div>'
                f'</div>'
                f'<div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 12px 20px; min-width: 120px; text-align: center;">'
                f'<div style="font-size: 0.8rem; color: #9CA3AF;">Total Size</div><div style="font-size: 1.5rem; font-weight: bold; color: #E2E8F0;">{total_kb:.1f} KB</div>'
                f'</div>'
                f'</div>'
            )
            st.markdown(meta_grid, unsafe_allow_html=True)
            
            # Show individual summaries
            for doc in st.session_state.uploaded_docs:
                with st.expander(f"📋 Summary: {doc['filename']}", expanded=True):
                    st.markdown(
                        f'<div style="background: rgba(255, 255, 255, 0.01); border-left: 3px solid #6C63FF; padding: 8px 12px;">'
                        f'{doc["summary"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

    # ── Tab 3: Chunk Explorer ────────────────────────────────────────────────
    with tab_chunks:
        st.markdown("### 🧩 Vector Index Chunks")
        st.caption("Inspect the exact fragments of text indexed inside ChromaDB:")
        
        try:
            chunks = api_get_chunks(st.session_state.doc_id)
            search_term = st.text_input("🔍 Search within indexed segments", "")
            if search_term:
                filtered_chunks = [c for c in chunks if search_term.lower() in c["text"].lower()]
            else:
                filtered_chunks = chunks
                
            st.write(f"Showing {len(filtered_chunks)} of {len(chunks)} parsed segments:")
            for c in filtered_chunks[:50]:  # Cap display at 50 for safety
                st.markdown(
                    f'<div class="citation-card">'
                    f'<div style="display: flex; justify-content: space-between; font-weight: bold; margin-bottom: 6px; font-size: 0.82rem; color: #9CA3AF;">'
                    f'<span>Chunk #{c["chunk_index"]}</span>'
                    f'<span>Page {c["page_number"]}</span>'
                    f'</div>'
                    f'<div style="font-family: monospace; font-size: 0.82rem; color: #CBD5E1; line-height: 1.4;">'
                    f'{c["text"]}'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            if len(filtered_chunks) > 50:
                st.info("💡 Display capped at top 50 chunks. Refine search query to see other chunks.")
        except Exception as e:
            st.error(f"Could not load chunks: {e}")
