# 📄 DocuBot — AI-Powered PDF Intelligence Platform

> **RAG · LangChain · ChromaDB · FastAPI · Streamlit · Claude AI**

DocuBot is a production-grade Retrieval-Augmented Generation (RAG) system that lets users upload any PDF and have a natural, contextually-aware conversation with its contents. Built with enterprise features: multi-PDF sessions, smart chunking, citation tracking, conversation memory, and confidence scoring.

---

## 🚀 Features

| Feature | Description |
|---|---|
| **Multi-PDF Upload** | Upload and switch between multiple documents in one session |
| **Citation Tracking** | Every answer cites the exact page & chunk it came from |
| **Conversation Memory** | Follow-up questions maintain context from prior turns |
| **Confidence Scoring** | Each answer shows a retrieval confidence score |
| **Smart Chunking** | Recursive character splitter with overlap for coherent context |
| **Document Summary** | Auto-generates a summary when you first upload a PDF |
| **Session Persistence** | ChromaDB persists embeddings so re-uploads are instant |
| **REST API** | FastAPI backend — fully decoupled from the UI |
| **Dark/Light Theme** | Clean Streamlit UI with custom styling |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit UI                       │
│  (Upload · Chat · Citations · Confidence · History)  │
└────────────────────┬────────────────────────────────┘
                     │ HTTP (REST)
┌────────────────────▼────────────────────────────────┐
│                  FastAPI Backend                      │
│   /upload  /chat  /summary  /sessions  /health       │
└──────┬─────────────────────────┬────────────────────┘
       │                         │
┌──────▼──────┐         ┌────────▼────────┐
│  RAG Engine │         │  ChromaDB       │
│  LangChain  │◄───────►│  Vector Store   │
│  + Memory   │         │  (Persistent)   │
└──────┬──────┘         └─────────────────┘
       │
┌──────▼──────┐
│  Claude API │
│  (claude-   │
│  sonnet-4)  │
└─────────────┘
```

---

## 📁 Project Structure

```
docubot/
├── backend/
│   ├── api/
│   │   └── routes.py          # FastAPI route handlers
│   ├── core/
│   │   ├── rag_engine.py      # RAG pipeline (embed, retrieve, generate)
│   │   ├── pdf_processor.py   # PDF parsing + smart chunking
│   │   └── memory_manager.py  # Conversation memory per session
│   └── utils/
│       └── helpers.py         # Confidence scoring, formatting
├── frontend/
│   └── app.py                 # Streamlit UI
├── tests/
│   └── test_rag.py            # Unit tests
├── .streamlit/
│   └── config.toml            # Streamlit theme config
├── main.py                    # FastAPI app entrypoint
├── config.py                  # Central config (env vars)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone & Install

```bash
git clone https://github.com/yourname/docubot
cd docubot
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file:

```env
ANTHROPIC_API_KEY=sk-ant-...
CHROMA_PERSIST_DIR=./chroma_db
MAX_CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5
```

### 3. Run the Backend

```bash
uvicorn main:app --reload --port 8000
```

### 4. Run the Frontend

```bash
streamlit run frontend/app.py
```

### 5. Open in Browser

- **UI:** http://localhost:8501
- **API Docs:** http://localhost:8000/docs

---

## 🧠 How RAG Works Here

1. **Ingest** — PDF is parsed with `pypdf`, split into overlapping chunks
2. **Embed** — Each chunk is embedded using `sentence-transformers` (local, free)
3. **Store** — Embeddings stored in ChromaDB with metadata (page, chunk index)
4. **Retrieve** — User query is embedded → cosine similarity search → top-K chunks
5. **Generate** — Retrieved chunks + conversation history → Claude prompt → answer
6. **Cite** — Source page numbers returned alongside every answer

---

## 🔑 Key Technical Decisions

- **ChromaDB over Pinecone/Weaviate** — zero cost, local-first, production-upgradeable
- **Claude Sonnet** — Best cost/quality tradeoff for document Q&A
- **Sentence Transformers** — Local embeddings, no API cost, fast
- **FastAPI** — Async, auto-docs, production-ready; not Flask
- **Session-scoped collections** — Multi-user safe; each session gets its own namespace

---

## 📊 Resume Talking Points

- Built end-to-end RAG pipeline covering ingestion → embedding → retrieval → generation
- Implemented citation-aware answers with page-level provenance tracking
- Designed stateful conversation memory with session isolation
- Engineered confidence scoring via cosine distance thresholding
- Decoupled UI from backend via REST API for independent scaling
