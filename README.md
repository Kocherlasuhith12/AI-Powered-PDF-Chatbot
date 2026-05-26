<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.45-FF4B4B?style=flat-square&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/ChromaDB-0.6-6C63FF?style=flat-square"/>
  <img src="https://img.shields.io/badge/Claude_Sonnet_4-Anthropic-191919?style=flat-square"/>
  <img src="https://img.shields.io/badge/Tests-26%20passed-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square"/>
</p>

<h1 align="center">📄 DocuBot — AI-Powered PDF Intelligence Platform</h1>

<p align="center">
  Upload any PDF. Ask anything. Get cited, confident answers — powered by RAG + Claude AI.
</p>

---

## What is this?

DocuBot is a full-stack Retrieval-Augmented Generation (RAG) system that lets you upload any PDF and have a natural, context-aware conversation with its contents. Unlike a basic "chat with PDF" tool, DocuBot adds **citation tracking**, **confidence scoring**, **conversation memory**, and an **auto-generated document summary** — all served through a clean Streamlit UI backed by a production-ready FastAPI REST API.

Every answer tells you *which page it came from* and *how confident the retrieval was*, so you always know whether to trust the response.

---

## Live Demo

> Start the backend, then the frontend, upload any PDF, and start asking questions.

```
Backend  →  http://localhost:8000/docs   (Swagger UI)
Frontend →  http://localhost:8501
```

---

## Features

| Feature | Details |
|---|---|
| **Cited answers** | Every response includes page numbers and source excerpts |
| **Confidence scoring** | Cosine similarity → 0–1 score with 🟢🟡🟠🔴 indicator |
| **Conversation memory** | Follow-up questions maintain context across turns |
| **Auto document summary** | Structured overview generated on every upload |
| **Idempotent ingestion** | Re-uploading the same PDF skips re-embedding (SHA-256 cache) |
| **Smart chunking** | Recursive splitter with overlap — no sentence ever gets cut at a boundary |
| **Local embeddings** | ONNX-based embedder via ChromaDB — zero API cost, runs offline |
| **Session isolation** | Each browser session gets its own conversation namespace |
| **REST API** | Fully documented FastAPI backend, decoupled from the UI |
| **26 unit tests** | Covers chunking, memory, confidence scoring, and helpers |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Streamlit UI                            │
│   Upload · Chat · Citations · Confidence · Summary · History │
└──────────────────────────┬──────────────────────────────────┘
                           │  HTTP REST
┌──────────────────────────▼──────────────────────────────────┐
│                     FastAPI Backend                           │
│          /upload    /chat    /health    /sessions             │
└──────┬───────────────────────────────┬───────────────────────┘
       │                               │
┌──────▼──────────┐          ┌─────────▼──────────┐
│   RAG Engine    │          │    ChromaDB         │
│   LangChain     │◄────────►│    Vector Store     │
│   + Memory Mgr  │          │    (Persistent)     │
└──────┬──────────┘          └────────────────────┘
       │
┌──────▼──────────┐
│   Claude API    │
│   Sonnet 4      │
└─────────────────┘
```

### Request lifecycle

```
User question
    │
    ▼
Embed question (ONNX)
    │
    ▼
Cosine search in ChromaDB  →  Top-5 most relevant chunks
    │
    ▼
Confidence score = 1 − mean_cosine_distance (top 3 chunks)
    │
    ├── score < 0.30  →  Fallback: "I couldn't find a confident answer"
    │
    └── score ≥ 0.30  →  Build prompt: [system] + [context chunks] + [history] + [question]
                              │
                              ▼
                         Claude Sonnet 4
                              │
                              ▼
                     Answer + citations returned
                              │
                              ▼
                    Saved to session memory  →  Available for follow-up questions
```

---

## Project Structure

```
docubot/
│
├── main.py                        # FastAPI app entrypoint + CORS middleware
├── config.py                      # All env vars in one place with defaults
├── requirements.txt               # Pinned dependency versions
├── .env.example                   # Copy to .env and add your API key
│
├── backend/
│   ├── core/
│   │   ├── pdf_processor.py       # PDF parsing, text cleaning, recursive chunking
│   │   ├── rag_engine.py          # Embed · retrieve · score · generate · summarise
│   │   └── memory_manager.py      # Per-session conversation history (in-memory)
│   ├── api/
│   │   └── routes.py              # POST /upload  POST /chat  GET /health  GET /sessions
│   └── utils/
│       └── helpers.py             # confidence_label(), format_file_size(), truncate()
│
├── frontend/
│   └── app.py                     # Streamlit UI: uploader, chat bubbles, citations
│
├── tests/
│   └── test_rag.py                # 26 unit tests across all core modules
│
└── .streamlit/
    └── config.toml                # Dark theme + 50 MB upload limit
```

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/docubot.git
cd docubot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Open `.env` and set your key:

```env
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Everything else has sensible defaults — no other changes needed to get started.

### 4. Start the backend

```bash
uvicorn main:app --reload --port 8000
```

The API is now live at `http://localhost:8000`. Open `http://localhost:8000/docs` to see the Swagger UI.

### 5. Start the frontend

Open a second terminal:

```bash
streamlit run frontend/app.py
```

Open `http://localhost:8501` in your browser.

### 6. Use it

1. Click **Browse files** in the sidebar and select any PDF
2. Click **Ingest Document** — DocuBot parses, chunks, embeds, and shows a summary
3. Type your first question in the chat input
4. Expand the **Sources** dropdown under any answer to see which pages were cited

---

## API Reference

All endpoints are prefixed with `/api/v1`.

### `POST /upload`

Upload and ingest a PDF. Re-uploading the same file (same SHA-256) skips re-embedding.

**Request:** `multipart/form-data`
| Field | Type | Description |
|---|---|---|
| `file` | `File` | The PDF to ingest (max 50 MB) |
| `session_id` | `string` (optional) | Pass an existing ID to continue a session; omit to generate a new one |

**Response:**
```json
{
  "doc_id": "a3f2c1b9e4d7",
  "session_id": "9f2e4a1c",
  "filename": "research_paper.pdf",
  "total_pages": 24,
  "total_chunks": 187,
  "file_size_kb": 412.5,
  "word_count": 14823,
  "title": "Attention Is All You Need",
  "author": "Vaswani et al.",
  "already_indexed": false,
  "summary": "**Document Overview**\n..."
}
```

---

### `POST /chat`

Ask a question about an ingested document.

**Request body:**
```json
{
  "question": "What evaluation metric was used?",
  "doc_id": "a3f2c1b9e4d7",
  "session_id": "9f2e4a1c",
  "filename": "research_paper.pdf"
}
```

**Response:**
```json
{
  "answer": "The authors used BLEU score as the primary evaluation metric [Page 8]...",
  "sources": [
    {
      "page_number": 8,
      "chunk_index": 63,
      "excerpt": "We evaluate on the WMT 2014 English-to-German translation task...",
      "relevance_score": 0.847
    }
  ],
  "confidence": 0.831,
  "confidence_label": "High",
  "confidence_emoji": "🟢",
  "tokens_used": 743,
  "fallback": false,
  "response_time_ms": 1842
}
```

---

### `GET /sessions/{session_id}/clear`

Clear the conversation history for a session without deleting the indexed document.

### `GET /health`

Returns `{ "status": "ok", "version": "1.0.0" }`. Use for uptime monitoring.

---

## Configuration

All settings live in `.env`. The table below shows every option with its default.

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | Your Anthropic API key |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Where ChromaDB stores vector data on disk |
| `MAX_CHUNK_SIZE` | `1000` | Maximum characters per chunk |
| `CHUNK_OVERLAP` | `200` | Character overlap between adjacent chunks |
| `TOP_K_RESULTS` | `5` | Number of chunks retrieved per query |
| `MIN_CONFIDENCE` | `0.3` | Below this score, answer with a fallback message |
| `CLAUDE_MODEL` | `claude-sonnet-4-20250514` | Claude model used for generation and summarisation |
| `MAX_TOKENS` | `1500` | Maximum tokens in Claude's response |
| `MAX_HISTORY_TURNS` | `10` | Conversation turns kept in memory per session |

---

## How the RAG Pipeline Works

### 1. Ingestion

When you upload a PDF, `PDFProcessor` extracts text page by page using `pypdf`, cleans it (normalises whitespace, strips non-printable characters), and passes it through LangChain's `RecursiveCharacterTextSplitter`. The splitter tries to break at paragraph boundaries first, then sentences, then words — ensuring no chunk cuts a sentence in half. Each chunk is tagged with its page number and position.

### 2. Embedding

Each chunk is converted to a 384-dimensional vector by ChromaDB's built-in ONNX embedding model (based on `all-MiniLM-L6-v2`). This runs entirely locally — no external API call, no cost, no rate limits. Vectors are stored in a per-document ChromaDB collection keyed by the document's SHA-256 hash.

### 3. Retrieval

Your question is embedded using the same model. ChromaDB performs an approximate nearest-neighbour search using cosine similarity (HNSW index) and returns the top-5 most relevant chunks along with their cosine distances. Confidence is computed as `1 − mean_distance` of the top 3 results — closer to 1.0 means the retrieved chunks are semantically very close to your question.

### 4. Generation

Retrieved chunks are assembled into a structured context block with page and relevance metadata. The last N conversation turns are appended as history. This full context is sent to Claude Sonnet 4 with a strict system prompt that instructs it to cite sources, stay grounded in the provided text, and maintain conversational continuity.

### 5. Memory

Each session has its own `SessionMemory` object that stores all `(user, assistant)` turn pairs. When you switch documents mid-session, the history is automatically cleared to avoid cross-document confusion. The memory store is in-process — for multi-instance deployments, swap it for Redis.

---

## Running the Tests

```bash
python -m pytest tests/ -v
```

```
tests/test_rag.py::TestCleanText::test_collapses_multiple_newlines PASSED
tests/test_rag.py::TestCleanText::test_strips_whitespace PASSED
tests/test_rag.py::TestCleanText::test_removes_non_printable PASSED
tests/test_rag.py::TestDocId::test_stable_hash PASSED
tests/test_rag.py::TestDocId::test_different_data_different_id PASSED
tests/test_rag.py::TestDocId::test_16_chars PASSED
...
============================== 26 passed in 6.03s ==============================
```

Test coverage spans: text cleaning, document ID generation, PDF rejection of invalid inputs, session memory lifecycle, conversation history formatting, document-switch clearing, confidence label thresholds, file size formatting, and text truncation.

---

## Design Decisions

**Why ChromaDB over Pinecone or Weaviate?**
Zero infrastructure cost, runs embedded in the Python process, persists to disk automatically, and the API is identical to cloud vector DBs — so migrating later requires changing one line.

**Why local ONNX embeddings over OpenAI's `text-embedding-ada-002`?**
No API cost, no latency, no rate limits, and the quality difference for document Q&A tasks is negligible. The model runs in under 100ms per batch on a CPU.

**Why FastAPI instead of Flask?**
Async request handling, automatic OpenAPI documentation, Pydantic validation on every request/response, and it scales to production with no changes.

**Why session-scoped ChromaDB collections?**
Each document gets its own named collection (`doc_{sha256_hash}`). This means multiple users can chat with different documents simultaneously without any cross-contamination, and the same document is only embedded once regardless of how many sessions use it.

**Why clear memory on document switch?**
Conversation history from document A is misleading context when the user switches to document B. Clearing on switch prevents Claude from referencing an answer grounded in the wrong document.

---

## Deployment

### Docker (recommended)

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t docubot .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... docubot
```

### Environment notes for production

- Mount a persistent volume to `CHROMA_PERSIST_DIR` so embeddings survive container restarts
- Replace the in-memory `MemoryManager` with Redis for multi-instance deployments
- Set `CORS` origins in `main.py` to your actual frontend domain
- The Streamlit frontend can be deployed separately to Streamlit Community Cloud

---

## Roadmap

- [ ] OCR support for scanned PDFs (Tesseract integration)
- [ ] Multi-PDF cross-document queries
- [ ] Redis-backed session memory for horizontal scaling
- [ ] Streaming responses via Server-Sent Events
- [ ] User authentication and per-user document libraries
- [ ] Export conversation as PDF or Markdown

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Anthropic Claude Sonnet 4 |
| Orchestration | LangChain (text splitting) |
| Vector DB | ChromaDB 0.6 (persistent, local) |
| Embeddings | ONNX all-MiniLM-L6-v2 (via ChromaDB default) |
| Backend | FastAPI 0.115 + Uvicorn |
| Frontend | Streamlit 1.45 |
| PDF parsing | pypdf 5.9 |
| Testing | pytest 8.3 |
| Config | python-dotenv |

---

## License

MIT — free to use, modify, and deploy.

---

<p align="center">Built with Python · FastAPI · Streamlit · ChromaDB · Claude AI</p>
