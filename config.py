"""
config.py — Central configuration for DocuBot.
Reads from environment variables with sensible defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_db"))
UPLOAD_DIR = os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads"))

# ── API Keys ─────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Chunking ─────────────────────────────────────────────────────────────────
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP",  "200"))

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K_RESULTS          = int(os.getenv("TOP_K_RESULTS", "5"))
MIN_CONFIDENCE         = float(os.getenv("MIN_CONFIDENCE", "0.3"))  # 0–1; lower = more confident

# ── Embedding model (local ONNX via ChromaDB default — no internet required) ──
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")  # informational only

# ── Claude model ─────────────────────────────────────────────────────────────
CLAUDE_MODEL   = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
MAX_TOKENS     = int(os.getenv("MAX_TOKENS", "1500"))

# ── Memory ───────────────────────────────────────────────────────────────────
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "10"))

# Ensure upload dir exists
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
