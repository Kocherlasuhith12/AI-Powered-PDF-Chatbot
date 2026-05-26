"""
main.py — FastAPI application entrypoint for DocuBot.

Run with:
    uvicorn main:app --reload --port 8000

API docs available at:
    http://localhost:8000/docs
"""
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import sys
sys.path.insert(0, str(Path(__file__).parent))

from backend.api.routes import router

app = FastAPI(
    title="DocuBot API",
    description=(
        "AI-Powered PDF Intelligence Platform. "
        "Upload any PDF and chat with its contents using RAG + Claude."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow Streamlit frontend to call the API in development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["root"])
async def root():
    return {
        "service": "DocuBot API",
        "version": "1.0.0",
        "docs": "/docs",
    }
