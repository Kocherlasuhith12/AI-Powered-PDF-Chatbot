"""
backend/utils/helpers.py
Utility functions used across the backend.
"""
from __future__ import annotations
import uuid


def generate_session_id() -> str:
    """Generate a short random session identifier."""
    return uuid.uuid4().hex[:12]


def confidence_label(score: float) -> tuple[str, str]:
    """
    Map a 0–1 confidence score to a human label and emoji.
    Returns (label, emoji).
    """
    if score >= 0.75:
        return "High", "🟢"
    elif score >= 0.50:
        return "Medium", "🟡"
    elif score >= 0.30:
        return "Low", "🟠"
    else:
        return "Very Low", "🔴"


def format_file_size(size_bytes: int) -> str:
    """Return a human-readable file size string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / 1024 ** 2:.1f} MB"


def truncate(text: str, max_chars: int = 150) -> str:
    """Truncate text to *max_chars*, appending '…' if needed."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"
