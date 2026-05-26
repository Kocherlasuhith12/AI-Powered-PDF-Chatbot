"""
tests/test_rag.py
Unit tests for DocuBot's core components.

Run with:
    python -m pytest tests/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from backend.core.pdf_processor import PDFProcessor, _clean_text, _compute_doc_id
from backend.core.memory_manager import MemoryManager, SessionMemory
from backend.utils.helpers import confidence_label, format_file_size, truncate


# ── PDF Processor tests ───────────────────────────────────────────────────────

class TestCleanText:
    def test_collapses_multiple_newlines(self):
        raw = "Hello\n\n\n\nWorld"
        assert "\n\n\n" not in _clean_text(raw)

    def test_strips_whitespace(self):
        assert _clean_text("  hello  ") == "hello"

    def test_removes_non_printable(self):
        result = _clean_text("hello\x00world")
        assert "\x00" not in result


class TestDocId:
    def test_stable_hash(self):
        data = b"hello world"
        assert _compute_doc_id(data) == _compute_doc_id(data)

    def test_different_data_different_id(self):
        assert _compute_doc_id(b"abc") != _compute_doc_id(b"xyz")

    def test_16_chars(self):
        assert len(_compute_doc_id(b"test")) == 16


class TestPDFProcessor:
    def test_rejects_empty_bytes(self):
        """Processing empty bytes should raise an error (not a valid PDF)."""
        processor = PDFProcessor()
        with pytest.raises(Exception):
            processor.process(b"", "empty.pdf")

    def test_rejects_non_pdf(self):
        """A plain text file is not a valid PDF."""
        processor = PDFProcessor()
        with pytest.raises(Exception):
            processor.process(b"This is not a PDF", "fake.pdf")


# ── Memory Manager tests ──────────────────────────────────────────────────────

class TestMemoryManager:
    def test_get_creates_session(self):
        mgr = MemoryManager()
        mem = mgr.get("sess_1")
        assert isinstance(mem, SessionMemory)
        assert mem.session_id == "sess_1"

    def test_add_exchange(self):
        mgr = MemoryManager()
        mgr.add_exchange("sess_2", "What is this?", "This is a test document.")
        mem = mgr.get("sess_2")
        assert len(mem.turns) == 2

    def test_clear_session(self):
        mgr = MemoryManager()
        mgr.add_exchange("sess_3", "Q", "A")
        mgr.clear_session("sess_3")
        assert len(mgr.get("sess_3").turns) == 0

    def test_history_text_format(self):
        mgr = MemoryManager()
        mgr.add_exchange("sess_4", "Hello", "Hi there!")
        text = mgr.get("sess_4").get_history_text()
        assert "User: Hello" in text
        assert "Assistant: Hi there!" in text

    def test_set_active_doc_clears_on_change(self):
        mgr = MemoryManager()
        mgr.add_exchange("sess_5", "Q", "A")
        mgr.set_active_doc("sess_5", "doc_old", "old.pdf")
        mgr.add_exchange("sess_5", "Q2", "A2")
        mgr.set_active_doc("sess_5", "doc_new", "new.pdf")  # changed doc
        assert len(mgr.get("sess_5").turns) == 0

    def test_max_history_turns(self):
        mgr = MemoryManager()
        for i in range(20):
            mgr.add_exchange("sess_6", f"Q{i}", f"A{i}")
        messages = mgr.get("sess_6").get_messages(max_turns=3)
        assert len(messages) <= 6  # 3 user + 3 assistant


# ── Helpers tests ─────────────────────────────────────────────────────────────

class TestConfidenceLabel:
    def test_high(self):
        label, emoji = confidence_label(0.85)
        assert label == "High"
        assert emoji == "🟢"

    def test_medium(self):
        label, emoji = confidence_label(0.60)
        assert label == "Medium"
        assert emoji == "🟡"

    def test_low(self):
        label, emoji = confidence_label(0.40)
        assert label == "Low"
        assert emoji == "🟠"

    def test_very_low(self):
        label, emoji = confidence_label(0.10)
        assert label == "Very Low"
        assert emoji == "🔴"

    def test_boundary_high(self):
        label, _ = confidence_label(0.75)
        assert label == "High"

    def test_boundary_medium(self):
        label, _ = confidence_label(0.50)
        assert label == "Medium"


class TestFormatFileSize:
    def test_bytes(self):
        assert format_file_size(512) == "512 B"

    def test_kilobytes(self):
        assert "KB" in format_file_size(2048)

    def test_megabytes(self):
        assert "MB" in format_file_size(2 * 1024 * 1024)


class TestTruncate:
    def test_short_string_unchanged(self):
        assert truncate("hello", 100) == "hello"

    def test_long_string_truncated(self):
        result = truncate("a" * 200, 50)
        assert len(result) <= 53  # 50 + ellipsis
        assert result.endswith("…")

    def test_exact_limit_unchanged(self):
        text = "a" * 150
        assert truncate(text, 150) == text


# ── Deletion & Caching tests ──────────────────────────────────────────────────

class TestSummaryCachingAndDeletion:
    def test_summary_caching(self, tmp_path):
        doc_id = "test_summary_123"
        summary_file = tmp_path / f"{doc_id}.summary"
        summary_text = "This is a test summary."
        
        # Write cache
        summary_file.write_text(summary_text, encoding="utf-8")
        assert summary_file.exists()
        
        # Read cache
        cached_summary = summary_file.read_text(encoding="utf-8")
        assert cached_summary == summary_text

    def test_delete_collection_fail_gracefully(self):
        from backend.core.rag_engine import rag_engine
        # Deleting a non-existent collection should return False and not raise an error
        res = rag_engine.delete_doc("non_existent_doc_id_12345")
        assert res is False


# ── Anthropic API Failure Fallback tests ─────────────────────────────────────

class TestAnthropicFallback:
    def test_summarise_graceful_fallback_on_api_error(self, monkeypatch):
        from backend.core.rag_engine import rag_engine
        
        # Mock _claude to raise an exception when messages.create is called
        class FailingMessages:
            def create(self, *args, **kwargs):
                raise Exception("API key invalid")
                
        class FailingClaude:
            messages = FailingMessages()
            
        original_claude = rag_engine._claude
        monkeypatch.setattr(rag_engine, "_claude", FailingClaude())
        
        try:
            # Call summarise and verify it returns a friendly error message instead of raising
            summary = rag_engine.summarise("dummy text", "dummy.pdf")
            assert "Summary generation unavailable" in summary
            assert "API key invalid" in summary
        finally:
            rag_engine._claude = original_claude

    def test_answer_graceful_fallback_on_api_error(self, monkeypatch):
        from backend.core.rag_engine import rag_engine
        
        # Mock retrieve_multiple to return a citation so it doesn't trigger low-confidence fallback
        from backend.core.rag_engine import SourceCitation
        mock_citations = [SourceCitation(1, 0, "dummy chunk text", "dummy...", 0.9)]
        monkeypatch.setattr(rag_engine, "retrieve_multiple", lambda *args, **kwargs: (mock_citations, 0.9))
        
        # Mock _claude to raise an exception
        class FailingMessages:
            def create(self, *args, **kwargs):
                raise Exception("API key invalid")
                
        class FailingClaude:
            messages = FailingMessages()
            
        original_claude = rag_engine._claude
        monkeypatch.setattr(rag_engine, "_claude", FailingClaude())
        
        try:
            # Call answer and verify it returns a friendly error message
            resp = rag_engine.answer("what is this?", "doc1", "sess1", "dummy.pdf")
            assert "LLM Generation Error" in resp.answer
            assert "API key invalid" in resp.answer
            assert resp.fallback is True
        finally:
            rag_engine._claude = original_claude

