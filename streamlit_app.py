"""
streamlit_app.py — Entrypoint file for Streamlit Community Cloud deployment.
"""
import sys
from pathlib import Path

# Ensure the root directory is on the path so internal imports function correctly
sys.path.insert(0, str(Path(__file__).parent))

# Run the frontend application
import frontend.app
