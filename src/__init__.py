# src/__init__.py
"""
AI Agent Toolkit - A multi-tool AI agent with web search, document retrieval, and SQL capabilities
"""

__version__ = "1.0.0"
__author__ = "FC"

from .agent import AIAgent
from .config import Config

# Make main classes available at package level
__all__ = ["AIAgent", "Config"]

# ============================================

# tests/__init__.py
"""
Test package for AI Agent Toolkit
"""