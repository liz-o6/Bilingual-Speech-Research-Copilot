"""Shared model connections and metadata logging only."""
from .client import create_llm
from .logging import LLMCallLogger

__all__ = ['create_llm', 'LLMCallLogger']
