"""LLM backends: local Gemma and the remote LangChain/OpenCode Go agent."""

from .base import Backend, BackendError

__all__ = ["Backend", "BackendError"]
