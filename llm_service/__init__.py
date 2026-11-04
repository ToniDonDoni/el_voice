"""Internal LLM service for EL Voice.

The service owns conversation history, prompt construction and language-model
inference for both launch modes. The browser voice application talks to it over
one internal WebSocket contract and never loads Gemma or contacts a remote
provider itself.
"""

__version__ = "1.0.0"
