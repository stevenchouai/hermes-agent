"""Ollama experimental web search plugin — bundled, auto-loaded.

Backed by the local Ollama daemon's experimental web search endpoint.
No API key required.
"""

from __future__ import annotations

from plugins.web.ollama.provider import OllamaWebSearchProvider


def register(ctx) -> None:
    """Register the Ollama provider with the plugin context."""
    ctx.register_web_search_provider(OllamaWebSearchProvider())
