"""Ollama experimental web search provider plugin."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

import httpx

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

OLLAMA_WEB_SEARCH_PATH = "/api/experimental/web_search"


class OllamaWebSearchProvider(WebSearchProvider):
    """Local Ollama experimental web search provider.

    Requires a running Ollama daemon with experimental web search enabled.
    """

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def display_name(self) -> str:
        return "Ollama Web Search"

    def _get_ollama_base_url(self) -> str:
        """Return the local Ollama host used for keyless web search."""
        try:
            from hermes_cli.config import load_config
            configured = (load_config().get("web", {}).get("ollama_base_url") or "").strip()
        except Exception:
            configured = ""
        env_host = os.getenv("OLLAMA_HOST", "").strip()
        base = configured or env_host or "http://127.0.0.1:11434"
        if not base.startswith(("http://", "https://")):
            base = f"http://{base}"
        return base.rstrip("/")

    def is_available(self) -> bool:
        """Return True when a local Ollama host is reachable.

        This runs at tool registration / setup repaint, so it must be reasonably cheap.
        We do a quick probe of the daemon.
        """
        try:
            with httpx.Client(timeout=1.0) as client:
                response = client.get(f"{self._get_ollama_base_url()}/api/tags")
            return response.status_code < 500
        except Exception:
            return False

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute search against Ollama's experimental web_search endpoint."""
        count = min(max(int(limit or 5), 1), 10)
        url = f"{self._get_ollama_base_url()}{OLLAMA_WEB_SEARCH_PATH}"
        try:
            with httpx.Client(timeout=20.0) as client:
                response = client.post(
                    url,
                    json={"query": query, "max_results": count},
                    headers={"Content-Type": "application/json"},
                )
        except Exception as exc:
            return {"success": False, "error": f"Ollama web search failed: {exc}"}

        if response.status_code == 401:
            return {"success": False, "error": "Ollama web search authentication failed. Run `ollama signin`."}
        if response.status_code == 403:
            return {
                "success": False,
                "error": "Ollama web search is unavailable. Ensure cloud-backed web search is enabled on the Ollama host."
            }
        if response.status_code >= 400:
            return {"success": False, "error": f"Ollama web search failed ({response.status_code}): {response.text[:1000]}"}

        try:
            payload = response.json()
        except Exception as exc:
            return {"success": False, "error": f"Could not parse Ollama response as JSON: {exc}"}

        web_results = []
        for idx, item in enumerate(payload.get("results") or [], start=1):
            url_value = str(item.get("url") or "").strip()
            if not url_value:
                continue
            web_results.append({
                "title": str(item.get("title") or ""),
                "url": url_value,
                "description": str(item.get("content") or "")[:500],
                "position": idx,
            })
        return {"success": True, "data": {"web": web_results}}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Ollama Web Search",
            "badge": "free · local · search only",
            "tag": "Ollama experimental local web search (requires cloud-enabled search on Ollama daemon)",
            "env_vars": [],
        }
