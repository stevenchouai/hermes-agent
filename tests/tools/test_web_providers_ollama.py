"""Tests for the Ollama web search provider."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tests.tools.conftest import register_all_web_providers


class TestOllamaWebSearchProvider:
    _SAMPLE_RESPONSE = {
        "results": [
            {
                "title": "Python Website",
                "url": "https://www.python.org",
                "content": "Python is a programming language that lets you work quickly."
            },
            {
                "title": "Python Wikipedia",
                "url": "https://en.wikipedia.org/wiki/Python",
                "content": "Python is an interpreted high-level programming language."
            }
        ]
    }

    def _mock_resp(self, status_code=200, json_data=None, text=""):
        mock = MagicMock()
        mock.status_code = status_code
        mock.json.return_value = json_data or {}
        mock.text = text
        return mock

    def test_name_and_display_name(self):
        from plugins.web.ollama.provider import OllamaWebSearchProvider
        provider = OllamaWebSearchProvider()
        assert provider.name == "ollama"
        assert provider.display_name == "Ollama Web Search"

    def test_is_available_true(self):
        from plugins.web.ollama.provider import OllamaWebSearchProvider
        provider = OllamaWebSearchProvider()
        
        with patch("httpx.Client.get", return_value=self._mock_resp(status_code=200)):
            assert provider.is_available() is True

    def test_is_available_false_on_timeout(self):
        from plugins.web.ollama.provider import OllamaWebSearchProvider
        provider = OllamaWebSearchProvider()
        
        with patch("httpx.Client.get", side_effect=Exception("Timeout")):
            assert provider.is_available() is False

    def test_search_happy_path(self):
        from plugins.web.ollama.provider import OllamaWebSearchProvider
        provider = OllamaWebSearchProvider()

        with patch("httpx.Client.post", return_value=self._mock_resp(status_code=200, json_data=self._SAMPLE_RESPONSE)):
            result = provider.search("python", limit=2)

        assert result["success"] is True
        assert len(result["data"]["web"]) == 2
        assert result["data"]["web"][0]["title"] == "Python Website"
        assert result["data"]["web"][0]["url"] == "https://www.python.org"
        assert "programming language" in result["data"]["web"][0]["description"]

    def test_search_http_401_returns_auth_error(self):
        from plugins.web.ollama.provider import OllamaWebSearchProvider
        provider = OllamaWebSearchProvider()

        with patch("httpx.Client.post", return_value=self._mock_resp(status_code=401)):
            result = provider.search("python")

        assert result["success"] is False
        assert "authentication failed" in result["error"]

    def test_search_http_403_returns_unavailable_error(self):
        from plugins.web.ollama.provider import OllamaWebSearchProvider
        provider = OllamaWebSearchProvider()

        with patch("httpx.Client.post", return_value=self._mock_resp(status_code=403)):
            result = provider.search("python")

        assert result["success"] is False
        assert "unavailable" in result["error"]


class TestOllamaBackendWiring:
    def test_is_backend_available_true_when_daemon_reachable(self, monkeypatch):
        from tools.web_tools import _is_backend_available
        
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("httpx.Client.get", return_value=mock_resp):
            assert _is_backend_available("ollama") is True

    def test_check_web_api_key_true_when_ollama_configured(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "ollama"})
        
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("httpx.Client.get", return_value=mock_resp):
            assert web_tools.check_web_api_key() is True


class TestOllamaSearchOnlyErrors:
    _register_providers = staticmethod(register_all_web_providers)

    @pytest.fixture(autouse=True)
    def _populate_web_registry(self):
        self._register_providers()
        yield
        from agent.web_search_registry import _reset_for_tests
        _reset_for_tests()

    def test_web_extract_returns_search_only_error(self, monkeypatch):
        import asyncio
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "ollama"})
        
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        monkeypatch.setattr("httpx.Client.get", lambda *args, **kwargs: mock_resp)
        monkeypatch.setattr(web_tools, "is_safe_url", lambda url: True)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False, raising=False)

        result_str = asyncio.get_event_loop().run_until_complete(
            web_tools.web_extract_tool(["https://example.com"])
        )
        result = json.loads(result_str)
        assert result["success"] is False
        assert "search-only" in result["error"].lower()
        assert "ollama" in result["error"].lower()
