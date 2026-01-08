import json

import pytest

from daystrom.components.tools import web_fetch


class TestWebFetchIntegration:
    """Integration tests for web_fetch tool with real network calls."""

    def test_fetch_json_from_httpbin(self):
        """Test fetching JSON from httpbin.org."""
        result = web_fetch("https://httpbin.org/json", format="json")

        parsed = json.loads(result)
        assert "slideshow" in parsed

    def test_fetch_html_from_httpbin(self):
        """Test fetching HTML content from httpbin.org."""
        result = web_fetch("https://httpbin.org/html", format="html")

        assert "<html>" in result or "<!DOCTYPE" in result.upper()
        assert "Herman Melville" in result

    def test_fetch_markdown_from_html(self):
        """Test fetching HTML and converting to markdown."""
        result = web_fetch("https://httpbin.org/html", format="markdown")

        # Should have text content but no HTML tags
        assert "Herman Melville" in result
        assert "<html>" not in result
        assert "<body>" not in result

    def test_fetch_text_from_html(self):
        """Test fetching HTML and converting to text."""
        result = web_fetch("https://httpbin.org/html", format="text")

        # Should have text content but no HTML tags
        assert "Herman Melville" in result
        assert "<html>" not in result

    def test_fetch_user_agent_endpoint(self):
        """Test fetching from httpbin user-agent endpoint."""
        result = web_fetch("https://httpbin.org/user-agent", format="json")

        parsed = json.loads(result)
        assert "user-agent" in parsed

    def test_fetch_headers_endpoint(self):
        """Test fetching from httpbin headers endpoint to verify accept header."""
        result = web_fetch("https://httpbin.org/headers", format="json")

        parsed = json.loads(result)
        assert "headers" in parsed
        # Should have Accept header set
        assert "Accept" in parsed["headers"]

    def test_fetch_status_404_raises(self):
        """Test that 404 responses raise an error."""
        import httpx

        with pytest.raises(httpx.HTTPStatusError):
            web_fetch("https://httpbin.org/status/404")

    def test_fetch_status_500_raises(self):
        """Test that 500 responses raise an error."""
        import httpx

        with pytest.raises(httpx.HTTPStatusError):
            web_fetch("https://httpbin.org/status/500")

    def test_redirect_raises_tool_call_error(self):
        """Test that redirect responses raise ToolCallError with redirect URL."""
        from daystrom.exceptions import ToolCallError

        with pytest.raises(ToolCallError) as exc_info:
            web_fetch("https://httpbin.org/redirect-to?url=https://example.com")

        assert "https://example.com" in str(exc_info.value)
