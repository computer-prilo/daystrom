import json

from markdownify import markdownify as md

from daystrom.components import tool
from daystrom.exceptions import ToolCallError


@tool
def web_fetch(url: str, format: str = "markdown") -> str:
    """Fetches content from a given URL.

    Args:
        url (str): The URL to fetch content from.
        format (str, optional): The format of the content to fetch, text, html, json, or markdown. Default "markdown".

    Returns:
        str: The fetched content as a string.
    """
    import httpx

    accept_header = "*/*"
    match format:
        case "text":
            accept_header = (
                "text/plain;q=1.0, text/markdown;q=0.9, text/html;q=0.8, */*;q=0.1"
            )
        case "html":
            accept_header = "text/html;q=1.0, application/xhtml+xml;q=0.9, text/plain;q=0.8, text/markdown;q=0.7, */*;q=0.1"
        case "json":
            accept_header = "application/json;q=1.0, text/markdown;q=0.9, text/x-markdown;q=0.8, text/plain;q=0.7, text/html;q=0.6, */*;q=0.1"
        case "markdown":
            accept_header = "text/markdown;q=1.0, text/x-markdown;q=0.9, text/plain;q=0.8, text/html;q=0.7, */*;q=0.1"
        case _:
            raise ToolCallError("web_fetch", f"Unsupported format: {format}")

    headers = {
        "accept": accept_header,
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
    }
    response = httpx.get(url, timeout=10.0, headers=headers, follow_redirects=False)
    if response.is_redirect:
        redirect_url = response.headers.get("location")
        raise ToolCallError(
            "web_fetch",
            f"Redirection not supported by tool - to get the content, call the web_fetch tool again with the new URL: {redirect_url}",
        )

    response.raise_for_status()
    response_type = response.headers.get("content-type") or ""

    match format:
        case "text":
            ans = response.text
            if "text/html" in response_type:
                ans = md(ans, convert=[])
            return ans
        case "html":
            return response.text
        case "json":
            ans = response.text
            if "application/json" in response_type:
                ans = json.dumps(response.json())
            return ans
        case "markdown":
            ans = response.text
            if "text/html" in response_type:
                ans = md(ans)
            return ans
        case _:
            raise ToolCallError("web_fetch", f"Unsupported format: {format}")
