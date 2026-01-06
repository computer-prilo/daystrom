from markdownify import markdownify as md

from daystrom.components import tool
from daystrom.exceptions.components import ToolCallError


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
            accept_header = "application/json;q=1.0 text/markdown;q=1.0, text/x-markdown;q=0.9, text/plain;q=0.8, text/html;q=0.7, */*;q=0.1"
        case "markdown":
            accept_header = "text/markdown;q=1.0, text/x-markdown;q=0.9, text/plain;q=0.8, text/html;q=0.7, */*;q=0.1"
        case _:
            # accept_header = "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
            pass

    headers = {
        "accept": accept_header,
    }
    response = httpx.get(url, headers=headers)
    response.raise_for_status()
    response_type = response.headers.get("content-type")

    match format:
        case "text":
            ans = response.text
            if "text/html" in response_type:
                # TODO: Convert HTML to text
                pass
            return ans
        case "html":
            return response.text
        case "json":
            ans = response.text
            if "application/json" in response_type:
                ans = response.json()
            return ans
        case "markdown":
            ans = response.text
            if "text/html" in response_type:
                ans = md(ans)
            return ans
        case _:
            raise ToolCallError(f"Unsupported format: {format}")
