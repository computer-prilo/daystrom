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

    response = httpx.get(url)
    response.raise_for_status()

    match format:
        case "text":
            return response.text
        case "html":
            return response.text
        case "json":
            return response.json()
        case "markdown":
            return response.text
        case _:
            raise ToolCallError(f"Unsupported format: {format}")
