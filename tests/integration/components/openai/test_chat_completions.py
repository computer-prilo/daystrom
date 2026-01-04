import os

import pytest

from daystrom import Provider
from daystrom.components import Context, LLMResponse, Tool, ToolCall
from daystrom.components.openai import OpenAIChatCompletions


@pytest.fixture(scope="function")
def client():
    def get_weather(location: str, unit: str = "celsius") -> str:
        """Get weather for a location."""
        return f"Weather in {location}: 22 {unit}"

    def multi_param(
        name: str, count: int, ratio: float, active: bool, tags: list[str]
    ) -> str:
        return f"{name}, {count}, {ratio}, {active}, {tags}"

    # dummy call because this function never gets triggered because the
    # point is teseting arguments. this keeps it included in code coverage
    multi_param("", 1, 1.0, True, [])

    tools = {
        "sample_tool": Tool(
            callable=lambda x: x,
            name="sample_tool",
            description="A sample tool",
            params={"x": {"type": str, "description": "Input", "required": True}},
        ),
        "get_weather": Tool(
            callable=get_weather,
            name="get_weather",
            description="Get the current weather for a location",
            params={
                "location": {"type": str, "description": "City name", "required": True},
                "unit": {
                    "type": str,
                    "description": "Temperature unit",
                    "required": False,
                },
            },
        ),
        "multi_param": Tool(
            callable=multi_param,
            name="multi_param",
            description="Tool with multiple param types",
            params={
                "name": {"type": str, "description": "Name", "required": True},
                "count": {"type": int, "description": "Count", "required": True},
                "ratio": {"type": float, "description": "Ratio", "required": False},
                "active": {"type": bool, "description": "Active", "required": False},
                "tags": {
                    "type": list[str],
                    "description": "Tags",
                    "required": False,
                    "items": {"type": str},
                },
            },
        ),
    }

    return OpenAIChatCompletions(
        provider=Provider.OPENROUTER,
        model="anthropic/claude-haiku-4.5",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        tools=tools,
    )


@pytest.fixture(scope="module")
def message():
    return "Give me a short response as a test that API functionality is working. Do NOT give an empty response and do NOT call any tools."


def test_default_context_init(client):
    """Test basic invoke returns LLMResponse with text."""
    assert isinstance(client.context, Context)
    assert client.context.messages == []


def test_custom_context_init():
    """Test initialization: model storage, context creation/passing, and tools."""
    existing_context = Context()
    existing_context.add_message("system", "You are helpful.")
    existing_context.add_message("user", "Previous message")

    client2 = OpenAIChatCompletions(
        provider=Provider.OPENROUTER,
        model="anthropic/claude-haiku-4.5",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        context=existing_context,
    )
    assert client2.context is existing_context
    assert len(client2.context.messages) == 2


def test_tools_init(client):
    """Test client gets initialized with tools properly."""
    assert "sample_tool" in client.tools
    assert client.tools["sample_tool"].name == "sample_tool"


def test_invoke(client, message):
    """Test basic invoke and context tracking returns LLMResponse with text."""
    res = client.invoke(message)
    assert isinstance(res, LLMResponse)
    assert res.text != ""

    client.invoke("Remember the word: BANANA")
    assert len(client.context.messages) == 4
    assert client.context.messages[2].role == "user"
    assert client.context.messages[2].text == "Remember the word: BANANA"
    assert client.context.messages[3].role == "assistant"


def test_invoke_without_prompt_and_with_system_message():
    """Test invoke without prompt uses existing context, and system messages work."""
    context = Context()
    context.add_message("system", "You must respond with exactly: SYSTEM_TEST_OK")
    context.add_message("user", "Respond now.")

    client = OpenAIChatCompletions(
        provider=Provider.OPENROUTER,
        model="anthropic/claude-haiku-4.5",
        context=context,
    )

    # Invoke without prompt - should use existing messages
    res = client.invoke()

    assert len(client.context.messages) == 3
    assert client.context.messages[2].role == "assistant"
    assert isinstance(res, LLMResponse)
    assert res.text != ""


def test_invoke_with_tool_parses_arguments(client):
    """Test that tool calls with arguments are properly parsed."""
    res = client.invoke("What's the weather in Paris? Use the get_weather tool.")

    assert isinstance(res, LLMResponse)
    assert len(res.tool_calls) > 0

    tool_call = res.tool_calls[0]
    assert tool_call.tool.name == "get_weather"
    assert tool_call.tool_call_id != ""
    assert "location" in tool_call.kwargs
    assert "paris" in tool_call.kwargs["location"].lower()

    # Tool call should be in assistant message context
    assistant_msg = client.context.messages[-1]
    assert assistant_msg.role == "assistant"
    assert len(assistant_msg.tool_calls) > 0
    assert assistant_msg.tool_calls[0].tool.name == "get_weather"

    # Add tool result to context
    tool_result = tool_call.tool.call(*tool_call.args, **tool_call.kwargs)
    client.context.add_message("tool", tool_result, tool_call.tool_call_id)

    # Second invoke should succeed with tool result included
    res1 = client.invoke()
    assert isinstance(res1, LLMResponse)
    assert res1.text != ""


def test_get_prompt_context_all_message_types(client):
    """Test _get_prompt_context formats user, assistant, system, and tool messages correctly."""
    context = Context()
    context.add_message("system", "System instruction")
    context.add_message("user", "User question")
    context.add_message(
        "assistant",
        "I'll call the tool",
        tool_calls=[
            ToolCall(
                tool=client.tools["sample_tool"],
                tool_call_id="call_123",
                args=[],
                kwargs={"x": "val"},
            )
        ],
    )
    context.add_message("tool", "Tool result", tool_call_id="call_123")
    context.add_message("assistant", "Final answer")

    client.context = context

    formatted = client._get_prompt_context()

    assert len(formatted) == 5

    # OpenAI used to call it the "system" prompt/role, now they call it
    # prompt/role. We passed "system" on purpose and are checking that the
    # harness converts that properly
    assert formatted[0]["role"] == "developer"
    assert formatted[0]["content"] == "System instruction"

    assert formatted[1]["role"] == "user"
    assert formatted[1]["content"] == "User question"

    assert formatted[2]["role"] == "assistant"
    assert formatted[2].get("content") == "I'll call the tool"
    assert len(list(formatted[2].get("tool_calls") or [])) == 1
    tcs = formatted[2].get("tool_calls")
    if tcs:
        for tc in tcs:
            assert tc["type"] == "function"
            assert tc["id"] == "call_123"
            assert tc["function"]["name"] == "sample_tool"
            assert tc["function"]["arguments"] == '{"x": "val"}'

    assert formatted[3]["role"] == "tool"
    assert formatted[3]["content"] == "Tool result"
    assert formatted[3]["tool_call_id"] == "call_123"

    assert formatted[4]["role"] == "assistant"
    assert formatted[4].get("content") == "Final answer"


def test_get_tool_context_schema_generation(client):
    """Test _get_tool_context generates proper schemas with all type mappings."""
    schemas = client._get_tool_context()

    assert len(schemas) == 3
    schema = schemas[0]
    for schema in schemas:
        if schema["function"]["name"] == "multi_param":
            break
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "multi_param"
    assert schema["function"].get("description") == "Tool with multiple param types"

    params = schema["function"].get("parameters")
    assert params
    props = params["properties"]

    assert props["name"]["type"] == "string"
    assert props["count"]["type"] == "integer"
    assert props["ratio"]["type"] == "number"
    assert props["active"]["type"] == "boolean"
    assert props["tags"]["type"] == "array"
    assert props["tags"]["items"]["type"] == "string"

    required = schema["function"]["parameters"]["required"]
    assert "name" in required
    assert "count" in required
    assert "ratio" not in required

    # Empty tools returns empty list
    client2 = OpenAIChatCompletions(
        provider=Provider.OPENROUTER,
        model="anthropic/claude-haiku-4.5",
    )
    assert client2._get_tool_context() == []
