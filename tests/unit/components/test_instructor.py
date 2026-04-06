from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from daystrom import Provider
from daystrom.components import Context, Instructor, Tool, ToolCall


@pytest.fixture
def client():
    class CityState(BaseModel):
        city_name: str
        state: str

    return Instructor(
        Provider.OPENROUTER,
        model="anthropic/claude-haiku-4.5",
        response_model=CityState,
    )


@pytest.fixture
def message():
    return "Give me information on a random city."


def test_init_with_context(message):
    existing_context = Context()
    existing_context.add_message("system", "You are helpful.")
    existing_context.add_message("user", message)
    instructor = Instructor(
        Provider.OPENROUTER,
        model="anthropic/claude-haiku-4.5",
        response_model=BaseModel,
        context=existing_context,
    )

    assert instructor.context is existing_context
    assert len(instructor.context.messages) == 2


def test_invoke_returns_response_model(client, message):
    mocked_response = client.response_model(city_name="Seattle", state="Washington")
    client.client.create = Mock(return_value=mocked_response)

    result = client.invoke(message)

    client.client.create.assert_called_once_with(
        response_model=client.response_model,
        messages=[{"role": "user", "content": message}],
        max_retries=3,
    )
    assert isinstance(result, client.response_model)
    assert isinstance(result.city_name, str)
    assert isinstance(result.state, str)


def test_get_prompt_context_all_message_types(client):
    """Test _get_prompt_context formats user, assistant, system, and tool messages correctly."""
    tool = Tool(
        callable=lambda x: x,
        name="sample_tool",
        description="A sample tool",
        params={"x": {"type": str, "description": "Input", "required": True}},
    )
    context = Context()
    context.add_message("system", "System instruction")
    context.add_message("user", "User question")
    context.add_message(
        "assistant",
        "I'll call the tool",
        tool_calls=[
            ToolCall(
                tool=tool,
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
    assert formatted[0]["role"] == "system"
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
