import os

import pytest

from daystrom.components import Context, LLMResponse, Tool, ToolCall
from daystrom.components.aws import BedrockConverse


@pytest.fixture
def tools():
    def get_weather(location: str, unit: str = "celsius") -> str:
        """Get weather for a location."""
        return f"Weather in {location}: 22 {unit}"

    def multi_param(
        dictarg: dict,
        tags: list[str],
        tuplearg: tuple[str, ...],
        name: str,
        count: int,
        ratio: float,
        active: bool,
    ) -> str:
        return (
            f"{dictarg}, {tags}, {tuplearg}, {name}, {count}, {ratio}, {active}, {tags}"
        )

    # dummy call for coverage
    multi_param({}, [], (), "", 1, 1.0, True)

    return {
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
                "dictarg": {"type": dict, "description": "DictArg", "required": True},
                "tags": {
                    "type": list[str],
                    "description": "Tags",
                    "required": False,
                    "items": {"type": str},
                },
                "tuplearg": {
                    "type": tuple[str, ...],
                    "description": "TupleArg",
                    "required": False,
                    "items": {"type": str},
                },
                "name": {"type": str, "description": "Name", "required": True},
                "count": {"type": int, "description": "Count", "required": True},
                "ratio": {"type": float, "description": "Ratio", "required": False},
                "active": {"type": bool, "description": "Active", "required": False},
            },
        ),
    }


@pytest.fixture(scope="function")
def client(tools):
    return BedrockConverse("anthropic/claude-haiku-4.5", tools=tools)


@pytest.fixture
def context():
    context = Context()
    context.add_message(
        role="user",
        text="Give me a short response as a test that API functionality is working. Do NOT give an empty response and do NOT call any tools.",
    )
    return context


class TestInvoke:
    def test_invoke_text_response(self, client):
        res = client.invoke(context)
        assert isinstance(res, LLMResponse)
        assert res.text != ""

    def test_invoke_tool_call_response(self, client):
        context = Context()
        context.add_message(
            role="user",
            text="What's the weather in Paris? Use the get_weather tool.",
        )
        res = client.invoke(context)

        assert isinstance(res, LLMResponse)
        assert len(res.tool_calls) > 0

        context.add_message("assistant", text=res.text, tool_calls=res.tool_calls)
        tool_call = res.tool_calls[0]
        assert tool_call.tool.name == "get_weather"
        assert tool_call.tool_call_id != ""
        assert "location" in tool_call.kwargs
        assert "paris" in tool_call.kwargs["location"].lower()

        # Tool call should be in assistant message context
        assistant_msg = context.messages[-1]
        assert assistant_msg.role == "assistant"
        assert len(assistant_msg.tool_calls) > 0
        assert assistant_msg.tool_calls[0].tool.name == "get_weather"

        # Add tool result to context
        tool_result = tool_call.tool.call(*tool_call.args, **tool_call.kwargs)
        context.add_message("tool", tool_result, tool_call.tool_call_id)

        # Second invoke should succeed with tool result included
        res1 = client.invoke(context)
        assert isinstance(res1, LLMResponse)
        assert res1.text != ""

    def test_invoke_passes_system_and_tools(self, client, mocker):
        mock_converse = mocker.MagicMock(
            return_value={
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "ok"}],
                    }
                },
                "usage": {"inputTokens": 5, "outputTokens": 2},
                "stopReason": "end_turn",
            }
        )
        client.client = mocker.MagicMock()
        client.client.converse = mock_converse

        ctx = Context()
        ctx.add_message("system", "Be concise")
        ctx.add_message("user", "Hi")
        client.invoke(ctx)

        call_kwargs = mock_converse.call_args[1]
        assert call_kwargs["modelId"] == client.model
        assert call_kwargs["system"] == [{"text": "Be concise"}]
        assert "toolConfig" in call_kwargs
        assert len(call_kwargs["toolConfig"]["tools"]) == 3

    def test_invoke_no_tools_omits_tool_config(self, client, mocker):
        client.tools = {}
        mock_converse = mocker.MagicMock(
            return_value={
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "ok"}],
                    }
                },
                "usage": {"inputTokens": 5, "outputTokens": 2},
                "stopReason": "end_turn",
            }
        )
        client.client = mocker.MagicMock()
        client.client.converse = mock_converse

        ctx = Context()
        ctx.add_message("user", "Hi")
        client.invoke(ctx)

        call_kwargs = mock_converse.call_args[1]
        assert "toolConfig" not in call_kwargs

    def test_invoke_no_system_omits_system(self, client, mocker):
        mock_converse = mocker.MagicMock(
            return_value={
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "ok"}],
                    }
                },
                "usage": {"inputTokens": 5, "outputTokens": 2},
                "stopReason": "end_turn",
            }
        )
        client.client = mocker.MagicMock()
        client.client.converse = mock_converse

        ctx = Context()
        ctx.add_message("user", "Hi")
        client.invoke(ctx)

        call_kwargs = mock_converse.call_args[1]
        assert "system" not in call_kwargs
