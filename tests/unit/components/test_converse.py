import pytest

from daystrom import Provider
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


@pytest.fixture
def client(tools):
    c = BedrockConverse.__new__(BedrockConverse)
    c.provider = Provider.AWS_BEDROCK
    c.model = "anthropic.claude-3-sonnet-20240229-v1:0"
    c.tools = tools
    c.input_tokens = 0
    c.output_tokens = 0
    c.input_cost = None
    c.output_cost = None
    c.context_limit = None
    c.output_limit = None
    c.client = None  # will be mocked per test as needed
    return c


class TestGetPromptContext:
    def test_empty_context(self, client):
        system, messages = client._get_prompt_context(None)
        assert system == []
        assert messages == []

    def test_user_message(self, client):
        ctx = Context()
        ctx.add_message("user", "Hello")
        system, messages = client._get_prompt_context(ctx)

        assert system == []
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": [{"text": "Hello"}]}

    def test_system_message_separated(self, client):
        ctx = Context()
        ctx.add_message("system", "Be helpful")
        ctx.add_message("user", "Hi")
        system, messages = client._get_prompt_context(ctx)

        assert system == [{"text": "Be helpful"}]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_assistant_message_with_text(self, client):
        ctx = Context()
        ctx.add_message("user", "Hi")
        ctx.add_message("assistant", "Hello there")
        system, messages = client._get_prompt_context(ctx)

        assert len(messages) == 2
        assert messages[1] == {
            "role": "assistant",
            "content": [{"text": "Hello there"}],
        }

    def test_assistant_message_with_tool_calls(self, client, tools):
        ctx = Context()
        ctx.add_message("user", "Weather?")
        ctx.add_message(
            "assistant",
            "Let me check",
            tool_calls=[
                ToolCall(
                    tool=tools["get_weather"],
                    tool_call_id="call_abc",
                    args=[],
                    kwargs={"location": "Paris"},
                )
            ],
        )
        _, messages = client._get_prompt_context(ctx)

        assistant_msg = messages[1]
        assert assistant_msg["role"] == "assistant"
        assert len(assistant_msg["content"]) == 2
        assert assistant_msg["content"][0] == {"text": "Let me check"}
        assert assistant_msg["content"][1] == {
            "toolUse": {
                "toolUseId": "call_abc",
                "name": "get_weather",
                "input": {"location": "Paris"},
            }
        }

    def test_tool_messages_grouped_into_user_message(self, client, tools):
        ctx = Context()
        ctx.add_message("user", "Weather?")
        ctx.add_message(
            "assistant",
            "",
            tool_calls=[
                ToolCall(
                    tool=tools["get_weather"],
                    tool_call_id="call_1",
                    args=[],
                    kwargs={"location": "Paris"},
                ),
                ToolCall(
                    tool=tools["sample_tool"],
                    tool_call_id="call_2",
                    args=[],
                    kwargs={"x": "test"},
                ),
            ],
        )
        ctx.add_message("tool", "Weather: sunny", tool_call_id="call_1")
        ctx.add_message("tool", "test result", tool_call_id="call_2")

        _, messages = client._get_prompt_context(ctx)

        # user, assistant, user (grouped tool results)
        assert len(messages) == 3
        tool_result_msg = messages[2]
        assert tool_result_msg["role"] == "user"
        assert len(tool_result_msg["content"]) == 2
        assert tool_result_msg["content"][0] == {
            "toolResult": {
                "toolUseId": "call_1",
                "content": [{"text": "Weather: sunny"}],
            }
        }
        assert tool_result_msg["content"][1] == {
            "toolResult": {
                "toolUseId": "call_2",
                "content": [{"text": "test result"}],
            }
        }

    def test_unsupported_role_raises(self, client):
        ctx = Context()
        ctx.add_message("unknown_role", "Bad message")
        with pytest.raises(ValueError, match="Unsupported message role: unknown_role"):
            client._get_prompt_context(ctx)


class TestGetToolContext:
    def test_empty_tools_returns_empty(self, client):
        client.tools = {}
        assert client._get_tool_context() == {}

    def test_tool_schema_structure(self, client, tools):
        result = client._get_tool_context()

        assert "tools" in result
        assert len(result["tools"]) == 3

        # Find the get_weather tool
        weather_spec = None
        for t in result["tools"]:
            if t["toolSpec"]["name"] == "get_weather":
                weather_spec = t["toolSpec"]
                break

        assert weather_spec is not None
        assert weather_spec["description"] == "Get the current weather for a location"
        assert "json" in weather_spec["inputSchema"]

        schema = weather_spec["inputSchema"]["json"]
        assert schema["type"] == "object"
        assert "location" in schema["properties"]
        assert schema["properties"]["location"]["type"] == "string"
        assert "location" in schema["required"]
        assert "unit" not in schema.get("required", [])

    def test_all_type_mappings(self, client):
        result = client._get_tool_context()

        # Find multi_param tool
        multi_spec = None
        for t in result["tools"]:
            if t["toolSpec"]["name"] == "multi_param":
                multi_spec = t["toolSpec"]
                break

        assert multi_spec is not None
        props = multi_spec["inputSchema"]["json"]["properties"]
        assert props["dictarg"]["type"] == "object"
        assert props["tags"]["type"] == "array"
        assert props["tags"]["items"]["type"] == "string"
        assert props["tuplearg"]["type"] == "array"
        assert props["tuplearg"]["items"]["type"] == "string"
        assert props["name"]["type"] == "string"
        assert props["count"]["type"] == "integer"
        assert props["ratio"]["type"] == "number"
        assert props["active"]["type"] == "boolean"

        required = multi_spec["inputSchema"]["json"]["required"]
        assert "name" in required
        assert "count" in required
        assert "ratio" not in required


class TestTrackUsage:
    def test_tracks_tokens(self, client):
        client.track_usage({"inputTokens": 100, "outputTokens": 50})
        client.track_usage({"inputTokens": 200, "outputTokens": 100})
        assert client.input_tokens == 300
        assert client.output_tokens == 150
