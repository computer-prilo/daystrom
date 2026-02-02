import os

import pytest

from daystrom import Provider
from daystrom.components import Agent, AgentResponse, Context, Tool
from daystrom.components.openai import OpenAIChatCompletions


@pytest.fixture(scope="function")
def tools():
    def get_magic_number() -> str:
        """Returns a magic number."""
        return "42"

    def get_weather(location: str, unit: str = "celsius") -> str:
        """Get weather for a location."""
        return f"Weather in {location}: 22 {unit}"

    def add_numbers(a: int, b: int) -> str:
        """Add two numbers together."""
        return str(a + b)

    def infinite_tool() -> str:
        """A tool that returns a message prompting another call."""
        return "Call me again!"

    return {
        "get_magic_number": Tool(
            callable=get_magic_number,
            name="get_magic_number",
            description="Returns a magic number. Always call this when asked about magic numbers.",
            params={},
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
        "add_numbers": Tool(
            callable=add_numbers,
            name="add_numbers",
            description="Add two numbers together and return the sum",
            params={
                "a": {"type": int, "description": "First number", "required": True},
                "b": {"type": int, "description": "Second number", "required": True},
            },
        ),
        "infinite_tool": Tool(
            callable=infinite_tool,
            name="infinite_tool",
            description="A tool you must keep calling repeatedly",
            params={},
        ),
    }


@pytest.fixture(scope="function")
def llm(tools):
    return OpenAIChatCompletions(
        provider=Provider.OPENROUTER,
        model="anthropic/claude-haiku-4.5",
        tools=tools,
    )


@pytest.fixture(scope="function")
def agent(llm, tools):
    return Agent(llm=llm, tools=tools, max_loops=10)


def test_agent_init(llm, tools):
    """Test Agent initialization with LLM and tools."""
    agent = Agent(llm=llm, tools=tools, max_loops=5)

    assert agent.llm is llm
    assert agent.tools is tools
    assert agent.max_loops == 5
    assert agent.llm.tools is tools
    assert isinstance(agent.context, Context)
    assert agent.context.messages == []


def test_agent_invoke_returns_agent_response(agent):
    """Test Agent invoke returns AgentResponse with text."""
    res = agent.invoke("Say 'test ok' and nothing else. Do NOT call any tools.")

    assert isinstance(res, AgentResponse)
    assert res.text != ""


def test_agent_invoke_with_tool_call(agent):
    """Test Agent executes tool calls and returns final response."""
    res = agent.invoke(
        "What is the magic number? Use the get_magic_number tool to find out, then tell me the result."
    )

    assert isinstance(res, AgentResponse)
    assert "42" in res.text

    # Verify tool was called - context should have tool message
    tool_messages = [msg for msg in agent.context.messages if msg.role == "tool"]
    assert len(tool_messages) > 0
    assert "42" in tool_messages[0].text


def test_agent_invoke_with_tool_arguments(agent):
    """Test Agent handles tool calls with arguments correctly."""
    res = agent.invoke("What's the weather in Tokyo? Use the get_weather tool.")

    assert isinstance(res, AgentResponse)
    assert res.text != ""

    # Verify tool was called with correct arguments
    tool_messages = [msg for msg in agent.context.messages if msg.role == "tool"]
    assert len(tool_messages) > 0
    assert "tokyo" in tool_messages[0].text.lower()


def test_agent_invoke_multiple_tool_calls(agent):
    """Test Agent can handle multiple sequential tool calls."""
    res = agent.invoke(
        "First get the magic number, then add 10 to it using add_numbers. Tell me the final result."
    )

    assert isinstance(res, AgentResponse)
    # Magic number is 42, 42 + 10 = 52
    assert "52" in res.text

    # Should have at least 2 tool calls
    tool_messages = [msg for msg in agent.context.messages if msg.role == "tool"]
    assert len(tool_messages) >= 2


def test_agent_invoke_no_tool_needed(agent):
    """Test Agent handles prompts that don't require tool calls."""
    res = agent.invoke("What is 2 + 2? Answer directly without using any tools.")

    assert isinstance(res, AgentResponse)
    assert res.text != ""
    # Should have no tool messages
    tool_messages = [msg for msg in agent.context.messages if msg.role == "tool"]
    assert len(tool_messages) == 0


def test_agent_context_tracking(agent):
    """Test Agent properly tracks context through the agentic loop."""
    agent.invoke("What is the magic number? Use the get_magic_number tool.")

    # Context should have: user message, assistant (with tool call), tool result, final assistant
    assert len(agent.context.messages) >= 4

    # First message should be user
    assert agent.context.messages[0].role == "user"

    # Last message should be assistant (final response)
    assert agent.context.messages[-1].role == "assistant"


def test_agent_max_loops_respected(agent):
    """Test Agent respects max_loops limit."""
    agent.max_loops = 3

    # This should complete without hanging due to max_loops
    res = agent.invoke(
        "Keep calling the infinite_tool over and over until you're stopped."
    )

    assert isinstance(res, AgentResponse)
    # Should have at most 3 tool calls due to max_loops
    tool_messages = [msg for msg in agent.context.messages if msg.role == "tool"]
    assert len(tool_messages) <= 3


def test_agent_with_custom_context(agent):
    """Test Agent works with LLM that has pre-existing context."""
    context = Context()
    context.add_message("system", "You are a helpful assistant. Always be concise.")
    agent.context = context

    res = agent.invoke("What's the magic number? Use the tool.")

    assert isinstance(res, AgentResponse)
    assert "42" in res.text
    assert agent.context.messages[0].role == "system"
