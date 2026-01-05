import functools
import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar, get_origin

from docstring_parser import parse

from daystrom import Provider

ComponentResponseT = TypeVar("ComponentResponseT")

log = logging.getLogger(__name__)


class Component(Generic[ComponentResponseT], ABC):
    @abstractmethod
    def invoke(self, *args, **kwargs) -> ComponentResponseT | None:
        pass  # pragma: no cover


class Tool:
    def __init__(
        self,
        callable,
        name: str = "",
        display_name: str = "",
        description: str = "",
        params: dict | None = None,
    ):
        self.callable = callable
        self.name = name or callable.__name__
        self.display_name = display_name or self.name.replace("_", " ").title()
        self.description = description or callable.__doc__ or ""
        self.params = params or {}

    def __str__(self):
        return f"Tool(name={self.name}, description={self.description}, params={self.params})"

    def call(self, *args, **kwargs):
        return self.callable(*args, **kwargs)


@dataclass
class ToolCall:
    tool: Tool
    tool_call_id: str
    args: list
    kwargs: dict


class Message:
    def __init__(
        self,
        role: str,
        text: str,
        tool_call_id: str = "",
        tool_calls: list[ToolCall] | None = None,
    ):
        self.role = role
        self.text = text
        self.tool_call_id = tool_call_id
        self.tool_calls: list[ToolCall] = tool_calls or []

    def __str__(self):
        parts = []
        parts.append(f"{self.role}: {self.text}")
        if self.tool_call_id:
            parts.append(f"    Tool Call ID: {self.tool_call_id}")
        if self.tool_calls:
            parts.append("    Tool Calls:")
            for tool in self.tool_calls:
                parts.append(f"    {str(tool)}")

        return "\n".join(parts)


class Context:
    def __init__(self):
        self.messages: list[Message] = []

    def add_message(
        self,
        role: str,
        text: str,
        tool_call_id: str = "",
        tool_calls: list[ToolCall] | None = None,
    ):
        self.messages.append(
            Message(
                text=text, role=role, tool_call_id=tool_call_id, tool_calls=tool_calls
            )
        )

    def print_feed(self):
        for message in self.messages:
            print(message)


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]


@dataclass
class AgentResponse:
    text: str


class LLM(Component[LLMResponse]):
    context: Context
    tools: dict[str, Tool]
    provider: Provider
    model: str
    input_tokens: int
    output_tokens: int
    input_cost: float | None
    output_cost: float | None
    context_limit: int | None
    output_limit: int | None

    def __init__(
        self,
        provider: Provider,
        model: str,
        context: Context | None = None,
        tools: dict[str, Tool] | None = None,
    ):
        self.provider = provider
        self.model = model

        if context:
            self.context = context
        else:
            self.context = Context()

        self.tools = tools or {}
        self.input_tokens: int = 0
        self.output_tokens: int = 0

        self.input_cost = None
        self.output_cost = None
        self.context_limit = None
        self.output_limit = None
        model_metadata = provider.value.models.get(self.model)
        if model_metadata:
            self.input_cost = model_metadata.input_cost
            self.output_cost = model_metadata.output_cost
            self.context_limit = model_metadata.context_limit
            self.output_limit = model_metadata.output_limit

    @abstractmethod
    def invoke(self, *args, **kwargs) -> LLMResponse:
        pass  # pragma: no cover

    @abstractmethod
    def track_usage(self, *args, **kwargs):
        pass  # pragma: no cover

    @property
    def total_cost(self) -> float:
        """Calculate total cost based on token usage and costs.

        Returns 0.0 if cost information is not available.
        """
        if self.input_cost is None and self.output_cost is None:
            return 0.0
        # Costs are per million tokens
        input_total = (self.input_tokens / 1_000_000) * (self.input_cost or 0.0)
        output_total = (self.output_tokens / 1_000_000) * (self.output_cost or 0.0)
        return input_total + output_total


DEFAULT_TOOLS = {}


# this is a decorator to be @tool above each tool function
def tool(func):
    docstring = parse(func.__doc__ or "")
    inspect_params = inspect.signature(func).parameters

    func_params = {}

    for idx, (name, param) in enumerate(inspect_params.items()):
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            raise TypeError("*args is not supported in tool parameters.")
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            raise TypeError("**kwargs is not supported in tool parameters.")

        required = False
        if param.default is inspect.Parameter.empty:
            required = True

        description = ""
        if len(docstring.params) >= idx + 1:
            description = docstring.params[idx].description

        func_params[name] = {
            "type": param.annotation,
            "description": description,
            "required": required,
        }

        if get_origin(param.annotation) in (list, tuple):
            if len(param.annotation.__args__) > 1:
                raise TypeError(
                    "Only single-type iterables are allowed as parameters to tool calls."
                )
            else:
                func_params[name]["items"] = {"type": param.annotation.__args__[0]}

    tool_desc = docstring.long_description or docstring.short_description or ""

    new_tool = Tool(func, name=func.__name__, description=tool_desc, params=func_params)

    DEFAULT_TOOLS[new_tool.name] = new_tool

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@tool
def fake_tool(a: str = "a", b: str = "b") -> str:
    return f"{a}:{b} - fake!"


class Agent(Component[AgentResponse]):
    def __init__(
        self, llm: LLM, tools: dict[str, Tool] | None = None, max_loops: int = 30
    ):
        self.llm = llm
        self.max_loops = max_loops
        self.tools = tools or DEFAULT_TOOLS
        self.llm.tools = self.tools

    def invoke(self, prompt, *args, **kwargs) -> AgentResponse:
        loop = 0
        res = self.llm.invoke(prompt)
        while loop < self.max_loops:
            loop += 1

            # if no tools were called, the agent loop is done
            if not res.tool_calls:
                break

            for tool_call in res.tool_calls:
                try:
                    tool_res = tool_call.tool.call(*tool_call.args, **tool_call.kwargs)
                    self.llm.context.add_message(
                        "tool", tool_res, tool_call.tool_call_id
                    )
                except Exception as e:
                    self.llm.context.add_message(
                        "tool", f"Tool call failed! Error: {e}", tool_call.tool_call_id
                    )
                    log.exception(f"Tool call failed: {tool_call.tool.name}")

            res = self.llm.invoke()

        agent_res = AgentResponse(text=res.text)
        return agent_res
