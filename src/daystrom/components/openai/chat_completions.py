import json
import os
from typing import get_origin

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionToolUnionParam,
    ChatCompletionUserMessageParam,
)
from openai.types.shared_params import FunctionDefinition

from daystrom.components import LLM, Context, LLMResponse, Tool, ToolCall
from daystrom.exceptions import InvalidComponentError


class OpenAIChatCompletions(LLM):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        url: str | None = None,
        context: Context | None = None,
        tools: dict[str, Tool] | None = None,
    ):
        self.model = model
        self.client = OpenAI(
            base_url=url,
            api_key=os.getenv("OPENAI_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or api_key,
        )
        super().__init__(context=context, tools=tools or {})

    def invoke(self, prompt: str | None = None) -> LLMResponse:
        if prompt:
            self.context.add_message("user", prompt)

        messages = self._get_prompt_context()
        completion = self.client.chat.completions.create(
            model=self.model,
            tools=self._get_tool_context(),
            messages=messages,
        )
        completion_text = completion.choices[0].message.content or ""

        # tool calls from openai api
        tool_calls = []
        for tool_call in completion.choices[0].message.tool_calls or []:
            if isinstance(tool_call, ChatCompletionMessageFunctionToolCall):
                tool = ToolCall(
                    tool=self.tools[tool_call.function.name],
                    tool_call_id=tool_call.id,
                    args=[],
                    kwargs=(
                        json.loads(tool_call.function.arguments)
                        if tool_call.function.arguments
                        else {}
                    ),
                )
                tool_calls.append(tool)
            else:
                raise InvalidComponentError(
                    self.__class__.__name__,
                    "Found unsupported tool call - missing 'function' attribute",
                )

        self.context.add_message(
            role="assistant", text=completion_text, tool_calls=tool_calls
        )
        response = LLMResponse(text=completion_text, tool_calls=tool_calls)
        return response

    def _get_prompt_context(self) -> list[ChatCompletionMessageParam]:
        """
        Returns the messages in the context formatted for OpenAI API
        """
        fmt_messages = []
        for msg in self.context.messages:
            match msg.role:
                case "user":
                    fmt_messages.append(
                        ChatCompletionUserMessageParam(role="user", content=msg.text)
                    )
                case "assistant":
                    tool_calls = []
                    for tool_call in msg.tool_calls:
                        # ChatCompletionMessageToolCallUnionParam
                        tool_calls.append(
                            {
                                "function": {
                                    "name": tool_call.tool.name,
                                    "arguments": json.dumps(tool_call.kwargs),
                                },
                                "type": "function",
                                "id": tool_call.tool_call_id,
                            }
                        )
                    if tool_calls:
                        fmt_messages.append(
                            ChatCompletionAssistantMessageParam(
                                role="assistant",
                                content=msg.text,
                                tool_calls=tool_calls,
                            )
                        )
                    else:
                        fmt_messages.append(
                            ChatCompletionAssistantMessageParam(
                                role="assistant", content=msg.text
                            )
                        )
                case "system":
                    fmt_messages.append(
                        ChatCompletionDeveloperMessageParam(
                            role="developer", content=msg.text
                        )
                    )
                case "tool":
                    fmt_messages.append(
                        ChatCompletionToolMessageParam(
                            role="tool", content=msg.text, tool_call_id=msg.tool_call_id
                        )
                    )
                case _:
                    raise ValueError(
                        f"Unsupported message role: {msg.role} for {self.__class__.__name__}"
                    )

        return fmt_messages

    def _get_tool_context(self) -> list[ChatCompletionToolUnionParam]:
        tool_schemas = []

        for tool in self.tools.values():
            function = FunctionDefinition(
                name=tool.name,
                description=tool.description,
                parameters=self._format_tool_params(tool),
            )
            tool_schema = ChatCompletionToolParam(function=function, type="function")
            tool_schemas.append(tool_schema)
        return tool_schemas

    def _format_tool_params(self, tool: Tool) -> dict:
        params = {"type": "object", "properties": {}}
        required_params = []
        type_map = {
            dict: "object",
            list: "array",
            tuple: "array",
            str: "string",
            int: "integer",
            float: "number",
            None: "null",
            bool: "boolean",
        }

        for pname, pinfo in tool.params.items():
            params["properties"][pname] = {
                "type": type_map[get_origin(pinfo["type"]) or pinfo["type"]],
                "description": pinfo["description"],
            }

            param_items = pinfo.get("items")
            if param_items is not None:
                param_type = param_items["type"]
                params["properties"][pname]["items"] = {
                    "type": type_map[get_origin(param_type) or param_type],
                }

            if pinfo["required"]:
                required_params.append(pname)

        if required_params:
            params["required"] = required_params

        return params
