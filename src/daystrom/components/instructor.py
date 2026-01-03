import json
import os
from dataclasses import dataclass
from typing import TypeVar

import instructor
from openai.types.chat import (  # ChatCompletionDeveloperMessageParam, # should probably use this one, it replaces system_message on some newer models apparently; ChatCompletionFunctionMessageParam,; ChatCompletionToolMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel

from daystrom.components import Component, Context
from daystrom.exceptions import InvalidComponentError


@dataclass
class Provider:
    name: str
    display_name: str


class Providers:
    openai = Provider(name="openai", display_name="Open AI")
    openrouter = Provider(name="openrouter", display_name="OpenRouter")


InstructorResponseT = TypeVar("InstructorResponseT", bound=BaseModel)


class Instructor(Component[InstructorResponseT]):
    client: instructor.Instructor
    response_model: type[InstructorResponseT]
    context: Context

    def __init__(
        self,
        provider: Provider,
        model: str,
        response_model: type[InstructorResponseT],
        api_key: str | None = None,
        context: Context | None = None,
    ):
        match provider:
            case Providers.openrouter:
                self.client = instructor.from_provider(
                    f"{provider.name}/{model}",
                    api_key=os.getenv("OPENROUTER_API_KEY") or api_key,
                )

        if not self.client:
            raise InvalidComponentError(
                self.__class__.__name__, f"Unsupported provider: {provider.name}"
            )

        self.response_model = response_model
        if context:
            self.context = context
        else:
            self.context = Context()

    def invoke(self, prompt: str) -> InstructorResponseT:
        self.context.add_message("user", prompt)
        messages = self._get_prompt_context()
        response = self.client.create(
            response_model=self.response_model, messages=messages, max_retries=3
        )
        return response

    def _get_prompt_context(self) -> list[ChatCompletionMessageParam]:
        """
        Returns the messages in the context formatted for OpenRouter API
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
                        ChatCompletionSystemMessageParam(
                            role="system", content=msg.text
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
