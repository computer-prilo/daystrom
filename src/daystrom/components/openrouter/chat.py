import os

from openrouter import OpenRouter
from openrouter.components import AssistantMessage
from openrouter.components import Message as OpenRouterMessage  # , ToolResponseMessage
from openrouter.components import SystemMessage, UserMessage

from daystrom import Provider
from daystrom.components import LLM, Context, LLMResponse


class OpenRouterChat(LLM):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        context: Context | None = None,
    ):
        super().__init__(model=model, context=context, provider=Provider.OPENROUTER)
        self.client = OpenRouter(api_key=api_key or self.provider.value.get_api_key())

    def invoke(self, prompt) -> LLMResponse:
        response = LLMResponse(text="".join(self.invoke_stream(prompt)), tool_calls=[])
        return response

    def invoke_stream(self, prompt):
        self.context.add_message("user", prompt)
        messages = self._get_prompt_context()
        res = self.client.chat.send(messages=messages, model=self.model, stream=True)

        response_content = ""
        for event in res:
            if isinstance(event.choices[0].delta.content, str):
                self.track_usage(event.usage)
                content_chunk = event.choices[0].delta.content
                response_content += content_chunk
                yield content_chunk
        self.context.add_message("assistant", response_content)

    async def ainvoke(self, prompt) -> str:
        return "".join([chunk async for chunk in self.ainvoke_stream(prompt)])

    async def ainvoke_stream(self, prompt):
        self.context.add_message("user", prompt)
        messages = self._get_prompt_context()
        res = await self.client.chat.send_async(
            messages=messages, model=self.model, stream=True
        )

        response_content = ""
        async for event in res:
            if isinstance(event.choices[0].delta.content, str):
                self.track_usage(event.usage)
                content_chunk = event.choices[0].delta.content
                response_content += content_chunk
                yield content_chunk
        self.context.add_message("assistant", response_content)

    def track_usage(self, usage):
        if usage:
            self.output_tokens += usage.completion_tokens
            self.input_tokens += usage.prompt_tokens

    def _get_prompt_context(self) -> list[OpenRouterMessage]:
        """
        Returns the messages in the context formatted for OpenRouter API
        """
        fmt_messages = []
        for msg in self.context.messages:
            match msg.role:
                case "user":
                    fmt_messages.append(UserMessage(content=msg.text))
                case "assistant":
                    fmt_messages.append(AssistantMessage(content=msg.text))
                case "system":
                    fmt_messages.append(SystemMessage(content=msg.text))

        return fmt_messages
