import asyncio
import logging
import signal
from typing import Callable

from daystrom.components.base import Agent
from daystrom.components.openai import OpenAIChatCompletions
from daystrom.providers import Provider

from .transports.base import Transport

log = logging.getLogger(__name__)


class Computer:
    def __init__(self, agent_factory: Callable[[], Agent]):
        self.agent_factory = agent_factory
        self.transports: list[Transport] = []
        self.sessions: dict[str, Agent] = {}
        self._stop_event = asyncio.Event()

    def register(self, transport: Transport):
        transport.computer = self
        self.transports.append(transport)

    def get_or_create_agent(self, session_id: str) -> Agent:
        if session_id not in self.sessions:
            self.sessions[session_id] = self.agent_factory()
        return self.sessions[session_id]

    async def handle_message(self, session_id: str, text: str) -> str:
        log.info(f"Received message for session {session_id}: {text[:50] + "..." if len(text) > 50 else ""}")

        text = text.strip()
        command, args = text.split(" ")
        #match text:
        match command:
            case "/new-session":
                model_dict = {
                    "claude": "anthropic/claude-opus-4.6",
                    "gpt": "openai/gpt-5.2",
                    "minimax": "minimax/minimax-m2.5",
                    "kimi": "moonshotai/kimi-k2.5",
                    "glm": "z-ai/glm-5",
                }
                if args:
                    model = model_dict.get(args[0])
                    if model:
                        def agent_factory():
                            llm = OpenAIChatCompletions(provider=Provider.OPENROUTER, model=model)
                            agent = Agent(llm=llm)
                            return agent
                        self.agent_factory = agent_factory

                self.sessions.pop(session_id, None)
                return "Session reset."

        agent = self.get_or_create_agent(session_id)
        response = await asyncio.to_thread(agent.invoke, text)
        return response.text

    def run(self):
        asyncio.run(self._run())

    async def _run(self):
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._stop_event.set)

        started_transports: list[Transport] = []
        try:
            for transport in self.transports:
                await transport.start()
                started_transports.append(transport)

            log.info("Computer is running")
            await self._stop_event.wait()
            log.info("Shutting down")
        finally:
            # Ensure all started transports are stopped, even if startup or run fails.
            for transport in reversed(started_transports):
                try:
                    await transport.stop()
                except Exception:
                    log.exception("Error while stopping transport %r", transport)
