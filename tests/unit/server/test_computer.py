import pytest
from unittest.mock import MagicMock

from daystrom.components.base import Agent, AgentResponse
from daystrom.server.computer import Computer
from daystrom.server.transports.base import Transport


class StubTransport(Transport):
    async def start(self):
        pass

    async def stop(self):
        pass


def make_computer():
    mock_agent = MagicMock(spec=Agent)
    mock_agent.invoke.return_value = AgentResponse(text="hello")
    factory = MagicMock(return_value=mock_agent)
    return Computer(agent_factory=factory), factory, mock_agent


class TestRegister:
    def test_sets_computer_on_transport(self):
        computer, _, _ = make_computer()
        transport = StubTransport()
        computer.register(transport)
        assert transport.computer is computer

    def test_appends_to_transports(self):
        computer, _, _ = make_computer()
        t1, t2 = StubTransport(), StubTransport()
        computer.register(t1)
        computer.register(t2)
        assert computer.transports == [t1, t2]


class TestHandleMessage:
    @pytest.mark.asyncio
    async def test_creates_agent_on_first_message(self):
        computer, factory, _ = make_computer()
        await computer.handle_message("telegram:123", "hi")
        factory.assert_called_once()
        assert "telegram:123" in computer.sessions

    @pytest.mark.asyncio
    async def test_returns_agent_response(self):
        computer, _, _ = make_computer()
        response = await computer.handle_message("telegram:123", "hi")
        assert response == "hello"

    @pytest.mark.asyncio
    async def test_reuses_agent_for_same_session(self):
        computer, factory, _ = make_computer()
        await computer.handle_message("telegram:123", "hi")
        await computer.handle_message("telegram:123", "there")
        factory.assert_called_once()

    @pytest.mark.asyncio
    async def test_creates_separate_agents_per_session(self):
        computer, factory, _ = make_computer()
        await computer.handle_message("telegram:123", "hi")
        await computer.handle_message("telegram:456", "hi")
        assert factory.call_count == 2
        assert "telegram:123" in computer.sessions
        assert "telegram:456" in computer.sessions


class TestNewSession:
    @pytest.mark.asyncio
    async def test_resets_session(self):
        computer, _, _ = make_computer()
        await computer.handle_message("telegram:123", "hi")
        assert "telegram:123" in computer.sessions

        response = await computer.handle_message("telegram:123", "/new-session")
        assert response == "Session reset."
        assert "telegram:123" not in computer.sessions

    @pytest.mark.asyncio
    async def test_strips_whitespace(self):
        computer, _, _ = make_computer()
        await computer.handle_message("telegram:123", "hi")
        response = await computer.handle_message("telegram:123", "  /new-session  ")
        assert response == "Session reset."
        assert "telegram:123" not in computer.sessions

    @pytest.mark.asyncio
    async def test_nonexistent_session(self):
        computer, _, _ = make_computer()
        response = await computer.handle_message("telegram:123", "/new-session")
        assert response == "Session reset."

    @pytest.mark.asyncio
    async def test_new_agent_after_reset(self):
        computer, factory, _ = make_computer()
        await computer.handle_message("telegram:123", "hi")
        await computer.handle_message("telegram:123", "/new-session")
        await computer.handle_message("telegram:123", "hi again")
        assert factory.call_count == 2
