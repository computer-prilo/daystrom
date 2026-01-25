import pytest

from daystrom.components import Context, LLMResponse
from daystrom.components.openrouter import OpenRouterChat


@pytest.fixture
def client():
    return OpenRouterChat(model="anthropic/claude-haiku-4.5")


@pytest.fixture
def context():
    context = Context()
    context.add_message(
        role="user",
        text="Give me a short response as a test that API functionality is working",
    )
    return context


def test_invoke_stream(client, context):
    res = "".join(client.invoke_stream(context))
    assert isinstance(res, str)
    assert res != ""


def test_invoke(client, context):
    res = client.invoke(context)
    assert isinstance(res, LLMResponse)
    assert res.text != ""


@pytest.mark.asyncio
async def test_ainvoke_stream(client, context):
    res = "".join([chunk async for chunk in client.ainvoke_stream(context)])
    assert isinstance(res, str)
    assert res != ""


@pytest.mark.asyncio
async def test_ainvoke(client, context):
    res = await client.ainvoke(context)
    assert isinstance(res, str)
    assert res != ""
