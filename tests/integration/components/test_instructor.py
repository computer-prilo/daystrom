import pytest
from pydantic import BaseModel

from daystrom import Provider
from daystrom.components import Instructor


@pytest.fixture
def client():
    class CityState(BaseModel):
        city_name: str
        state: str

    return Instructor(
        Provider.OPENROUTER,
        model="anthropic/claude-haiku-4.5",
        response_model=CityState,
    )


@pytest.fixture
def message():
    return "Give me information on a random city."


def test_invoke_returns_response_model(client, message):
    result = client.invoke(message)

    assert isinstance(result, client.response_model)
    assert isinstance(result.city_name, str)
    assert isinstance(result.state, str)

