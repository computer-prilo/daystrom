import pytest

from daystrom.exceptions import InvalidComponentError


def test_invalid_component_error():
    with pytest.raises(InvalidComponentError):
        raise InvalidComponentError("TestComponent", "This is a test error message.")
