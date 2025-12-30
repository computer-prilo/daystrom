import pytest

from daystrom.components import DEFAULT_TOOLS, tool


def test_tool_decorator_basic():
    """Test @tool decorator creates Tool with correct name and description."""

    @tool
    def my_test_tool(x: str) -> str:
        """Short description.

        This is the long description.
        """
        return x

    assert "my_test_tool" in DEFAULT_TOOLS
    t = DEFAULT_TOOLS["my_test_tool"]
    assert t.name == "my_test_tool"
    assert t.description == "This is the long description."
    assert t.callable is not None

    # Cleanup
    del DEFAULT_TOOLS["my_test_tool"]


def test_tool_decorator_short_description_fallback():
    """Test @tool decorator uses short description when no long description."""

    @tool
    def short_desc_tool() -> str:
        """Only a short description."""
        return "ok"

    t = DEFAULT_TOOLS["short_desc_tool"]
    assert t.description == "Only a short description."

    # Cleanup
    del DEFAULT_TOOLS["short_desc_tool"]


def test_tool_decorator_extracts_params():
    """Test @tool decorator extracts parameter info correctly."""

    @tool
    def param_tool(required_param: str, optional_param: int = 10) -> str:
        """A tool with params.

        Args:
            required_param: A required string parameter.
            optional_param: An optional integer parameter.
        """
        return f"{required_param}-{optional_param}"

    t = DEFAULT_TOOLS["param_tool"]

    assert "required_param" in t.params
    assert t.params["required_param"]["type"] == str
    assert t.params["required_param"]["required"] is True

    assert "optional_param" in t.params
    assert t.params["optional_param"]["type"] == int
    assert t.params["optional_param"]["required"] is False

    # Cleanup
    del DEFAULT_TOOLS["param_tool"]


def test_tool_decorator_list_param():
    """Test @tool decorator handles list type parameters with items."""

    @tool
    def list_tool(tags: list[str]) -> str:
        """A tool with list param."""
        return ",".join(tags)

    t = DEFAULT_TOOLS["list_tool"]

    assert t.params["tags"]["type"] == list[str]
    assert "items" in t.params["tags"]
    assert t.params["tags"]["items"]["type"] == str

    # Cleanup
    del DEFAULT_TOOLS["list_tool"]


def test_tool_decorator_rejects_args():
    """Test @tool decorator raises TypeError for *args."""
    with pytest.raises(TypeError, match="\\*args is not supported"):

        @tool
        def bad_tool(*args) -> str:
            """Bad tool."""
            return str(args)


def test_tool_decorator_rejects_kwargs():
    """Test @tool decorator raises TypeError for **kwargs."""
    with pytest.raises(TypeError, match="\\*\\*kwargs is not supported"):

        @tool
        def bad_tool(**kwargs) -> str:
            """Bad tool."""
            return str(kwargs)


def test_tool_decorator_rejects_multi_type_tuple():
    """Test @tool decorator raises TypeError for multi-type tuple."""
    with pytest.raises(TypeError, match="Only single-type iterables"):

        @tool
        def bad_tool(data: tuple[str, int]) -> str:
            """Bad tool."""
            return str(data)


def test_tool_decorator_callable_still_works():
    """Test decorated function still works as expected."""

    @tool
    def working_tool(a: str, b: str = "default") -> str:
        """A working tool."""
        return f"{a}-{b}"

    # Direct call should work
    result = working_tool("hello", "world")
    assert result == "hello-world"

    # Default argument should work
    result = working_tool("hello")
    assert result == "hello-default"

    # Cleanup
    del DEFAULT_TOOLS["working_tool"]
