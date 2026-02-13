from daystrom.server.transports.telegram import _split_message, TELEGRAM_MESSAGE_LIMIT


class TestSplitMessage:
    def test_short_message(self):
        assert _split_message("hello") == ["hello"]

    def test_exact_limit(self):
        text = "a" * TELEGRAM_MESSAGE_LIMIT
        assert _split_message(text) == [text]

    def test_over_limit_splits_at_newline(self):
        line = "a" * 2000
        text = f"{line}\n{line}\n{line}"
        chunks = _split_message(text)
        assert len(chunks) == 2
        assert all(len(c) <= TELEGRAM_MESSAGE_LIMIT for c in chunks)
        assert "\n".join(chunks) == text

    def test_over_limit_no_newline_splits_at_limit(self):
        text = "a" * 5000
        chunks = _split_message(text)
        assert len(chunks) == 2
        assert chunks[0] == "a" * TELEGRAM_MESSAGE_LIMIT
        assert chunks[1] == "a" * (5000 - TELEGRAM_MESSAGE_LIMIT)

    def test_empty_string(self):
        assert _split_message("") == [""]

    def test_preserves_all_content(self):
        lines = [f"line {i}: {'x' * 200}" for i in range(30)]
        text = "\n".join(lines)
        chunks = _split_message(text)
        reassembled = "\n".join(chunks)
        assert reassembled == text
