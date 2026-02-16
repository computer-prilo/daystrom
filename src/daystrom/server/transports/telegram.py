import logging

try:
    from telegram import Update
    from telegram.ext import Application, MessageHandler, filters
except ImportError:
    raise ImportError(
        "python-telegram-bot is required for TelegramTransport. "
        "Install with: uv add daystrom[telegram] (or run with --extra telegram)"
    )

from .base import Transport

log = logging.getLogger(__name__)

TELEGRAM_MESSAGE_LIMIT = 4096


class TelegramTransport(Transport):
    def __init__(self, token: str):
        self.token = token
        self._app: Application | None = None

    async def start(self):
        self._app = Application.builder().token(self.token).build()

        if not self._app or not self._app.updater:
            raise RuntimeError("Failed to create Telegram application")

        self._app.add_handler(MessageHandler(filters.TEXT, self._on_message))
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()
        log.info("Telegram transport started")

    async def stop(self):
        if self._app and self._app.updater:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            log.info("Telegram transport stopped")

    async def _on_message(self, update: Update, _context):
        if not update.message or not update.message.text or not update.effective_chat:
            return

        session_id = f"telegram:{update.effective_chat.id}"
        response = await self.computer.handle_message(session_id, update.message.text)

        if not response:
            return

        for chunk in _split_message(response):
            await update.message.reply_text(chunk)


def _split_message(text: str) -> list[str]:
    if len(text) <= TELEGRAM_MESSAGE_LIMIT:
        return [text]

    chunks = []
    while text:
        if len(text) <= TELEGRAM_MESSAGE_LIMIT:
            chunks.append(text)
            break

        split_at = text.rfind("\n", 0, TELEGRAM_MESSAGE_LIMIT)
        if split_at == -1:
            split_at = TELEGRAM_MESSAGE_LIMIT

        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")

    if not chunks:
        chunks.append("Something went wrong, no response from the Computer")

    return chunks
