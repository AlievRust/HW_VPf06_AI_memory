from __future__ import annotations

import asyncio
import logging
import mimetypes
import os

import aiohttp
from aiogram import Bot, Dispatcher, Router
from aiogram.enums import ChatAction
from aiogram.types import Message

from .config import settings
from .content import normalize_text, sanitize_filename
from .memory import LongTermMemory, ShortTermMemory, build_dialog_prompt
from .openai_service import OpenAIService


logging.basicConfig(
    level=os.getenv("BOT_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


router = Router()

openai_service = OpenAIService.create(
    api_key=settings.openai_api_key,
    chat_model=settings.openai_model,
    embedding_model=settings.openai_embedding_model,
    temperature=settings.openai_temperature,
    max_output_tokens=settings.openai_max_output_tokens,
)
short_term_memory = ShortTermMemory(settings.max_short_term_messages)
long_term_memory = LongTermMemory(
    storage_dir=settings.chroma_path,
    uploads_dir=settings.uploads_path,
    openai_service=openai_service,
    chunk_size=settings.store_chunk_size,
    chunk_overlap=settings.store_chunk_overlap,
)


async def download_telegram_file_bytes(bot_token: str, bot: Bot, file_id: str) -> bytes:
    telegram_file = await bot.get_file(file_id)
    if not telegram_file.file_path:
        raise RuntimeError("Telegram did not return a file path")

    file_url = f"https://api.telegram.org/file/bot{bot_token}/{telegram_file.file_path}"
    async with aiohttp.ClientSession() as session:
        async with session.get(file_url) as response:
            response.raise_for_status()
            return await response.read()


def _message_text(message: Message) -> str:
    return (message.text or message.caption or "").strip()


async def _answer_with_memory(message: Message, user_text: str) -> None:
    user_id = message.from_user.id if message.from_user else message.chat.id
    history = short_term_memory.get(user_id)

    relevant_memories = await long_term_memory.search(
        user_id=user_id,
        query=user_text,
        top_k=settings.long_term_top_k,
    )

    prompt = build_dialog_prompt(
        system_prompt=settings.system_prompt,
        history=history,
        memories=relevant_memories,
        user_message=user_text,
    )

    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    answer = await openai_service.answer(prompt)
    if not answer:
        answer = "Я не смог сформировать ответ. Попробуйте переформулировать запрос."

    short_term_memory.add(user_id, "user", user_text)
    short_term_memory.add(user_id, "assistant", answer)
    await message.answer(answer)


async def _store_text_payload(message: Message, payload: str) -> None:
    user_id = message.from_user.id if message.from_user else message.chat.id
    cleaned = payload.strip()
    if cleaned.lower().startswith("/store"):
        cleaned = cleaned[len("/store") :].strip()

    if not cleaned:
        await message.answer("Добавьте текст после /store, например: /store мой профиль и интересы.")
        return

    result = await long_term_memory.store_text(
        user_id=user_id,
        source_name=f"manual_{message.message_id}",
        text=cleaned,
    )
    await message.answer(
        f"Текст сохранен.\nИсточник: {result.source_path.name}\nФрагментов в векторной БД: {result.chunks_stored}"
    )


async def _store_document_payload(message: Message) -> None:
    user_id = message.from_user.id if message.from_user else message.chat.id
    document = message.document
    if not document:
        await message.answer("Не вижу файла. Пришлите PDF или текстовый документ с подписью /store.")
        return

    file_name = sanitize_filename(document.file_name or f"document_{message.message_id}")
    mime_type = document.mime_type or mimetypes.guess_type(file_name)[0] or ""
    raw_bytes = await download_telegram_file_bytes(settings.telegram_bot_token, bot=message.bot, file_id=document.file_id)

    if "pdf" in mime_type.lower() or file_name.lower().endswith(".pdf"):
        result = await long_term_memory.store_pdf(
            user_id=user_id,
            source_name=file_name,
            pdf_bytes=raw_bytes,
        )
        await message.answer(
            f"PDF сохранен.\nИсточник: {result.source_path.name}\nФрагментов в векторной БД: {result.chunks_stored}"
        )
        return

    text = raw_bytes.decode("utf-8", errors="ignore")
    result = await long_term_memory.store_text(
        user_id=user_id,
        source_name=file_name,
        text=text,
    )
    await message.answer(
        f"Текстовый файл сохранен.\nИсточник: {result.source_path.name}\nФрагментов в векторной БД: {result.chunks_stored}"
    )


@router.message()
async def handle_message(message: Message) -> None:
    text = _message_text(message)

    if not text and not message.document:
        await message.answer("Отправьте /start, текстовое сообщение или документ с подписью /store.")
        return

    if text.startswith("/start"):
        await message.answer(
            "Привет! Я бот с краткосрочной и долговременной памятью.\n\n"
            "Команды:\n"
            "/start - справка\n"
            "/store <текст> - сохранить текст в долговременную память\n"
            "/store + PDF/текстовый документ - сохранить файл в долговременную память\n\n"
            "Обычные сообщения я отвечаю с учетом последних 20 сообщений и релевантных фрагментов из памяти."
        )
        return

    if text.startswith("/store"):
        if message.document:
            await _store_document_payload(message)
        else:
            await _store_text_payload(message, text)
        return

    if message.document:
        await message.answer("Чтобы сохранить документ, пришлите его с подписью /store.")
        return

    await _answer_with_memory(message, normalize_text(text))


async def main() -> None:
    bot = Bot(token=settings.telegram_bot_token)
    dispatcher = Dispatcher()
    dispatcher.include_router(router)

    logger.info("Bot started")
    await dispatcher.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
