from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _default_system_prompt() -> str:
    return (
        "Ты полезный Telegram-бот с краткосрочной и долговременной памятью. "
        "Используй только то, что есть в текущем диалоге и в найденных фрагментах "
        "из долговременной памяти. Если данных недостаточно, честно скажи об этом "
        "и не выдумывай факты. Отвечай на языке пользователя."
    )


@dataclass(slots=True)
class Settings:
    telegram_bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", "").strip())
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", "").strip())
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-5.4-nano").strip())
    openai_embedding_model: str = field(default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip())
    openai_temperature: float = field(default_factory=lambda: float(os.getenv("OPENAI_TEMPERATURE", "0.3")))
    openai_max_output_tokens: int = field(default_factory=lambda: int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "800")))
    max_short_term_messages: int = field(default_factory=lambda: int(os.getenv("MAX_SHORT_TERM_MESSAGES", "20")))
    long_term_top_k: int = field(default_factory=lambda: int(os.getenv("LONG_TERM_TOP_K", "4")))
    store_chunk_size: int = field(default_factory=lambda: int(os.getenv("STORE_CHUNK_SIZE", "1200")))
    store_chunk_overlap: int = field(default_factory=lambda: int(os.getenv("STORE_CHUNK_OVERLAP", "200")))
    chroma_path: Path = field(default_factory=lambda: Path(os.getenv("CHROMA_PATH", str(BASE_DIR / "data" / "chroma"))))
    uploads_path: Path = field(default_factory=lambda: Path(os.getenv("UPLOADS_PATH", str(BASE_DIR / "data" / "uploads"))))
    system_prompt: str = field(default_factory=lambda: os.getenv("OPENAI_SYSTEM_PROMPT", _default_system_prompt()).strip())

    def __post_init__(self) -> None:
        if not self.telegram_bot_token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is required")
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required")


settings = Settings()
