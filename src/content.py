from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader


@dataclass(slots=True)
class TextChunk:
    text: str
    metadata: dict[str, object]


def sanitize_filename(name: str) -> str:
    # Убираем символы, которые могут быть опасны или неудобны в именах файлов.
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return cleaned or "file"


def normalize_text(text: str) -> str:
    # Приводим текст к более компактному виду: убираем NUL-символы и лишние пробелы.
    cleaned = text.replace("\x00", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def split_text(text: str, max_chars: int, overlap: int) -> list[str]:
    # Разбиваем длинный текст на куски, чтобы векторная БД хранила его частями.
    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + max_chars, length)
        if end < length:
            # Стараемся резать по пробелу, чтобы не ломать слова на середине.
            split_at = text.rfind(" ", start, end)
            if split_at <= start + max_chars // 2:
                split_at = end
            end = split_at
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(end - overlap, 0)

    return chunks


def extract_pdf_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    # Извлекаем текст постранично, чтобы потом можно было хранить источник и номер страницы.
    reader = PdfReader(BytesIO(pdf_bytes))
    pages: list[tuple[int, str]] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = normalize_text(text)
        if text:
            pages.append((index, text))
    return pages


def chunk_text_source(
    *,
    text: str,
    source_name: str,
    source_type: str,
    max_chars: int,
    overlap: int,
) -> list[TextChunk]:
    # Оборачиваем обычный текст в чанки с метаданными об источнике.
    chunks: list[TextChunk] = []
    for chunk_index, chunk in enumerate(split_text(text, max_chars, overlap), start=1):
        chunks.append(
            TextChunk(
                text=chunk,
                metadata={
                    "source_name": source_name,
                    "source_type": source_type,
                    "chunk_index": chunk_index,
                },
            )
        )
    return chunks


def chunk_pdf_source(
    *,
    pdf_bytes: bytes,
    source_name: str,
    max_chars: int,
    overlap: int,
) -> list[TextChunk]:
    # Для PDF сохраняем и текст, и номер страницы, чтобы потом удобно ссылаться на источник.
    chunks: list[TextChunk] = []
    for page_number, page_text in extract_pdf_pages(pdf_bytes):
        page_chunks = split_text(page_text, max_chars, overlap)
        for page_chunk_index, chunk in enumerate(page_chunks, start=1):
            chunks.append(
                TextChunk(
                    text=chunk,
                    metadata={
                        "source_name": source_name,
                        "source_type": "pdf",
                        "page_number": page_number,
                        "page_chunk_index": page_chunk_index,
                    },
                )
            )
    return chunks


def trim_text(text: str, limit: int = 500) -> str:
    # Обрезаем длинные фрагменты, чтобы ответы и логи оставались читабельными.
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def format_messages(messages: Iterable[tuple[str, str]]) -> str:
    lines: list[str] = []
    for role, content in messages:
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
