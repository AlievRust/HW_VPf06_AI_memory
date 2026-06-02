from __future__ import annotations

import asyncio
import hashlib
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import chromadb

from .content import (
    TextChunk,
    chunk_pdf_source,
    chunk_text_source,
    ensure_parent_dir,
    sanitize_filename,
    trim_text,
)
from .openai_service import OpenAIService


@dataclass(slots=True)
class MemorySearchResult:
    text: str
    metadata: dict[str, object]
    distance: float | None


@dataclass(slots=True)
class StoreResult:
    source_path: Path
    chunks_stored: int
    source_name: str


class ShortTermMemory:
    def __init__(self, max_messages: int) -> None:
        self.max_messages = max_messages
        self._messages: dict[int, deque[tuple[str, str]]] = defaultdict(lambda: deque(maxlen=self.max_messages))

    def add(self, user_id: int, role: str, content: str) -> None:
        self._messages[user_id].append((role, trim_text(content, 900)))

    def get(self, user_id: int) -> list[tuple[str, str]]:
        return list(self._messages[user_id])


class LongTermMemory:
    def __init__(
        self,
        *,
        storage_dir: Path,
        uploads_dir: Path,
        openai_service: OpenAIService,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self.storage_dir = storage_dir
        self.uploads_dir = uploads_dir
        self.openai_service = openai_service
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.storage_dir))
        self.collection = self.client.get_or_create_collection(
            name="user_memories",
            metadata={"hnsw:space": "cosine"},
        )

    def _user_dir(self, user_id: int) -> Path:
        path = self.uploads_dir / str(user_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _make_source_path(self, user_id: int, source_name: str, suffix: str) -> Path:
        timestamp = time.time_ns()
        safe_name = sanitize_filename(Path(source_name).stem)
        file_name = f"{timestamp}_{safe_name}{suffix}"
        return self._user_dir(user_id) / file_name

    def _index_name(self, user_id: int, source_name: str, text: str, chunk_index: int) -> str:
        fingerprint = hashlib.sha1(f"{user_id}:{source_name}:{chunk_index}:{text}".encode("utf-8")).hexdigest()
        return f"{user_id}-{fingerprint}"

    async def store_text(self, user_id: int, source_name: str, text: str) -> StoreResult:
        source_path = self._make_source_path(user_id, source_name, ".txt")
        ensure_parent_dir(source_path)
        source_path.write_text(text, encoding="utf-8")

        chunks = chunk_text_source(
            text=text,
            source_name=source_name,
            source_type="text",
            max_chars=self.chunk_size,
            overlap=self.chunk_overlap,
        )
        await self._store_chunks(user_id=user_id, source_name=source_name, source_path=source_path, chunks=chunks)
        return StoreResult(source_path=source_path, chunks_stored=len(chunks), source_name=source_name)

    async def store_pdf(self, user_id: int, source_name: str, pdf_bytes: bytes) -> StoreResult:
        source_path = self._make_source_path(user_id, source_name, ".pdf")
        ensure_parent_dir(source_path)
        source_path.write_bytes(pdf_bytes)

        chunks = chunk_pdf_source(
            pdf_bytes=pdf_bytes,
            source_name=source_name,
            max_chars=self.chunk_size,
            overlap=self.chunk_overlap,
        )
        await self._store_chunks(user_id=user_id, source_name=source_name, source_path=source_path, chunks=chunks)
        return StoreResult(source_path=source_path, chunks_stored=len(chunks), source_name=source_name)

    async def _store_chunks(
        self,
        *,
        user_id: int,
        source_name: str,
        source_path: Path,
        chunks: list[TextChunk],
    ) -> None:
        if not chunks:
            return

        embeddings = await self.openai_service.embed_texts([chunk.text for chunk in chunks])

        ids = [self._index_name(user_id, source_name, chunk.text, index) for index, chunk in enumerate(chunks, start=1)]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "user_id": str(user_id),
                "source_name": source_name,
                "source_path": str(source_path),
                **chunk.metadata,
            }
            for chunk in chunks
        ]

        def _upsert() -> None:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )

        await asyncio.to_thread(_upsert)

    async def search(self, user_id: int, query: str, top_k: int) -> list[MemorySearchResult]:
        if not query.strip():
            return []

        query_embedding = (await self.openai_service.embed_texts([query]))[0]

        def _query() -> dict[str, object]:
            return self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where={"user_id": str(user_id)},
            )

        result = await asyncio.to_thread(_query)
        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("distances") or [[]]

        hits: list[MemorySearchResult] = []
        for text, metadata, distance in zip(documents[0], metadatas[0], distances[0], strict=False):
            hits.append(
                MemorySearchResult(
                    text=str(text),
                    metadata=dict(metadata or {}),
                    distance=float(distance) if distance is not None else None,
                )
            )
        return hits


def format_long_term_hits(hits: list[MemorySearchResult]) -> str:
    if not hits:
        return "Релевантные фрагменты не найдены."

    lines: list[str] = []
    for index, hit in enumerate(hits, start=1):
        meta = hit.metadata
        origin_bits = [str(meta.get("source_name", "unknown source"))]
        if meta.get("source_type") == "pdf" and meta.get("page_number"):
            origin_bits.append(f"page {meta['page_number']}")
        if meta.get("chunk_index"):
            origin_bits.append(f"chunk {meta['chunk_index']}")
        if hit.distance is not None:
            origin_bits.append(f"distance={hit.distance:.4f}")
        origin = ", ".join(origin_bits)
        lines.append(f"{index}. [{origin}] {trim_text(hit.text, 700)}")
    return "\n".join(lines)


def build_dialog_prompt(
    *,
    system_prompt: str,
    history: list[tuple[str, str]],
    memories: list[MemorySearchResult],
    user_message: str,
) -> str:
    history_text = "\n".join(f"{role.title()}: {content}" for role, content in history) or "История пуста."
    memories_text = format_long_term_hits(memories)
    return (
        f"System instructions:\n{system_prompt}\n\n"
        f"Recent dialogue:\n{history_text}\n\n"
        f"Long-term memory:\n{memories_text}\n\n"
        f"User message:\n{user_message}\n\n"
        "Answer using the memories when relevant. If the memory does not contain the answer, say so plainly."
    )
