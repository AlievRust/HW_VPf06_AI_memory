from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _cosine_distance(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 1.0

    dot_product = sum(l * r for l, r in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 1.0
    similarity = dot_product / (left_norm * right_norm)
    return 1.0 - similarity


def _matches_where(metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
    if not where:
        return True
    for key, expected in where.items():
        if metadata.get(key) != expected:
            return False
    return True


@dataclass(slots=True)
class _StoredRecord:
    id: str
    document: str
    metadata: dict[str, Any]
    embedding: list[float]


class Collection:
    def __init__(self, storage_file: Path) -> None:
        self.storage_file = storage_file
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._records: dict[str, _StoredRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.storage_file.exists():
            return

        payload = json.loads(self.storage_file.read_text(encoding="utf-8"))
        records = payload.get("records", [])
        for item in records:
            self._records[str(item["id"])] = _StoredRecord(
                id=str(item["id"]),
                document=str(item.get("document", "")),
                metadata=dict(item.get("metadata") or {}),
                embedding=[float(value) for value in item.get("embedding") or []],
            )

    def _save(self) -> None:
        payload = {
            "records": [
                {
                    "id": record.id,
                    "document": record.document,
                    "metadata": record.metadata,
                    "embedding": record.embedding,
                }
                for record in self._records.values()
            ]
        }
        self.storage_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def upsert(
        self,
        *,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        if not (len(ids) == len(documents) == len(metadatas) == len(embeddings)):
            raise ValueError("ids, documents, metadatas, and embeddings must have the same length")

        with self._lock:
            for record_id, document, metadata, embedding in zip(ids, documents, metadatas, embeddings, strict=False):
                self._records[str(record_id)] = _StoredRecord(
                    id=str(record_id),
                    document=str(document),
                    metadata=dict(metadata),
                    embedding=[float(value) for value in embedding],
                )
            self._save()

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict[str, Any] | None = None,
    ) -> dict[str, list[list[Any]]]:
        with self._lock:
            candidates = [record for record in self._records.values() if _matches_where(record.metadata, where)]

        ids_results: list[list[str]] = []
        documents_results: list[list[str]] = []
        metadatas_results: list[list[dict[str, Any]]] = []
        distances_results: list[list[float]] = []

        for query_embedding in query_embeddings:
            scored = [
                (_cosine_distance(query_embedding, record.embedding), record)
                for record in candidates
            ]
            scored.sort(key=lambda item: item[0])
            top_records = scored[: max(n_results, 0)]

            ids_results.append([record.id for _, record in top_records])
            documents_results.append([record.document for _, record in top_records])
            metadatas_results.append([record.metadata for _, record in top_records])
            distances_results.append([distance for distance, _ in top_records])

        return {
            "ids": ids_results,
            "documents": documents_results,
            "metadatas": metadatas_results,
            "distances": distances_results,
        }


class PersistentClient:
    def __init__(self, path: str) -> None:
        self.root = Path(path)
        self.root.mkdir(parents=True, exist_ok=True)

    def get_or_create_collection(self, name: str, metadata: dict[str, Any] | None = None) -> Collection:
        del metadata
        return Collection(self.root / f"{name}.json")
