from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from openai import AsyncOpenAI


@dataclass(slots=True)
class OpenAIService:
    client: AsyncOpenAI
    chat_model: str
    embedding_model: str
    temperature: float
    max_output_tokens: int

    @classmethod
    def create(
        cls,
        *,
        api_key: str,
        chat_model: str,
        embedding_model: str,
        temperature: float,
        max_output_tokens: int,
    ) -> "OpenAIService":
        return cls(
            client=AsyncOpenAI(api_key=api_key),
            chat_model=chat_model,
            embedding_model=embedding_model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        response = await self.client.embeddings.create(
            model=self.embedding_model,
            input=list(texts),
        )
        return [item.embedding for item in response.data]

    async def answer(self, prompt: str) -> str:
        response = await self.client.responses.create(
            model=self.chat_model,
            input=prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        output_text = getattr(response, "output_text", "")
        return str(output_text).strip()
