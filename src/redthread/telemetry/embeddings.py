"""Text Embeddings Client for Telemetry (Phase 4.5).

Provides a lightweight, dependency-free (no PyTorch/sentence-transformers) 
interface to generate embeddings by calling the backend provider directly
(Ollama or OpenAI).
"""

from __future__ import annotations

import logging

from redthread.config.settings import RedThreadSettings, TargetBackend

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Generates vector embeddings for text using the target's backend."""

    def __init__(self, settings: RedThreadSettings) -> None:
        self.settings = settings
        self.backend = settings.target_backend
        self.api_key = settings.openai_api_key
        
        if self.backend == TargetBackend.LLAMA_CPP:
            self.base_url = settings.llama_cpp_base_url
        else:
            self.base_url = settings.target_base_url

        if settings.telemetry_embedding_model:
            self.model = settings.telemetry_embedding_model
        elif self.backend in [TargetBackend.OLLAMA, TargetBackend.LLAMA_CPP]:
            self.model = "gemma4:e4b"
        else:
            self.model = "text-embedding-3-small"

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text."""
        if not text.strip():
            return []

        if self.settings.dry_run:
            import random
            random.seed(hash(text))
            base = [random.uniform(-1, 1) for _ in range(1536)]
            norm = sum(x*x for x in base) ** 0.5
            return [x/norm for x in base]

        if self.backend in [TargetBackend.OLLAMA, TargetBackend.LLAMA_CPP]:
            return await self._embed_generic_openai_compatible(text)
        elif self.backend == TargetBackend.OPENAI:
            return await self._embed_openai(text)
        else:
            raise NotImplementedError(f"Embeddings not supported for {self.backend}")

    async def _embed_generic_openai_compatible(self, text: str) -> list[float]:
        """Call OpenAI-compatible /v1/embeddings endpoint (Ollama or llama.cpp)."""
        import httpx
        
        url = f"{self.base_url.rstrip('/')}/v1/embeddings"
        payload = {
            "model": self.model,
            "input": text,
        }
        
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
            return data["data"][0]["embedding"]  # type: ignore[no-any-return]

    async def _embed_openai(self, text: str) -> list[float]:
        """Call OpenAI /v1/embeddings API."""
        import httpx
        
        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": text,
        }
        
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=headers, json=payload, timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
            return data["data"][0]["embedding"]  # type: ignore[no-any-return]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts sequentially or batched if supported."""
        import asyncio
        if self.backend == TargetBackend.OPENAI and not self.settings.dry_run:
            import httpx
            url = "https://api.openai.com/v1/embeddings"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "input": texts,
            }
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, headers=headers, json=payload, timeout=30.0)
                resp.raise_for_status()
                data = resp.json()
                sorted_data = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in sorted_data]
        
        return await asyncio.gather(*(self.embed(t) for t in texts))
