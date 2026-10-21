from __future__ import annotations

import json
import math
import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from src.vector_store import EMBEDDING_MODEL, VECTOR_SIZE


class OllamaEmbedder:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")

    def _request(self, texts: list[str]) -> list[list[float]]:
        request = Request(
            f"{self.base_url}/api/embed",
            data=json.dumps({"model": EMBEDDING_MODEL, "input": texts, "truncate": False}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=120) as response:
            vectors = json.load(response).get("embeddings", [])
        if len(vectors) != len(texts):
            raise ValueError("Ollama returned an unexpected embedding count")
        for vector in vectors:
            if len(vector) != VECTOR_SIZE or not all(math.isfinite(value) for value in vector):
                raise ValueError("Ollama returned invalid embedding dimensions or values")
            if not any(vector):
                raise ValueError("Ollama returned a zero vector")
        return vectors

    def embed_passages(self, texts: list[str]) -> list[tuple[str, list[float]]]:
        """Return one embedding per input; caller must chunk long text first."""
        if not texts:
            return []
        try:
            vectors = self._request(texts)
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama embedding failed ({error.code}). "
                "Chunk text before embedding and check the model context length. "
                f"Details: {detail[:300]}"
            ) from error
        return list(zip(texts, vectors))
