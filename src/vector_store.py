from __future__ import annotations

import os
from pathlib import Path

from qdrant_client import QdrantClient, models


EMBEDDING_MODEL = "all-minilm"
VECTOR_SIZE = 384
COLLECTIONS = ("products", "reviews", "documents_chunks")


def default_storage_path() -> Path:
    configured = os.getenv("QDRANT_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "qdrant_storage"


def open_vector_store(path: Path | None = None) -> QdrantClient:
    storage_path = path if path is not None else default_storage_path()
    storage_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(storage_path))


def ensure_collections(client: QdrantClient) -> None:
    for collection_name in COLLECTIONS:
        if client.collection_exists(collection_name):
            vectors = client.get_collection(collection_name).config.params.vectors
            if not isinstance(vectors, models.VectorParams) or (
                vectors.size != VECTOR_SIZE or vectors.distance != models.Distance.COSINE
            ):
                raise ValueError(
                    f"Collection {collection_name!r} has incompatible vector settings. "
                    f"Expected {VECTOR_SIZE} dimensions and cosine distance for {EMBEDDING_MODEL}. "
                    "Use a separate database path and re-embed when changing models."
                )
        else:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE, distance=models.Distance.COSINE
                ),
            )


def main() -> None:
    storage_path = default_storage_path()
    client = open_vector_store(storage_path)
    try:
        ensure_collections(client)
        print(f"Local Qdrant storage: {storage_path}")
        print(f"Embedding configuration: {EMBEDDING_MODEL}, {VECTOR_SIZE} dimensions, cosine")
        for collection_name in COLLECTIONS:
            count = client.count(collection_name, exact=True).count
            print(f"{collection_name}: {count} points")
    finally:
        client.close()


if __name__ == "__main__":
    main()