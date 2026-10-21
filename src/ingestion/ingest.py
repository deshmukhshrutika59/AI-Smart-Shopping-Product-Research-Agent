from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from src.ingestion import load_amazon_json, load_documents, load_synthetic_csv
from src.ingestion.embeddings import OllamaEmbedder
from src.ingestion.text_processing import chunk_text, product_text
from src.vector_store import COLLECTIONS, EMBEDDING_MODEL, ensure_collections, open_vector_store


def store_record(
    client: QdrantClient,
    embedder: OllamaEmbedder,
    collection: str,
    record_key: str,
    text: str,
    payload: dict,
) -> int:
    passages = embedder.embed_passages(chunk_text(text))
    if not passages:
        return 0
    points = [models.PointStruct(
        id=str(uuid5(NAMESPACE_URL, f"shopping/{collection}/{record_key}/{index}")),
        vector=vector,
        payload={
            **payload,
            "record_key": record_key,
            "chunk_index": index,
            "text": passage,
            "embedding_model": EMBEDDING_MODEL,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "freshness": "synthetic" if payload.get("is_synthetic") else "source_date_unknown",
        },
    ) for index, (passage, vector) in enumerate(passages)]
    client.upsert(collection, points=points, wait=True)
    client.delete(
        collection,
        points_selector=models.FilterSelector(filter=models.Filter(
            must=[models.FieldCondition(key="record_key", match=models.MatchValue(value=record_key))],
            must_not=[models.HasIdCondition(has_id=[point.id for point in points])],
        )),
        wait=True,
    )
    return len(points)


def ingest(data_dir: Path, limit: int | None = None) -> dict[str, int]:
    products = load_amazon_json.load_products(data_dir) + load_synthetic_csv.load_products(data_dir)
    if limit is not None:
        products = products[:limit]
    product_lookup = {product.product_id: product for product in products}
    reviews = [review for review in load_amazon_json.load_reviews(data_dir)
               if review.product_id in product_lookup]
    documents = load_documents.load_all_chunks(data_dir)
    embedder = OllamaEmbedder()
    client = open_vector_store()
    counts = dict.fromkeys(COLLECTIONS, 0)
    try:
        ensure_collections(client)
        for index, product in enumerate(products, start=1):
            counts["products"] += store_record(
                client, embedder, "products", f"{product.source_file}/{product.product_id}",
                product_text(product),
                {**product.model_dump(mode="json"),
                 "is_synthetic": product.source_file == load_synthetic_csv.SOURCE_FILE},
            )
            if index % 50 == 0 or index == len(products):
                print(f"Embedded {index}/{len(products)} products", flush=True)
        for review in reviews:
            product = product_lookup[review.product_id]
            counts["reviews"] += store_record(
                client, embedder, "reviews", f"{product.source_file}/{review.review_id}",
                "\n".join(part for part in (review.review_title, review.review_text) if part),
                {**review.model_dump(mode="json"), "source_file": product.source_file,
                 "product_url": product.product_url, "is_synthetic": False},
            )
        for document in documents:
            counts["documents_chunks"] += store_record(
                client, embedder, "documents_chunks", document.chunk_id, document.text,
                {**document.model_dump(mode="json"), "is_synthetic": True},
            )
        for collection in COLLECTIONS:
            print(f"{collection}: processed {counts[collection]} vectors; "
                  f"stored {client.count(collection, exact=True).count} points", flush=True)
    finally:
        client.close()
    return counts


def verify_ingestion(data_dir: Path) -> None:
    products = load_amazon_json.load_products(data_dir) + load_synthetic_csv.load_products(data_dir)
    expected = {
        "products": {f"{product.source_file}/{product.product_id}" for product in products},
        "reviews": {f"{load_amazon_json.SOURCE_FILE}/{review.review_id}"
                    for review in load_amazon_json.load_reviews(data_dir)},
        "documents_chunks": {document.chunk_id for document in load_documents.load_all_chunks(data_dir)},
    }
    client = open_vector_store()
    try:
        ensure_collections(client)
        for collection in COLLECTIONS:
            stored_keys = set()
            offset = None
            while True:
                points, offset = client.scroll(collection, limit=256, offset=offset, with_vectors=False)
                stored_keys.update(point.payload["record_key"] for point in points)
                if offset is None:
                    break
            missing = expected[collection] - stored_keys
            if missing:
                raise ValueError(f"{collection}: {len(missing)} source records have not been ingested")
            print(f"{collection}: verified {len(expected[collection])} source records; "
                  f"{client.count(collection, exact=True).count} stored vectors", flush=True)

        vector = OllamaEmbedder().embed_passages(["USB accessories for laptop and online meetings"])[0][1]
        for currency, budget, synthetic in (("USD", 50, False), ("INR", 5000, True)):
            matches = client.query_points(
                collection_name="products", query=vector, limit=5,
                query_filter=models.Filter(must=[
                    models.FieldCondition(key="currency", match=models.MatchValue(value=currency)),
                    models.FieldCondition(key="price", range=models.Range(lte=budget)),
                    models.FieldCondition(key="is_synthetic", match=models.MatchValue(value=synthetic)),
                ]),
            ).points
            if not matches:
                raise ValueError(f"No products found for verification filter: {currency} <= {budget}")
            for match in matches:
                payload = match.payload
                if payload["currency"] != currency or not 0 <= payload["price"] <= budget or payload["is_synthetic"] != synthetic:
                    raise ValueError("Search returned a product violating the verification filters")
            best = matches[0].payload
            print(f"Semantic search ({currency} <= {budget}, synthetic={synthetic}): "
                  f"{best['name']} | {best['price']} {currency}", flush=True)
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[2] / "Data")
    parser.add_argument("--limit", type=int, help="Limit products for a smoke run; documents are still included")
    parser.add_argument("--verify-only", action="store_true", help="Check coverage and retrieval without ingesting")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.verify_only:
        verify_ingestion(args.data_dir)
    else:
        ingest(args.data_dir, args.limit)


if __name__ == "__main__":
    main()