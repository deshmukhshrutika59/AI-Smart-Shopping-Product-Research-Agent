"""Load the real Amazon electronics product dataset (Data/Amazon products.json)
into normalized Product and Review records.

Source: a real scraped Amazon.com listings export (1000 records, verified via
real ASINs/URLs/multilingual review text). Not fabricated. Field coverage is
uneven (see README/analysis) - missing price/rating is left as None rather
than defaulted to zero, per the project's "don't invent missing data" rule.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.schemas import Product, Review

SOURCE_FILE = "Amazon products.json"
SOURCE_NAME = "Real Amazon.com electronics listings (scraped export)"


def _parse_product_details(details: list[dict[str, Any]] | None) -> dict[str, Any]:
    if not details:
        return {}
    specs: dict[str, Any] = {}
    for item in details:
        key = item.get("type")
        if key:
            specs[key] = item.get("value")
    return specs


def _parse_categories(categories: list[str] | None) -> tuple[str | None, str | None]:
    if not categories:
        return None, None
    category = categories[0]
    subcategory = categories[-1] if len(categories) > 1 else None
    return category, subcategory


def load_products(data_dir: Path) -> list[Product]:
    raw = json.loads((data_dir / SOURCE_FILE).read_text(encoding="utf-8"))
    products = []
    for item in raw:
        category, subcategory = _parse_categories(item.get("categories"))
        products.append(Product(
            product_id=item["asin"],
            name=item.get("title") or "(no title in source data)",
            brand=item.get("brand"),
            category=category,
            subcategory=subcategory,
            price=item.get("final_price"),
            currency=item.get("currency") or "USD",
            rating=item.get("rating") or None,
            review_count=item.get("reviews_count") or None,
            merchant=item.get("seller_name") or item.get("manufacturer"),
            product_url=item.get("url"),
            availability=item.get("availability"),
            specifications=_parse_product_details(item.get("product_details")),
            description=item.get("description"),
            features=item.get("features") or [],
            source_name=SOURCE_NAME,
            source_file=SOURCE_FILE,
            dataset_snapshot=None,
        ))
    return products


def load_reviews(data_dir: Path) -> list[Review]:
    raw = json.loads((data_dir / SOURCE_FILE).read_text(encoding="utf-8"))
    reviews = []
    for item in raw:
        asin = item["asin"]
        top_review = item.get("top_review")
        if top_review:
            reviews.append(Review(
                review_id=f"{asin}_top_review",
                product_id=asin,
                review_text=top_review,
                is_aggregate=False,
                source=SOURCE_NAME,
            ))
        customers_say = item.get("customers_say") or {}
        summary_text = customers_say.get("text")
        if summary_text:
            reviews.append(Review(
                review_id=f"{asin}_customers_say",
                product_id=asin,
                review_text=summary_text,
                is_aggregate=True,
                source=f"{SOURCE_NAME} - Amazon 'Customers say' aggregated summary",
            ))
    return reviews


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    products = load_products(base_dir / "Data")
    reviews = load_reviews(base_dir / "Data")
    print(f"Loaded {len(products)} products, {len(reviews)} reviews from {SOURCE_FILE}")
