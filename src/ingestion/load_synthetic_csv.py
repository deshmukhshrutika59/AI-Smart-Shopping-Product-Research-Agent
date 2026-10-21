"""Load the synthetic demo electronics inventory (Data/synthetic_electronics_inventory.csv).

This is explicitly fictional data (self-labeled in the source files) used as a
small, controlled fixture for testing the ingestion/comparison pipeline - not
real product data. Keep it clearly tagged via source_name so downstream
agents/UI can distinguish it from the real Amazon dataset.
"""
from __future__ import annotations

import csv
from pathlib import Path

from src.schemas import Product

SOURCE_FILE = "synthetic_electronics_inventory.csv"
SOURCE_NAME = "Synthetic demo electronics inventory (fictional, for pipeline testing only)"


def load_products(data_dir: Path) -> list[Product]:
    products = []
    with open(data_dir / SOURCE_FILE, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            products.append(Product(
                product_id=row["product_id"],
                name=row["title"],
                brand=row.get("brand") or None,
                category=row.get("category") or None,
                subcategory=row.get("subcategory") or None,
                price=float(row["price_inr"]) if row.get("price_inr") else None,
                currency="INR",
                rating=float(row["rating"]) if row.get("rating") else None,
                review_count=int(row["reviews_count"]) if row.get("reviews_count") else None,
                merchant=row.get("seller") or None,
                product_url=None,
                availability=row.get("availability") or None,
                stock=int(row["stock"]) if row.get("stock") else None,
                warranty_months=int(row["warranty_months"]) if row.get("warranty_months") else None,
                specifications={
                    "sku": row.get("sku"),
                    "model": row.get("model"),
                    "connectivity": row.get("connectivity"),
                    "compatible_with": row.get("compatible_with"),
                },
                description=None,
                features=[],
                source_name=SOURCE_NAME,
                source_file=SOURCE_FILE,
                dataset_snapshot="fictional - no real-world date applies",
            ))
    return products


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    products = load_products(base_dir / "Data")
    print(f"Loaded {len(products)} products from {SOURCE_FILE}")
