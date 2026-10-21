from __future__ import annotations

import re

from src.schemas import Product


def chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> list[str]:
    if max_chars <= 0 or not 0 <= overlap < max_chars:
        raise ValueError("Require max_chars > 0 and 0 <= overlap < max_chars")
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + max_chars // 2, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def product_text(product: Product) -> str:
    parts = [product.name]
    for label, value in (
        ("Brand", product.brand),
        ("Category", product.category),
        ("Subcategory", product.subcategory),
    ):
        if value:
            parts.append(f"{label}: {value}")
    for key, value in product.specifications.items():
        if value is not None and value != "":
            parts.append(f"{key}: {value}")
    if product.description:
        parts.append(product.description)
    parts.extend(feature for feature in product.features if feature.strip())
    return "\n".join(parts)