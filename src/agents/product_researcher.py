from __future__ import annotations

import argparse
import re
from threading import Lock
from time import perf_counter

from agent_framework import Executor, WorkflowContext, handler
from pydantic import ValidationError
from qdrant_client import QdrantClient, models

from src.ingestion.embeddings import OllamaEmbedder
from src.ingestion.text_processing import chunk_text
from src.schemas import (
    Product,
    ProductResearchResult,
    ResearchCandidate,
    ResearchEvidence,
    RequirementAnalysis,
    ShoppingBrief,
)
from src.state import ShoppingState
from src.tools.scoring import SPEC_FIELDS, measurement, numeric_quantity
from src.vector_store import ensure_collections, open_vector_store


STORE_ACCESS = Lock()
BATCH_SIZE = 64
MAX_EVIDENCE_PER_PRODUCT = 3

CATEGORY_ALIASES = (
    ("headphone", "headphones", "earphone", "earphones", "earbud", "earbuds", "headset", "headsets"),
    ("laptop", "laptops", "notebook", "notebooks"),
    ("smartphone", "smartphones", "cell phone", "cell phones", "mobile phone", "mobile phones"),
    ("smartwatch", "smartwatches", "smart watch", "smart watches"),
    ("keyboard", "keyboards"),
    ("mouse", "mice"),
    ("webcam", "webcams"),
    ("microphone", "microphones"),
    ("monitor", "monitors"),
    ("router", "routers"),
    ("ssd", "solid state drive", "solid state drives"),
    ("usb hub", "usb hubs", "usb c hub", "usb c hubs"),
    ("power bank", "power banks"),
    ("charger", "chargers", "wall charger", "wall chargers"),
)

ACCESSORY_TERMS = {
    "accessories", "case", "cases", "cover", "covers", "skin", "skins",
    "stand", "stands", "mount", "mounts", "warranty", "warranties",
    "replacement",
}


def words(text: str) -> set[str]:
    """Normalize text into lowercase alphanumeric tokens."""
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def category_matches(product: Product, category: str) -> bool:
    requested = words(category)

    aliases = next(
        (
            group
            for group in CATEGORY_ALIASES
            if any(words(alias) <= requested for alias in group)
        ),
        (category,),
    )

    classification = words(product.subcategory or product.category or "")

    if classification & ACCESSORY_TERMS and not requested & ACCESSORY_TERMS:
        return False

    return any(
        words(alias) <= classification
        or words(alias) <= words(product.category or "")
        for alias in aliases
    )


QUANTITY_PATTERN = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]+)")


def parse_quantity_requirement(feature: str) -> tuple[str, float] | None:
    """Interpret a must-have like '16 GB RAM' as a (criterion, threshold) pair."""
    match = QUANTITY_PATTERN.search(feature)
    if match is None:
        return None

    feature_words = words(feature)
    if feature_words & {"ram", "memory"}:
        criterion = "ram"
    elif feature_words & {"storage", "ssd", "hdd", "rom"}:
        criterion = "storage"
    elif feature_words & {"battery"} and feature_words & {"hour", "hours", "life"}:
        criterion = "battery_life"
    elif feature_words & {"mah"} or (feature_words & {"battery"} and feature_words & {"capacity"}):
        criterion = "battery_capacity"
    elif feature_words & {"weight", "weigh", "lightweight"}:
        criterion = "weight"
    else:
        return None

    unit, _ = SPEC_FIELDS[criterion]
    threshold = numeric_quantity(f"{match.group(1)} {match.group(2)}", unit, False)
    if threshold is None:
        return None

    return criterion, threshold


def verify_feature(product: Product, feature: str) -> str | None:
    """Return evidence for a must-have feature, or None if unverified."""
    requested = words(feature)
    specifications = product.specifications

    for key, value in specifications.items():
        if words(key) != requested:
            continue

        if isinstance(value, bool):
            return f"{key}: {value}" if value else None

        if str(value).casefold() in {"no", "false", "unsupported"}:
            return None

    connection_values = [
        str(value)
        for key, value in specifications.items()
        if words(key) & {"connectivity", "connection", "wireless", "interface"}
        and value
    ]

    connection_text = "; ".join(
        f"{key}: {value}"
        for key, value in specifications.items()
        if words(key) & {"connectivity", "connection", "wireless", "interface"}
        and value
    )
    connection_words = words(" ".join(connection_values))

    if connection_words & {"no", "not", "false", "without", "unsupported"}:
        return None

    if requested == {"wireless"}:
        if any(
            words(key) == {"wireless"} and value is False
            for key, value in specifications.items()
        ):
            return None

        supports_wireless = (
            bool(connection_words & {"bluetooth", "wireless", "wifi"})
            or {"wi", "fi"} <= connection_words
            or any("2.4" in value for value in connection_values)
        )
        return connection_text if supports_wireless else None

    if requested in ({"wired"}, {"bluetooth"}, {"usb", "c"}):
        return connection_text if requested <= connection_words else None

    for key, value in specifications.items():
        if isinstance(value, bool) or value is None:
            continue

        if (
            words(key) == requested
            and str(value).casefold() in {"yes", "supported", "included"}
        ):
            return f"{key}: {value}"

    return None


FEATURE_STOPWORDS = {
    "compatible", "compatibility", "with", "for", "device", "phone",
    "model", "the", "a", "an", "and", "or", "to", "of", "support",
    "supports", "supported", "only",
}


def verify_text_feature(product: Product, feature: str) -> str | None:
    """Verify a non-spec must-have (e.g. device compatibility) against listing text."""
    significant = words(feature) - FEATURE_STOPWORDS
    if not significant:
        return None

    haystack = words(" ".join(filter(None, [
        product.name, product.description, *product.features,
    ])))

    if significant <= haystack:
        return f"Mentioned in product listing: {feature!r}"

    return None


def eligible_product(
    product: Product,
    brief: ShoppingBrief,
) -> tuple[bool, dict[str, str]]:
    """Check budget, currency, category, availability, and must-have features."""
    if (
        brief.category is None
        or brief.budget_max is None
        or brief.currency is None
    ):
        return False, {}

    if (
        product.price is None
        or product.price > brief.budget_max
        or product.currency != brief.currency
    ):
        return False, {}

    if brief.excluded_brands:
        if not product.brand:
            return False, {}

        excluded = {brand.casefold() for brand in brief.excluded_brands}
        if product.brand.strip().casefold() in excluded:
            return False, {}

    if not category_matches(product, brief.category):
        return False, {}

    availability = (product.availability or "").casefold()

    if product.stock == 0 or "unavailable" in availability or "out of stock" in availability:
        return False, {}

    verified = {}

    for feature in brief.must_have:
        quantity_requirement = parse_quantity_requirement(feature)
        if quantity_requirement is not None:
            criterion, threshold = quantity_requirement
            value, unit, evidence, _problem = measurement(
                ResearchCandidate(product=product, retrieval_score=0.0), criterion
            )
            if value is None or value < threshold:
                return False, {}
            verified[feature] = f"{evidence} (requires >= {threshold} {unit})"
            continue

        evidence = verify_feature(product, feature)
        if evidence is None:
            evidence = verify_text_feature(product, feature)
        if evidence is None:
            return False, {}
        verified[feature] = evidence

    return True, verified


def evidence_from_point(point) -> ResearchEvidence:
    payload = point.payload or {}

    return ResearchEvidence(
        point_id=str(point.id),
        text=payload["text"],
        source_file=payload["source_file"],
        product_url=payload.get("product_url"),
        source_location=payload.get("source_location"),
        ingested_at=payload.get("ingested_at"),
        freshness=payload.get("freshness", "source_date_unknown"),
        is_synthetic=payload.get("is_synthetic", False),
        is_aggregate=payload.get("is_aggregate", False),
    )


def build_query(brief: ShoppingBrief) -> str:
    terms = (
        brief.category,
        brief.use_case,
        *brief.must_have,
        *brief.preferred_features,
        *brief.priority_order,
    )
    return "\n".join(term for term in terms if term)


def create_query_vector(query: str, embedder: OllamaEmbedder) -> list[float]:
    passages = embedder.embed_passages(chunk_text(query))

    if not passages:
        raise ValueError("Embedding service returned no query vectors")

    vectors = [vector for _, vector in passages]
    dimensions = len(vectors[0])

    if not dimensions or any(len(vector) != dimensions for vector in vectors):
        raise ValueError("Embedding vectors have inconsistent dimensions")

    return [
        sum(vector[index] for vector in vectors) / len(vectors)
        for index in range(dimensions)
    ]


def product_filter(
    brief: ShoppingBrief,
    include_synthetic: bool,
) -> list:
    conditions = [
        models.FieldCondition(
            key="currency",
            match=models.MatchValue(value=brief.currency),
        ),
        models.FieldCondition(
            key="price",
            range=models.Range(gte=0, lte=brief.budget_max),
        ),
    ]

    if not include_synthetic:
        conditions.append(
            models.FieldCondition(
                key="is_synthetic",
                match=models.MatchValue(value=False),
            )
        )

    return conditions


def retrieve_reviews(
    client: QdrantClient,
    candidate: ResearchCandidate,
    vector: list[float],
) -> None:
    product = candidate.product

    response = client.query_points(
        "reviews",
        query=vector,
        limit=6,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="product_id",
                    match=models.MatchValue(value=product.product_id),
                ),
                models.FieldCondition(
                    key="source_file",
                    match=models.MatchValue(value=product.source_file),
                ),
                models.FieldCondition(
                    key="is_synthetic",
                    match=models.MatchValue(value=candidate.is_synthetic),
                ),
            ]
        ),
    )

    seen = set()

    for point in response.points:
        payload = point.payload or {}
        key = payload.get("record_key", str(point.id))

        if key not in seen:
            candidate.reviews.append(evidence_from_point(point))
            seen.add(key)

    if not candidate.reviews:
        candidate.warnings.append("No review text was retrieved for this product.")


def research_products(
    analysis: RequirementAnalysis,
    *,
    client: QdrantClient | None = None,
    embedder: OllamaEmbedder | None = None,
    top_k: int = 5,
    include_synthetic: bool = False,
    max_batches: int = 10,
) -> ProductResearchResult:
    if not 1 <= top_k <= 20 or not 1 <= max_batches <= 20:
        raise ValueError("Require top_k and max_batches between 1 and 20")

    analysis = RequirementAnalysis(
        requirements=analysis.requirements,
        clarification_questions=analysis.clarification_questions,
        warnings=analysis.warnings,
    )
    brief = analysis.requirements

    result = ProductResearchResult(
        requirements=brief,
        warnings=list(analysis.warnings),
    )

    if analysis.status != "ready":
        result.clarification_questions = analysis.clarification_questions
        return result

    embedder = embedder or OllamaEmbedder()
    vector = create_query_vector(build_query(brief), embedder)

    owns_client = client is None
    client = client or open_vector_store()

    try:
        ensure_collections(client)

        conditions = product_filter(brief, include_synthetic)
        seen = set()
        candidates = {}
        exhausted = False

        for _ in range(max_batches):
            excluded = (
                [
                    models.FieldCondition(
                        key="record_key",
                        match=models.MatchAny(any=list(seen)),
                    )
                ]
                if seen
                else []
            )

            response = client.query_points(
                "products",
                query=vector,
                query_filter=models.Filter(
                    must=conditions,
                    must_not=excluded,
                ),
                limit=BATCH_SIZE,
            )
            points = response.points

            if not points:
                exhausted = True
                break

            for point in points:
                payload = point.payload or {}
                record_key = payload.get("record_key")

                if not record_key:
                    raise ValueError(
                        "Stored product lacks its ingestion record key; "
                        "rerun ingestion."
                    )

                if record_key in seen:
                    candidate = candidates.get(record_key)
                    if (
                        candidate
                        and len(candidate.product_evidence)
                        < MAX_EVIDENCE_PER_PRODUCT
                    ):
                        candidate.product_evidence.append(evidence_from_point(point))
                    continue

                seen.add(record_key)
                result.examined_products += 1

                try:
                    product = Product.model_validate(payload)
                    eligible, verified = eligible_product(product, brief)
                except ValidationError:
                    eligible, verified = False, {}

                if not eligible:
                    result.rejected_products += 1
                    continue

                candidate = ResearchCandidate(
                    product=product,
                    retrieval_score=point.score,
                    is_synthetic=payload.get("is_synthetic", False),
                    product_evidence=[evidence_from_point(point)],
                    verified_features=verified,
                    warnings=[
                        "Source price/specification date is unknown; "
                        "ingestion time is not source freshness."
                    ],
                )

                if candidate.is_synthetic:
                    candidate.warnings = [
                        "Fictional demo product; not a real shopping recommendation."
                    ]

                candidates[record_key] = candidate

                if len(candidates) >= top_k:
                    break

            if len(candidates) >= top_k:
                break

            if len(points) < BATCH_SIZE:
                exhausted = True
                break

        result.candidates = list(candidates.values())

        for candidate in result.candidates:
            retrieve_reviews(client, candidate, vector)

        if result.candidates:
            document_filter = None

            if not include_synthetic:
                document_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="is_synthetic",
                            match=models.MatchValue(value=False),
                        )
                    ]
                )

            documents = client.query_points(
                "documents_chunks",
                query=vector,
                query_filter=document_filter,
                limit=3,
            ).points

            result.reference_documents = [
                evidence_from_point(point) for point in documents
            ]

            if result.reference_documents:
                result.warnings.append(
                    "Reference documents are general context, "
                    "not verified evidence for individual products."
                )

        if len(result.candidates) < top_k:
            result.warnings.append(
                f"Found {len(result.candidates)} eligible products, "
                f"fewer than the requested {top_k}. Missing prices, "
                "category mismatches, excluded brands, unavailability, "
                "or unverified must-haves can exclude records."
            )

        if not exhausted and len(result.candidates) < top_k:
            result.warnings.append(
                "The bounded search limit was reached; results are not exhaustive."
            )

        result.warnings.append(
            "Retrieval similarity is not a recommendation or quality score."
        )

        return result

    finally:
        if owns_client:
            client.close()


def research_state(
    state: ShoppingState,
    **kwargs,
) -> ShoppingState:
    if state.requirement_analysis is None:
        raise ValueError("Run the Requirement Analyst before Product Research")

    result = research_products(state.requirement_analysis, **kwargs)

    return state.model_copy(update={"research_result": result})


class ProductResearcher(Executor):
    def __init__(
        self,
        *,
        client: QdrantClient | None = None,
        embedder: OllamaEmbedder | None = None,
        top_k: int = 5,
        include_synthetic: bool = False,
    ):
        super().__init__(id="product_researcher")
        self.client = client
        self.embedder = embedder
        self.top_k = top_k
        self.include_synthetic = include_synthetic

    @handler
    async def handle(
        self,
        state: ShoppingState,
        context: WorkflowContext[ShoppingState],
    ) -> None:
        started = perf_counter()

        with STORE_ACCESS:
            updated = research_state(
                state,
                client=self.client,
                embedder=self.embedder,
                top_k=self.top_k,
                include_synthetic=self.include_synthetic,
            )

        updated = updated.model_copy(
            update={
                "stage_timings": {
                    **updated.stage_timings,
                    "research": perf_counter() - started,
                }
            }
        )

        await context.send_message(updated)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Research eligible electronics products using Qdrant."
    )
    parser.add_argument("--category", required=True)
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--currency", required=True)
    parser.add_argument("--must-have", action="append", default=[])
    parser.add_argument("--exclude-brand", action="append", default=[])
    parser.add_argument("--include-synthetic", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    analysis = RequirementAnalysis(
        requirements=ShoppingBrief(
            category=args.category,
            budget_max=args.budget,
            currency=args.currency,
            must_have=args.must_have,
            excluded_brands=args.exclude_brand,
        )
    )

    result = research_products(
        analysis,
        top_k=args.top_k,
        include_synthetic=args.include_synthetic,
    )

    print(
        f"Status: {result.status}; "
        f"examined: {result.examined_products}; "
        f"rejected: {result.rejected_products}"
    )

    for candidate in result.candidates:
        product = candidate.product
        print(
            f"{product.name} | {product.price} {product.currency} | "
            f"synthetic={candidate.is_synthetic}"
        )
        print(f"Source: {product.product_url or product.source_file}")
        print(
            f"Verified features: {candidate.verified_features}; "
            f"review records: {len(candidate.reviews)}"
        )

    print(f"Reference document passages: {len(result.reference_documents)}")

    for warning in result.warnings:
        print(f"Warning: {warning}")


if __name__ == "__main__":
    main()