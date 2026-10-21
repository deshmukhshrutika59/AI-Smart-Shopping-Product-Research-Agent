"""Pydantic data models shared across ingestion, agents, and the API layer."""
from __future__ import annotations

import math
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, computed_field, field_validator, model_validator


class Product(BaseModel):
    """A normalized product record, regardless of which source file it came from."""

    product_id: str
    name: str
    brand: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    price: Optional[float] = Field(default=None, ge=0)
    currency: str = "USD"
    rating: Optional[float] = Field(default=None, ge=0, le=5)
    review_count: Optional[int] = Field(default=None, ge=0)
    merchant: Optional[str] = None
    product_url: Optional[str] = None
    availability: Optional[str] = None
    stock: Optional[int] = None
    warranty_months: Optional[int] = None
    specifications: dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = None
    features: list[str] = Field(default_factory=list)
    source_name: str
    source_file: str
    dataset_snapshot: Optional[str] = None


class Review(BaseModel):
    """A single review or an aggregated review-summary record for a product."""

    review_id: str
    product_id: str
    rating: Optional[float] = Field(default=None, ge=0, le=5)
    review_title: Optional[str] = None
    review_text: Optional[str] = None
    user_id: Optional[str] = None
    verified_purchase: Optional[bool] = None
    is_aggregate: bool = False
    source: str


class DocumentChunk(BaseModel):
    """A chunk of extracted text from a PDF/DOCX source document, pre-embedding."""

    chunk_id: str
    source_file: str
    doc_type: str
    chunk_index: int
    text: str
    product_id: Optional[str] = None
    source_location: Optional[str] = None


class PriorityWeight(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1)
    weight: float = Field(gt=0, allow_inf_nan=False)


class ShoppingBrief(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    category: Optional[str] = Field(default=None, min_length=1)
    budget_max: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    currency: Optional[str] = Field(default=None, pattern=r"^[A-Z]{3}$")
    use_case: Optional[str] = None
    must_have: list[str] = Field(
        default_factory=list,
        description="Explicit required features, including wireless when wireless headphones are requested.",
    )
    preferred_features: list[str] = Field(default_factory=list)
    excluded_brands: list[str] = Field(default_factory=list)
    priority_order: list[str] = Field(default_factory=list)
    priorities: list[PriorityWeight] = Field(
        default_factory=list,
        description="Numeric weights as {name, weight} pairs, only when the user explicitly supplied numbers.",
    )

    @field_validator("must_have", "preferred_features", "excluded_brands", "priority_order")
    @classmethod
    def clean_items(cls, items: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(item.strip() for item in items))
        if any(not item for item in cleaned):
            raise ValueError("List entries must not be empty")
        return cleaned

    @field_validator("priorities")
    @classmethod
    def normalize_priorities(cls, priorities: list[PriorityWeight]) -> list[PriorityWeight]:
        if not priorities:
            return []
        merged: dict[str, float] = {}
        for item in priorities:
            key = item.name.strip()
            if not key or not math.isfinite(item.weight) or item.weight < 0:
                raise ValueError("Priority names must be nonempty and weights finite and nonnegative")
            merged[key] = merged.get(key, 0) + item.weight
        total = sum(merged.values())
        if not math.isfinite(total) or total <= 0:
            raise ValueError("Priority weights must have a positive finite total")
        return [PriorityWeight(name=name, weight=weight / total) for name, weight in merged.items()]


class RequirementAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    requirements: ShoppingBrief
    clarification_questions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_essential_fields(self, info: ValidationInfo) -> RequirementAnalysis:
        query = info.context.get("user_query") if info.context else None
        use_case = self.requirements.use_case
        if query is not None and use_case and use_case.casefold() not in query.casefold():
            raise ValueError("use_case must be an exact excerpt from the user's request")
        questions = [question.strip() for question in self.clarification_questions if question.strip()]
        for field_name, question in (
            ("category", "Which electronics product category are you looking for?"),
            ("budget_max", "What is your maximum budget?"),
            ("currency", "Which currency should the budget use, for example INR or USD?"),
        ):
            if getattr(self.requirements, field_name) is None:
                questions.append(question)
        self.clarification_questions = list(dict.fromkeys(questions))
        return self

    @computed_field
    @property
    def status(self) -> Literal["ready", "needs_clarification"]:
        return "needs_clarification" if self.clarification_questions else "ready"


class ResearchEvidence(BaseModel):
    point_id: str
    text: str
    source_file: str
    product_url: Optional[str] = None
    source_location: Optional[str] = None
    ingested_at: Optional[str] = None
    freshness: str = "source_date_unknown"
    is_synthetic: bool = False
    is_aggregate: bool = False


class ResearchCandidate(BaseModel):
    product: Product
    retrieval_score: float = Field(allow_inf_nan=False)
    is_synthetic: bool = False
    product_evidence: list[ResearchEvidence] = Field(default_factory=list)
    reviews: list[ResearchEvidence] = Field(default_factory=list)
    verified_features: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ProductResearchResult(BaseModel):
    requirements: ShoppingBrief
    candidates: list[ResearchCandidate] = Field(default_factory=list)
    reference_documents: list[ResearchEvidence] = Field(default_factory=list)
    clarification_questions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    examined_products: int = Field(default=0, ge=0)
    rejected_products: int = Field(default=0, ge=0)

    @computed_field
    @property
    def status(self) -> Literal["results", "no_matches", "needs_clarification"]:
        if self.clarification_questions:
            return "needs_clarification"
        return "results" if self.candidates else "no_matches"


class ScoringConfig(BaseModel):
    targets: dict[str, float] = Field(default_factory=lambda: {
        "battery_life": 40, "battery_capacity": 5000, "ram": 16,
        "storage": 1024, "warranty": 36, "weight": 1000,
    })

    @field_validator("targets")
    @classmethod
    def validate_targets(cls, targets: dict[str, float]) -> dict[str, float]:
        if any(not math.isfinite(value) or value <= 0 for value in targets.values()):
            raise ValueError("Scoring targets must be finite and positive")
        return targets


class CriterionScore(BaseModel):
    criterion: str
    weight: float = Field(ge=0, le=1)
    raw_value: Optional[float] = Field(default=None, allow_inf_nan=False)
    unit: Optional[str] = None
    score: Optional[float] = Field(default=None, ge=0, le=100, allow_inf_nan=False)
    contribution: float = Field(ge=0, le=100, allow_inf_nan=False)
    method: str
    evidence: Optional[str] = None


class ReviewCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    point_id: str
    quote: str = Field(min_length=3, max_length=600)


REVIEW_TOPIC_CUES = {
    "battery": ("battery", "batteries", "charging", "charge"),
    "comfort": ("comfort", "fit", "wear", "ear canal", "pressure"),
    "microphone": ("microphone", "mic", "voice", "call"),
    "sound": ("sound", "audio", "bass", "treble", "volume"),
    "connectivity": ("bluetooth", "connect", "pair", "signal", "disconnect"),
    "build_quality": ("build", "material", "sturdy", "durab", "plastic", "metal", "broken", "construction"),
    "value": ("price", "cost", "budget", "value", "money", "expensive", "cheap", "affordable"),
}


class ReviewTheme(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topic: Literal["battery", "comfort", "microphone", "sound", "connectivity", "build_quality", "value", "other"]
    sentiment: Literal["positive", "negative", "mixed"]
    citations: list[ReviewCitation] = Field(min_length=1, max_length=6)
    recurring: bool = False

    @model_validator(mode="after")
    def validate_citations(self, info: ValidationInfo) -> ReviewTheme:
        evidence = info.context.get("review_evidence") if info.context else None
        if evidence is not None:
            self.recurring = False
            individual_sources = set()
            for citation in self.citations:
                source = evidence.get(citation.point_id)
                if source is None or citation.quote not in source.text:
                    raise ValueError("Every review citation must quote a supplied passage exactly")
                if self.topic != "other" and not any(
                    cue in citation.quote.casefold() for cue in REVIEW_TOPIC_CUES[self.topic]
                ):
                    raise ValueError("Review topic must have a literal cue in each cited quote; use other if uncertain")
                if not source.is_aggregate:
                    individual_sources.add(source.point_id)
            self.recurring = len(individual_sources) >= 2
        return self


class ReviewThemes(BaseModel):
    model_config = ConfigDict(extra="forbid")
    themes: list[ReviewTheme] = Field(default_factory=list, max_length=8)


class ReviewSummary(BaseModel):
    status: Literal["no_reviews", "summarized", "no_supported_themes", "unavailable", "skipped"]
    evidence_count: int = Field(default=0, ge=0)
    aggregate_count: int = Field(default=0, ge=0)
    themes: list[ReviewTheme] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ComparedProduct(BaseModel):
    candidate: ResearchCandidate
    criteria: list[CriterionScore]
    total_score: Optional[float] = Field(default=None, ge=0, le=100)
    data_coverage: float = Field(ge=0, le=1)
    review_summary: ReviewSummary = Field(default_factory=lambda: ReviewSummary(status="no_reviews"))
    warnings: list[str] = Field(default_factory=list)


class ComparisonResult(BaseModel):
    requirements: ShoppingBrief
    products: list[ComparedProduct] = Field(default_factory=list)
    applied_weights: dict[str, float] = Field(default_factory=dict)
    weighting_method: str
    clarification_questions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @computed_field
    @property
    def status(self) -> Literal["results", "no_matches", "needs_clarification", "insufficient_data"]:
        if self.clarification_questions:
            return "needs_clarification"
        if not self.products:
            return "no_matches"
        return "results" if any(product.total_score is not None for product in self.products) else "insufficient_data"


class RecommendationReason(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    type: Literal["strength", "trade_off"]
    text: str = Field(min_length=3, max_length=300)
    criterion: Optional[str] = None
    review_topic: Optional[str] = None

    @model_validator(mode="after")
    def validate_grounding(self, info: ValidationInfo) -> RecommendationReason:
        if (self.criterion is None) == (self.review_topic is None):
            raise ValueError("Each reason must reference exactly one of criterion or review_topic")
        context = info.context or {}
        criteria = context.get("criteria")
        review_topics = context.get("review_topics")
        if self.criterion is not None and criteria is not None:
            if self.criterion not in criteria:
                raise ValueError(f"Unknown criterion {self.criterion!r}; not present for this product")
            score = criteria[self.criterion]
            if self.type == "strength" and not (score is not None and score >= 50):
                raise ValueError("A strength must cite a criterion with a known score of at least 50")
            if self.type == "trade_off" and not (score is None or score < 50):
                raise ValueError("A trade_off must cite a criterion that is missing or scores below 50")
        if self.review_topic is not None and review_topics is not None:
            sentiments = review_topics.get(self.review_topic)
            if not sentiments:
                raise ValueError(f"Unknown review topic {self.review_topic!r}; not present for this product")
            needed = {"positive"} if self.type == "strength" else {"negative", "mixed"}
            if not sentiments & needed:
                raise ValueError("Review topic sentiment does not support this reason type")
        return self


class ProductRecommendation(BaseModel):
    role: Literal["best_overall", "best_value", "best_alternative"]
    product_id: str
    name: str
    price: Optional[float] = None
    currency: str
    total_score: Optional[float] = Field(default=None, ge=0, le=100)
    data_coverage: float = Field(ge=0, le=1)
    product_url: Optional[str] = None
    source_file: str
    is_synthetic: bool = False
    reasoning_status: Literal["grounded", "unavailable"] = "grounded"
    reasons: list[RecommendationReason] = Field(default_factory=list, max_length=6)
    caveats: list[str] = Field(default_factory=list)


class ComparisonRow(BaseModel):
    product_id: str
    name: str
    price: Optional[float] = None
    currency: str
    total_score: Optional[float] = Field(default=None, ge=0, le=100)
    data_coverage: float = Field(ge=0, le=1)
    review_status: str
    is_synthetic: bool = False


class BuyingGuidance(BaseModel):
    label: str = "General buying guidance, not verified product recommendations"
    points: list[str] = Field(default_factory=list)
    next_step: str
    used_llm: bool = False


class RecommendationResult(BaseModel):
    requirements: ShoppingBrief
    picks: list[ProductRecommendation] = Field(default_factory=list)
    comparison_table: list[ComparisonRow] = Field(default_factory=list)
    clarification_questions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    general_guidance: Optional[BuyingGuidance] = None

    @computed_field
    @property
    def status(self) -> Literal["results", "no_matches", "needs_clarification", "insufficient_data"]:
        if self.clarification_questions:
            return "needs_clarification"
        if not self.comparison_table:
            return "no_matches"
        return "results" if self.picks else "insufficient_data"
