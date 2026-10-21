"""Shared state passed between shopping workflow stages."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.schemas import (
    ComparisonResult,
    ProductResearchResult,
    RecommendationResult,
    RequirementAnalysis,
    ShoppingBrief,
)


class ShoppingState(BaseModel):
    """Stores user requirements and results produced by each workflow stage."""

    user_query: str = Field(min_length=1)

    # Results from individual workflow stages
    requirement_analysis: RequirementAnalysis | None = None
    research_result: ProductResearchResult | None = None
    comparison_result: ComparisonResult | None = None
    recommendation_result: RecommendationResult | None = None

    # Requirements retained across conversation turns
    requirement_controls: ShoppingBrief | None = None
    previous_requirements: ShoppingBrief | None = None

    # Execution time recorded per workflow stage, in seconds
    stage_timings: dict[str, float] = Field(default_factory=dict)
