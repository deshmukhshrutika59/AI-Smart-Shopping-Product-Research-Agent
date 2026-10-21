"""Orchestrates the shopping workflow from requirements to recommendations."""
from __future__ import annotations

import json
from time import perf_counter
from typing import Callable

from agent_framework import (
    Agent,
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    handler,
)
from qdrant_client import QdrantClient
from src.agents.buying_guidance import buying_guidance
from src.agents.comparison_analyst import ComparisonReviewAnalyst
from src.agents.product_researcher import ProductResearcher
from src.agents.recommendation_assistant import RecommendationAssistant
from src.agents.requirement_analyst import (
    analyze_requirements,
    create_requirement_agent,
)
from src.ingestion.embeddings import OllamaEmbedder
from src.schemas import (
    RecommendationResult,
    RequirementAnalysis,
    ShoppingBrief,
)
from src.state import ShoppingState


ProgressCallback = Callable[[str], None]


def _report(
    callback: ProgressCallback | None,
    message: str,
) -> None:
    """Send a progress update when a callback is configured."""
    if callback is not None:
        callback(message)


def _merge_requirements(
    analysis: RequirementAnalysis,
    previous: ShoppingBrief | None,
) -> RequirementAnalysis:
    """Preserve unchanged requirements across conversation turns."""
    if previous is None:
        return analysis

    current = analysis.requirements
    values = current.model_dump()

    # Retain category-specific fields unless the category changes.
    if current.category is None or current.category == previous.category:
        for field in (
            "must_have",
            "excluded_brands",
            "preferred_features",
            "priority_order",
            "priorities",
            "use_case",
        ):
            if field not in current.model_fields_set:
                values[field] = getattr(previous, field)

    # Preserve essential fields when the latest message omits them.
    for field in ("category", "budget_max", "currency"):
        if values[field] is None:
            values[field] = getattr(previous, field)

    requirements = ShoppingBrief.model_validate(values)

    generic_questions = {
        "Which electronics product category are you looking for?",
        "What is your maximum budget?",
        "Which currency should the budget use, for example INR or USD?",
    }

    return RequirementAnalysis(
        requirements=requirements,
        clarification_questions=[
            question
            for question in analysis.clarification_questions
            if question not in generic_questions
        ],
        warnings=analysis.warnings,
    )


def _apply_controls(
    analysis: RequirementAnalysis,
    controls: ShoppingBrief | None,
) -> RequirementAnalysis:
    """Apply explicit shopping controls to analyzed requirements."""
    if controls is None:
        return analysis

    updates = controls.model_dump(
        exclude_none=True,
        exclude_defaults=True,
    )
    values = analysis.requirements.model_dump()

    # Merge list-based controls without creating duplicates.
    for field in ("must_have", "excluded_brands"):
        if field in updates:
            values[field] = list(
                dict.fromkeys(values[field] + updates[field])
            )

    # Explicit controls take precedence over analyzed values.
    values.update(updates)
    requirements = ShoppingBrief.model_validate(values)

    resolved_questions = set()

    if requirements.category is not None:
        resolved_questions.add(
            "Which electronics product category are you looking for?"
        )

    if requirements.budget_max is not None:
        resolved_questions.add("What is your maximum budget?")

    if requirements.currency is not None:
        resolved_questions.add(
            "Which currency should the budget use, for example INR or USD?"
        )

    return RequirementAnalysis(
        requirements=requirements,
        clarification_questions=[
            question
            for question in analysis.clarification_questions
            if question not in resolved_questions
        ],
        warnings=analysis.warnings,
    )


class RequirementStep(Executor):
    """Analyze the request and resolve shopping requirements."""

    def __init__(
        self,
        agent: Agent | None = None,
        *,
        fast_mode: bool = False,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        super().__init__(id="requirement_analyst")
        self.agent = agent
        self.fast_mode = fast_mode
        self.on_progress = on_progress

    @handler
    async def handle(
        self,
        state: ShoppingState,
        context: WorkflowContext[ShoppingState],
    ) -> None:
        started = perf_counter()
        _report(self.on_progress, "Understanding your request")

        query = state.user_query
        previous = state.previous_requirements

        if previous is not None:
            query = (
                "Previous shopping requirements:\n"
                f"{json.dumps(previous.model_dump())}\n"
                "Latest user message:\n"
                f"{query}\n"
                "Update the previous requirements using the latest message. "
                "Preserve unchanged fields. If the product category changes, "
                "discard category-specific features and priorities. Respect "
                "explicit removal requests and ask only for genuinely "
                "missing requirements."
            )

        controls = state.requirement_controls

        if controls is not None:
            query += (
                "\nExplicit shopping form controls:\n"
                + json.dumps(
                    controls.model_dump(
                        exclude_none=True,
                        exclude_defaults=True,
                    )
                )
            )

        agent = self.agent

        if agent is None and self.fast_mode:
            agent = create_requirement_agent(fast_mode=True)

        analysis = await analyze_requirements(
            query,
            agent=agent,
            timeout_seconds=120,
            max_attempts=2,
        )

        analysis = _merge_requirements(analysis, previous)
        analysis = _apply_controls(analysis, controls)

        await context.send_message(
            state.model_copy(
                update={
                    "requirement_analysis": analysis,
                    "stage_timings": {
                        **state.stage_timings,
                        "requirements": perf_counter() - started,
                    },
                }
            )
        )


class ProgressResearcher(ProductResearcher):
    """Add progress reporting to product research."""

    def __init__(self,*,on_progress: ProgressCallback | None = None,**kwargs,) -> None:
        self.on_progress = on_progress
        super().__init__(**kwargs)

    @handler
    async def handle(
        self,
        state: ShoppingState,
        context: WorkflowContext[ShoppingState],
    ) -> None:
        _report(self.on_progress, "Searching catalogue records")
        await super().handle(state, context)


class ProgressComparison(ComparisonReviewAnalyst):
    """Add progress reporting to product comparison."""

    def __init__(
        self,
        *,
        on_progress: ProgressCallback | None = None,
        **kwargs,
    ) -> None:
        self.on_progress = on_progress
        super().__init__(**kwargs)

    @handler
    async def handle(
        self,
        state: ShoppingState,
        context: WorkflowContext[ShoppingState],
    ) -> None:
        _report(self.on_progress, "Comparing product evidence")
        await super().handle(state, context)


class ProgressRecommendation(RecommendationAssistant):
    """Add progress reporting to recommendation generation."""

    def __init__(
        self,
        *,
        on_progress: ProgressCallback | None = None,
        **kwargs,
    ) -> None:
        self.on_progress = on_progress
        super().__init__(**kwargs)

    @handler
    async def handle(
        self,
        state: ShoppingState,
        context: WorkflowContext[ShoppingState],
    ) -> None:
        _report(self.on_progress, "Preparing your shortlist")
        await super().handle(state, context)


class GuidanceStep(Executor):
    """Generate general buying advice when evidence is insufficient."""

    def __init__(
        self,
        *,
        agent: Agent | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        super().__init__(id="buying_guidance")
        self.agent = agent
        self.on_progress = on_progress

    @handler
    async def handle(
        self,
        state: ShoppingState,
        context: WorkflowContext[ShoppingState],
    ) -> None:
        started = perf_counter()
        _report(self.on_progress, "Preparing general buying guidance")

        analysis = state.requirement_analysis

        if analysis is None:
            raise ValueError(
                "Buying guidance requires requirement analysis"
            )

        result = state.recommendation_result

        if result is None:
            warnings = list(analysis.warnings)

            if state.research_result is not None:
                warnings.extend(state.research_result.warnings)

            result = RecommendationResult(
                requirements=analysis.requirements,
                warnings=list(dict.fromkeys(warnings)),
            )

        guidance = await buying_guidance(
            result.requirements,
            agent=self.agent,
        )

        result = result.model_copy(
            update={"general_guidance": guidance}
        )

        await context.send_message(
            state.model_copy(
                update={
                    "recommendation_result": result,
                    "stage_timings": {
                        **state.stage_timings,
                        "guidance": perf_counter() - started,
                    },
                }
            )
        )


class FinishStep(Executor):
    """Ensure that the workflow returns a final shopping state."""

    def __init__(self) -> None:
        super().__init__(id="finish")

    @handler
    async def handle(
        self,
        state: ShoppingState,
        context: WorkflowContext[ShoppingState, ShoppingState],
    ) -> None:
        if state.recommendation_result is None:
            analysis = state.requirement_analysis

            if analysis is None:
                raise ValueError(
                    "Workflow finished without requirement analysis"
                )

            warnings = list(analysis.warnings)

            if state.research_result is not None:
                warnings.extend(state.research_result.warnings)

            result = RecommendationResult(
                requirements=analysis.requirements,
                clarification_questions=analysis.clarification_questions,
                warnings=list(dict.fromkeys(warnings)),
            )

            if state.research_result is not None:
                result.next_steps = [
                    "Review your requirements and try another search."
                ]

            state = state.model_copy(
                update={"recommendation_result": result}
            )

        await context.yield_output(state)


def build_workflow(
    *,
    requirement_agent: Agent | None = None,
    review_agent: Agent | None = None,
    reason_agent: Agent | None = None,
    guidance_agent: Agent | None = None,
    client: QdrantClient | None = None,
    embedder: OllamaEmbedder | None = None,
    top_k: int = 5,
    include_synthetic: bool = False,
    fast_mode: bool = False,
    enable_guidance: bool = False,
    on_progress: ProgressCallback | None = None,
):
    """Build and connect the shopping workflow stages."""
    requirement = RequirementStep(
        requirement_agent,
        fast_mode=fast_mode,
        on_progress=on_progress,
    )

    researcher = ProgressResearcher(
        client=client,
        embedder=embedder,
        top_k=top_k,
        include_synthetic=include_synthetic,
        on_progress=on_progress,
    )

    comparison = ProgressComparison(
        review_agent=review_agent,
        analyze_reviews=not fast_mode,
        on_progress=on_progress,
    )

    recommendation = ProgressRecommendation(
        reason_agent=reason_agent,
        fast_mode=fast_mode,
        on_progress=on_progress,
    )

    finish = FinishStep()

    builder = (
        WorkflowBuilder(
            start_executor=requirement,
            max_iterations=8,
            output_from=[finish],
            name="shopping_research",
        )
        .add_edge(
            requirement,
            researcher,
            condition=lambda state: (
                state.requirement_analysis.status == "ready"
            ),
        )
        .add_edge(
            requirement,
            finish,
            condition=lambda state: (
                state.requirement_analysis.status != "ready"
            ),
        )
        .add_edge(
            researcher,
            comparison,
            condition=lambda state: (
                state.research_result.status == "results"
            ),
        )
    )

    if enable_guidance:
        guidance = GuidanceStep(
            agent=guidance_agent,
            on_progress=on_progress,
        )

        builder.add_edge(
            researcher,
            guidance,
            condition=lambda state: (
                state.research_result.status == "no_matches"
            ),
        )

        builder.add_edge(
            researcher,
            finish,
            condition=lambda state: (
                state.research_result.status == "needs_clarification"
            ),
        )

        builder.add_edge(
            recommendation,
            guidance,
            condition=lambda state: (
                state.recommendation_result.status == "insufficient_data"
            ),
        )

        builder.add_edge(
            recommendation,
            finish,
            condition=lambda state: (
                state.recommendation_result.status != "insufficient_data"
            ),
        )

        builder.add_edge(guidance, finish)

    else:
        builder.add_edge(
            researcher,
            finish,
            condition=lambda state: (
                state.research_result.status != "results"
            ),
        )

        builder.add_edge(recommendation, finish)

    builder.add_edge(comparison, recommendation)

    return builder.build()


async def run_shopping(
    query: str,
    *,
    controls: ShoppingBrief | None = None,
    previous_requirements: ShoppingBrief | None = None,
    **workflow_options,
) -> ShoppingState:
    """Run the workflow and return exactly one validated result."""
    query = query.strip()

    if not query:
        raise ValueError("Shopping request must not be blank")

    workflow = build_workflow(**workflow_options)
    started = perf_counter()

    response = await workflow.run(
        ShoppingState(
            user_query=query,
            requirement_controls=controls,
            previous_requirements=previous_requirements,
        )
    )

    outputs = response.get_outputs()

    if len(outputs) != 1 or not isinstance(outputs[0], ShoppingState):
        raise RuntimeError(
            "Workflow did not produce exactly one validated shopping result"
        )

    state = outputs[0]

    return state.model_copy(
        update={
            "stage_timings": {
                **state.stage_timings,
                "total": perf_counter() - started,
            }
        }
    )
