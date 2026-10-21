from __future__ import annotations

import argparse
import asyncio

from agent_framework import Agent
from agent_framework.exceptions import ChatClientException
from httpx import HTTPError
from pydantic import ValidationError

from src.agents.llm_client import create_chat_client
from src.schemas import RequirementAnalysis
from src.state import ShoppingState


INSTRUCTIONS = """
You are a shopping requirement analyst for electronics.

Extract only requirements explicitly stated by the user.
Do not recommend products, invent facts, or make purchases.

Return the JSON structure required by the RequirementAnalysis schema.

Rules:
- category: requested electronics category, or null if missing.
- budget_max: positive maximum budget, or null if missing or invalid.
- currency: explicit currency code, or null if ambiguous or missing.
  Indian rupees and INR mean INR. US dollars and USD mean USD.
  A bare $ or "dollars" is ambiguous.
- use_case: exact excerpt of the user's intended use, or null.
- must_have: explicit requirements and product modifiers.
- preferred_features: explicitly stated nice-to-have features.
- excluded_brands: explicitly excluded brands.
- priority_order: named priorities in the order provided.
- priorities: list of {name, weight} numeric pairs, only when explicitly supplied; otherwise [].

Never guess missing information, country, currency, or product specifications.
Do not invent equal priority weights or scores.
Do not ask optional preference questions.

Python handles questions for missing category, budget, and currency.
Use clarification_questions for contradictions, ambiguous currency,
invalid budgets, or non-electronics requests.

If category, budget, and currency are clear and there are no contradictions,
return an empty clarification_questions list.

"Under", "up to", and "less than" specify the maximum budget.
"Meetings and music" means both use cases.
For "wireless headphones", include "wireless" in must_have.
For "wired keyboard", include "wired" in must_have.
When a product must work with or fit a specific device, brand, or model
(for example "for Redmi 13 5G phone", "compatible with iPhone 15",
"for Dell XPS 13"), include that exact device/model phrase in must_have.
Never drop a stated device or model name; it is a hard requirement, not
an optional preference.
When the user prioritizes multiple features, preserve their stated order.
Use readable words rather than underscores.

Do not include a status field. Python determines readiness.
Return only the required structured output.
"""


FAST_INSTRUCTIONS = """
Extract the user's electronics shopping requirements as structured JSON.

Use the RequirementAnalysis schema.
Extract only explicit information; never guess missing values.

Preserve:
- Category, maximum budget, and explicit currency.
- Exact use-case excerpts.
- Explicit must-have features and product modifiers.
- Device, brand, or model compatibility phrases (for example "for Redmi 13
  5G phone", "compatible with iPhone 15") as exact must_have entries; never
  drop a stated device or model name.
- Preferred features and excluded brands.
- Priority order, with numeric weights (as {name, weight} pairs) only when explicitly supplied.

A bare $ or "dollars" is ambiguous. INR and USD must be supported by
explicit wording or an unambiguous currency symbol.

Keep unknown essentials null. Python handles standard missing-field questions.
Use clarification_questions for contradictions, ambiguous currency,
invalid budgets, and non-electronics requests.

For follow-ups, preserve previous requirements and change only what the
latest message changes.

Do not invent product facts, ask optional questions, or include a status field.
Return only the required structured output.
"""


def create_requirement_agent(*, fast_mode: bool = False) -> Agent:
    return Agent(
        client=create_chat_client(),
        name="shopping_requirement_analyst",
        instructions=FAST_INSTRUCTIONS if fast_mode else INSTRUCTIONS,
        default_options={
            "response_format": RequirementAnalysis,
            "temperature": 0,
            "max_tokens": 700,
        },
    )


async def analyze_requirements(
    user_query: str,
    *,
    agent: Agent | None = None,
    fast_mode: bool = False,
    timeout_seconds: float = 120,
    max_attempts: int = 2,
) -> RequirementAnalysis:
    if not user_query.strip():
        raise ValueError("Shopping request must not be blank")

    if not 1 <= max_attempts <= 3 or timeout_seconds <= 0:
        raise ValueError("Require 1-3 attempts and a positive timeout")

    agent = agent or create_requirement_agent(fast_mode=fast_mode)

    prompt = user_query

    for attempt in range(max_attempts):
        try:
            response = await asyncio.wait_for(
                agent.run(prompt),
                timeout=timeout_seconds,
            )

            return RequirementAnalysis.model_validate_json(
                response.text,
                context={"user_query": user_query},
            )

        except (
            ValidationError,
            ChatClientException,
            HTTPError,
            asyncio.TimeoutError,
        ) as error:
            if attempt == max_attempts - 1:
                raise RuntimeError(
                    f"Requirement analysis failed after {max_attempts} "
                    f"attempts ({type(error).__name__}). Check Azure OpenAI "
                    "connectivity and structured output."
                ) from error

            prompt = (
                f"Original shopping request:\n{user_query}\n\n"
                "Return valid structured output matching the schema. "
                "Preserve unknown values as null. Do not guess. "
                "Quote the use case exactly from the original request."
            )

    raise RuntimeError("Requirement analysis did not complete")


async def analyze_state(state: ShoppingState) -> ShoppingState:
    analysis = await analyze_requirements(state.user_query)

    return state.model_copy(
        update={"requirement_analysis": analysis}
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "query",
        help="Electronics shopping request to interpret",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use the shorter requirement extraction instructions",
    )
    args = parser.parse_args()

    result = asyncio.run(
        analyze_requirements(args.query, fast_mode=args.fast)
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
