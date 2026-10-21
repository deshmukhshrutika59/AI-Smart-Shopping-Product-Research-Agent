"""UI-only presentation of verified workflow results."""

from __future__ import annotations

import streamlit as st

from src.state import ShoppingState


ROLE_LABELS = {
    "best_overall": "Best Match",
    "best_value": "Best Value",
    "best_alternative": "Alternative",
}

def price_label(price: float | None, currency: str | None) -> str:
    """Format a price, clearly identifying unavailable values."""
    if price is None:
        return "Price unavailable"

    amount = f"{price:,.2f}"
    return f"{amount} {currency}" if currency else amount


def answer_text(state: ShoppingState) -> str:
    """Return a concise, user-facing summary of the workflow result."""
    result = state.recommendation_result

    if result is None:
        return "No recommendation result was produced."

    if result.status == "needs_clarification":
        if result.clarification_questions:
            questions = "\n".join(
                f"- {question}"
                for question in result.clarification_questions
            )
            return f"Before I search, I need:\n{questions}"

        return "Please clarify your requirements before searching."

    if result.status == "no_matches":
        return (
            "I couldn't verify a matching product in the current catalogue. "
            "This doesn't mean suitable products don't exist."
        )

    if result.status == "insufficient_data":
        return (
            "I found candidate products, but there isn't enough evidence "
            "to rank them reliably against your priorities."
        )

    if result.picks:
        return "Here are the strongest catalogue matches based on the available evidence."

    if result.general_guidance:
        return (
            "I couldn't establish a reliable product recommendation. "
            "Here is some general buying guidance instead."
        )

    return "The workflow completed without producing a product recommendation."


STATUS_MESSAGES = {
    "no_matches": (
        "I couldn't verify a matching product in the current catalogue. "
        "This does not mean that no suitable products exist on the market."
    ),
    "insufficient_data": (
        "I found candidate products, but the available evidence is "
        "insufficient to rank them reliably against your priorities."
    ),
}

MAX_CAVEATS = 8


def _format_price(price: float | None, currency: str | None) -> str:
    """Format a price without inventing missing information."""
    if price is None:
        return "Price unavailable"

    amount = f"{price:,.2f}"
    return f"{amount} {currency}" if currency else amount


def _format_score(score: float | None) -> str:
    """Format a score only when one is available."""
    return f"{score:.1f}/100" if score is not None else "Not scored"


def _render_clarification(result) -> None:
    """Display questions required before product research."""
    st.info("I need a little more information before searching.")

    if result.clarification_questions:
        for question in result.clarification_questions:
            st.markdown(f"- {question}")
    else:
        st.caption("Please clarify your product requirements and try again.")


def _render_pick(pick) -> None:
    """Render one recommendation using its existing evidence."""
    role = ROLE_LABELS.get(pick.role, "Option")

    with st.container(border=True):
        st.markdown(f"**{role}: {pick.name}**")

        price = _format_price(pick.price, pick.currency)
        score = _format_score(pick.total_score)

        st.caption(
            f"{price} · Score: {score} · "
            f"Evidence coverage: {pick.data_coverage:.0%}"
        )

        if pick.is_synthetic:
            st.warning(
                "Fictional demo product — not a real-world recommendation."
            )

        if pick.product_url:
            st.link_button("View product source", pick.product_url)

        if pick.reasons:
            for reason in pick.reasons:
                if reason.type == "strength":
                    marker = "✓"
                else:
                    marker = "△"

                st.write(f"{marker} {reason.text}")
        else:
            st.caption("No evidence-based explanation is available.")

        if pick.caveats:
            with st.expander("Evidence limitations"):
                for caveat in pick.caveats[:MAX_CAVEATS]:
                    st.caption(f"• {caveat}")

                remaining = len(pick.caveats) - MAX_CAVEATS
                if remaining > 0:
                    st.caption(f"{remaining} additional limitations omitted.")


def _render_buying_guidance(guidance) -> None:
    """Keep general buying advice separate from product recommendations."""
    if not guidance:
        return

    st.markdown("**General buying guidance**")
    st.caption("General advice, not a verified product recommendation.")

    for point in guidance.points:
        st.markdown(f"- {point}")

    if guidance.next_step:
        st.write(guidance.next_step)


def _render_comparison_table(products) -> None:
    """Display the available comparison data without inventing values."""
    if not products:
        return

    with st.expander("Compare candidate products"):
        rows = []

        for product in products:
            rows.append(
                {
                    "Product": product.name,
                    "Price": _format_price(
                        product.price, product.currency
                    ),
                    "Score (/100)": (
                        round(product.total_score, 1)
                        if product.total_score is not None
                        else None
                    ),
                    "Evidence coverage": (
                        f"{product.data_coverage:.0%}"
                    ),
                    "Review status": product.review_status,
                    "Fictional demo": product.is_synthetic,
                }
            )

        st.dataframe(
            rows,
            hide_index=True,
            use_container_width=True,
        )


def _render_diagnostics(result, state: ShoppingState) -> None:
    """Keep workflow diagnostics and raw JSON optional."""
    with st.expander("Sources, warnings and diagnostic output"):
        if result.warnings:
            st.markdown("**Workflow warnings**")
            for warning in result.warnings:
                st.caption(f"• {warning}")
        else:
            st.caption("No workflow warnings were reported.")

        st.download_button(
            label="Download result as JSON",
            data=result.model_dump_json(indent=2),
            file_name="shopping-result.json",
            mime="application/json",
            key=f"shopping_json_{id(state)}",
        )


def render_response(state: ShoppingState) -> None:
    """Render workflow output; do not perform research or scoring here."""
    result = state.recommendation_result

    if result is None:
        st.warning("No recommendation result was produced.")
        return

    status = result.status

    # Handle clarification before rendering any recommendation content.
    if status == "needs_clarification":
        _render_clarification(result)
        return

    # Give unsuccessful or limited-evidence results explicit messaging.
    if status in STATUS_MESSAGES:
        st.info(STATUS_MESSAGES[status])
    elif status == "success":
        st.markdown(
            "**Here are the strongest catalogue matches based on "
            "the available evidence:**"
        )
    else:
        st.warning(
            f"The workflow returned status: {status}. "
            "Review the available results and warnings carefully."
        )

    # Handle empty picks independently of the workflow status.
    if result.picks:
        for pick in result.picks:
            _render_pick(pick)
    elif status not in {"no_matches", "insufficient_data"}:
        st.info("No product recommendations are available for this result.")

    if result.comparison_table:
        _render_comparison_table(result.comparison_table)

    if result.general_guidance:
        _render_buying_guidance(result.general_guidance)

    if result.next_steps:
        st.caption(f"Next step: {result.next_steps[0]}")

    # Explain the score so users don't mistake it for verified quality.
    if result.picks or result.comparison_table:
        st.caption(
            "Scores reflect the configured criteria and available evidence. "
            "Evidence coverage indicates the proportion of weighted criteria "
            "that could be evaluated; it does not guarantee product quality."
        )

    _render_diagnostics(result, state)
