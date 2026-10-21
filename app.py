"""Minimal multi-agent shopping chatbot. Run: streamlit run app.py"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
import asyncio
import streamlit as st
from src.ui.presentation import ROLE_LABELS, answer_text, price_label
from src.schemas import ShoppingBrief
from src.service import ShoppingOptions, shop

st.set_page_config(page_title="Smart Shopping Agent", page_icon="🛍️", layout="centered")
st.title("🛍️ Smart Shopping Agent")

with st.sidebar:
    st.subheader("Search settings")
    detailed = st.toggle("Detailed AI explanations and review analysis", value=True)
    synthetic = st.toggle("Include fictional demo products", value=False)
    if synthetic:
        st.warning("Fictional products are included for testing only.")
    top_k = 5
    currency = st.selectbox("Currency", ["From message", "USD", "INR", "EUR", "GBP"])
    if st.button("New conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["previous"] = None
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "previous" not in st.session_state:
    st.session_state.previous = None


def show_result(state, index: int) -> None:
    result = state.recommendation_result
    st.write(answer_text(state))
    if result is None:
        return
    if result.general_guidance:
        st.markdown("**General buying guidance (not verified product recommendations)**")
        for item in result.general_guidance.points:
            st.write("•", item)
        st.caption(result.general_guidance.next_step)
    for pick in result.picks:
        with st.container(border=True):
            st.markdown(f"**{ROLE_LABELS.get(pick.role, 'Option')}: {pick.name}**")
            st.write(f"{price_label(pick.price, pick.currency)} · Score {pick.total_score:.1f}/100 · Evidence coverage {pick.data_coverage:.0%}")
            if pick.is_synthetic:
                st.warning("Fictional demo product")
            if pick.product_url:
                st.link_button("View source listing", pick.product_url)
            for reason in pick.reasons:
                st.write(f"{'✓' if reason.type == 'strength' else '△'} {reason.text}")
            if pick.caveats:
                with st.expander("Limitations"):
                    for item in pick.caveats:
                        st.write("•", item)
    if result.comparison_table:
        with st.expander("Compare products"):
            st.dataframe([{
                "Product": p.name, "Price": price_label(p.price, p.currency),
                "Score": p.total_score, "Coverage": f"{p.data_coverage:.0%}",
                "Reviews": p.review_status, "Demo": p.is_synthetic,
            } for p in result.comparison_table], hide_index=True, use_container_width=True)
    with st.expander("Data quality and sources"):
        for warning in result.warnings:
            st.write("•", warning)
        if state.comparison_result:
            for product in state.comparison_result.products:
                st.markdown(f"**{product.candidate.product.name}**")
                for evidence in product.candidate.product_evidence + product.candidate.reviews:
                    st.caption(f"{evidence.source_file} · {evidence.point_id} · {evidence.freshness}")
    if result.next_steps:
        st.caption("Next: " + " · ".join(result.next_steps))
    st.download_button("Export JSON", result.model_dump_json(indent=2),
                       file_name=f"shopping_result_{index}.json", mime="application/json", key=f"export_{index}")

for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if "state" in message:
            show_result(message["state"], index)
        else:
            st.write(message["content"])

prompt = st.chat_input("e.g. Wireless headphones under USD 50 for meetings")
if prompt and prompt.strip():
    prompt = prompt.strip()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        try:
            controls = ShoppingBrief(currency=None if currency == "From message" else currency)
            with st.status("Researching", expanded=False) as status:
                state = asyncio.run(shop(
                    prompt, previous=st.session_state.previous, controls=controls,
                    options=ShoppingOptions(include_synthetic=synthetic, detailed=detailed, top_k=top_k),
                    on_progress=lambda label: status.update(label=label),
                ))
                status.update(label="Complete", state="complete")
            st.session_state.previous = state.recommendation_result.requirements
            index = len(st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "state": state})
            show_result(state, index)
        except Exception as exc:
            error = f"The request failed ({type(exc).__name__}): {exc}"
            st.error(error)
            st.session_state.messages.append({"role": "assistant", "content": error})
