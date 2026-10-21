"""Application service connecting the UI to the shopping workflow."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from src.schemas import ShoppingBrief
from src.state import ShoppingState
from src.workflow import run_shopping


@dataclass(frozen=True)
class ShoppingOptions:
    """Configuration for a shopping request."""

    include_synthetic: bool = False
    detailed: bool = True
    top_k: int = 5

async def shop(
    query: str,
    *,
    previous: ShoppingBrief | None = None,
    controls: ShoppingBrief | None = None,
    options: ShoppingOptions | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> ShoppingState:
    """Run a shopping request through the workflow."""
    config = options or ShoppingOptions()

    return await run_shopping(
        query,  # Pass the required positional argument
        previous_requirements=previous,
        controls=controls,
        fast_mode=not config.detailed,
        enable_guidance=True,
        include_synthetic=config.include_synthetic,
        top_k=config.top_k,
        on_progress=on_progress,
    )