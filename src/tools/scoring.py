from __future__ import annotations

import math
import re

from src.schemas import (
    ComparedProduct,
    CriterionScore,
    ResearchCandidate,
    ScoringConfig,
    ShoppingBrief,
)


ALIASES = {
    "value_for_money": "affordability",
    "value": "affordability",
    "price": "affordability",
    "battery_hours": "battery_life",
    "battery": "battery_life",
    "battery_mah": "battery_capacity",
    "ram_gb": "ram",
    "storage_gb": "storage",
    "warranty_months": "warranty",
    "weight_grams": "weight",
    "mic_quality": "microphone_quality",
}

SPEC_FIELDS = {
    "battery_life": (
        "hours",
        {"battery_hours", "battery_life", "battery_runtime"},
    ),
    "battery_capacity": (
        "mAh",
        {"battery_mah", "battery_capacity"},
    ),
    "ram": (
        "GB",
        {"ram_gb", "ram", "ram_memory_installed_size", "ram_memory_installed"},
    ),
    "storage": (
        "GB",
        {
            "storage_gb",
            "storage_capacity",
            "digital_storage_capacity",
            "memory_storage_capacity",
            "flash_memory_size",
        },
    ),
    "weight": (
        "g",
        {"weight_grams", "item_weight", "weight"},
    ),
}

UNIT_FACTORS = {
    "hours": {
        "h": 1, "hr": 1, "hrs": 1, "hour": 1, "hours": 1,
        "min": 1 / 60, "minutes": 1 / 60,
    },
    "mAh": {"mah": 1, "ah": 1000},
    "GB": {"gb": 1, "tb": 1024, "mb": 1 / 1024},
    "g": {
        "g": 1, "gram": 1, "grams": 1, "kg": 1000,
        "oz": 28.349523125, "ounce": 28.349523125,
        "ounces": 28.349523125, "lb": 453.59237,
        "pound": 453.59237, "pounds": 453.59237,
    },
}


def normalize_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_")
    return ALIASES.get(normalized, normalized)


RECOGNIZED_CRITERIA = {"affordability", "rating", "warranty"} | set(SPEC_FIELDS)


def scoring_weights(brief: ShoppingBrief) -> tuple[dict[str, float], str]:
    if brief.priorities:
        raw = {item.name: item.weight for item in brief.priorities}
        method = "User-provided numeric weights, normalized to sum to 1."

    elif brief.priority_order or brief.preferred_features:
        priorities = list(
            dict.fromkeys(brief.priority_order or brief.preferred_features)
        )
        raw = {
            name: len(priorities) - index
            for index, name in enumerate(priorities)
        }
        method = (
            "Configured rank policy: descending N..1 weights, normalized; "
            "not user-specified numeric weights."
        )

    else:
        raw = {"affordability": 0.5, "rating": 0.5}
        method = (
            "Prototype default: 50% affordability and 50% source rating; "
            "neither proves product quality."
        )

    normalized = {}
    dropped = []

    for name, weight in raw.items():
        key = normalize_name(name)
        if key not in RECOGNIZED_CRITERIA:
            dropped.append(name)
            continue
        normalized[key] = normalized.get(key, 0) + weight

    if not normalized:
        # None of the requested priorities can be measured quantitatively;
        # fall back instead of silently producing an unscored result.
        normalized = {"affordability": 0.5, "rating": 0.5}
        method = (
            f"None of the requested priorities ({', '.join(raw)}) map to a "
            "quantitative criterion, so affordability and source rating were "
            "used instead; those preferences were not scored numerically."
        )
    elif dropped:
        method += (
            f" Not quantitatively scorable and excluded from weighting: {', '.join(dropped)}."
        )

    total = sum(normalized.values())

    if total <= 0 or not math.isfinite(total):
        raise ValueError("Scoring weights must have a positive finite total")

    return {
        name: weight / total
        for name, weight in normalized.items()
    }, method


def numeric_quantity(
    value,
    unit: str,
    implicit_unit: bool,
) -> float | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        if implicit_unit and math.isfinite(value) and value > 0:
            return float(value)
        return None

    match = re.match(
        r"\s*([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]+)",
        str(value),
    )

    if not match:
        return None

    amount = float(match.group(1))
    factor = UNIT_FACTORS.get(unit, {}).get(match.group(2).casefold())

    if factor is None or amount <= 0:
        return None

    return amount * factor


def measurement(
    candidate: ResearchCandidate,
    criterion: str,
) -> tuple[float | None, str | None, str | None, str | None]:
    product = candidate.product

    direct_measurements = {
        "affordability": (
            product.price,
            product.currency,
            f"price: {product.price} {product.currency}",
        ),
        "rating": (
            product.rating,
            "stars / 5",
            f"source rating: {product.rating}",
        ),
        "warranty": (
            product.warranty_months,
            "months",
            f"warranty_months: {product.warranty_months}",
        ),
    }

    if criterion in direct_measurements:
        value, unit, evidence = direct_measurements[criterion]
        return value, unit, evidence, None

    if criterion not in SPEC_FIELDS:
        return (
            None,
            None,
            None,
            "No supported quantitative scoring rule for this criterion.",
        )

    unit, fields = SPEC_FIELDS[criterion]
    measurements = []

    for key, value in product.specifications.items():
        normalized = re.sub(
            r"[^a-z0-9]+", "_", key.casefold()
        ).strip("_")

        if normalized not in fields:
            continue

        implicit_unit = normalized.endswith(
            ("_hours", "_mah", "_gb", "_grams")
        )
        amount = numeric_quantity(value, unit, implicit_unit)

        if amount is not None:
            measurements.append((amount, f"{key}: {value}"))

    if not measurements:
        return (
            None,
            unit,
            None,
            "No measurement with verified units is available.",
        )

    first_value = measurements[0][0]

    if any(
        not math.isclose(value, first_value, rel_tol=1e-6)
        for value, _ in measurements[1:]
    ):
        evidence = "; ".join(text for _, text in measurements)
        return (
            None,
            unit,
            evidence,
            "Conflicting source measurements; criterion left unscored.",
        )

    return (
        first_value,
        unit,
        "; ".join(text for _, text in measurements),
        None,
    )


def normalize_score(
    criterion: str,
    value: float,
    unit: str | None,
    brief: ShoppingBrief,
    config: ScoringConfig,
) -> tuple[float | None, str]:
    if criterion == "affordability":
        if not brief.budget_max or brief.budget_max <= 0:
            return None, "A positive budget is required to score affordability."

        score = 100 * (1 - value / brief.budget_max)
        return max(0.0, min(100.0, score)), (
            "100 * (1 - price / budget), clamped to 0..100. "
            "Price-only proxy, not value or quality."
        )

    if criterion == "rating":
        if not 0 <= value <= 5:
            return None, "Source rating is outside the expected 0..5 range."

        return min(100.0, value / 5 * 100), (
            "Source rating / 5 * 100; source ratings do not prove "
            "verified product quality."
        )

    if criterion not in config.targets:
        return None, "No configured normalization target; zero contribution."

    target = config.targets[criterion]

    if not math.isfinite(target) or target <= 0:
        return None, "Normalization target must be positive and finite."

    if criterion == "weight":
        score = 100 * (1 - value / target)
    else:
        score = 100 * value / target

    return max(0.0, min(100.0, score)), (
        f"Configured scale: target={target} {unit}; "
        f"{'lower' if criterion == 'weight' else 'higher'} is better; "
        "clamped to 0..100."
    )


def score_candidate(
    candidate: ResearchCandidate,
    brief: ShoppingBrief,
    weights: dict[str, float],
    config: ScoringConfig,
) -> ComparedProduct:
    criteria = []
    warnings = list(candidate.warnings)
    coverage = 0.0

    for criterion, weight in weights.items():
        value, unit, evidence, problem = measurement(candidate, criterion)
        score = None

        if problem:
            method = problem
        else:
            method = "Missing source data; zero contribution without redistributing weight."

            if (
                value is not None
                and math.isfinite(value)
                and value >= 0
            ):
                score, method = normalize_score(
                    criterion,
                    value,
                    unit,
                    brief,
                    config,
                )

        if score is None:
            if weight > 0:
                warnings.append(f"{criterion}: {method}")
            contribution = 0.0
        else:
            coverage += weight
            contribution = weight * score

        criteria.append(
            CriterionScore(
                criterion=criterion,
                weight=weight,
                raw_value=value,
                unit=unit,
                score=score,
                contribution=contribution,
                method=method,
                evidence=evidence,
            )
        )

    total_score = (
        sum(item.contribution for item in criteria)
        if coverage > 0
        else None
    )

    return ComparedProduct(
        candidate=candidate,
        criteria=criteria,
        total_score=total_score,
        data_coverage=min(1.0, coverage),
        warnings=list(dict.fromkeys(warnings)),
    )
