from src.schemas import ShoppingBrief
from src.tools.scoring import scoring_weights

def test_weights_normalized():
    weights, _ = scoring_weights(ShoppingBrief(category="headphones", budget_max=50, currency="USD", priority_order=["battery life", "rating"]))
    assert abs(sum(weights.values()) - 1) < 1e-9
