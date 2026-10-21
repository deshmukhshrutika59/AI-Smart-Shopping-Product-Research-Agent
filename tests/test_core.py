from src.ingestion.text_processing import chunk_text
from src.schemas import Product, ShoppingBrief
from src.agents.product_researcher import eligible_product
from src.ui.presentation import price_label

def test_chunks_have_content():
    chunks = chunk_text("Hello world " * 150)
    assert chunks and all(chunks)

def test_hard_budget_filter():
    product = Product(product_id="test", name="Keyboard", category="Keyboard", price=80,
                      currency="USD", source_name="test", source_file="test.csv")
    eligible, _ = eligible_product(product, ShoppingBrief(category="keyboard", budget_max=50, currency="USD"))
    assert not eligible

def test_missing_price_label():
    assert price_label(None, "USD") == "Price unavailable"
