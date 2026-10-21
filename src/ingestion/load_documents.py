"""Load the synthetic PDF/DOCX reference documents (Data/electronics_marketplace_report.pdf,
Data/smart_device_buying_guide.docx) and split them into DocumentChunk records.

Both documents are explicitly self-labeled fictional/synthetic in their own
text. They exist to test the PDF/DOCX ingestion + chunking + embedding
pipeline, not as a source of real product facts.
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.table import Table
from pypdf import PdfReader

from src.ingestion.text_processing import chunk_text
from src.schemas import DocumentChunk

PDF_FILE = "electronics_marketplace_report.pdf"
DOCX_FILE = "smart_device_buying_guide.docx"
MIN_CHUNK_CHARS = 40


def _merge_short_fragments(fragments: list[str]) -> list[str]:
    """Merge fragments shorter than MIN_CHUNK_CHARS into the previous chunk."""
    merged: list[str] = []
    for fragment in fragments:
        fragment = fragment.strip()
        if not fragment:
            continue
        if merged and len(fragment) < MIN_CHUNK_CHARS:
            merged[-1] = f"{merged[-1]} {fragment}"
        else:
            merged.append(fragment)
    return merged


def load_pdf_chunks(data_dir: Path) -> list[DocumentChunk]:
    reader = PdfReader(data_dir / PDF_FILE)
    chunks = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for passage in chunk_text(text):
            chunk_index = len(chunks)
            chunks.append(DocumentChunk(
                chunk_id=f"{PDF_FILE}_{chunk_index}",
                source_file=PDF_FILE,
                doc_type="marketplace_report",
                chunk_index=chunk_index,
                text=passage,
                source_location=f"page {page_number}",
            ))
    return chunks


def load_docx_chunks(data_dir: Path) -> list[DocumentChunk]:
    doc = Document(data_dir / DOCX_FILE)
    chunks = []
    for block_number, block in enumerate(doc.iter_inner_content(), start=1):
        if isinstance(block, Table):
            text = "\n".join(" | ".join(cell.text for cell in row.cells) for row in block.rows)
        else:
            text = block.text
        for passage in chunk_text(text):
            chunk_index = len(chunks)
            chunks.append(DocumentChunk(
                chunk_id=f"{DOCX_FILE}_{chunk_index}",
                source_file=DOCX_FILE,
                doc_type="buying_guide",
                chunk_index=chunk_index,
                text=passage,
                source_location=f"block {block_number}",
            ))
    return chunks


def load_all_chunks(data_dir: Path) -> list[DocumentChunk]:
    return load_pdf_chunks(data_dir) + load_docx_chunks(data_dir)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    chunks = load_all_chunks(base_dir / "Data")
    print(f"Loaded {len(chunks)} document chunks from {PDF_FILE} and {DOCX_FILE}")
