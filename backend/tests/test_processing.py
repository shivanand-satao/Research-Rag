import io

import pytest
from docx import Document
from pypdf import PdfWriter

from app.processing import extract, split_text
from app.chunking import SemanticChunker


def test_long_text_keeps_complete_sentences_and_parent_context():
    sentence = "Evidence matters when comparing research findings."
    text = (sentence + " ") * 200
    chunker = SemanticChunker(child_tokens=80, parent_tokens=400)
    chunks = split_text(text, page=3, chunker=chunker)
    assert len(chunks) > 2
    assert all(chunker.count(chunk.text) <= 80 for chunk in chunks)
    assert all(chunk.page == 3 for chunk in chunks)
    assert all(chunk.text.endswith(".") for chunk in chunks)
    assert all(len(chunk.parent_text) >= len(chunk.text) for chunk in chunks)


def test_unicode_chunk_boundaries_preserve_characters():
    text = "研究 प्रमाण evidence " * 80
    chunker = SemanticChunker(child_tokens=40, parent_tokens=160)
    chunks = split_text(text, chunker=chunker)
    assert all(chunker.count(chunk.text) <= 40 for chunk in chunks)
    assert not any("�" in chunk.text for chunk in chunks)


def test_a_coherent_paragraph_is_not_cut_into_300_byte_fragments():
    paragraph = "The experiment measured recall across a representative dataset. " * 12
    chunks = split_text(paragraph)
    assert len(paragraph.encode()) > 300
    assert len(chunks) == 1
    assert chunks[0].text == paragraph.strip()


def test_semantic_topic_changes_create_variable_sized_children():
    paragraphs = ["Biological findings and cellular respiration. " * 20,
        "Additional biological results and ATP synthesis. " * 15,
        "Financial analysis of bond prices and interest rates. " * 20]
    chunker = SemanticChunker(semantic_embed=lambda windows: [[1, 0], [1, 0], [0, 1]], minimum_tokens=50)
    chunks = chunker.chunk("\n\n".join(paragraphs), page=2)
    assert len(chunks) == 2
    assert "Financial" not in chunks[0].text
    assert "Financial" in chunks[1].text
    assert chunks[0].parent_id == chunks[1].parent_id
    assert len(chunks[0].text) != len(chunks[1].text)


def test_markdown_keeps_section_metadata():
    chunks = extract(b"# Methods\nWe compare three studies.\n# Results\nThe evidence is mixed.", "notes.md")
    assert {chunk.section for chunk in chunks} == {"Methods", "Results"}


def test_word_extracts_headings_and_tables():
    document = Document()
    document.add_heading("Findings", level=1)
    document.add_paragraph("The study reports improved recall.")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Recall"
    table.cell(0, 1).text = "87%"
    stream = io.BytesIO()
    document.save(stream)
    chunks = extract(stream.getvalue(), "paper.docx")
    assert all(chunk.section == "Findings" for chunk in chunks)
    assert "Recall | 87%" in "\n".join(chunk.text for chunk in chunks)


def test_empty_pdf_and_unsupported_files_fail_cleanly():
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    stream = io.BytesIO()
    writer.write(stream)
    with pytest.raises(ValueError, match="No readable text"):
        extract(stream.getvalue(), "blank.pdf")
    with pytest.raises(ValueError, match="PDF, DOCX"):
        extract(b"anything", "file.exe")

