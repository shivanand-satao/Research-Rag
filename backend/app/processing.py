"""Extract source structure and create semantic, hierarchical research chunks."""
import io
import logging
import re
import zipfile
from pathlib import Path

from .config import settings
from .chunking import Chunk, SemanticChunker, structured_chunks

logger = logging.getLogger(__name__)
MIME_TYPES = {".pdf": "application/pdf", ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
              ".txt": "text/plain", ".md": "text/markdown"}


def split_text(text: str, *, page=None, section=None, chunker=None) -> list[Chunk]:
    return structured_chunks(text, chunker or SemanticChunker(), page=page, section=section)


def ocr_image(image) -> str:
    try:
        import pytesseract
    except ImportError as exc:
        raise ValueError("OCR requires requirements-ocr.txt and Tesseract. Install them or set OCR=false.") from exc
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
    try:
        return pytesseract.image_to_string(image, lang=settings.ocr_language, timeout=30)
    except (RuntimeError, pytesseract.TesseractNotFoundError, pytesseract.TesseractError):
        logger.warning("OCR unavailable for an image; native text is retained.")
        return ""


def extract(content: bytes, filename: str, *, tokenizer=None, progress=None) -> list[Chunk]:
    suffix = Path(filename).suffix.lower()
    if suffix not in MIME_TYPES:
        raise ValueError("Upload a PDF, DOCX, TXT, or Markdown file.")
    chunks: list[Chunk] = []
    chunker = SemanticChunker(tokenizer, child_tokens=settings.embedding_child_tokens,
        parent_tokens=settings.parent_tokens)
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            raise ValueError("Remove the PDF password before uploading.")
        if len(reader.pages) > 500:
            raise ValueError("Please split PDFs longer than 500 pages.")
        ocr_pdf = None
        if settings.ocr:
            try:
                import pdfplumber
            except ImportError as exc:
                raise ValueError("OCR requires requirements-ocr.txt and Tesseract. Install them or set OCR=false.") from exc
            ocr_pdf = pdfplumber.open(io.BytesIO(content))
        try:
            for number, page in enumerate(reader.pages, 1):
                native = page.extract_text() or ""
                chunks.extend(split_text(native, page=number, chunker=chunker))
                if ocr_pdf and (not native.strip() or bool(page.images)):
                    rendered = ocr_pdf.pages[number - 1].to_image(resolution=120).original
                    recognized = ocr_image(rendered)
                    native_lines = {line.strip().lower() for line in native.splitlines()}
                    extra = "\n".join(line for line in recognized.splitlines() if line.strip().lower() not in native_lines)
                    chunks.extend(split_text(extra, page=number, section="Image text", chunker=chunker))
                if progress:
                    progress(number, len(reader.pages))
        finally:
            if ocr_pdf:
                ocr_pdf.close()
    elif suffix == ".docx":
        from docx import Document
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            if sum(item.file_size for item in archive.infolist()) > 100 * 1024 * 1024:
                raise ValueError("The expanded Word document is too large.")
        document = Document(io.BytesIO(content))
        section, parts = "Document", []
        for block in document.iter_inner_content():
            if hasattr(block, "text"):
                if block.style and block.style.name.startswith("Heading"):
                    chunks.extend(split_text("\n\n".join(parts), section=section, chunker=chunker))
                    section, parts = block.text.strip() or "Document", [block.text]
                else:
                    parts.append(block.text)
            else:
                parts.append("\n".join(" | ".join(cell.text for cell in row.cells) for row in block.rows))
        chunks.extend(split_text("\n\n".join(parts), section=section, chunker=chunker))
        if settings.ocr:
            from PIL import Image
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                for entry in archive.infolist():
                    if entry.filename.startswith("word/media/") and entry.filename.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
                        with Image.open(io.BytesIO(archive.read(entry))) as picture:
                            chunks.extend(split_text(ocr_image(picture), section="Image text", chunker=chunker))
    else:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("Save text files with UTF-8 encoding before uploading.") from exc
        section = "Document"
        for part in re.split(r"(?m)(?=^#{1,6} )", text):
            if part.startswith("#"):
                section = part.splitlines()[0].lstrip("# ")
            chunks.extend(split_text(part, section=section, chunker=chunker))
    if not chunks:
        raise ValueError("No readable text was found. Scanned documents require OCR and Tesseract.")
    if len(chunks) > 6000:
        raise ValueError("This document is too large to index. Split it into smaller files.")
    return chunks

