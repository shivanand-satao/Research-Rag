"""Keep uploaded PDFs original; provide readable PDF previews for text/Word sources."""
import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader, PdfWriter

from .processing import extract


def source_pdf(content: bytes, filename: str) -> bytes:
    if Path(filename).suffix.lower() == ".pdf":
        return content
    chunks = extract(content, filename)
    seen, paragraphs = set(), []
    for chunk in chunks:
        if chunk.parent_id not in seen:
            paragraphs.append(chunk.parent_text or chunk.text)
            seen.add(chunk.parent_id)
    font_paths = ["C:/Windows/Fonts/segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    font_path = next((path for path in font_paths if Path(path).exists()), None)
    font = ImageFont.truetype(font_path, 23) if font_path else ImageFont.load_default(size=23)
    heading = ImageFont.truetype(font_path, 29) if font_path else ImageFont.load_default(size=29)
    writer = PdfWriter()
    page_count, image, draw, y = 0, None, None, 0

    def flush():
        if image is not None:
            buffer = io.BytesIO()
            image.save(buffer, format="PDF", resolution=150)
            writer.add_page(PdfReader(buffer).pages[0])
            image.close()

    def page():
        nonlocal image, draw, y, page_count
        flush()
        if page_count >= 100:
            raise ValueError("This source is too long for a PDF preview.")
        page_count += 1
        image = Image.new("RGB", (1240, 1754), "white")
        draw = ImageDraw.Draw(image)
        draw.text((85, 65), filename[:70], font=heading, fill="#202020")
        draw.line((85, 118, 1155, 118), fill="#cccccc", width=1)
        draw.text((85, 1680), f"Research RAG | Source preview | Page {page_count}", font=font, fill="#777777")
        y = 155

    page()
    for paragraph in paragraphs:
        for line in paragraph.splitlines():
            wrapped, current = [], ""
            for word in line.split():
                candidate = f"{current} {word}".strip()
                if current and draw.textlength(candidate, font=font) > 1070:
                    wrapped.append(current); current = word
                else:
                    current = candidate
            wrapped.append(current)
            for line in wrapped:
                if y > 1620:
                    page()
                draw.text((85, y), line, font=font, fill="#111111")
                y += 34
        y += 20
    output = io.BytesIO()
    flush()
    writer.write(output)
    return output.getvalue()
