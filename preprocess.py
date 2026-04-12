"""
Pre-processing pipeline for Vulcan OmniPro 220 documentation.
Converts PDFs into AI-ready formats:
  - Per-page PNG images (300 DPI)
  - Docling Markdown extraction
"""

import os
import json
from pathlib import Path
from pdf2image import convert_from_path

BASE = Path(__file__).parent
FILES_DIR = BASE / "files"
KB_DIR = BASE / "knowledge_base"
PAGES_DIR = KB_DIR / "pages"
MARKDOWN_DIR = KB_DIR / "markdown"

PDFS = [
    "owner-manual.pdf",
    "quick-start-guide.pdf",
    "selection-chart.pdf",
]

DPI = 300


def rasterize_pdfs():
    """Convert each PDF page to a PNG image."""
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}

    for pdf_name in PDFS:
        pdf_path = FILES_DIR / pdf_name
        if not pdf_path.exists():
            print(f"  SKIP: {pdf_name} not found")
            continue

        stem = pdf_path.stem
        print(f"  Rasterizing {pdf_name}...")
        images = convert_from_path(str(pdf_path), dpi=DPI)

        page_files = []
        for i, img in enumerate(images, start=1):
            out_name = f"{stem}_p{i:03d}.png"
            out_path = PAGES_DIR / out_name
            img.save(str(out_path), "PNG")
            page_files.append(out_name)

        manifest[pdf_name] = {
            "total_pages": len(images),
            "files": page_files,
        }
        print(f"    -> {len(images)} pages")

    manifest_path = KB_DIR / "pages_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest written to {manifest_path}")


def extract_with_docling():
    """Use Docling to extract Markdown from each PDF."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        print("  Docling not installed. Run: pip install docling")
        return

    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    converter = DocumentConverter()

    for pdf_name in PDFS:
        pdf_path = FILES_DIR / pdf_name
        if not pdf_path.exists():
            print(f"  SKIP: {pdf_name} not found")
            continue

        stem = pdf_path.stem
        print(f"  Extracting {pdf_name} with Docling...")
        result = converter.convert(str(pdf_path))
        md_content = result.document.export_to_markdown()

        out_path = MARKDOWN_DIR / f"{stem}.md"
        with open(out_path, "w") as f:
            f.write(md_content)
        print(f"    -> {out_path} ({len(md_content)} chars)")


if __name__ == "__main__":
    print("=== Step 1: Rasterize PDFs ===")
    rasterize_pdfs()

    print("\n=== Step 2: Docling extraction ===")
    extract_with_docling()

    print("\nDone.")
