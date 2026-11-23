"""1-2. Input layer + document processing pipeline: parse -> chunk -> embed -> store.

Run:  python -m veriguide.ingest
Supports .pdf, .md and .txt files in data/docs/.
"""

import re
from pathlib import Path

from . import config, store
from .embeddings import embed

SUPPORTED = {".pdf", ".md", ".txt"}


# ---------- Parsing (text + structure) ----------

def _ocr_page(pdf_path: Path, page_no: int) -> str:
    """Optional OCR for scanned pages. Needs `pip install pytesseract pdf2image` + Tesseract/Poppler."""
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError:
        return ""
    images = convert_from_path(str(pdf_path), first_page=page_no, last_page=page_no)
    return pytesseract.image_to_string(images[0]) if images else ""


def parse_pdf(path: Path) -> list[dict]:
    from pypdf import PdfReader

    blocks = []
    for i, page in enumerate(PdfReader(str(path)).pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            text = _ocr_page(path, i).strip()
        if text:
            blocks.append({"page": i, "section": None, "text": text})
    return blocks


def parse_text(path: Path) -> list[dict]:
    """Split Markdown/plain text into sections using '#' headings (or ALL-CAPS lines)."""
    blocks, section, buf = [], None, []

    def flush():
        if any(line.strip() for line in buf):
            blocks.append({"page": None, "section": section, "text": "\n".join(buf).strip()})

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        heading = re.match(r"^#{1,6}\s+(.*)", line)
        if heading or (line.isupper() and 3 < len(line.strip()) < 80):
            flush()
            section, buf = (heading.group(1) if heading else line).strip(), []
        else:
            buf.append(line)
    flush()
    return blocks


def parse(path: Path) -> list[dict]:
    return parse_pdf(path) if path.suffix.lower() == ".pdf" else parse_text(path)


# ---------- Chunking ----------

def chunk_block(text: str, size: int, overlap: int) -> list[str]:
    """Pack whole sentences into ~`size`-word chunks; carry ~`overlap` words of sentences forward."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", " ".join(text.split())) if s]
    chunks, cur = [], []
    for s in sentences:
        if cur and sum(len(x.split()) for x in cur) + len(s.split()) > size:
            chunks.append(" ".join(cur))
            keep = []
            while cur and sum(len(x.split()) for x in keep) + len(cur[-1].split()) <= overlap:
                keep.insert(0, cur.pop())
            cur = keep
        cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def build_chunks(docs_dir: Path) -> list[dict]:
    chunks = []
    for path in sorted(docs_dir.rglob("*")):
        if path.suffix.lower() not in SUPPORTED:
            continue
        for block in parse(path):
            for piece in chunk_block(block["text"], config.CHUNK_WORDS, config.CHUNK_OVERLAP):
                # Prefix the section title so the embedding knows the context of the chunk.
                header = f"{block['section']}: " if block["section"] else ""
                chunks.append({**block, "doc": path.name, "text": header + piece})
    return chunks


def run() -> int:
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chunks = build_chunks(config.DOCS_DIR)
    if not chunks:
        store.save([], None)  # clear any stale index
        print(f"No documents found in {config.DOCS_DIR}")
        return 0
    print(f"Embedding {len(chunks)} chunks from {len({c['doc'] for c in chunks})} documents...")
    vectors = embed([c["text"] for c in chunks])
    store.save(chunks, vectors)
    print(f"Saved index to {config.STORE_DIR}")
    return len(chunks)


if __name__ == "__main__":
    run()
