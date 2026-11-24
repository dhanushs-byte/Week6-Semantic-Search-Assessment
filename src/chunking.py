"""Document extraction, chunking, metadata and deterministic chunk identifiers."""
from __future__ import annotations

import hashlib
import uuid
from datetime import date
from pathlib import Path
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

NAMESPACE = uuid.UUID("c2a2a8a1-b26c-4f7d-a0e8-8a8e8c76ab12")


def extract_document(path: str | Path) -> list[dict]:
    """Return page-like records: {text, page}. Supports PDF, DOCX and TXT."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import fitz
        doc = fitz.open(path)
        return [{"text": page.get_text("text"), "page": i + 1} for i, page in enumerate(doc)]
    if suffix == ".docx":
        from docx import Document
        doc = Document(path)
        # DOCX has no reliable page boundaries; represent its content as page 1.
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [{"text": text, "page": 1}]
    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8")
        pages = text.split("\f")
        return [{"text": page, "page": i + 1} for i, page in enumerate(pages)]
    raise ValueError(f"Unsupported file type: {suffix}")


def _section_for(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line and (line.isupper() or line.endswith(":") or line.startswith("#")):
            return line.lstrip("# ").rstrip(":")[:100]
    return "General"


def fixed_token_chunks(text: str, size: int = 512, overlap: int = 50) -> list[str]:
    """Approximate token chunks using whitespace tokens, with overlap."""
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step)]


def sentence_chunks(text: str, max_words: int = 100) -> list[str]:
    """Group complete sentences where possible."""
    import re
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    chunks, current = [], []
    count = 0
    for sentence in sentences:
        n = len(sentence.split())
        if current and count + n > max_words:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def recursive_chunks(text: str, size: int = 512, overlap: int = 50) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)


def stable_chunk_id(source: str, page: int, section: str, chunk_index: int) -> str:
    """Stable identity is based on location, not content, so edits upsert in place."""
    key = f"{Path(source).name}|{page}|{section}|{chunk_index}"
    return str(uuid.uuid5(NAMESPACE, key))


def describe_chunk(metadata: dict) -> str:
    return (f"{metadata.get('source', 'unknown')} | page {metadata.get('page', '?')} | "
            f"{metadata.get('section', 'General')} | chunk {metadata.get('chunk_index', '?')}")


def build_chunks(paths: Iterable[str | Path], strategy: str = "recursive",
                 size: int = 512, overlap: int = 50) -> list[dict]:
    chunks = []
    for path in paths:
        path = Path(path)
        for page_record in extract_document(path):
            text = page_record["text"].strip()
            if not text:
                continue
            section = _section_for(text)
            if strategy == "fixed":
                pieces = fixed_token_chunks(text, size, overlap)
            elif strategy == "sentence":
                pieces = sentence_chunks(text, max_words=size)
            elif strategy == "recursive":
                pieces = recursive_chunks(text, size, overlap)
            else:
                raise ValueError("strategy must be fixed, sentence, or recursive")
            for idx, piece in enumerate(pieces):
                metadata = {
                    "source": path.name, "page": page_record["page"],
                    "section": section, "created_date": date.today().isoformat(),
                    "chunk_index": idx, "doc_type": path.suffix.lower().lstrip(".") or "txt"
                }
                chunks.append({
                    "id": stable_chunk_id(path.name, page_record["page"], section, idx),
                    "text": piece, "metadata": metadata
                })
    return chunks


def content_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
