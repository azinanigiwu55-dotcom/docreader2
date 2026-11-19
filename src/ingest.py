"""Parse documents (PDF, DOCX, TXT/MD) into citation-ready chunks.

Every chunk remembers which document it came from, where in that document
(page for PDFs, section number otherwise) and which groups may read it.
That metadata is what makes citations and permission-aware retrieval possible.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

SUPPORTED = {".pdf", ".docx", ".txt", ".md"}


@dataclass
class Chunk:
    chunk_id: str
    doc: str
    loc: str            # "p.2" for PDFs, "§3" for other formats
    text: str
    acl: list[str] = field(default_factory=lambda: ["all"])

    @property
    def doc_title(self) -> str:
        return Path(self.doc).stem.replace("_", " ")

    @property
    def index_text(self) -> str:
        """Text used for search. The title helps tell near-identical documents apart."""
        return f"{self.doc_title}. {self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ parsing
def read_units(path: Path) -> list[tuple[str, str]]:
    """Return [(location_label, text)] for a file. Label is '' when unknown."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [(f"p.{i + 1}", page.extract_text() or "") for i, page in enumerate(reader.pages)]
    if suffix == ".docx":
        from docx import Document

        doc = Document(str(path))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text.strip() for cell in row.cells))
        return [("", "\n".join(parts))]
    if suffix in {".txt", ".md"}:
        return [("", path.read_text(encoding="utf-8", errors="ignore"))]
    raise ValueError(f"Unsupported file type: {path.suffix}")


# ----------------------------------------------------------------- chunking
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def chunk_text(text: str, max_chars: int = 900, overlap: int = 150) -> list[str]:
    """Greedy sentence packing with a small overlap between neighbouring chunks."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for sent in sentences:
        if current and size + len(sent) + 1 > max_chars:
            chunks.append("\n".join(current))
            # carry the tail of the previous chunk forward as overlap
            tail: list[str] = []
            tail_size = 0
            for prev in reversed(current):
                if tail_size + len(prev) > overlap:
                    break
                tail.insert(0, prev)
                tail_size += len(prev) + 1
            current, size = tail, tail_size
        current.append(sent)
        size += len(sent) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


# ------------------------------------------------------------------- folder
def load_acl(path: str | Path | None) -> dict[str, list[str]]:
    if path and Path(path).exists():
        return json.loads(Path(path).read_text())
    return {}


def ingest_folder(
    docs_dir: str | Path,
    acl: dict[str, list[str]] | None = None,
    default_acl: list[str] | None = None,
    max_chars: int = 900,
    overlap: int = 150,
) -> list[Chunk]:
    """Chunk every supported file in docs_dir.

    Files missing from `acl` get `default_acl` (default: readable by everyone).
    Change that default to ["admin"] if you prefer default-deny.
    """
    acl = acl or {}
    default_acl = default_acl or ["all"]
    chunks: list[Chunk] = []
    for path in sorted(Path(docs_dir).iterdir()):
        if path.suffix.lower() not in SUPPORTED:
            continue
        groups = acl.get(path.name, default_acl)
        section = 0
        for loc, text in read_units(path):
            for piece in chunk_text(text, max_chars, overlap):
                section += 1
                label = loc or f"§{section}"
                chunks.append(Chunk(
                    chunk_id=f"{path.name}::{len(chunks)}",
                    doc=path.name, loc=label, text=piece, acl=list(groups),
                ))
    return chunks
