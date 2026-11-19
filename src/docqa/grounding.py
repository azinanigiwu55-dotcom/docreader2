"""Cheap, model-free checks that an answer really comes from its cited sources.

A model can cite [1] and still state a figure that is not in source [1]. For
contracts, invoices and policies a wrong number is the costly failure, so every
figure in an answer must appear in the sources it cites. This does not prove an
answer is right, but it catches invented and hijacked numbers for free.
"""
from __future__ import annotations

import re

from .index import tokenize

_CITE = re.compile(r"\[\d+\]")
# "48,500", "48 500" and "48500" should all count as the same figure.
_THOUSANDS = re.compile(r"(?<=\d)[ , ](?=\d{3}(?!\d))")
_FIGURE = re.compile(r"\d+(?:\.\d+)?")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_STOP = {"the", "a", "an", "of", "for", "to", "in", "is", "are", "was", "were", "by", "and",
         "or", "with", "be", "at", "this", "that", "it", "its", "on", "as", "per"}


def figures(text: str) -> set[str]:
    """Numbers in a text, normalised (thousands separators and citation marks removed)."""
    return set(_FIGURE.findall(_THOUSANDS.sub("", _CITE.sub("", text))))


def ungrounded_figures(answer_text: str, question: str, sources: list[str]) -> list[str]:
    """Figures the answer states that are in neither the question nor the given sources."""
    allowed = figures(question)
    for s in sources:
        allowed |= figures(s)
    return sorted(figures(answer_text) - allowed)


def best_quote(answer_text: str, chunk_text: str, limit: int = 240) -> str:
    """The sentence of a source that best matches the answer, for showing next to a citation."""
    wanted = {t for t in tokenize(_CITE.sub("", answer_text))} - _STOP
    sentences = [s.strip() for s in _SENTENCE.split(chunk_text) if s.strip()]
    if not sentences:
        return ""
    best = max(sentences, key=lambda s: len(wanted & set(tokenize(s))))
    return best if len(best) <= limit else best[: limit - 1].rstrip() + "…"
