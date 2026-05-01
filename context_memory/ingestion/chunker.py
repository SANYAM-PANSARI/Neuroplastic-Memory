"""
Markdown chunker — splits documents by headers into semantic chunks.
Used for .md files, design docs, and structured text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from context_memory.agents.llm import count_tokens
from context_memory.config import settings


@dataclass
class Chunk:
    """A single chunk of text with metadata."""
    text: str
    source_uri: str
    chunk_index: int
    token_count: int


def chunk_markdown(text: str, source_uri: str = "") -> list[Chunk]:
    """
    Split markdown text by top-level headers (## and above).
    If a section exceeds chunk_size_tokens, it's split further by sub-headers.
    Fallback: splits by token count if no headers found.
    """
    max_tokens = settings.chunk_size_tokens
    chunks: list[Chunk] = []

    # Split by ## headers (keep the header with its content)
    sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)

    for section in sections:
        section = section.strip()
        if not section:
            continue

        tokens = count_tokens(section)

        if tokens <= max_tokens:
            chunks.append(Chunk(
                text=section,
                source_uri=source_uri,
                chunk_index=len(chunks),
                token_count=tokens,
            ))
        else:
            # Section too large — split by ### sub-headers
            sub_sections = re.split(r"(?=^### )", section, flags=re.MULTILINE)
            for sub in sub_sections:
                sub = sub.strip()
                if not sub:
                    continue
                sub_tokens = count_tokens(sub)
                if sub_tokens <= max_tokens:
                    chunks.append(Chunk(
                        text=sub,
                        source_uri=source_uri,
                        chunk_index=len(chunks),
                        token_count=sub_tokens,
                    ))
                else:
                    # Still too large — hard split by token count
                    for piece in _hard_split(sub, max_tokens):
                        chunks.append(Chunk(
                            text=piece,
                            source_uri=source_uri,
                            chunk_index=len(chunks),
                            token_count=count_tokens(piece),
                        ))

    # Edge case: no headers found at all
    if not chunks and text.strip():
        for piece in _hard_split(text.strip(), max_tokens):
            chunks.append(Chunk(
                text=piece,
                source_uri=source_uri,
                chunk_index=len(chunks),
                token_count=count_tokens(piece),
            ))

    return chunks


def _hard_split(text: str, max_tokens: int) -> list[str]:
    """
    Fallback: split text into pieces of roughly max_tokens.
    Splits on paragraph boundaries, then sentences for oversized paragraphs.
    """
    paragraphs = text.split("\n\n")
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        # Bug 9 fix: if a single paragraph exceeds max_tokens, split by sentences
        if para_tokens > max_tokens:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sentence in sentences:
                s_tokens = count_tokens(sentence)
                if current_tokens + s_tokens > max_tokens and current:
                    pieces.append("\n\n".join(current))
                    current = []
                    current_tokens = 0
                current.append(sentence)
                current_tokens += s_tokens
            continue

        if current_tokens + para_tokens > max_tokens and current:
            pieces.append("\n\n".join(current))
            current = []
            current_tokens = 0
        current.append(para)
        current_tokens += para_tokens

    if current:
        pieces.append("\n\n".join(current))

    return pieces
