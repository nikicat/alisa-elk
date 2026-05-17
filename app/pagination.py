import re

# Alice limit is 1024 chars; we reserve some space for a continuation suffix.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+|\n+")


def chunk_for_alice(text: str, *, chunk_chars: int = 1000) -> list[str]:
    """Split text into <=chunk_chars pieces, preferring sentence boundaries."""
    text = text.strip()
    if not text:
        return [""]
    if len(text) <= chunk_chars:
        return [text]
    parts = _SENTENCE_BOUNDARY.split(text)
    chunks: list[str] = []
    buf = ""
    for part in parts:
        if not part:
            continue
        candidate = (buf + " " + part).strip() if buf else part
        if len(candidate) <= chunk_chars:
            buf = candidate
            continue
        if buf:
            chunks.append(buf)
            buf = ""
        # Single sentence longer than the limit: hard-split.
        while len(part) > chunk_chars:
            chunks.append(part[:chunk_chars])
            part = part[chunk_chars:]
        buf = part
    if buf:
        chunks.append(buf)
    return chunks
