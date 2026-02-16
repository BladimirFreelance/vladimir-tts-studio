from __future__ import annotations

import re
from typing import Iterable


def split_to_segments(text: str, split_regex: str, max_chars: int) -> list[str]:
    chunks = re.split(split_regex, text)
    chunks = [c.strip() for c in chunks if c.strip()]

    out: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            out.append(chunk)
            continue
        words = chunk.split()
        buf: list[str] = []
        for word in words:
            probe = " ".join(buf + [word])
            if len(probe) <= max_chars:
                buf.append(word)
            else:
                if buf:
                    out.append(" ".join(buf))
                buf = [word]
        if buf:
            out.append(" ".join(buf))
    return out


def indexed_segments(segments: Iterable[str], prefix: str = "utt") -> list[tuple[str, str]]:
    return [(f"{prefix}_{idx:05d}", text) for idx, text in enumerate(segments, start=1)]
