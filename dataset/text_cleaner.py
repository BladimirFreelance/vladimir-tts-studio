from __future__ import annotations

import re
from typing import Iterable

ABBREVIATIONS = {
    "т.д.": "и так далее",
    "т.п.": "и тому подобное",
    "г.": "город",
    "%": " процентов",
}


def normalize_text(
    raw: str, replacements: dict[str, str], expand_abbreviations: bool = True
) -> str:
    text = raw.replace("\ufeff", "")
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(\d+)[ ]*%", r"\1 процентов", text)
    if expand_abbreviations:
        for old, new in ABBREVIATIONS.items():
            text = text.replace(old, new)
    return text


def remove_garbage_lines(lines: Iterable[str], min_chars: int = 6) -> list[str]:
    output: list[str] = []
    for line in lines:
        cleaned = line.strip()
        if len(cleaned) < min_chars:
            continue
        if cleaned.count("\ufffd"):
            continue
        output.append(cleaned)
    return output
