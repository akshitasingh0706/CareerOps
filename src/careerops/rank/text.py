"""Small text helpers shared by filters and components."""
from __future__ import annotations

import re

_STOP = {
    "and", "the", "for", "with", "that", "this", "from", "into", "our", "you", "your",
    "are", "will", "have", "has", "work", "working", "team", "teams", "using", "use",
}


def contains_term(text: str, term: str) -> bool:
    """True if term appears in text as a whole word or phrase, ignoring case."""
    term = term.strip().lower()
    if not term:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])"
    return re.search(pattern, text.lower()) is not None


def tokens(text: str) -> set[str]:
    """Lowercase word tokens longer than two characters, minus a few filler words."""
    return {t for t in re.findall(r"[a-z0-9+#]+", text.lower()) if len(t) > 2 and t not in _STOP}
