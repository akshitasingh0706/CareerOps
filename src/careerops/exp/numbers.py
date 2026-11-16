"""Find numbers, percentages and dollar amounts in text, in a form that can be compared."""
from __future__ import annotations

import re

_SCALE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "b": 1e9, "billion": 1e9}

_NUM = re.compile(
    r"(?<![\w.$])(?P<cur>\$)?(?P<num>\d[\d,]*(?:\.\d+)?)"
    r"(?:\s?(?P<suf>%|percent\b|thousand\b|million\b|billion\b|[kKmMbB]\b|x\b|X\b))?"
)
_IDENT = re.compile(r"\b[A-Za-z]+\d+[A-Za-z0-9]*\b")


def _fmt(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:.6g}"


def extract_numbers(text: str) -> list[str]:
    """Return normalized number tokens, e.g. '40%', '$1200000', '3x', '15', 'id:p95'.

    Magnitude suffixes are expanded (1.2M == 1,200,000). Identifiers such as p95 or S3 are
    returned as 'id:<lowercase>' so a changed identifier is caught.
    """
    out: list[str] = []
    for m in _NUM.finditer(text):
        raw = m.group("num").rstrip(",").replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        suf = (m.group("suf") or "").lower()
        cur = bool(m.group("cur"))
        if suf in _SCALE:
            value *= _SCALE[suf]
            suf = ""
        if suf == "percent":
            suf = "%"
        token = _fmt(value) + suf
        out.append(("$" if cur else "") + token)
    out.extend("id:" + t.lower() for t in _IDENT.findall(text))
    return out


def source_number_pool(texts: list[str]) -> set[str]:
    """All number tokens in the sources. A '$' token also matches its bare form and the reverse."""
    pool: set[str] = set()
    for t in texts:
        for n in extract_numbers(t):
            pool.add(n)
            if n.startswith("$"):
                pool.add(n[1:])
    return pool
