"""Small helpers shared by the ingestion modules."""
from __future__ import annotations

import hashlib
import html
import re
from html.parser import HTMLParser

from careerops.domain import Source

_BLOCK_TAGS = {"p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "table"}
_SKIP_TAGS = {"script", "style", "head"}


def make_job_id(source: Source, source_id: str) -> str:
    """Stable job id: sha1(source + source_id)."""
    return hashlib.sha1(f"{source.value}{source_id}".encode()).hexdigest()


def slugify(name: str) -> str:
    """Lowercase company name as a slug, e.g. 'Acme Corp.' -> 'acme-corp'."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")
            if tag == "li":
                self.parts.append("- ")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(raw: str | None) -> str:
    """Convert HTML (possibly entity-escaped, as Greenhouse returns it) to plain text."""
    if not raw:
        return ""
    if "&lt;" in raw and "<" not in raw:
        raw = html.unescape(raw)
    parser = _TextParser()
    parser.feed(raw)
    text = "".join(parser.parts).replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    joined = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return re.sub(r"(- [^\n]*)\n\n(?=- )", r"\1\n", joined).strip()
