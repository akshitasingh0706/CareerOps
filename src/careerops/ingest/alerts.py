"""Parse job alert emails (LinkedIn, Wellfound, Indeed style) into Jobs.

This reads email the user already received. It never visits any website.
Fetching mail from Gmail is a separate concern.
"""
from __future__ import annotations

import email
import re
from datetime import UTC, datetime
from email import policy
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from careerops.domain import Job, Source
from careerops.ingest.common import make_job_id, slugify

_LINKEDIN = re.compile(r"linkedin\.com/(?:comm/)?jobs/view/(?:[^/?#]*?-)?(\d{6,})")
_INDEED = re.compile(r"indeed\.com/(?:m/)?(?:viewjob|rc/clk|pagead/clk)\?[^\s\"']*")
_WELLFOUND = re.compile(r"(?:wellfound|angel)\.(?:com|co)/(?:l/|jobs/)([A-Za-z0-9]+)")
_URL = re.compile(r"https?://[^\s<>\"')\]]+")
_BUTTONS = {"view job", "view", "apply", "apply now", "see job", "see details", "view details",
            "easy apply", "view all jobs", "see all jobs", "view more jobs"}
_SKIP_LINES = ("unsubscribe", "you are receiving", "manage", "privacy", "©", "terms", "help center")


def _match_job_url(url: str) -> tuple[str, str, str] | None:
    """Return (provider, source_id, clean_url) if url is a job link."""
    if m := _LINKEDIN.search(url):
        return "linkedin", m.group(1), f"https://www.linkedin.com/jobs/view/{m.group(1)}"
    if _INDEED.search(url):
        jk = parse_qs(urlparse(url).query).get("jk", [None])[0]
        if jk:
            return "indeed", jk, f"https://www.indeed.com/viewjob?jk={jk}"
    if m := _WELLFOUND.search(url):
        path = url.split("?")[0].split("#")[0]
        return "wellfound", m.group(1), path
    return None


class _Tokens(HTMLParser):
    """Flatten HTML into ('text', line) and ('link', text, href) items."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.items: list[tuple] = []
        self._buf: list[str] = []
        self._href: str | None = None
        self._link_buf: list[str] = []
        self._skip = 0

    def _flush(self) -> None:
        line = re.sub(r"\s+", " ", "".join(self._buf)).strip()
        if line:
            self.items.append(("text", line))
        self._buf = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head"):
            self._skip += 1
        elif tag == "a":
            self._flush()
            self._href = dict(attrs).get("href")
            self._link_buf = []
        elif tag in ("p", "div", "br", "tr", "td", "li", "h1", "h2", "h3", "h4", "table"):
            self._flush()

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head"):
            self._skip = max(0, self._skip - 1)
        elif tag == "a":
            text = re.sub(r"\s+", " ", "".join(self._link_buf)).strip()
            if self._href:
                self.items.append(("link", text, self._href))
            self._href = None
        elif tag in ("p", "div", "tr", "td", "li", "h1", "h2", "h3", "h4", "table"):
            self._flush()

    def handle_data(self, data):
        if self._skip:
            return
        if self._href is not None:
            self._link_buf.append(data)
        else:
            self._buf.append(data)


def _tokens_from_html(raw: str) -> list[tuple]:
    p = _Tokens()
    p.feed(raw)
    p._flush()
    return p.items


def _tokens_from_text(raw: str) -> list[tuple]:
    """Plain text: a URL line becomes a link whose text is the first line of its block."""
    items: list[tuple] = []
    block: list[str] = []
    lines = raw.splitlines() + [""]
    for line in lines:
        line = line.strip()
        if not line:
            block = []
            continue
        urls = _URL.findall(line)
        if urls and _match_job_url(urls[0]):
            # The block above the URL holds title, company, location.
            if block:
                items.append(("link", block[0], urls[0]))
                items.extend(("text", b) for b in block[1:])
            else:
                items.append(("link", "", urls[0]))
            block = []
        else:
            block.append(line)
    return items


def _split_context(lines: list[str]) -> tuple[str, str | None, str]:
    """Pull company, location and a snippet out of the lines after a job title."""
    lines = [ln for ln in lines if ln.lower() not in _BUTTONS
             and not any(s in ln.lower() for s in _SKIP_LINES)]
    company, location, rest = "", None, []
    if lines:
        first = lines[0]
        parts = [p.strip() for p in re.split(r"\s[·•|\-–]\s", first) if p.strip()]
        if len(parts) >= 2:
            company, location = parts[0], parts[1]
            rest = lines[1:]
        else:
            company = first
            if len(lines) > 1:
                location = lines[1]
            rest = lines[2:]
    return company, location, " ".join(rest[:4])


def _job_from_item(provider: str, sid: str, url: str, title: str, ctx: list[str],
                   fetched_at: datetime) -> Job:
    company, location, snippet = _split_context(ctx)
    company = company or "Unknown"
    desc = snippet or f"{title} at {company}"
    return Job(
        id=make_job_id(Source.EMAIL_ALERT, f"{provider}:{sid}"), source=Source.EMAIL_ALERT,
        source_id=f"{provider}:{sid}", url=url, company_id=slugify(company),
        company_name=company, title=title, location=location,
        remote=True if location and "remote" in location.lower() else None,
        description=desc, fetched_at=fetched_at,
    )


def _jobs_from_tokens(items: list[tuple], fetched_at: datetime) -> list[Job]:
    # Mark job links, then give each titled link the text lines up to the next job link.
    marks = []
    for i, it in enumerate(items):
        if it[0] == "link" and (m := _match_job_url(it[2])):
            marks.append((i, m, it[1]))
    jobs: dict[str, Job] = {}
    for n, (i, (prov, sid, url), text) in enumerate(marks):
        if not text or text.lower() in _BUTTONS or key_seen(jobs, prov, sid):
            continue
        end = marks[n + 1][0] if n + 1 < len(marks) else len(items)
        ctx = [it[1] for it in items[i + 1:end] if it[0] == "text"][:5]
        jobs[f"{prov}:{sid}"] = _job_from_item(prov, sid, url, text, ctx, fetched_at)
    return list(jobs.values())


def key_seen(jobs: dict[str, Job], prov: str, sid: str) -> bool:
    return f"{prov}:{sid}" in jobs


def _looks_like_eml(raw: str) -> bool:
    head = raw.lstrip()[:2000]
    return bool(re.match(r"(?im)^(from|subject|mime-version|received|return-path|date|to):", head))


def _body_from_eml(raw: str) -> tuple[str, bool, datetime | None]:
    msg = email.message_from_string(raw, policy=policy.default)
    body = msg.get_body(preferencelist=("html", "plain"))
    content = body.get_content() if body else ""
    is_html = bool(body and body.get_content_type() == "text/html")
    date = None
    if msg["date"]:
        try:
            date = msg["date"].datetime
        except (ValueError, TypeError, AttributeError):
            date = None
    return content, is_html, date


def parse_alert_email(raw: str) -> list[Job]:
    """Parse a job alert email (raw .eml text, HTML or plain text) into Jobs.

    Jobs get source EMAIL_ALERT and the original listing URL.
    """
    fetched_at = datetime.now(UTC)
    if _looks_like_eml(raw):
        raw, is_html, _ = _body_from_eml(raw)
    else:
        is_html = bool(re.search(r"<(a|p|div|table|html|body)\b", raw, re.IGNORECASE))
    items = _tokens_from_html(raw) if is_html else _tokens_from_text(raw)
    return _jobs_from_tokens(items, fetched_at)


def read_eml(path: str | Path) -> list[Job]:
    """Read an .eml file and parse its alert jobs."""
    return parse_alert_email(Path(path).read_text(encoding="utf-8", errors="replace"))
