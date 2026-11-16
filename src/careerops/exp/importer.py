"""Import experience items from resume text. Items are saved as unapproved drafts."""
from __future__ import annotations

import sqlite3
import uuid
from datetime import date

from pydantic import BaseModel, Field

from careerops.domain import ExperienceItem, ExperienceKind
from careerops.exp import store
from careerops.exp.numbers import extract_numbers
from careerops.llm import LLM, Embedder

SYSTEM = (
    "You extract facts from a resume. Copy facts as written. Never add numbers, tools, "
    "scope or titles that are not in the text. One item per accomplishment, project or skill group."
)


class ExtractedItem(BaseModel):
    kind: ExperienceKind
    title: str
    employer: str | None = None
    start: date | None = None
    end: date | None = None
    text: str
    metrics: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    situation: str | None = None
    task: str | None = None
    action: str | None = None
    result: str | None = None


class ExtractedItems(BaseModel):
    items: list[ExtractedItem]


class ImportResult(BaseModel):
    items: list[ExperienceItem]  # saved as drafts
    rejected: list[str] = Field(default_factory=list)  # reasons, for items not saved


def import_resume_text(
    text: str,
    llm: LLM,
    *,
    conn: sqlite3.Connection,
    embedder: Embedder | None = None,
) -> ImportResult:
    """Extract items with the LLM and save them as drafts (approved=False).

    An item is rejected if it contains a number that is not in the resume text.
    """
    result = llm.structured(ExtractedItems, SYSTEM, f"Resume text:\n\n{text}")
    source_numbers = set(extract_numbers(text))
    saved: list[ExperienceItem] = []
    rejected: list[str] = []
    for ex in result.items:
        body = " ".join([ex.title, ex.text, *ex.metrics])
        bad = [n for n in extract_numbers(body) if n not in source_numbers]
        if bad:
            rejected.append(f"'{ex.title}': numbers not in resume text: {sorted(set(bad))}")
            continue
        item = ExperienceItem(id=uuid.uuid4().hex[:12], **ex.model_dump())
        store.save_item(conn, item, approved=False)
        saved.append(item)
    if embedder is not None and saved:
        vecs = embedder.embed([store.embedding_text(i) for i in saved])
        for item, vec in zip(saved, vecs, strict=True):
            store.put_embedding(conn, item, vec)
    return ImportResult(items=saved, rejected=rejected)
