"""Grounding check. Code decides what is verified. The LLM judge can only add failures."""
from __future__ import annotations

import sqlite3
from collections.abc import Collection

from pydantic import BaseModel, Field

from careerops.domain import ExperienceItem, ResumeVersion
from careerops.exp import store
from careerops.exp.numbers import extract_numbers, source_number_pool
from careerops.exp.tailor import format_item
from careerops.llm import LLM

JUDGE_SYSTEM = (
    "You check resume bullets against their source items. For each bullet, list every claim "
    "that the sources do not support: tools or technologies, scope, scale, seniority or title, "
    "team size, ownership, outcomes. Rewording is fine. Be strict. Mark supported=true only "
    "if every claim is in the sources."
)


class BulletJudgement(BaseModel):
    bullet_index: int  # 0-based
    supported: bool
    unsupported_claims: list[str] = Field(default_factory=list)


class JudgeResult(BaseModel):
    judgements: list[BulletJudgement]


def _item_texts(item: ExperienceItem) -> list[str]:
    parts = [item.title, item.text, *item.metrics]
    parts += [p for p in (item.employer, item.situation, item.task, item.action, item.result) if p]
    parts += [str(d.year) for d in (item.start, item.end) if d]
    return parts


def check_grounding_code(
    resume: ResumeVersion,
    items_by_id: dict[str, ExperienceItem],
    approved_ids: Collection[str] | None = None,
) -> list[str]:
    """Code-only checks. Returns failure notes. An empty list means the code checks passed."""
    notes: list[str] = []
    for n, b in enumerate(resume.bullets, 1):
        if not b.source_item_ids:
            notes.append(f"bullet {n}: no source items cited")
            continue
        usable: list[ExperienceItem] = []
        for sid in b.source_item_ids:
            if sid not in items_by_id:
                notes.append(f"bullet {n}: cited id '{sid}' does not exist")
            elif approved_ids is not None and sid not in approved_ids:
                notes.append(f"bullet {n}: cited item '{sid}' is not approved")
            else:
                usable.append(items_by_id[sid])
        pool = source_number_pool([t for it in usable for t in _item_texts(it)])
        for num in dict.fromkeys(extract_numbers(b.text)):
            if num not in pool:
                shown = num.removeprefix("id:")
                notes.append(
                    f"bullet {n}: number '{shown}' is not in the cited items' text or metrics"
                )
    return notes


def verify(
    resume: ResumeVersion,
    items_by_id: dict[str, ExperienceItem],
    llm: LLM,
    *,
    approved_ids: Collection[str] | None = None,
) -> ResumeVersion:
    """Return a copy of resume with verified and verification_notes set.

    verified is True only if the code checks pass and the judge flags nothing. Pass
    approved_ids to require that every cited item is approved. If it is None, items_by_id
    must hold approved items only.
    """
    notes = check_grounding_code(resume, items_by_id, approved_ids)
    judged: dict[int, BulletJudgement] = {}
    if resume.bullets:
        blocks = []
        for i, b in enumerate(resume.bullets):
            srcs = "\n".join(format_item(items_by_id[s]) for s in b.source_item_ids
                             if s in items_by_id) or "(no valid sources)"
            blocks.append(f"Bullet {i}: {b.text}\nSources:\n{srcs}")
        # If the judge call raises, the error propagates and nothing is marked verified.
        result = llm.structured(JudgeResult, JUDGE_SYSTEM, "\n\n".join(blocks))
        judged = {j.bullet_index: j for j in result.judgements}
        for i in range(len(resume.bullets)):
            j = judged.get(i)
            if j is None:
                notes.append(f"bullet {i + 1}: judge returned no result")
            elif not j.supported or j.unsupported_claims:
                claims = "; ".join(j.unsupported_claims) or "unspecified claim"
                notes.append(f"bullet {i + 1}: judge found unsupported claims: {claims}")
    if not resume.bullets:
        notes.append("resume has no bullets")
    return resume.model_copy(update={"verified": not notes, "verification_notes": notes})


def verify_with_store(resume: ResumeVersion, llm: LLM, *, conn: sqlite3.Connection) -> ResumeVersion:
    """Verify against the experience bank. Approval is read from the store."""
    items = {i.id: i for i in store.list_items(conn)}
    return verify(resume, items, llm, approved_ids=store.approved_ids(conn))
