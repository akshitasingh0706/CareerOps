"""Feedback log in table rank_feedback. Component values are stored with each action so later
changes to scoring do not alter old training rows."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from pydantic import BaseModel

from careerops.domain import Feedback, FeedbackAction, FitScore

POSITIVE = {FeedbackAction.SAVED, FeedbackAction.APPLIED, FeedbackAction.INTERVIEWED}
NEGATIVE = {FeedbackAction.SKIPPED, FeedbackAction.REJECTED_BY_ME}


def label(action: FeedbackAction) -> int:
    """1 for positive actions, 0 for negative ones."""
    return 1 if action in POSITIVE else 0


class FeedbackRow(BaseModel):
    feedback: Feedback
    components: dict[str, float]
    rules_score: float
    scorer_version: str

    @property
    def label(self) -> int:
        return label(self.feedback.action)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS rank_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    action TEXT NOT NULL,
    at TEXT NOT NULL,
    reason TEXT,
    components TEXT NOT NULL,
    rules_score REAL NOT NULL,
    scorer_version TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS rank_feedback_job ON rank_feedback(job_id);
"""


class FeedbackStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        conn.executescript(_SCHEMA)

    def record(self, feedback: Feedback, fit: FitScore) -> None:
        """Log one action with the component values and rules score the job had at that time."""
        self.conn.execute(
            "INSERT INTO rank_feedback (job_id, action, at, reason, components, rules_score,"
            " scorer_version) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                feedback.job_id,
                feedback.action.value,
                feedback.at.isoformat(),
                feedback.reason,
                json.dumps(fit.components, sort_keys=True),
                fit.overall,
                fit.scorer_version,
            ),
        )
        self.conn.commit()

    def rows(self) -> list[FeedbackRow]:
        cur = self.conn.execute("SELECT * FROM rank_feedback ORDER BY id")
        return [
            FeedbackRow(
                feedback=Feedback(
                    job_id=r["job_id"],
                    action=FeedbackAction(r["action"]),
                    at=datetime.fromisoformat(r["at"]),
                    reason=r["reason"],
                ),
                components=json.loads(r["components"]),
                rules_score=r["rules_score"],
                scorer_version=r["scorer_version"],
            )
            for r in cur.fetchall()
        ]

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM rank_feedback").fetchone()[0])
