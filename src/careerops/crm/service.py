"""CRM service. All state lives in sqlite tables prefixed crm_."""
from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from careerops.crm.rules import (
    DEFAULT_RULES,
    REOPEN_TARGETS,
    TERMINAL,
    TRANSITIONS,
    FollowUpRule,
)
from careerops.db import connect
from careerops.domain import Application, Contact, FollowUp, Stage

SCHEMA = """
CREATE TABLE IF NOT EXISTS crm_applications (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    deadline TEXT,
    updated_at TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS crm_applications_job ON crm_applications(job_id);
CREATE TABLE IF NOT EXISTS crm_stage_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id TEXT NOT NULL REFERENCES crm_applications(id),
    from_stage TEXT,
    to_stage TEXT NOT NULL,
    at TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS crm_submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id TEXT NOT NULL REFERENCES crm_applications(id),
    channel TEXT NOT NULL,
    resume_version_id TEXT,
    at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS crm_contacts (
    id TEXT PRIMARY KEY,
    company_id TEXT,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS crm_contacts_company ON crm_contacts(company_id);
CREATE TABLE IF NOT EXISTS crm_contact_applications (
    contact_id TEXT NOT NULL REFERENCES crm_contacts(id),
    application_id TEXT NOT NULL REFERENCES crm_applications(id),
    PRIMARY KEY (contact_id, application_id)
);
CREATE TABLE IF NOT EXISTS crm_followups (
    id TEXT PRIMARY KEY,
    application_id TEXT NOT NULL REFERENCES crm_applications(id),
    due_on TEXT NOT NULL,
    done INTEGER NOT NULL DEFAULT 0,
    rule_key TEXT,
    completed_on TEXT,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS crm_followups_due ON crm_followups(done, due_on);
"""


class CRMError(Exception):
    """Base error for CRM problems."""


class NotFound(CRMError):
    """An id does not exist."""


class IllegalTransition(CRMError):
    """A stage change that the rules do not allow."""


@dataclass(frozen=True)
class Submission:
    application_id: str
    channel: str
    resume_version_id: str | None
    at: datetime


@dataclass(frozen=True)
class DueItem:
    """Something to act on: a follow-up or an application deadline."""

    kind: str  # "followup" or "deadline"
    application_id: str
    due_on: date
    note: str
    followup_id: str | None = None
    contact_id: str | None = None


@dataclass(frozen=True)
class TimelineEvent:
    at: datetime
    kind: str  # "stage", "submission" or "followup"
    summary: str


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class CRM:
    """Application CRM backed by sqlite.

    Pass a path (or None for the default db). Pass `rules` to override the default
    follow-up rules. Rules given for a stage replace that stage's defaults.
    """

    def __init__(
        self,
        path: str | None = None,
        rules: dict[Stage, list[FollowUpRule]] | None = None,
        conn: sqlite3.Connection | None = None,
        auto_followups: bool = True,
    ):
        self.conn = conn or connect(path)
        self.conn.executescript(SCHEMA)
        self.rules: dict[Stage, list[FollowUpRule]] = {**DEFAULT_RULES, **(rules or {})}
        self.auto_followups = auto_followups

    def close(self) -> None:
        self.conn.close()

    # ---- applications -------------------------------------------------

    def create_application(
        self,
        job_id: str,
        at: datetime | None = None,
        deadline: date | None = None,
        notes: str = "",
    ) -> Application:
        at = at or datetime.now()
        app = Application(
            id=_new_id(), job_id=job_id, stage=Stage.SAVED, deadline=deadline,
            notes=notes, updated_at=at,
        )
        with self.conn:
            self._save_app(app)
            self._add_history(app.id, None, Stage.SAVED, at, "created")
        return app

    def get_application(self, app_id: str) -> Application:
        row = self.conn.execute(
            "SELECT data FROM crm_applications WHERE id = ?", (app_id,)
        ).fetchone()
        if row is None:
            raise NotFound(f"application {app_id!r} not found")
        return Application.model_validate_json(row["data"])

    def application_for_job(self, job_id: str) -> Application | None:
        row = self.conn.execute(
            "SELECT data FROM crm_applications WHERE job_id = ? ORDER BY updated_at LIMIT 1",
            (job_id,),
        ).fetchone()
        return Application.model_validate_json(row["data"]) if row else None

    def list_applications(self, stage: Stage | None = None) -> list[Application]:
        if stage is None:
            rows = self.conn.execute("SELECT data FROM crm_applications ORDER BY updated_at")
        else:
            rows = self.conn.execute(
                "SELECT data FROM crm_applications WHERE stage = ? ORDER BY updated_at",
                (stage.value,),
            )
        return [Application.model_validate_json(r["data"]) for r in rows]

    def set_deadline(self, app_id: str, deadline: date | None) -> Application:
        app = self.get_application(app_id)
        app.deadline = deadline
        with self.conn:
            self._save_app(app)
        return app

    def advance(
        self, app_id: str, stage: Stage, at: datetime | None = None, note: str = ""
    ) -> Application:
        """Move an application to a new stage. Raises IllegalTransition if not allowed."""
        at = at or datetime.now()
        app = self.get_application(app_id)
        if stage == app.stage:
            raise IllegalTransition(f"application is already in stage {stage.value!r}")
        if stage not in TRANSITIONS[app.stage]:
            hint = ""
            if app.stage in TERMINAL:
                hint = f". {app.stage.value} is terminal; use reopen() first"
            raise IllegalTransition(
                f"cannot go from {app.stage.value} to {stage.value}{hint}"
            )
        return self._move(app, stage, at, note)

    def reopen(
        self, app_id: str, stage: Stage = Stage.APPLIED, at: datetime | None = None, note: str = ""
    ) -> Application:
        """Move a REJECTED or WITHDRAWN application back to an active stage."""
        at = at or datetime.now()
        app = self.get_application(app_id)
        if app.stage not in TERMINAL:
            raise IllegalTransition(f"application is {app.stage.value}, not rejected or withdrawn")
        if stage not in REOPEN_TARGETS:
            raise IllegalTransition(f"cannot reopen into {stage.value}")
        return self._move(app, stage, at, note or "reopened")

    def _move(self, app: Application, stage: Stage, at: datetime, note: str) -> Application:
        old = app.stage
        app.stage = stage
        app.updated_at = at
        with self.conn:
            self._save_app(app)
            self._add_history(app.id, old, stage, at, note)
        if self.auto_followups:
            self.schedule_default_followups(app.id)
        return app

    def stage_history(self, app_id: str) -> list[dict]:
        self.get_application(app_id)
        rows = self.conn.execute(
            "SELECT from_stage, to_stage, at, note FROM crm_stage_history "
            "WHERE application_id = ? ORDER BY id",
            (app_id,),
        )
        return [
            {
                "from_stage": Stage(r["from_stage"]) if r["from_stage"] else None,
                "to_stage": Stage(r["to_stage"]),
                "at": datetime.fromisoformat(r["at"]),
                "note": r["note"],
            }
            for r in rows
        ]

    # ---- submissions --------------------------------------------------

    def record_submission(
        self, app_id: str, channel: str, resume_version_id: str | None, at: datetime | None = None
    ) -> Application:
        """Log a submission and move the application to APPLIED.

        A repeat submission for an application already in APPLIED is logged
        and updates the submission fields. Later stages raise IllegalTransition.
        """
        at = at or datetime.now()
        app = self.get_application(app_id)
        if app.stage not in (Stage.SAVED, Stage.PREPARING, Stage.APPLIED):
            raise IllegalTransition(f"cannot submit an application in stage {app.stage.value}")
        app.submitted_at = at
        app.submission_channel = channel
        app.resume_version_id = resume_version_id
        with self.conn:
            self.conn.execute(
                "INSERT INTO crm_submissions (application_id, channel, resume_version_id, at) "
                "VALUES (?, ?, ?, ?)",
                (app_id, channel, resume_version_id, at.isoformat()),
            )
            self._save_app(app)
        if app.stage == Stage.APPLIED:
            return app
        return self._move(app, Stage.APPLIED, at, f"submitted via {channel}")

    def submissions(self, app_id: str) -> list[Submission]:
        self.get_application(app_id)
        rows = self.conn.execute(
            "SELECT * FROM crm_submissions WHERE application_id = ? ORDER BY id", (app_id,)
        )
        return [
            Submission(
                r["application_id"], r["channel"], r["resume_version_id"],
                datetime.fromisoformat(r["at"]),
            )
            for r in rows
        ]

    # ---- contacts -----------------------------------------------------

    def create_contact(
        self,
        name: str,
        role: str | None = None,
        company_id: str | None = None,
        email: str | None = None,
        linkedin_url: str | None = None,
    ) -> Contact:
        c = Contact(
            id=_new_id(), name=name, role=role, company_id=company_id,
            email=email, linkedin_url=linkedin_url,
        )
        with self.conn:
            self.conn.execute(
                "INSERT INTO crm_contacts (id, company_id, data) VALUES (?, ?, ?)",
                (c.id, c.company_id, c.model_dump_json()),
            )
        return c

    def get_contact(self, contact_id: str) -> Contact:
        row = self.conn.execute(
            "SELECT data FROM crm_contacts WHERE id = ?", (contact_id,)
        ).fetchone()
        if row is None:
            raise NotFound(f"contact {contact_id!r} not found")
        return self._contact_from_row(row["data"])

    def link_contact(self, contact_id: str, app_id: str) -> Contact:
        self.get_contact(contact_id)
        self.get_application(app_id)
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO crm_contact_applications VALUES (?, ?)",
                (contact_id, app_id),
            )
        return self.get_contact(contact_id)

    def set_contact_company(self, contact_id: str, company_id: str | None) -> Contact:
        c = self.get_contact(contact_id)
        c.company_id = company_id
        with self.conn:
            self.conn.execute(
                "UPDATE crm_contacts SET company_id = ?, data = ? WHERE id = ?",
                (company_id, c.model_dump_json(), contact_id),
            )
        return c

    def contacts_by_company(self, company_id: str) -> list[Contact]:
        rows = self.conn.execute(
            "SELECT data FROM crm_contacts WHERE company_id = ? ORDER BY rowid", (company_id,)
        )
        return [self._contact_from_row(r["data"]) for r in rows]

    def contacts_for_application(self, app_id: str) -> list[Contact]:
        rows = self.conn.execute(
            "SELECT c.data FROM crm_contacts c JOIN crm_contact_applications l "
            "ON l.contact_id = c.id WHERE l.application_id = ? ORDER BY c.rowid",
            (app_id,),
        )
        return [self._contact_from_row(r["data"]) for r in rows]

    def _contact_from_row(self, data: str) -> Contact:
        c = Contact.model_validate_json(data)
        rows = self.conn.execute(
            "SELECT application_id FROM crm_contact_applications WHERE contact_id = ? "
            "ORDER BY rowid",
            (c.id,),
        )
        c.application_ids = [r["application_id"] for r in rows]
        return c

    # ---- follow-ups ---------------------------------------------------

    def add_followup(
        self, app_id: str, due_on: date, note: str, contact_id: str | None = None,
        rule_key: str | None = None,
    ) -> FollowUp:
        self.get_application(app_id)
        if contact_id:
            self.get_contact(contact_id)
        f = FollowUp(id=_new_id(), application_id=app_id, due_on=due_on, note=note,
                     contact_id=contact_id)
        with self.conn:
            self.conn.execute(
                "INSERT INTO crm_followups (id, application_id, due_on, done, rule_key, data) "
                "VALUES (?, ?, ?, 0, ?, ?)",
                (f.id, app_id, due_on.isoformat(), rule_key, f.model_dump_json()),
            )
        return f

    def get_followup(self, followup_id: str) -> FollowUp:
        row = self.conn.execute(
            "SELECT data FROM crm_followups WHERE id = ?", (followup_id,)
        ).fetchone()
        if row is None:
            raise NotFound(f"follow-up {followup_id!r} not found")
        return FollowUp.model_validate_json(row["data"])

    def followups(self, app_id: str) -> list[FollowUp]:
        rows = self.conn.execute(
            "SELECT data FROM crm_followups WHERE application_id = ? ORDER BY due_on, rowid",
            (app_id,),
        )
        return [FollowUp.model_validate_json(r["data"]) for r in rows]

    def schedule_default_followups(self, app_id: str) -> list[FollowUp]:
        """Schedule the default follow-ups for the application's current stage.

        Rules with no `after` are dated from when the application entered the stage.
        A rule already scheduled for this application is not scheduled again.
        Rules with `after` are scheduled later, when the earlier follow-up is completed.
        Returns the follow-ups created.
        """
        app = self.get_application(app_id)
        if app.stage in TERMINAL:
            return []
        entered = self._entered_at(app)
        created = []
        for rule in self.rules.get(app.stage, []):
            if rule.after is not None or self._has_rule(app_id, rule.key):
                continue
            due_on = entered.date() + timedelta(days=rule.days)
            created.append(self.add_followup(app_id, due_on, rule.note, rule_key=rule.key))
        return created

    def complete(self, followup_id: str, on: date | None = None) -> list[FollowUp]:
        """Mark a follow-up done. Schedules any rule chained to it.

        `on` is the completion date (default today). Returns the new follow-ups.
        """
        on = on or date.today()
        f = self.get_followup(followup_id)
        row = self.conn.execute(
            "SELECT rule_key FROM crm_followups WHERE id = ?", (followup_id,)
        ).fetchone()
        f.done = True
        with self.conn:
            self.conn.execute(
                "UPDATE crm_followups SET done = 1, completed_on = ?, data = ? WHERE id = ?",
                (on.isoformat(), f.model_dump_json(), followup_id),
            )
        created = []
        app = self.get_application(f.application_id)
        if row["rule_key"] and app.stage not in TERMINAL:
            for rule in self.rules.get(app.stage, []):
                if rule.after == row["rule_key"] and not self._has_rule(app.id, rule.key):
                    due_on = on + timedelta(days=rule.days)
                    created.append(
                        self.add_followup(app.id, due_on, rule.note, rule_key=rule.key)
                    )
        return created

    def snooze(self, followup_id: str, days: int, today: date | None = None) -> FollowUp:
        """Push a follow-up back by `days`.

        The new date is today + days if `today` is given, else the old due date + days.
        """
        if days <= 0:
            raise CRMError("days must be positive")
        f = self.get_followup(followup_id)
        if f.done:
            raise CRMError("follow-up is already done")
        f.due_on = (today or f.due_on) + timedelta(days=days)
        with self.conn:
            self.conn.execute(
                "UPDATE crm_followups SET due_on = ?, data = ? WHERE id = ?",
                (f.due_on.isoformat(), f.model_dump_json(), followup_id),
            )
        return f

    def due(self, today: date, within_days: int = 0) -> list[DueItem]:
        """Open follow-ups and application deadlines due on or before today + within_days.

        Includes overdue items. Items for rejected or withdrawn applications are left out.
        Deadlines count only until the application is submitted.
        """
        return self._open_items(today + timedelta(days=within_days))

    def overdue(self, today: date) -> list[DueItem]:
        """Open items whose date is before today."""
        return self._open_items(today - timedelta(days=1))

    def _open_items(self, last_day: date) -> list[DueItem]:
        items: list[DueItem] = []
        rows = self.conn.execute(
            "SELECT f.data FROM crm_followups f JOIN crm_applications a "
            "ON a.id = f.application_id "
            "WHERE f.done = 0 AND f.due_on <= ? AND a.stage NOT IN ('rejected', 'withdrawn')",
            (last_day.isoformat(),),
        )
        for r in rows:
            f = FollowUp.model_validate_json(r["data"])
            items.append(
                DueItem("followup", f.application_id, f.due_on, f.note, f.id, f.contact_id)
            )
        rows = self.conn.execute(
            "SELECT data FROM crm_applications WHERE deadline IS NOT NULL AND deadline <= ? "
            "AND stage IN ('saved', 'preparing')",
            (last_day.isoformat(),),
        )
        for r in rows:
            a = Application.model_validate_json(r["data"])
            items.append(DueItem("deadline", a.id, a.deadline, f"Deadline for job {a.job_id}"))
        items.sort(key=lambda i: (i.due_on, i.kind, i.application_id))
        return items

    def _has_rule(self, app_id: str, key: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM crm_followups WHERE application_id = ? AND rule_key = ?",
            (app_id, key),
        ).fetchone()
        return row is not None

    def _entered_at(self, app: Application) -> datetime:
        row = self.conn.execute(
            "SELECT at FROM crm_stage_history WHERE application_id = ? AND to_stage = ? "
            "ORDER BY id DESC LIMIT 1",
            (app.id, app.stage.value),
        ).fetchone()
        if app.stage == Stage.APPLIED and app.submitted_at:
            return app.submitted_at
        return datetime.fromisoformat(row["at"]) if row else app.updated_at

    # ---- reports ------------------------------------------------------

    def pipeline_summary(self) -> dict[Stage, int]:
        """Count of applications in each stage. Every stage appears, even with zero."""
        counts = {s: 0 for s in Stage}
        for r in self.conn.execute("SELECT stage, COUNT(*) AS n FROM crm_applications GROUP BY stage"):
            counts[Stage(r["stage"])] = r["n"]
        return counts

    def stale_applications(self, today: date, days: int) -> list[Application]:
        """Active applications with no change in `days` days or more, oldest first.

        Rejected and withdrawn applications are not active.
        """
        cutoff = today - timedelta(days=days)
        return [
            a for a in self.list_applications()
            if a.stage not in TERMINAL and a.updated_at.date() <= cutoff
        ]

    def application_timeline(self, app_id: str) -> list[TimelineEvent]:
        """Stage changes, submissions and follow-ups for one application, in time order.

        A follow-up appears at its due date (start of day), and again when completed.
        """
        events: list[TimelineEvent] = []
        for h in self.stage_history(app_id):
            frm = h["from_stage"].value if h["from_stage"] else "none"
            text = f"{frm} -> {h['to_stage'].value}"
            if h["note"]:
                text += f" ({h['note']})"
            events.append(TimelineEvent(h["at"], "stage", text))
        for s in self.submissions(app_id):
            text = f"submitted via {s.channel}"
            if s.resume_version_id:
                text += f", resume {s.resume_version_id}"
            events.append(TimelineEvent(s.at, "submission", text))
        rows = self.conn.execute(
            "SELECT due_on, done, completed_on, data FROM crm_followups "
            "WHERE application_id = ?",
            (app_id,),
        )
        for r in rows:
            f = FollowUp.model_validate_json(r["data"])
            due_at = datetime.combine(date.fromisoformat(r["due_on"]), time.min)
            events.append(TimelineEvent(due_at, "followup", f"follow-up due: {f.note}"))
            if r["completed_on"]:
                done_at = datetime.combine(date.fromisoformat(r["completed_on"]), time.max)
                events.append(TimelineEvent(done_at, "followup", f"follow-up done: {f.note}"))
        events.sort(key=lambda e: e.at)  # stable: keeps insertion order for ties
        return events

    # ---- internals ----------------------------------------------------

    def _save_app(self, app: Application) -> None:
        self.conn.execute(
            "INSERT INTO crm_applications (id, job_id, stage, deadline, updated_at, data) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
            "stage = excluded.stage, deadline = excluded.deadline, "
            "updated_at = excluded.updated_at, data = excluded.data",
            (
                app.id, app.job_id, app.stage.value,
                app.deadline.isoformat() if app.deadline else None,
                app.updated_at.isoformat(), app.model_dump_json(),
            ),
        )

    def _add_history(
        self, app_id: str, old: Stage | None, new: Stage, at: datetime, note: str
    ) -> None:
        self.conn.execute(
            "INSERT INTO crm_stage_history (application_id, from_stage, to_stage, at, note) "
            "VALUES (?, ?, ?, ?, ?)",
            (app_id, old.value if old else None, new.value, at.isoformat(), note),
        )
