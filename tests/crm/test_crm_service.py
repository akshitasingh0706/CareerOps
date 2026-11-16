from datetime import date, datetime, timedelta

import pytest

from careerops.crm import (
    CRM,
    TRANSITIONS,
    FollowUpRule,
    IllegalTransition,
    NotFound,
)
from careerops.domain import Stage

T0 = datetime(2026, 3, 1, 9, 0)


@pytest.fixture
def crm(tmp_path):
    c = CRM(str(tmp_path / "crm.db"))
    yield c
    c.close()


def make_app(crm, job="job1", at=T0):
    return crm.create_application(job, at=at)


# ---- transitions -------------------------------------------------------


def test_create_records_history(crm):
    app = make_app(crm)
    assert app.stage == Stage.SAVED
    h = crm.stage_history(app.id)
    assert len(h) == 1 and h[0]["from_stage"] is None and h[0]["to_stage"] == Stage.SAVED


@pytest.mark.parametrize(
    "src,dst", [(s, d) for s, ds in TRANSITIONS.items() for d in ds]
)
def test_every_allowed_transition_works(crm, src, dst):
    app = make_app(crm)
    # walk to src by the shortest path, using reopen for terminal states
    path = {
        Stage.SAVED: [],
        Stage.PREPARING: [Stage.PREPARING],
        Stage.APPLIED: [Stage.APPLIED],
        Stage.SCREEN: [Stage.APPLIED, Stage.SCREEN],
        Stage.INTERVIEW: [Stage.APPLIED, Stage.INTERVIEW],
        Stage.OFFER: [Stage.APPLIED, Stage.OFFER],
    }.get(src)
    if path is None:
        pytest.skip("terminal stages have no outgoing moves")
    for s in path:
        crm.advance(app.id, s, at=T0)
    assert crm.advance(app.id, dst, at=T0 + timedelta(days=1)).stage == dst


def test_every_disallowed_transition_raises(crm):
    for src, allowed in TRANSITIONS.items():
        for dst in Stage:
            if dst in allowed or dst == src:
                continue
            app = make_app(crm, job=f"{src}-{dst}")
            _force_stage(crm, app.id, src)
            with pytest.raises(IllegalTransition):
                crm.advance(app.id, dst, at=T0)
            assert crm.get_application(app.id).stage == src


def _force_stage(crm, app_id, stage):
    app = crm.get_application(app_id)
    app.stage = stage
    with crm.conn:
        crm._save_app(app)


def test_same_stage_rejected(crm):
    app = make_app(crm)
    with pytest.raises(IllegalTransition, match="already"):
        crm.advance(app.id, Stage.SAVED)


def test_rejected_to_interview_needs_reopen(crm):
    app = make_app(crm)
    crm.advance(app.id, Stage.APPLIED, at=T0)
    crm.advance(app.id, Stage.REJECTED, at=T0 + timedelta(days=5))
    with pytest.raises(IllegalTransition, match="reopen"):
        crm.advance(app.id, Stage.INTERVIEW)
    crm.reopen(app.id, Stage.APPLIED, at=T0 + timedelta(days=6))
    assert crm.advance(app.id, Stage.INTERVIEW, at=T0 + timedelta(days=7)).stage == Stage.INTERVIEW


def test_withdrawn_is_terminal(crm):
    app = make_app(crm)
    crm.advance(app.id, Stage.WITHDRAWN, at=T0)
    with pytest.raises(IllegalTransition):
        crm.advance(app.id, Stage.APPLIED)


def test_reopen_rules(crm):
    app = make_app(crm)
    with pytest.raises(IllegalTransition, match="not rejected"):
        crm.reopen(app.id)
    crm.advance(app.id, Stage.REJECTED, at=T0)
    with pytest.raises(IllegalTransition, match="cannot reopen into"):
        crm.reopen(app.id, Stage.OFFER)
    got = crm.reopen(app.id, Stage.SAVED, at=T0 + timedelta(days=1), note="recruiter came back")
    assert got.stage == Stage.SAVED
    assert crm.stage_history(app.id)[-1]["note"] == "recruiter came back"


def test_unknown_application(crm):
    with pytest.raises(NotFound):
        crm.advance("nope", Stage.APPLIED)
    with pytest.raises(NotFound):
        crm.get_application("nope")


def test_history_is_in_order_with_notes(crm):
    app = make_app(crm)
    crm.advance(app.id, Stage.APPLIED, at=T0 + timedelta(days=1), note="sent")
    crm.advance(app.id, Stage.SCREEN, at=T0 + timedelta(days=3))
    h = crm.stage_history(app.id)
    assert [x["to_stage"] for x in h] == [Stage.SAVED, Stage.APPLIED, Stage.SCREEN]
    assert h[1]["note"] == "sent" and h[1]["from_stage"] == Stage.SAVED
    assert crm.get_application(app.id).updated_at == T0 + timedelta(days=3)


# ---- submissions -------------------------------------------------------


def test_record_submission(crm):
    app = make_app(crm)
    at = datetime(2026, 3, 2, 14, 30)
    got = crm.record_submission(app.id, "company site", "rv1", at)
    assert got.stage == Stage.APPLIED
    assert got.submitted_at == at
    assert got.submission_channel == "company site"
    assert got.resume_version_id == "rv1"
    subs = crm.submissions(app.id)
    assert len(subs) == 1 and subs[0].at == at and subs[0].resume_version_id == "rv1"


def test_resubmission_logged_and_stays_applied(crm):
    app = make_app(crm)
    crm.record_submission(app.id, "company site", "rv1", T0)
    got = crm.record_submission(app.id, "referral", "rv2", T0 + timedelta(days=2))
    assert got.stage == Stage.APPLIED and got.resume_version_id == "rv2"
    assert [s.channel for s in crm.submissions(app.id)] == ["company site", "referral"]
    assert len(crm.stage_history(app.id)) == 2  # no second APPLIED entry


def test_submission_after_applied_stage_rejected(crm):
    app = make_app(crm)
    crm.record_submission(app.id, "x", None, T0)
    crm.advance(app.id, Stage.SCREEN, at=T0)
    with pytest.raises(IllegalTransition):
        crm.record_submission(app.id, "x", None, T0)


# ---- contacts ----------------------------------------------------------


def test_contacts(crm):
    a1, a2 = make_app(crm, "j1"), make_app(crm, "j2")
    c1 = crm.create_contact("Ann", role="Recruiter", company_id="acme", email="a@acme.com")
    c2 = crm.create_contact("Bob", company_id="acme")
    crm.create_contact("Cy", company_id="other")
    crm.link_contact(c1.id, a1.id)
    crm.link_contact(c1.id, a2.id)
    crm.link_contact(c1.id, a2.id)  # idempotent
    assert [c.name for c in crm.contacts_by_company("acme")] == ["Ann", "Bob"]
    assert crm.get_contact(c1.id).application_ids == [a1.id, a2.id]
    assert [c.name for c in crm.contacts_for_application(a1.id)] == ["Ann"]
    assert crm.contacts_for_application(a1.id)[0].email == "a@acme.com"
    assert crm.contacts_by_company("nobody") == []
    crm.set_contact_company(c2.id, "other")
    assert len(crm.contacts_by_company("acme")) == 1
    with pytest.raises(NotFound):
        crm.link_contact("bad", a1.id)
    with pytest.raises(NotFound):
        crm.link_contact(c1.id, "bad")


# ---- follow-ups --------------------------------------------------------


def test_applied_schedules_followup_seven_days_after_submission(crm):
    app = make_app(crm)
    crm.record_submission(app.id, "site", None, datetime(2026, 3, 10, 23, 0))
    fus = crm.followups(app.id)
    assert [f.due_on for f in fus] == [date(2026, 3, 17)]


def test_schedule_is_idempotent(crm):
    app = make_app(crm)
    crm.advance(app.id, Stage.APPLIED, at=T0)
    assert len(crm.followups(app.id)) == 1
    assert crm.schedule_default_followups(app.id) == []
    assert len(crm.followups(app.id)) == 1


def test_interview_thank_you_after_three_days(crm):
    app = make_app(crm)
    crm.advance(app.id, Stage.APPLIED, at=T0)
    crm.advance(app.id, Stage.INTERVIEW, at=datetime(2026, 3, 30, 10, 0))
    due_dates = {f.note: f.due_on for f in crm.followups(app.id)}
    assert due_dates["Send a thank-you note."] == date(2026, 4, 2)


def test_chain_second_followup_then_fourteen_days(crm):
    app = make_app(crm)
    crm.advance(app.id, Stage.APPLIED, at=T0)  # applied_1 due 3/8
    first = crm.followups(app.id)[0]
    new = crm.complete(first.id, on=date(2026, 3, 9))
    assert [f.due_on for f in new] == [date(2026, 3, 16)]  # applied_2: 7 days after completion
    new = crm.complete(new[0].id, on=date(2026, 3, 16))
    assert [f.due_on for f in new] == [date(2026, 3, 30)]  # applied_3: 14 days after
    assert crm.complete(new[0].id, on=date(2026, 3, 30)) == []


def test_complete_does_not_chain_for_terminal_app(crm):
    app = make_app(crm)
    crm.advance(app.id, Stage.APPLIED, at=T0)
    first = crm.followups(app.id)[0]
    crm.advance(app.id, Stage.REJECTED, at=T0 + timedelta(days=1))
    assert crm.complete(first.id, on=date(2026, 3, 9)) == []
    assert crm.get_followup(first.id).done


def test_custom_rules_override(tmp_path):
    rules = {Stage.APPLIED: [FollowUpRule("quick", 2, "Ping them")]}
    c = CRM(str(tmp_path / "x.db"), rules=rules)
    app = c.create_application("j", at=T0)
    c.advance(app.id, Stage.APPLIED, at=T0)
    f = c.followups(app.id)
    assert [(x.due_on, x.note) for x in f] == [(date(2026, 3, 3), "Ping them")]
    c.advance(app.id, Stage.INTERVIEW, at=T0)  # default INTERVIEW rules still apply
    assert len(c.followups(app.id)) == 2


def test_auto_followups_off(tmp_path):
    c = CRM(str(tmp_path / "x.db"), auto_followups=False)
    app = c.create_application("j", at=T0)
    c.advance(app.id, Stage.APPLIED, at=T0)
    assert c.followups(app.id) == []
    assert len(c.schedule_default_followups(app.id)) == 1


def test_due_and_overdue_boundaries(crm):
    app = make_app(crm)
    crm.advance(app.id, Stage.APPLIED, at=T0)  # due 2026-03-08
    assert crm.due(date(2026, 3, 7)) == []
    assert len(crm.due(date(2026, 3, 8))) == 1
    assert crm.overdue(date(2026, 3, 8)) == []
    assert len(crm.overdue(date(2026, 3, 9))) == 1
    assert len(crm.due(date(2026, 3, 9))) == 1
    assert len(crm.due(date(2026, 3, 6), within_days=2)) == 1


def test_done_followup_not_listed(crm):
    app = make_app(crm)
    crm.advance(app.id, Stage.APPLIED, at=T0)
    crm.complete(crm.followups(app.id)[0].id, on=date(2026, 3, 8))
    assert crm.due(date(2026, 3, 8)) == []


def test_snooze_from_due_date_and_from_today(crm):
    app = make_app(crm)
    crm.advance(app.id, Stage.APPLIED, at=T0)
    f = crm.followups(app.id)[0]  # due 3/8
    assert crm.snooze(f.id, 3).due_on == date(2026, 3, 11)
    assert crm.overdue(date(2026, 3, 10)) == []
    assert len(crm.overdue(date(2026, 3, 12))) == 1
    assert crm.snooze(f.id, 2, today=date(2026, 3, 20)).due_on == date(2026, 3, 22)
    assert crm.get_followup(f.id).due_on == date(2026, 3, 22)


def test_snooze_errors(crm):
    app = make_app(crm)
    crm.advance(app.id, Stage.APPLIED, at=T0)
    f = crm.followups(app.id)[0]
    with pytest.raises(Exception, match="positive"):
        crm.snooze(f.id, 0)
    crm.complete(f.id, on=date(2026, 3, 8))
    with pytest.raises(Exception, match="done"):
        crm.snooze(f.id, 1)
    with pytest.raises(NotFound):
        crm.complete("nope")


def test_terminal_application_hides_followups(crm):
    app = make_app(crm)
    crm.advance(app.id, Stage.APPLIED, at=T0)
    crm.advance(app.id, Stage.REJECTED, at=T0)
    assert crm.due(date(2027, 1, 1)) == []


def test_deadlines_surface_in_due(crm):
    app = crm.create_application("j", at=T0, deadline=date(2026, 3, 20))
    assert crm.due(date(2026, 3, 19)) == []
    items = crm.due(date(2026, 3, 20))
    assert [(i.kind, i.application_id, i.due_on) for i in items] == [
        ("deadline", app.id, date(2026, 3, 20))
    ]
    assert len(crm.overdue(date(2026, 3, 21))) == 1
    crm.record_submission(app.id, "site", None, T0)
    assert [i.kind for i in crm.due(date(2026, 3, 21))] == ["followup"]
    crm.set_deadline(app.id, None)


def test_due_sorted_by_date(crm):
    a = crm.create_application("a", at=T0, deadline=date(2026, 3, 5))
    b = make_app(crm, "b")
    crm.advance(b.id, Stage.APPLIED, at=T0)
    items = crm.due(date(2026, 3, 30))
    assert [i.due_on for i in items] == [date(2026, 3, 5), date(2026, 3, 8)]
    assert items[0].application_id == a.id


def test_manual_followup_with_contact(crm):
    app = make_app(crm)
    c = crm.create_contact("Ann")
    f = crm.add_followup(app.id, date(2026, 4, 1), "Email Ann", contact_id=c.id)
    item = crm.due(date(2026, 4, 1))[0]
    assert item.contact_id == c.id and item.followup_id == f.id


# ---- reports -----------------------------------------------------------


def test_pipeline_summary(crm):
    a, b, _ = make_app(crm, "a"), make_app(crm, "b"), make_app(crm, "c")
    crm.advance(a.id, Stage.APPLIED, at=T0)
    crm.advance(b.id, Stage.APPLIED, at=T0)
    crm.advance(b.id, Stage.SCREEN, at=T0)
    s = crm.pipeline_summary()
    assert s[Stage.SAVED] == 1 and s[Stage.APPLIED] == 1 and s[Stage.SCREEN] == 1
    assert s[Stage.OFFER] == 0 and set(s) == set(Stage)


def test_stale_applications(crm):
    old = make_app(crm, "old", at=datetime(2026, 1, 1))
    mid = make_app(crm, "mid", at=datetime(2026, 2, 1))
    make_app(crm, "new", at=datetime(2026, 2, 25))
    dead = make_app(crm, "dead", at=datetime(2026, 1, 1))
    crm.advance(dead.id, Stage.REJECTED, at=datetime(2026, 1, 2))
    got = crm.stale_applications(date(2026, 3, 1), days=14)
    assert [a.id for a in got] == [old.id, mid.id]
    # boundary: exactly `days` old counts as stale
    assert [a.id for a in crm.stale_applications(date(2026, 3, 1), days=28)] == [old.id, mid.id]
    assert [a.id for a in crm.stale_applications(date(2026, 3, 1), days=29)] == [old.id]


def test_timeline_order(crm):
    app = make_app(crm)
    crm.record_submission(app.id, "site", "rv1", datetime(2026, 3, 2, 10, 0))
    crm.complete(crm.followups(app.id)[0].id, on=date(2026, 3, 9))
    crm.advance(app.id, Stage.SCREEN, at=datetime(2026, 3, 5, 9, 0))
    tl = crm.application_timeline(app.id)
    assert [e.at for e in tl] == sorted(e.at for e in tl)
    kinds = [e.kind for e in tl]
    assert kinds.count("stage") == 3 and kinds.count("submission") == 1
    assert kinds.count("followup") >= 2
    assert tl[0].summary.startswith("none -> saved")
    assert any("resume rv1" in e.summary for e in tl)
    # follow-up due 3/9 (submission 3/2 + 7) comes after the 3/5 stage change
    due = next(e for e in tl if e.summary.startswith("follow-up due: No response"))
    assert due.at == datetime(2026, 3, 9)
    assert tl.index(due) > kinds.index("stage", 2)


# ---- persistence -------------------------------------------------------


def test_persistence_across_connections(tmp_path):
    path = str(tmp_path / "p.db")
    c1 = CRM(path)
    app = c1.create_application("job9", at=T0, deadline=date(2026, 4, 1))
    c1.record_submission(app.id, "referral", "rv7", T0 + timedelta(days=1))
    contact = c1.create_contact("Ann", company_id="acme")
    c1.link_contact(contact.id, app.id)
    c1.close()

    c2 = CRM(path)
    got = c2.get_application(app.id)
    assert got.stage == Stage.APPLIED and got.resume_version_id == "rv7"
    assert got.deadline == date(2026, 4, 1)
    assert c2.application_for_job("job9").id == app.id
    assert c2.application_for_job("zzz") is None
    assert [s.channel for s in c2.submissions(app.id)] == ["referral"]
    assert len(c2.stage_history(app.id)) == 2
    assert c2.contacts_by_company("acme")[0].application_ids == [app.id]
    assert len(c2.followups(app.id)) == 1
    assert c2.list_applications(Stage.APPLIED)[0].id == app.id
    assert c2.list_applications(Stage.OFFER) == []
    c2.close()
