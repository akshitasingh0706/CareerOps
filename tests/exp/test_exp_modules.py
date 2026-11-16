from datetime import UTC, datetime

from careerops.domain import ResumeBullet, ResumeVersion
from careerops.exp import (
    approved_ids,
    coverage_report,
    delete_item,
    get_item,
    get_resume,
    import_resume_text,
    is_approved,
    list_items,
    save_item,
    save_resume,
    select_items,
    set_approved,
    tailor_resume,
    verify_with_store,
)
from careerops.exp.numbers import extract_numbers
from careerops.llm import FakeLLM, HashEmbedder
from tests.exp.conftest import make_item

RESUME = """Acme, Senior Engineer 2020-2023
- Cut p95 latency 40% on the checkout API using Python and Postgres.
- Led migration of 12 services to AWS."""


def test_crud_and_approval(conn):
    it = make_item("a", "Did a thing", skills=["Python"])
    save_item(conn, it)
    assert get_item(conn, "a") == it
    assert not is_approved(conn, "a")
    assert list_items(conn, approved=True) == []
    set_approved(conn, "a")
    assert approved_ids(conn) == {"a"}
    delete_item(conn, "a")
    assert get_item(conn, "a") is None


def test_import_saves_drafts_and_rejects_invented_numbers(conn):
    llm = FakeLLM(structured={"ExtractedItems": {"items": [
        {"kind": "accomplishment", "title": "Checkout latency", "employer": "Acme",
         "text": "Cut p95 latency 40% on the checkout API", "metrics": ["40%"],
         "skills": ["Python", "Postgres"]},
        {"kind": "accomplishment", "title": "AWS migration", "text": "Migrated 30 services",
         "metrics": ["30 services"]},
    ]}})
    res = import_resume_text(RESUME, llm, conn=conn, embedder=HashEmbedder())
    assert len(res.items) == 1 and len(res.rejected) == 1
    assert list_items(conn, approved=True) == []
    assert len(list_items(conn, approved=False)) == 1


def test_select_items_ranks_and_skips_unapproved(conn):
    emb = HashEmbedder()
    a = make_item("a", "Built Python services on AWS", skills=["Python", "AWS"])
    b = make_item("b", "Organized the office party", skills=["Planning"])
    c = make_item("c", "Python AWS Postgres latency work", skills=["Python"])
    for it in (a, b, c):
        save_item(conn, it)
    set_approved(conn, "a")
    set_approved(conn, "b")
    out = select_items("Python services on AWS", 5, emb, conn=conn)
    assert [s.item.id for s in out] == ["a", "b"]
    assert out[0].score > out[1].score
    assert out[0].matched_skills == ["Python", "AWS"]
    assert len(select_items("Python", 1, emb, conn=conn)) == 1


def test_tailor_notes_bad_citations(job):
    items = [make_item("a", "x")]
    llm = FakeLLM(structured={"TailoredBullets": {"bullets": [
        {"text": "ok", "source_item_ids": ["a"]},
        {"text": "bad", "source_item_ids": ["zzz"]},
        {"text": "none", "source_item_ids": []},
    ]}})
    r = tailor_resume(job, items, llm)
    assert not r.verified and len(r.verification_notes) == 2


def test_verify_with_store_requires_approval(conn):
    save_item(conn, make_item("a", "Cut latency 40%", metrics=["40%"]))
    r = ResumeVersion(id="r", job_id="j", created_at=datetime.now(UTC),
                      bullets=[ResumeBullet(text="Cut latency 40%", source_item_ids=["a"])])
    ok = FakeLLM(structured={"JudgeResult": {"judgements": [{"bullet_index": 0, "supported": True}]}})
    assert not verify_with_store(r, ok, conn=conn).verified
    set_approved(conn, "a")
    assert verify_with_store(r, ok, conn=conn).verified


def test_judge_missing_result_fails():
    it = make_item("a", "Did work")
    r = ResumeVersion(id="r", job_id="j", created_at=datetime.now(UTC),
                      bullets=[ResumeBullet(text="Did work", source_item_ids=["a"])])
    from careerops.exp import verify
    out = verify(r, {"a": it}, FakeLLM(structured={"JudgeResult": {"judgements": []}}))
    assert not out.verified and "judge returned no result" in out.verification_notes[0]


def test_resume_roundtrip(conn):
    r = ResumeVersion(id="r", job_id="j", created_at=datetime.now(UTC), bullets=[])
    save_resume(conn, r)
    assert get_resume(conn, "r") == r


def test_number_normalization():
    assert extract_numbers("$1.2M and 1,200,000") == ["$1200000", "1200000"]
    assert "40%" in extract_numbers("up 40 percent")
    assert "id:p95" in extract_numbers("p95 latency")
    assert extract_numbers("Python3") == ["id:python3"]


def test_coverage_gaps_are_honest(job):
    r = ResumeVersion(id="r", job_id="j", created_at=datetime.now(UTC), bullets=[
        ResumeBullet(text="Built Python services", source_item_ids=["a"])])
    llm = FakeLLM(structured={"RequirementMatches": {"requirements": [
        {"requirement": "Python", "bullet_indexes": [0], "evidence": "Python services"},
        {"requirement": "Kubernetes", "bullet_indexes": [], "evidence": None},
        {"requirement": "AWS", "bullet_indexes": [0], "evidence": "AWS Lambda"},  # not in bullet
        {"requirement": "Go", "bullet_indexes": [7], "evidence": "Go"},  # bad index
    ]}})
    rep = coverage_report(job, r, llm)
    assert rep.gaps == ["Kubernetes", "AWS", "Go"]
    assert rep.coverage_ratio == 0.25
    assert rep.requirements[0].bullet_indexes == [0]
