"""Eval for dedupe on a labeled set of pairs, using HashEmbedder.

Duplicate pairs mix sources, title variants and location variants.
Non-duplicate pairs include hard negatives: same company, similar title or same title elsewhere.

We measure the code path only. Pairs that land in the middle band would go to the LLM judge,
so we report how many there are and score the auto decisions on the rest.
"""
from datetime import UTC, datetime

from careerops.domain import Job, Source
from careerops.ingest.common import make_job_id, slugify
from careerops.ingest.dedupe import _embed_text, decide, normalize_company, score_pair
from careerops.llm import HashEmbedder

G, L, A, E, W, I = (Source.GREENHOUSE, Source.LEVER, Source.ASHBY, Source.EMAIL_ALERT,
                    Source.AGGREGATOR, Source.CAREER_PAGE)

ML = ("Build and ship machine learning models for search ranking. You will own training pipelines "
      "in Python and PyTorch, run experiments, and work with product teams. 5 years of experience.")
DE = ("Design and maintain batch and streaming data pipelines using Spark and Airflow. Own the "
      "warehouse, data quality and the analytics platform for the whole company.")
BE = ("Build backend services in Go and Postgres for our payments platform. Design APIs, review "
      "code, and improve reliability and latency of core systems.")
FE = ("Build the customer web app in React and TypeScript. Work with designers on components, "
      "accessibility and performance of the frontend.")
PM = ("Own the product roadmap for our marketplace. Talk to customers, write specs, and work with "
      "engineering and design to ship features.")
SEC = ("Lead security reviews, threat modeling and incident response for our cloud infrastructure. "
       "Run penetration tests and improve detection.")
SNIP = lambda d: d.split(".")[0] + "."


def mk(company, title, loc, desc, source):
    sid = f"{company}|{title}|{loc}|{source.value}"
    return Job(id=make_job_id(source, sid), source=source, source_id=sid, url="u",
               company_id=slugify(company), company_name=company, title=title, location=loc,
               description=desc, fetched_at=datetime.now(UTC))


# (job_a, job_b, is_duplicate)
PAIRS = [
    # Duplicates: cross-source and title variants
    (mk("Acme AI", "Senior Machine Learning Engineer", "San Francisco, CA", ML, G),
     mk("Acme AI", "Sr. ML Engineer", "San Francisco, CA", SNIP(ML), E), True),
    (mk("Acme AI", "Senior Machine Learning Engineer", "San Francisco, CA", ML, G),
     mk("Acme AI", "Senior ML Engineer", "SF", ML, A), True),
    (mk("Beta Inc", "Staff Software Engineer, Platform", "New York, NY", BE, L),
     mk("Beta", "Staff Software Engineer - Platform", "NYC", SNIP(BE), E), True),
    (mk("Gamma Labs", "Data Engineer", "Seattle, WA", DE, G),
     mk("Gamma Labs", "Data Engineer (Remote)", "Seattle, WA", DE, W), True),
    (mk("Gamma Labs", "Sr. Data Engineer", "Seattle, WA", DE, I),
     mk("Gamma Labs", "Senior Data Engineer", "Seattle", SNIP(DE), E), True),
    (mk("Delta Health", "Senior Frontend Engineer", "Remote", FE, A),
     mk("Delta Health", "Sr Front-End Engineer", "Remote - US", SNIP(FE), E), True),
    (mk("Delta Health", "Frontend Software Engineer", "Boston, MA", FE, G),
     mk("Delta Health", "Frontend Software Engineer", "Boston", FE, L), True),
    (mk("Epsilon Corp.", "Product Manager", "Austin, TX", PM, G),
     mk("Epsilon Corp", "Product Manager", "Austin, TX", SNIP(PM), E), True),
    (mk("Epsilon", "Sr. Product Manager", "Austin, TX", PM, L),
     mk("Epsilon, Inc.", "Senior Product Manager", "Austin", PM, W), True),
    (mk("Zeta Retail", "Senior Backend Engineer", "Chicago, IL", BE, G),
     mk("Zeta Retail", "Senior Backend Developer", "Chicago, IL", SNIP(BE), E), True),
    (mk("Zeta Retail", "ML Engineer", "Remote", ML, A),
     mk("Zeta Retail", "Machine Learning Engineer", "Remote", ML, L), True),
    (mk("Eta Security", "Principal Security Engineer", "Washington, DC", SEC, G),
     mk("Eta Security", "Principal Security Engineer (Hybrid)", "Washington, DC", SNIP(SEC), E), True),
    (mk("Theta Robotics", "Software Engineer II", "Los Angeles, CA", BE, G),
     mk("Theta Robotics", "Software Engineer 2", "LA", BE, E), True),
    (mk("Iota Systems", "Senior Data Scientist", "Remote", ML, L),
     mk("Iota Systems", "Sr. Data Scientist", "Remote", ML, I), True),
    (mk("Kappa Media", "Engineering Manager", "New York, NY", BE, G),
     mk("Kappa Media", "Engineering Mgr", "New York City", SNIP(BE), E), True),
    # Non-duplicates: same company, different role
    (mk("Acme AI", "Senior Machine Learning Engineer", "San Francisco, CA", ML, G),
     mk("Acme AI", "Senior Data Engineer", "San Francisco, CA", DE, G), False),
    (mk("Acme AI", "Senior Machine Learning Engineer", "San Francisco, CA", ML, G),
     mk("Acme AI", "Product Manager", "San Francisco, CA", PM, E), False),
    (mk("Gamma Labs", "Data Engineer", "Seattle, WA", DE, G),
     mk("Gamma Labs", "Security Engineer", "Seattle, WA", SEC, G), False),
    (mk("Delta Health", "Senior Frontend Engineer", "Remote", FE, A),
     mk("Delta Health", "Senior Backend Engineer", "Remote", BE, A), False),
    # Hard negatives: same title, other location
    (mk("Beta", "Software Engineer", "New York, NY", BE, L),
     mk("Beta", "Software Engineer", "London, UK", BE, L), False),
    (mk("Epsilon", "Product Manager", "Austin, TX", PM, G),
     mk("Epsilon", "Product Manager", "Berlin, Germany", PM, G), False),
    (mk("Zeta Retail", "Senior Backend Engineer", "Chicago, IL", BE, G),
     mk("Zeta Retail", "Senior Backend Engineer", "Denver, CO", BE, E), False),
    # Hard negatives: same role family, different level
    (mk("Eta Security", "Security Engineer", "Washington, DC", SEC, G),
     mk("Eta Security", "Principal Security Engineer", "Washington, DC", SEC, G), False),
    (mk("Theta Robotics", "Software Engineer", "Los Angeles, CA", BE, G),
     mk("Theta Robotics", "Staff Software Engineer", "Los Angeles, CA", BE, G), False),
    (mk("Iota Systems", "Data Scientist", "Remote", ML, L),
     mk("Iota Systems", "Senior Data Scientist", "Remote", ML, L), False),
    (mk("Kappa Media", "Engineering Manager", "New York, NY", BE, G),
     mk("Kappa Media", "Senior Software Engineer", "New York, NY", BE, G), False),
    # Different companies
    (mk("Acme AI", "Senior Machine Learning Engineer", "San Francisco, CA", ML, G),
     mk("Beta", "Senior Machine Learning Engineer", "San Francisco, CA", ML, G), False),
    (mk("Gamma Labs", "Data Engineer", "Seattle, WA", DE, G),
     mk("Gamma Robotics", "Data Engineer", "Seattle, WA", DE, G), False),
    # Same company, similar title, different team text
    (mk("Zeta Retail", "Frontend Engineer", "Chicago, IL", FE, G),
     mk("Zeta Retail", "Frontend Engineer, Design Systems", "Chicago, IL", PM, G), False),
]


def classify(a: Job, b: Job, emb: HashEmbedder) -> str:
    """Return 'merge', 'split' or 'judge' the way dedupe_jobs decides."""
    if normalize_company(a.company_name) != normalize_company(b.company_name):
        return "split"
    va, vb = emb.embed([_embed_text(a), _embed_text(b)])
    return decide(score_pair(a, b, va, vb))


def test_dedupe_eval():
    emb = HashEmbedder()
    assert len(PAIRS) >= 20
    assert sum(1 for *_, d in PAIRS if d) >= 10 and sum(1 for *_, d in PAIRS if not d) >= 10
    rows = [(classify(a, b, emb), dup) for a, b, dup in PAIRS]

    merged = [(v, d) for v, d in rows if v == "merge"]
    tp = sum(1 for v, d in rows if v == "merge" and d)
    fp = sum(1 for v, d in rows if v == "merge" and not d)
    n_dup = sum(1 for _, d in rows if d)
    judged = sum(1 for v, _ in rows if v == "judge")
    judged_dup = sum(1 for v, d in rows if v == "judge" and d)
    missed_by_split = sum(1 for v, d in rows if v == "split" and d)

    precision = tp / (tp + fp) if merged else 1.0
    recall_auto = tp / n_dup
    # If the judge answers borderline pairs correctly, recall rises to this bound.
    recall_with_judge = (tp + judged_dup) / n_dup
    print(f"\npairs={len(PAIRS)} duplicates={n_dup}")
    print(f"auto-merge precision={precision:.3f} (tp={tp}, fp={fp})")
    print(f"auto recall={recall_auto:.3f}  judge band={judged} pairs  "
          f"recall ceiling with a correct judge={recall_with_judge:.3f}")
    print(f"duplicates wrongly auto-split={missed_by_split}")

    assert precision >= 0.95          # a bad merge hides a real job, so we keep this strict
    assert missed_by_split == 0       # no duplicate may be lost without the judge seeing it
    assert recall_with_judge == 1.0
    assert judged <= 0.4 * len(PAIRS)  # the judge should see a minority of pairs
