from datetime import UTC, datetime

import pytest

from careerops.db import connect
from careerops.domain import ExperienceItem, ExperienceKind, Job, Source


def make_item(id, text, metrics=(), skills=(), title="Engineer", kind=ExperienceKind.ACCOMPLISHMENT):
    return ExperienceItem(id=id, kind=kind, title=title, employer="Acme", text=text,
                          metrics=list(metrics), skills=list(skills))


@pytest.fixture
def conn():
    c = connect(":memory:")
    yield c
    c.close()


@pytest.fixture
def job():
    return Job(id="j1", source=Source.MANUAL, source_id="1", url="http://x", company_id="c",
               company_name="Co", title="Backend Engineer",
               description="Build Python services on AWS. Reduce latency. Postgres experience.",
               fetched_at=datetime.now(UTC))
