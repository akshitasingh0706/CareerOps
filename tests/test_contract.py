import numpy as np

from careerops.db import connect
from careerops.domain import ResumeBullet
from careerops.llm import FakeLLM, HashEmbedder


def test_hash_embedder_is_normalized_and_deterministic():
    e = HashEmbedder()
    a = e.embed(["machine learning engineer", "machine learning engineer"])
    assert np.allclose(a[0], a[1])
    assert abs(np.linalg.norm(a[0]) - 1.0) < 1e-5


def test_fake_llm_structured():
    llm = FakeLLM(structured={"ResumeBullet": {"text": "x", "source_item_ids": ["a"]}})
    assert llm.structured(ResumeBullet, "s", "p").source_item_ids == ["a"]


def test_connect_memory():
    conn = connect(":memory:")
    conn.execute("create table t (x int)")
