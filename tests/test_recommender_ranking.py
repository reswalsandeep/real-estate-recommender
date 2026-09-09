"""Ranking logic in SimilarProjectsRecommender.recommend.

`recommend()` only reads self.names / self._index / self.blended /
self.components_raw / self.components_norm, so a stub instance with a small
hand-built blended matrix isolates the "exclude self + top-k descending" logic
from the (expensive, data-dependent) similarity-matrix construction.
"""
import numpy as np
import pytest

from src.recommender.recommender import SimilarProjectsRecommender


@pytest.fixture
def rec():
    r = object.__new__(SimilarProjectsRecommender)
    r.names = ["A", "B", "C"]
    r._index = {"A": 0, "B": 1, "C": 2}
    #        A    B    C
    r.blended = np.array([
        [1.0, 0.9, 0.2],   # A: closest to B, then C
        [0.9, 1.0, 0.5],   # B
        [0.2, 0.5, 1.0],   # C: closest to B, then A
    ])
    r.components_raw = {}   # -> the per-axis columns in recommend() expand to nothing
    r.components_norm = {}
    return r


def test_query_is_excluded_from_a_normal_top_k(rec):
    # the realistic case: k is smaller than the candidate pool.
    out = rec.recommend("A", k=2)
    assert "A" not in list(out["PropertyName"])


def test_self_exclusion_is_deprioritise_not_filter(rec):
    # recommend() sets the query's own score to -inf and takes the top k; it
    # does not drop the row. So if k reaches past every other candidate, the
    # query reappears last (with score -inf), not omitted. Not a bug for real
    # use (k << n_projects) — pinned so the mechanism is explicit.
    out = rec.recommend("A", k=3)
    assert list(out["PropertyName"]) == ["B", "C", "A"]
    assert out["score"].iloc[-1] == -np.inf


def test_top_k_ordering_descending(rec):
    out = rec.recommend("A", k=2)
    assert list(out["PropertyName"]) == ["B", "C"]
    assert list(out["score"]) == sorted(out["score"], reverse=True)
    assert list(out["rank"]) == [1, 2]


def test_k_limits_result_length(rec):
    assert list(rec.recommend("A", k=1)["PropertyName"]) == ["B"]


def test_ordering_from_a_different_query_row(rec):
    # C's row is [0.2, 0.5, 1.0] -> drop self -> B (0.5) ranks above A (0.2)
    assert list(rec.recommend("C", k=2)["PropertyName"]) == ["B", "A"]


def test_unknown_project_raises(rec):
    with pytest.raises(KeyError):
        rec.recommend("does-not-exist")
