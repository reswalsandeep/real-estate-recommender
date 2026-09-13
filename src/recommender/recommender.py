"""``SimilarProjectsRecommender`` -- "which developments are most like this one".

Item-to-item similarity over the 246 Gurgaon apartment *projects* in
``data/raw/appartments.csv``. This is deliberately **not** a preference /
budget / bedroom recommender over individual listings -- that is a separate
component, ``src/recommender/listing_recommender.py``.

    rec = SimilarProjectsRecommender.from_csv()
    rec.recommend("DLF The Arbour", k=5)
"""

from __future__ import annotations

import difflib
from pathlib import Path

import numpy as np
import pandas as pd

from .similarity import (
    DEFAULT_APPARTMENTS_CSV,
    DEFAULT_WEIGHTS,
    blend,
    build_components,
    load_appartments,
    minmax_offdiag,
)


class SimilarProjectsRecommender:
    """Rank apartment projects by blended similarity to a query project.

    Parameters
    ----------
    df
        Output of :func:`src.recommender.similarity.load_appartments` (must
        have a ``PropertyName`` column; the row order defines the index).
    weights
        Axis weights for the blend; defaults to
        :data:`src.recommender.similarity.DEFAULT_WEIGHTS`
        (``structural 0.5 / location 0.3 / facilities 0.2``). Renormalised to
        sum to 1.
    """

    def __init__(self, df: pd.DataFrame, weights: dict[str, float] | None = None):
        self.df = df.reset_index(drop=True)
        self.names: list[str] = self.df["PropertyName"].tolist()
        self._index = {name: i for i, name in enumerate(self.names)}
        self.weights = dict(DEFAULT_WEIGHTS if weights is None else weights)

        self.components_raw = build_components(self.df)
        # per-axis, min-max-normalised -- kept so recommend() can show *why*
        self.components_norm = {k: minmax_offdiag(v) for k, v in self.components_raw.items()}
        self.blended = blend(self.components_raw, self.weights)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_csv(
        cls,
        path: str | Path = DEFAULT_APPARTMENTS_CSV,
        weights: dict[str, float] | None = None,
    ) -> "SimilarProjectsRecommender":
        return cls(load_appartments(path), weights=weights)

    # ------------------------------------------------------------------ #
    def _resolve(self, project_name: str) -> int:
        """Exact-match a project name to its row index, or raise with hints."""
        if project_name in self._index:
            return self._index[project_name]
        hints = difflib.get_close_matches(project_name, self.names, n=5, cutoff=0.4)
        suffix = f" Did you mean: {hints}?" if hints else ""
        raise KeyError(f"project not found: {project_name!r}.{suffix}")

    def recommend(self, project_name: str, k: int = 5) -> pd.DataFrame:
        """Top-``k`` most similar projects to ``project_name``.

        Columns: ``rank``, ``PropertyName``, ``score`` (blended, [0, 1]), and
        one column per axis (``structural`` / ``location`` / ``facilities``)
        giving that pair's min-max-normalised similarity on the axis -- so the
        blended score is explainable.
        """
        idx = self._resolve(project_name)

        scores = self.blended[idx].copy()
        scores[idx] = -np.inf  # never recommend the query itself
        top = np.argsort(scores)[::-1][:k]

        return pd.DataFrame(
            {
                "rank": np.arange(1, len(top) + 1),
                "PropertyName": [self.names[j] for j in top],
                "score": scores[top],
                **{
                    axis: [self.components_norm[axis][idx, j] for j in top]
                    for axis in self.components_raw
                },
            }
        )

    def pair_components(self, project_a: str, project_b: str) -> dict[str, float]:
        """The per-axis normalised similarities (and the blend) for one pair."""
        i, j = self._resolve(project_a), self._resolve(project_b)
        out = {axis: float(self.components_norm[axis][i, j]) for axis in self.components_raw}
        out["blended"] = float(self.blended[i, j])
        return out
