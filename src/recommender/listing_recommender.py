"""Preference-based listing recommender (``PROJECT_PLAN.md`` §10, §12).

Third of the app's recommender features. Takes a **preference vector** (no
seed-listing mode -- that is `SimilarProjectsRecommender` at the project level),
applies **hard filters**, then ranks the survivors by **weighted closeness** on
the soft axes the user actually specified. The price model and project-similarity
are attached to each result as **display columns only** -- never folded into the
ranking score.

Ranking method
--------------
For a candidate ``c`` and preference ``p``, over the set ``S`` of soft axes where
``p`` gave a value::

    d_a(c)   = |p_a - value_a(c)| / scale_a          for each a in S
    score(c) = ( Σ_{a in S} w_a * d_a(c) ) / ( Σ_{a in S} w_a )

    scale_a = IQR (built_up_area, luxury_score)  |  std (bathroom, furnishing
              code 0/1/2, agePossession ordinal 0-4), each over the FULL listings
              table -- so score means "how many typical spreads from what you
              asked for", stable across queries.
    w_a     = fixed SOFT_WEIGHTS[a], renormalised over the active axes.

Ranked ascending; ties broken by ``price`` ascending. If **no** soft axis is
given, ``match_score`` is ``NaN`` and results are ordered by ``price`` ascending.

Hard filters: ``budget_max_cr`` (price ceiling), ``min_bedrooms``, ``sector``,
``property_type`` -- each applied only when set. No candidates after filtering ->
an empty DataFrame with the normal columns.

Deliberate scope choices (v1) -- written down so they are not later mistaken for gaps
----------------------------------------------------------------------------------
* **``luxury_score`` carries a high soft weight (0.25)** even though
  ``reports/model/explainability_summary.md`` found ``luxury_category`` contributes
  almost nothing to *actual price*. Preference-weight (how much a buyer cares
  about a luxury level when choosing) and price-importance (how much it moves the
  model's price) are treated here as **separate questions** -- this is intentional,
  not an unreconciled inconsistency between the two pieces of work.
* **``sector`` is a hard filter only, never a soft-ranking axis**, despite being
  the #2 price driver in the explainability summary. A buyer either fixes a
  sector or leaves it open; "somewhat near the preferred sector" needs adjacency
  data we do not have. Deliberate v1 simplification.

Enrichment (display only)
-------------------------
* ``price_vs_model_pct`` -- ``100 * (actual_price / price_pipeline.pkl prediction
  - 1)``; negative = priced **below** the model. Uses the row-aligned
  ``post_feature_selection_v2`` table (it carries ``luxury_category`` /
  ``floor_category`` that the listings table lacks; rows are position-aligned so
  no re-binning).
* ``similar_projects`` -- for listings whose ``society`` exact-matches an
  ``appartments.csv`` project (~49 % of rows; the §10 society bridge),
  ``SimilarProjectsRecommender``'s top 3. ``None`` otherwise.

Both are computed for the top ``k`` only and cached; the price model and the
project recommender load lazily on first enriched call.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.predict import load_model, predict
from src.models.train import FEATURE_COLS

from .recommender import SimilarProjectsRecommender
from .similarity import REPO_ROOT, load_appartments

DEFAULT_LISTINGS_CSV = REPO_ROOT / "data" / "processed" / "gurgaon_properties_missing_value_imputation.csv"
DEFAULT_MODEL_FEATURES_CSV = REPO_ROOT / "data" / "processed" / "gurgaon_properties_post_feature_selection_v2.csv"

# mirrors src/models/train.ORDINAL_SPEC["agePossession"] — newest first, oldest last
_AGE_ORDER = ["Under Construction", "New Property", "Relatively New", "Moderately Old", "Old Property"]
_AGE_RANK = {label: i for i, label in enumerate(_AGE_ORDER)}

_FURNISHING_CODE = {
    "unfurnished": 0,
    "semi-furnished": 1,
    "semifurnished": 1,
    "semi furnished": 1,
    "furnished": 2,
}

# soft axis (Preferences field) -> (listings column, weight, scale statistic)
_SOFT_AXES: dict[str, tuple[str, float, str]] = {
    "area_sqft": ("built_up_area", 0.35, "iqr"),
    "luxury_score": ("luxury_score", 0.25, "iqr"),
    "bathrooms": ("bathroom", 0.15, "std"),
    "age_possession": ("agePossession", 0.15, "std"),
    "furnishing": ("furnishing_type", 0.10, "std"),
}
SOFT_WEIGHTS = {name: w for name, (_col, w, _stat) in _SOFT_AXES.items()}

_RESULT_COLUMNS = [
    "society", "sector", "property_type", "price", "price_per_sqft",
    "bedRoom", "bathroom", "built_up_area", "agePossession", "furnishing_type",
    "floorNum", "luxury_score", "match_score",
    "predicted_price_cr", "price_vs_model_pct", "similar_projects",
]


@dataclass
class Preferences:
    """A buyer's preference vector. Every field is optional.

    Hard filters (applied only when set): ``budget_max_cr``, ``min_bedrooms``,
    ``sector``, ``property_type``. Soft ranking axes (contribute only when set):
    ``area_sqft``, ``bathrooms``, ``furnishing`` (``"unfurnished"`` /
    ``"semifurnished"`` / ``"furnished"`` or ``0`` / ``1`` / ``2``),
    ``age_possession`` (one of the five ``agePossession`` labels),
    ``luxury_score``.
    """

    budget_max_cr: float | None = None
    min_bedrooms: int | None = None
    sector: str | None = None
    property_type: str | None = None

    area_sqft: float | None = None
    bathrooms: float | None = None
    furnishing: str | int | None = None
    age_possession: str | None = None
    luxury_score: float | None = None


class ListingRecommender:
    """Filter + soft-rank listings against a :class:`Preferences` vector."""

    def __init__(self, listings: pd.DataFrame, model_features: pd.DataFrame):
        if len(listings) != len(model_features):
            raise ValueError(
                f"listings ({len(listings)}) and model_features ({len(model_features)}) "
                "must be row-aligned"
            )
        self.df = listings.reset_index(drop=True)
        self._mf = model_features.reset_index(drop=True)
        self._scale = self._compute_scales()
        self._society_to_project = self._build_society_bridge()
        self._model = None
        self._recommender: SimilarProjectsRecommender | None = None
        self._sim_cache: dict[str, str] = {}

    @classmethod
    def from_csv(
        cls,
        listings_path=DEFAULT_LISTINGS_CSV,
        model_features_path=DEFAULT_MODEL_FEATURES_CSV,
    ) -> "ListingRecommender":
        return cls(pd.read_csv(listings_path), pd.read_csv(model_features_path))

    # -- setup -------------------------------------------------------------- #
    def _compute_scales(self) -> dict[str, float]:
        scales: dict[str, float] = {}
        for pref_name, (col, _w, stat) in _SOFT_AXES.items():
            series = self.df[col].map(_AGE_RANK) if col == "agePossession" else self.df[col]
            spread = (
                series.quantile(0.75) - series.quantile(0.25) if stat == "iqr" else series.std()
            )
            scales[pref_name] = float(spread) if spread else 1.0
        return scales

    def _build_society_bridge(self) -> dict[str, str]:
        by_lower = {n.lower().strip(): n for n in load_appartments()["PropertyName"]}
        matched = set(self.df["society"].astype(str).str.lower().str.strip()) & set(by_lower)
        return {key: by_lower[key] for key in matched}

    # -- preference / candidate values on a common numeric scale ----------- #
    def _pref_numeric(self, pref_name: str, value) -> float:
        if pref_name == "age_possession":
            if value not in _AGE_RANK:
                raise ValueError(f"age_possession must be one of {_AGE_ORDER}, got {value!r}")
            return float(_AGE_RANK[value])
        if pref_name == "furnishing":
            if isinstance(value, str):
                code = _FURNISHING_CODE.get(value.lower().strip())
                if code is None:
                    raise ValueError(
                        f"furnishing must be one of {sorted(set(_FURNISHING_CODE))} or 0/1/2"
                    )
                return float(code)
            return float(value)
        return float(value)

    def _candidate_numeric(self, pref_name: str, col: str, frame: pd.DataFrame) -> pd.Series:
        if pref_name == "age_possession":
            return frame[col].map(_AGE_RANK).astype(float)
        return frame[col].astype(float)

    # -- main ------------------------------------------------------------- #
    def recommend(self, prefs: Preferences, k: int = 10, enrich: bool = True) -> pd.DataFrame:
        cand = self.df
        if prefs.budget_max_cr is not None:
            cand = cand[cand["price"] <= prefs.budget_max_cr]
        if prefs.min_bedrooms is not None:
            cand = cand[cand["bedRoom"] >= prefs.min_bedrooms]
        if prefs.sector is not None:
            cand = cand[cand["sector"].astype(str).str.lower().str.strip() == prefs.sector.lower().strip()]
        if prefs.property_type is not None:
            cand = cand[cand["property_type"].astype(str).str.lower().str.strip() == prefs.property_type.lower().strip()]

        if cand.empty:
            return pd.DataFrame(columns=_RESULT_COLUMNS)

        active = {p: cfg for p, cfg in _SOFT_AXES.items() if getattr(prefs, p) is not None}
        if active:
            weight_sum = sum(w for _col, w, _stat in active.values())
            distance = pd.Series(0.0, index=cand.index)
            for pref_name, (col, weight, _stat) in active.items():
                pref_val = self._pref_numeric(pref_name, getattr(prefs, pref_name))
                cand_val = self._candidate_numeric(pref_name, col, cand)
                distance = distance + weight * (pref_val - cand_val).abs() / self._scale[pref_name]
            cand = cand.assign(match_score=distance / weight_sum)
            top = cand.nsmallest(k, "match_score").copy()
        else:
            top = cand.assign(match_score=np.nan).nsmallest(k, "price").copy()

        if enrich:
            top["predicted_price_cr"] = self._predict_prices(top.index)
            top["price_vs_model_pct"] = (100.0 * (top["price"] / top["predicted_price_cr"] - 1.0)).round(1)
            top["similar_projects"] = [self._similar_projects(s) for s in top["society"]]
        else:
            top["predicted_price_cr"] = np.nan
            top["price_vs_model_pct"] = np.nan
            top["similar_projects"] = None

        return top.reindex(columns=_RESULT_COLUMNS).reset_index(drop=True)

    # -- enrichment ----------------------------------------------------- #
    def _predict_prices(self, index) -> np.ndarray:
        if self._model is None:
            self._model = load_model()
        return predict(self._model, self._mf.loc[index, FEATURE_COLS]).round(2)

    def _similar_projects(self, society, k: int = 3) -> str | None:
        project = self._society_to_project.get(str(society).lower().strip())
        if project is None:
            return None
        if project not in self._sim_cache:
            if self._recommender is None:
                self._recommender = SimilarProjectsRecommender.from_csv()
            names = self._recommender.recommend(project, k=k)["PropertyName"].tolist()
            self._sim_cache[project] = ", ".join(names)
        return self._sim_cache[project]


if __name__ == "__main__":
    rec = ListingRecommender.from_csv()
    demo = Preferences(budget_max_cr=2.5, min_bedrooms=3, area_sqft=1600,
                       furnishing="semifurnished", age_possession="Relatively New")
    print(rec.recommend(demo, k=5).to_string(index=False))
