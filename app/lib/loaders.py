"""Cached loaders for the Streamlit app.

Rule (from the project spec): models must not retrain or reload on every
interaction.

- ``@st.cache_data``   -- JSON / text reports and any DataFrame. Cached by value.
- ``@st.cache_resource`` -- unpicklable singletons: the fitted price pipeline and
  the recommender objects. Loaded once per server process.

Paths are given relative to the repo root.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]


@st.cache_data
def load_json(rel_path: str) -> dict:
    """Read a JSON artifact (e.g. ``models/model_metrics.json``)."""
    return json.loads((REPO_ROOT / rel_path).read_text(encoding="utf-8"))


@st.cache_data
def load_text(rel_path: str) -> str:
    """Read a text artifact by repo-relative path."""
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


@st.cache_data
def load_listings() -> pd.DataFrame:
    """The imputed listing-level table (``gurgaon_properties_missing_value_imputation.csv``)."""
    return pd.read_csv(REPO_ROOT / "data/processed/gurgaon_properties_missing_value_imputation.csv")


@st.cache_data
def load_sector_options() -> list[str]:
    """Sorted unique ``sector`` values the price model was trained on (all 104,
    including the four road/area names like ``sohna road``). Raw lowercase — the
    Price Prediction form title-cases them for display only."""
    sectors = pd.read_csv(
        REPO_ROOT / "data/processed/gurgaon_properties_post_feature_selection_v2.csv",
        usecols=["sector"],
    )["sector"].unique()
    return sorted(sectors)


@st.cache_data
def load_sector_map_frame() -> tuple[pd.DataFrame, dict]:
    """Per-sector aggregates + centroid coords, via ``src.features.geo``.

    Returns ``(group_df, join_report)``. The join reuses the tested suffix
    fallback in ``geo.join_sector_coordinates`` (recovers ``sector 70a``); the
    report lists the sectors with no coordinates (``sohna road``,
    ``dwarka expressway``) and how many listings that drops.
    """
    from src.features.geo import sector_map_frame

    return sector_map_frame(write_csv=False)


@st.cache_data
def load_amenities_text() -> str:
    """Space-joined ``TopFacilities`` tokens from every project in ``appartments.csv``.

    Rebuilt from source each run — not the reference project's ``feature_text.pkl``.
    """
    from src.recommender.similarity import load_appartments, parse_facilities

    tokens: list[str] = []
    for raw in load_appartments()["TopFacilities"]:
        tokens.extend(parse_facilities(raw))
    return " ".join(tokens)


def _price_range_cr(price_details_raw: str) -> str | None:
    """A plain "₹1.49–4.34 Cr" spread from a project's raw ``PriceDetails`` blob.

    Takes the overall low→high across every per-BHK ``price-range`` string,
    each parsed by ``src.recommender.similarity.price_range_to_crore`` (the
    shared per-number Lakh/Cr parser).
    """
    from src.recommender.similarity import price_range_to_crore

    try:
        details = json.loads(str(price_details_raw).replace("'", '"'))
    except (json.JSONDecodeError, TypeError):
        return None

    values: list[float] = []
    for detail in details.values():
        values.extend(price_range_to_crore(str(detail.get("price-range", ""))))

    if not values:
        return None
    lo, hi = min(values), max(values)
    return f"₹{lo:.2f} Cr" if abs(hi - lo) < 0.005 else f"₹{lo:.2f}–{hi:.2f} Cr"


@st.cache_data
def load_project_meta() -> pd.DataFrame:
    """Per-project display metadata for the Recommendations page, indexed by
    ``PropertyName``: ``description`` (the raw ``PropertySubName`` — config,
    type and location, already human-readable) and ``price_range`` (a display
    string from :func:`_price_range_cr`, or ``None``)."""
    from src.recommender.similarity import load_appartments

    df = load_appartments()[["PropertyName", "PropertySubName", "PriceDetails"]].copy()
    df["description"] = df["PropertySubName"].astype(str).str.strip()
    df["price_range"] = df["PriceDetails"].map(_price_range_cr)
    return df.set_index("PropertyName")[["description", "price_range"]]


@st.cache_data
def load_landmark_matrix() -> pd.DataFrame:
    """Projects × landmarks distance matrix (metres), from the cached CSV
    (``data/processed/landmark_distance_matrix.csv``)."""
    from src.recommender.landmark_search import load_distance_matrix

    return load_distance_matrix()


# --- resource singletons -------------------------------------------------- #
# Defined now so the caching contract is in one place; wired up when the
# Price Prediction / Recommendations pages land. Each loads once per process.

@st.cache_resource
def get_price_model():
    """The exported price pipeline (``models/price_pipeline.pkl``)."""
    from src.models.predict import load_model

    return load_model()


@st.cache_resource
def get_similar_projects_recommender():
    """``SimilarProjectsRecommender`` over ``appartments.csv`` (builds 3 matrices)."""
    from src.recommender.recommender import SimilarProjectsRecommender

    return SimilarProjectsRecommender.from_csv()


@st.cache_resource
def get_listing_recommender():
    """``ListingRecommender`` over the imputed listings table.

    Reuses the already-cached price model and project recommender for its
    enrichment columns, rather than loading its own second copies (~120 MB).
    """
    from src.recommender.listing_recommender import ListingRecommender

    return ListingRecommender.from_csv(
        price_model=get_price_model(),
        project_recommender=get_similar_projects_recommender(),
    )
