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
    """Read a text/Markdown artifact (e.g. ``reports/model/model_selection_log.md``)."""
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
    """``ListingRecommender`` over the imputed listings table."""
    from src.recommender.listing_recommender import ListingRecommender

    return ListingRecommender.from_csv()
