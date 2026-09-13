"""Recommendations page — three buyer tasks, one per recommender tool.

Tabs:
- "Homes like this one"              → ``SimilarProjectsRecommender``
- "Near a place I care about"        → ``src.recommender.landmark_search``
- "Find homes matching what I want"  → ``ListingRecommender``

Every tab is a form: nothing computes until the user submits. No algorithm
names, similarity scores or raw column names reach the UI. Recommender objects
load once via ``@st.cache_resource`` in ``lib.loaders``.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from lib.loaders import (
    get_listing_recommender,
    get_similar_projects_recommender,
    load_landmark_matrix,
    load_project_meta,
    load_sector_options,
)

_ANY = "Any"
_NO_PREF = "No preference"

_TYPE_LABEL = {"flat": "Flat", "house": "House"}
_FURNISHING_LABEL = {0.0: "Unfurnished", 1.0: "Semi-furnished", 2.0: "Furnished"}

_AGE_DISPLAY_TO_MODEL = {
    "Under construction": "Under Construction",
    "New": "New Property",
    "Relatively new": "Relatively New",
    "Moderately old": "Moderately Old",
    "Old": "Old Property",
}
_AGE_MODEL_TO_DISPLAY = {v: k for k, v in _AGE_DISPLAY_TO_MODEL.items()}
_FURNISHING_DISPLAY_TO_MODEL = {
    "Unfurnished": "unfurnished",
    "Semi-furnished": "semifurnished",
    "Furnished": "furnished",
}


def render() -> None:
    st.title("Recommendations")
    tab_similar, tab_landmark, tab_pref = st.tabs(
        [
            "Homes like this one",
            "Near a place I care about",
            "Find homes matching what I want",
        ]
    )
    with tab_similar:
        _similar_projects_tab()
    with tab_landmark:
        _landmark_tab()
    with tab_pref:
        _preference_tab()


# --------------------------------------------------------------------------- #
def _meta_row(meta: pd.DataFrame, name: str) -> tuple[str, str]:
    """(description, price_range) for a project, each falling back to an em dash."""
    if name not in meta.index:
        return "—", "—"
    desc = meta.at[name, "description"]
    price = meta.at[name, "price_range"]
    return (
        desc if isinstance(desc, str) and desc else "—",
        price if isinstance(price, str) and price else "—",
    )


# --------------------------------------------------------------------------- #
def _similar_projects_tab() -> None:
    st.subheader("Homes like this one")
    st.caption(
        "Pick a project to see the developments most like it — in size, layout, "
        "price band, location and amenities."
    )

    meta = load_project_meta()
    project_names = sorted(meta.index)

    with st.form("similar_form"):
        project = st.selectbox("Project", project_names)
        count = st.slider("How many to show", 3, 10, 5)
        submitted = st.form_submit_button("Show similar projects", type="primary")

    if not submitted:
        return

    names = (
        get_similar_projects_recommender()
        .recommend(project, k=count)["PropertyName"]
        .tolist()
    )

    rows = []
    for name in names:
        desc, price = _meta_row(meta, name)
        rows.append({"Project": name, "What it offers": desc, "Price range": price})

    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


# --------------------------------------------------------------------------- #
def _landmark_tab() -> None:
    st.subheader("Near a place I care about")
    st.caption("Pick a landmark and a distance to see which projects are within reach.")

    from src.recommender.landmark_search import list_landmarks, search_by_landmark

    matrix = load_landmark_matrix()
    landmarks = list_landmarks(matrix, min_projects=5)
    meta = load_project_meta()

    with st.form("landmark_form"):
        landmark = st.selectbox("Landmark", landmarks)
        radius_km = st.slider("Within how many km", 0.5, 15.0, 3.0, step=0.5)
        submitted = st.form_submit_button("Find nearby projects", type="primary")

    if not submitted:
        return

    results = search_by_landmark(landmark, radius_km, matrix=matrix)

    if results.empty:
        st.info(
            f"No projects mention {landmark} within {radius_km:g} km. "
            "Try a larger distance."
        )
        return

    rows = []
    for r in results.itertuples():
        desc, price = _meta_row(meta, r.project)
        rows.append(
            {
                "Project": r.project,
                "What it offers": desc,
                "Price range": price,
                "Distance": f"{r.distance_km:g} km away",
            }
        )

    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "These are projects whose own listing mentions this landmark within the "
        "distance you picked — not every project physically nearby. One that didn't "
        "list this landmark won't show up here even if it's close."
    )


# --------------------------------------------------------------------------- #
def _preference_tab() -> None:
    st.subheader("Find homes matching what I want")
    st.caption(
        "Set your must-haves and your preferences. Must-haves rule listings out; "
        "preferences only affect the order."
    )

    sectors = load_sector_options()

    with st.form("preference_form"):
        st.markdown("**Must-haves**")
        c1, c2, c3, c4 = st.columns(4)
        budget = c1.number_input(
            "Max budget (₹ Cr)", min_value=0.25, max_value=40.0, value=2.5, step=0.25
        )
        min_beds = c2.selectbox("Min. bedrooms", [_ANY, 1, 2, 3, 4, 5, 6], index=3)
        sector = c3.selectbox(
            "Sector",
            [_ANY, *sectors],
            format_func=lambda s: s if s == _ANY else s.title(),
        )
        ptype = c4.selectbox("Property type", [_ANY, "Flat", "Independent house"])

        st.markdown("**Preferences** — used to rank, not to exclude")
        c5, c6, c7 = st.columns(3)
        area = c5.number_input(
            "Preferred size (sq ft)",
            min_value=0,
            max_value=10_000,
            value=1_500,
            step=50,
            help="Set to 0 for no size preference.",
        )
        furnishing = c6.selectbox(
            "Furnishing", [_NO_PREF, "Unfurnished", "Semi-furnished", "Furnished"]
        )
        age = c7.selectbox(
            "Age / possession",
            [_NO_PREF, "Under construction", "New", "Relatively new", "Moderately old", "Old"],
        )

        submitted = st.form_submit_button("Find matches", type="primary")

    if not submitted:
        return

    from src.recommender.listing_recommender import Preferences

    prefs = Preferences(
        budget_max_cr=float(budget),
        min_bedrooms=None if min_beds == _ANY else int(min_beds),
        sector=None if sector == _ANY else sector,
        property_type=None if ptype == _ANY else ("flat" if ptype == "Flat" else "house"),
        area_sqft=None if area == 0 else float(area),
        furnishing=None if furnishing == _NO_PREF else _FURNISHING_DISPLAY_TO_MODEL[furnishing],
        age_possession=None if age == _NO_PREF else _AGE_DISPLAY_TO_MODEL[age],
    )

    results = get_listing_recommender().recommend(prefs, k=10, enrich=True)

    if results.empty:
        st.info(
            "No listings match those must-haves. Try raising the budget, lowering "
            "the minimum bedrooms, or clearing the sector."
        )
        return

    display = pd.DataFrame(
        {
            "Project": results["society"].replace("independent", "—"),
            "Sector": results["sector"].astype(str).str.title(),
            "Type": results["property_type"].map(_TYPE_LABEL).fillna(results["property_type"]),
            "Price (₹ Cr)": results["price"].round(2),
            "Beds": results["bedRoom"].astype("Int64"),
            "Baths": results["bathroom"].astype("Int64"),
            "Size (sq ft)": results["built_up_area"].round().astype("Int64"),
            "Age": results["agePossession"].map(_AGE_MODEL_TO_DISPLAY).fillna(results["agePossession"]),
            "Furnishing": results["furnishing_type"].map(_FURNISHING_LABEL).fillna("—"),
            "Priced vs. our estimate": results["price_vs_model_pct"].map(_pct_str),
            "Similar projects": results["similar_projects"].fillna("—"),
        }
    )

    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config={
            "Priced vs. our estimate": st.column_config.TextColumn(
                "Priced vs. our estimate",
                help=(
                    "How the asking price compares to our price model's estimate for this "
                    "home. Negative = cheaper than the model predicts. Treat a big negative "
                    "with care: the model tends to over-estimate cheaper homes, so part of a "
                    "large discount can be model error, not a real bargain."
                ),
            )
        },
    )
    st.caption(
        "**Priced vs. our estimate** — negative means the asking price is below our price "
        "model's estimate, but the model over-estimates cheaper homes, so a big negative is "
        "partly model error, not purely a bargain."
    )
    st.caption(
        "Some near-identical listings can show up more than once: the source data lost its "
        "listing IDs, so exact duplicates can't be fully removed."
    )


def _pct_str(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:+.0f}%"
