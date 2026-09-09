"""Recommendations page — stub.

Will hold three tabs:
- Similar Projects  — `SimilarProjectsRecommender`
- Landmark Search   — `src.recommender.landmark_search`
- Preference Match  — `ListingRecommender`
"""

import streamlit as st


def render() -> None:
    st.title("🔍 Recommendations")
    tab_similar, tab_landmark, tab_preference = st.tabs(
        ["Similar Projects", "Landmark Search", "Preference Match"]
    )
    for tab in (tab_similar, tab_landmark, tab_preference):
        with tab:
            st.info("Not built yet — coming after the skeleton is confirmed.")
