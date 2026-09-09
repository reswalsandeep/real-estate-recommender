"""Sector Map page — mean price per sq ft across Gurgaon's sectors.

Data: ``data/raw/latlong.csv`` joined onto the processed listings via
``src.features.geo`` (the tested join with the ``sector 70a`` suffix fallback).

The map is an interactive OpenStreetMap render built here with
``plotly.express.scatter_mapbox``. **This is the shipped version**, confirmed
working in a live browser check — scroll-zoom, drag-pan and hover all verified.
Constraints, learned the hard way:

- ``scatter_mapbox`` only exists in Plotly < 7 (``scatter_map`` replaced it and
  paints blank inside Streamlit here); ``requirements.txt`` pins ``plotly<6``.
- ``st.plotly_chart`` needs ``config={"scrollZoom": True}`` or both scroll-zoom
  *and* drag-pan are dead — Streamlit's default Plotly config disables them.
- The figure ``height`` is honoured; ``width`` is overridden to the container.

``src.features.geo.build_sector_map_static`` (matplotlib) is kept as a documented
fallback and is what ``notebooks/05_eda.ipynb`` embeds, but it is no longer
rendered on this page.
"""

from __future__ import annotations

import streamlit as st

from lib.loaders import load_sector_map_frame


def render() -> None:
    st.title("Sector Map")
    st.caption(
        "Average price per square foot across Gurgaon's sectors. Each dot sits at the "
        "centre of a sector; bigger dots are sectors with larger homes on average. "
        "Sectors are shown as dots rather than shaded areas because there are no "
        "official sector boundaries to draw."
    )

    group_df, report = load_sector_map_frame()

    # config={"scrollZoom": True} is required — without it Streamlit's default
    # Plotly config leaves both scroll-zoom and drag-pan disabled.
    st.plotly_chart(_sector_map(group_df), config={"scrollZoom": True})

    dropped = report.get("dropped_rows_by_sector", {})
    if dropped:
        st.caption(
            f"{sum(dropped.values())} listings in "
            + ", ".join(f"**{s}** ({n})" for s, n in dropped.items())
            + " aren't shown — those areas don't have location data."
        )

    price_columns = {
        "sector": st.column_config.TextColumn("Sector"),
        "price_per_sqft": st.column_config.NumberColumn("₹ / sq ft", format="%d"),
        "price": st.column_config.NumberColumn("Typical price", format="₹%.2f Cr"),
        "n_properties": st.column_config.NumberColumn("Listings", format="%d"),
    }
    keep = ["sector", "price_per_sqft", "price", "n_properties"]
    highest = group_df.sort_values("price_per_sqft", ascending=False).head(8).reset_index()[keep]
    lowest = group_df.sort_values("price_per_sqft").head(8).reset_index()[keep]
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Priciest sectors**")
        st.dataframe(highest, width="stretch", hide_index=True, column_config=price_columns)
    with cols[1]:
        st.markdown("**Most affordable sectors**")
        st.dataframe(lowest, width="stretch", hide_index=True, column_config=price_columns)


def _sector_map(group_df):
    """Interactive OpenStreetMap dot map. See the module docstring for why this
    is ``scatter_mapbox`` (Plotly < 7) and not ``scatter_map``."""
    import plotly.express as px

    g = group_df.reset_index()
    fig = px.scatter_mapbox(
        g,
        lat="latitude",
        lon="longitude",
        color="price_per_sqft",
        size="built_up_area",
        size_max=28,
        hover_name="sector",
        hover_data={"price": ":.2f", "n_properties": True, "latitude": False, "longitude": False},
        color_continuous_scale="Plasma",
        zoom=10.2,   # frame Gurgaon; user can scroll-zoom out
        center={"lat": 28.44, "lon": 77.03},
        width=1200,   # Streamlit overrides this to the container width; height is kept
        height=700,
        mapbox_style="open-street-map",
        title="Mean price per sq ft by Gurgaon sector",
        labels={
            "price_per_sqft": "₹ / sq ft",
            "price": "Typical price (₹ Cr)",
            "n_properties": "Listings",
            "built_up_area": "Avg. size (sq ft)",
        },
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    return fig
