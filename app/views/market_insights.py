"""Market Insights page.

Plain-language aggregate view of the Gurgaon market. No model output here — every
chart is drawn from our own processed data:

- the amenities word cloud is rebuilt from ``appartments.csv``'s
  ``TopFacilities`` (not the reference project's ``feature_text.pkl``),
- distributions use ``plotly`` throughout — the reference's ``sns.distplot`` is
  removed, not ported.

The sector price map lives on its own page (``sector_map``), not here.
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st
from wordcloud import WordCloud

from lib.loaders import load_amenities_text, load_listings

_PLASMA = "Plasma"


def render() -> None:
    st.title("\U0001F4C8 Market Insights")
    st.caption(
        "An overview of the Gurgaon market from the listings we have. "
        "Nothing on this page is a price prediction."
    )

    listings = load_listings()

    # ---- What are homes typically like? ------------------------------- #
    st.subheader("What are homes typically like?")
    c1, c2, c3 = st.columns(3)
    c1.metric("Typical price", f"₹{listings['price'].median():.2f} Cr")
    c2.metric("Typical size", f"{listings['built_up_area'].median():,.0f} sq ft")
    c3.metric("Most common", f"{int(listings['bedRoom'].mode().iloc[0])} BHK")

    bedrooms = listings["bedRoom"].clip(upper=6).astype(int)
    labels = {1: "1 BHK", 2: "2 BHK", 3: "3 BHK", 4: "4 BHK", 5: "5 BHK", 6: "6+ BHK"}
    bhk_share = (
        bedrooms.map(labels).value_counts(normalize=True).mul(100).round(1)
        .reindex(list(labels.values()))
        .rename_axis("Bedrooms")
        .reset_index(name="% of listings")
    )
    fig_bhk = px.bar(
        bhk_share,
        x="Bedrooms",
        y="% of listings",
        color_discrete_sequence=[px.colors.sequential.Plasma[4]],
    )
    fig_bhk.update_xaxes(type="category")
    st.plotly_chart(fig_bhk, width="stretch")

    type_col, furnish_col = st.columns(2)
    with type_col:
        share = (listings["property_type"].value_counts(normalize=True) * 100).round(0)
        st.write(
            f"**Flats vs. independent houses:** {share.get('flat', 0):.0f}% flats, "
            f"{share.get('house', 0):.0f}% houses."
        )
    with furnish_col:
        labels = {0.0: "unfurnished", 1.0: "semi-furnished", 2.0: "furnished"}
        fshare = (listings["furnishing_type"].map(labels).value_counts(normalize=True) * 100).round(0)
        st.write(
            "**Furnishing:** "
            + ", ".join(f"{v:.0f}% {k}" for k, v in fshare.items())
            + "."
        )

    # ---- Price by home size ----------------------------------------- #
    st.subheader("Price by home size")
    size_df = listings[listings["built_up_area"].between(200, 6000)]
    fig_size = px.scatter(
        size_df,
        x="built_up_area",
        y="price",
        color="bedRoom",
        color_continuous_scale=_PLASMA,
        opacity=0.45,
        labels={"built_up_area": "Built-up area (sq ft)", "price": "Price (₹ Cr)", "bedRoom": "BHK"},
    )
    st.plotly_chart(fig_size, width="stretch")
    hidden = len(listings) - len(size_df)
    if hidden:
        st.caption(f"{hidden} listings outside 200–6,000 sq ft are hidden so the bulk is readable.")

    # ---- Price ranges by bedroom count ---------------------------- #
    st.subheader("Price ranges by bedroom count")
    box_df = listings[listings["bedRoom"].between(1, 5)].copy()
    box_df["Bedrooms"] = box_df["bedRoom"].astype(int).astype(str) + " BHK"
    fig_box = px.box(
        box_df.sort_values("bedRoom"),
        x="Bedrooms",
        y="price",
        color="Bedrooms",
        color_discrete_sequence=px.colors.sequential.Plasma,
        labels={"price": "Price (₹ Cr)"},
    )
    fig_box.update_layout(showlegend=False)
    st.plotly_chart(fig_box, width="stretch")
    st.caption("1–5 BHK shown; the handful of 6+ BHK listings are excluded (too few to be a range).")

    # ---- What amenities are common? ------------------------------ #
    st.subheader("What amenities are common?")
    cloud = WordCloud(
        width=1200,
        height=450,
        background_color="white",
        colormap="plasma",
        collocations=False,
    ).generate(load_amenities_text())
    st.image(cloud.to_array(), width="stretch")
    st.caption("Amenities advertised across the Gurgaon apartment projects in the dataset.")
