"""Price Prediction page — a plain calculator for a Gurgaon home's market price.

Buyer-facing form → one estimate from ``models/price_pipeline.pkl`` (loaded once
via ``@st.cache_resource``) → one plain accuracy sentence, its number read live
from ``models/model_metrics.json``. No SHAP, no importances, no charts.

``luxury_category`` is one of the model's 12 features but is not a form field:
it barely moves the price (mean |SHAP| ~0.006, last of 12 in the explainability
report) and a buyer can't meaningfully answer it in any framing, so it is fixed
to ``"Medium"``.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from lib.loaders import get_price_model, load_json, load_sector_options
from src.models.predict import predict

_PROPERTY_TYPE = {"Flat": "flat", "Independent house": "house"}
_FURNISHING = {"Unfurnished": 0.0, "Semi-furnished": 1.0, "Furnished": 2.0}
_FLOOR = {"Low": "Low Floor", "Middle": "Mid Floor", "High": "High Floor"}
_AGE = {
    "Under construction": "Under Construction",
    "New": "New Property",
    "Relatively new": "Relatively New",
    "Moderately old": "Moderately Old",
    "Old": "Old Property",
}
_BALCONY = ["0", "1", "2", "3", "3+"]


def render() -> None:
    st.title("\U0001F4B0 Price Prediction")
    st.caption("Estimate the market price of a Gurgaon home from its main details.")

    sectors = load_sector_options()
    default_sector = sectors.index("sector 57") if "sector 57" in sectors else 0

    with st.form("price_form"):
        c1, c2 = st.columns(2)
        property_type = c1.radio("Property type", list(_PROPERTY_TYPE), horizontal=True)
        sector = c2.selectbox("Location", sectors, index=default_sector, format_func=str.title)

        c3, c4, c5 = st.columns(3)
        bedrooms = c3.number_input("Bedrooms", min_value=1, max_value=8, value=3, step=1)
        bathrooms = c4.number_input("Bathrooms", min_value=1, max_value=8, value=2, step=1)
        area = c5.number_input("Built-up area (sq ft)", min_value=200, max_value=10_000, value=1_600, step=50)

        c6, c7, c8 = st.columns(3)
        balcony = c6.selectbox("Balconies", _BALCONY, index=2)
        furnishing = c7.selectbox("Furnishing", list(_FURNISHING))
        age = c8.selectbox("Age / possession", list(_AGE), index=1)

        c9, c10, c11 = st.columns(3)
        floor = c9.selectbox("Floor level", list(_FLOOR), index=1)
        servant_room = c10.checkbox("Servant room")
        store_room = c11.checkbox("Store room")

        submitted = st.form_submit_button("Estimate price", type="primary")

    if not submitted:
        return

    features = pd.DataFrame([{
        "bedRoom": float(bedrooms),
        "bathroom": float(bathrooms),
        "built_up_area": float(area),
        "servant room": 1.0 if servant_room else 0.0,
        "store room": 1.0 if store_room else 0.0,
        "sector": sector,
        "balcony": balcony,
        "agePossession": _AGE[age],
        "furnishing_type": _FURNISHING[furnishing],
        "luxury_category": "Medium",
        "floor_category": _FLOOR[floor],
        "property_type": _PROPERTY_TYPE[property_type],
    }])

    price_cr = float(predict(get_price_model(), features)[0])

    st.metric("Estimated price", f"₹{price_cr:.2f} Cr")

    metrics = load_json("models/model_metrics.json")
    mae_lakh = round(metrics["exported_model"]["test_metrics"]["mae_crore"] * 100)
    st.caption(f"On average, these estimates are off by about ₹{mae_lakh} lakh.")
    if price_cr >= 3:
        band_mae = metrics["error_by_price_band"]["bands"][-1]["mae_crore"]
        st.caption(
            f"Above ₹5 crore the estimate is rougher — off by around ₹{band_mae:.1f} crore on average."
        )
