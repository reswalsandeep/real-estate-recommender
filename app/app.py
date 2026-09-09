"""Gurgaon real-estate app — entry point and page routing.

    streamlit run app/app.py

v1 pages: Price Prediction, Recommendations (tabs: Similar Projects / Landmark
Search / Preference Match), Sector Map, Market Insights. Sector Map and Market
Insights are implemented; Price Prediction and Recommendations are stubs.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Standard Streamlit layout: `app/` is a script root, not a package. Put it on
# the path so `views` / `lib` import, and the repo root so `src.*` imports.
# (`from app.views import …` breaks under `streamlit run` because Streamlit
# registers the entry script in sys.modules as `app`, shadowing the folder.)
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
for _p in (str(REPO_ROOT), str(APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

from views import market_insights, price_prediction, recommendations, sector_map

st.set_page_config(
    page_title="Gurgaon Real Estate",
    page_icon="🏙️",
    layout="wide",
)

# Explicit ``url_path`` per page: ``st.navigation`` otherwise infers it from the
# callable name, and every view's entry point is ``render`` — which collides.
PAGES = [
    st.Page(price_prediction.render, title="Price Prediction", icon="💰", url_path="price-prediction", default=True),
    st.Page(recommendations.render, title="Recommendations", icon="🔍", url_path="recommendations"),
    st.Page(sector_map.render, title="Sector Map", icon="🗺️", url_path="sector-map"),
    st.Page(market_insights.render, title="Market Insights", icon="📈", url_path="market-insights"),
]

st.navigation(PAGES).run()
