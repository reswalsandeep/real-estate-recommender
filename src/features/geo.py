"""Sector coordinates: join ``data/raw/latlong.csv`` onto listings and aggregate
per sector for the price map (see ``notebooks/19_sector_price_map.ipynb``).

Ported from ``notebooks_original/data-visualization.ipynb`` (the sector map cells).
The core join/aggregation functions have **no plotting dependency**;
:func:`build_sector_map` imports ``plotly`` lazily.

``latlong.csv`` has 129 sectors, the modelling data 104 -- mostly a superset,
but three modelling sectors have no coordinates:

* ``sector 70a`` -- recovered by :func:`join_sector_coordinates` via a
  trailing-letter-suffix fallback (``sector 70a`` -> ``sector 70``), the same
  ``37c``/``37`` inconsistency noted in ``docs/data_dictionary.md``. Reported as a
  ``fallback`` match, distinct from an exact match.
* ``dwarka expressway``, ``sohna road`` -- road names, not sector-suffix
  variants; no fallback applies, left genuinely unmatched.

Nothing is dropped silently: :func:`join_sector_coordinates` returns a report of
what matched exactly, what matched by fallback, what was dropped, and how many
rows each dropped sector cost.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LATLONG_CSV = REPO_ROOT / "data" / "raw" / "latlong.csv"
DEFAULT_PROPERTIES_CSV = REPO_ROOT / "data" / "processed" / "gurgaon_properties_missing_value_imputation.csv"
DEFAULT_SECTOR_FRAME_CSV = REPO_ROOT / "data" / "processed" / "sector_price_map.csv"

# Gurgaon bounding box, for a sanity check on the parsed coordinates.
_LAT_RANGE = (28.0, 29.0)
_LON_RANGE = (76.5, 77.5)

_SUFFIX_RE = re.compile(r"^(sector\s+\d+)[a-z]$")

_MAP_VALUE_COLS = ("price", "price_per_sqft", "built_up_area")


def parse_latlong(path: str | Path = DEFAULT_LATLONG_CSV) -> pd.DataFrame:
    """Read ``latlong.csv`` and parse ``"28.3663° N, 76.9456° E"`` -> floats.

    Returns columns ``sector``, ``latitude``, ``longitude``. Raises if any row
    fails to parse or lands outside the Gurgaon bounding box.
    """
    df = pd.read_csv(path)
    coords = df["coordinates"].str.split(",", expand=True)
    lat = coords[0].str.split("°").str[0].str.strip().astype(float)
    lon = coords[1].str.split("°").str[0].str.strip().astype(float)

    out = pd.DataFrame(
        {"sector": df["sector"].str.strip().str.lower(), "latitude": lat, "longitude": lon}
    )
    if out[["latitude", "longitude"]].isna().any().any():
        bad = df.loc[out["latitude"].isna() | out["longitude"].isna(), "sector"].tolist()
        raise ValueError(f"latlong.csv: unparseable coordinates for {bad}")
    off = ~out["latitude"].between(*_LAT_RANGE) | ~out["longitude"].between(*_LON_RANGE)
    if off.any():
        raise ValueError(f"latlong.csv: coordinates outside Gurgaon bbox for {out.loc[off, 'sector'].tolist()}")
    return out


def join_sector_coordinates(
    df: pd.DataFrame,
    latlong: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Attach ``latitude`` / ``longitude`` to ``df`` by ``sector``.

    Exact match first; for anything left over, strip a trailing letter suffix
    (``sector 70a`` -> ``sector 70``) and retry. Returns ``(merged_df, report)``
    where ``merged_df`` has only the rows that matched (exactly or by fallback)
    and ``report`` records everything that happened.
    """
    latlong = parse_latlong() if latlong is None else latlong
    coords = latlong.set_index("sector")[["latitude", "longitude"]]
    known = set(coords.index)

    sectors = list(pd.Index(df["sector"].astype(str).str.strip().str.lower()).unique())
    resolved: dict[str, str] = {}       # df sector -> latlong sector
    fallback: dict[str, str] = {}
    unmatched: list[str] = []
    for s in sectors:
        if s in known:
            resolved[s] = s
            continue
        stripped = _SUFFIX_RE.sub(r"\1", s)
        if stripped != s and stripped in known:
            resolved[s] = stripped
            fallback[s] = stripped
        else:
            unmatched.append(s)

    work = df.copy()
    work["sector"] = work["sector"].astype(str).str.strip().str.lower()
    keep = work["sector"].isin(resolved)
    merged = work[keep].copy()
    merged["_ll"] = merged["sector"].map(resolved)
    merged = merged.merge(coords, left_on="_ll", right_index=True, how="left").drop(columns="_ll")

    dropped = work[~keep]
    report = {
        "n_input_rows": int(len(df)),
        "n_joined_rows": int(len(merged)),
        "n_dropped_rows": int(len(dropped)),
        "n_sectors_input": len(sectors),
        "n_exact_matches": sum(1 for s in resolved if s not in fallback),
        "n_fallback_matches": len(fallback),
        "fallback_matches": dict(sorted(fallback.items())),
        "unmatched_sectors": sorted(unmatched),
        "dropped_rows_by_sector": (
            dropped["sector"].value_counts().sort_index().astype(int).to_dict()
        ),
    }
    return merged, report


def aggregate_by_sector(
    merged_df: pd.DataFrame,
    value_cols: tuple[str, ...] = _MAP_VALUE_COLS,
) -> pd.DataFrame:
    """Per-sector means of ``value_cols`` + coordinates + ``n_properties`` count.

    Index is ``sector``; ready to hand to :func:`build_sector_map`.
    """
    value_cols = [c for c in value_cols if c in merged_df.columns]
    grp = merged_df.groupby("sector")
    out = grp[value_cols + ["latitude", "longitude"]].mean()
    out["n_properties"] = grp.size().astype(int)
    return out.round(2)


def sector_map_frame(
    properties_df: pd.DataFrame | None = None,
    latlong: pd.DataFrame | None = None,
    write_csv: bool | str | Path = True,
) -> tuple[pd.DataFrame, dict]:
    """Load listings, join coordinates, aggregate -> ``(group_df, join_report)``.

    ``properties_df`` defaults to
    ``data/processed/gurgaon_properties_missing_value_imputation.csv`` (the only
    processed table carrying ``price_per_sqft`` and ``built_up_area`` alongside
    ``sector``). If ``write_csv`` is truthy, the per-sector frame is written to
    ``data/processed/sector_price_map.csv`` (or the given path) for reuse by the
    app's map section.
    """
    if properties_df is None:
        properties_df = pd.read_csv(DEFAULT_PROPERTIES_CSV)
    merged, report = join_sector_coordinates(properties_df, latlong=latlong)
    group_df = aggregate_by_sector(merged)

    if write_csv:
        path = DEFAULT_SECTOR_FRAME_CSV if write_csv is True else Path(write_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        group_df.to_csv(path)
        report["sector_frame_csv"] = str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)
    return group_df, report


def build_sector_map(
    group_df: pd.DataFrame,
    *,
    color: str = "price_per_sqft",
    size: str = "built_up_area",
    zoom: float = 9.3,
):
    """OpenStreetMap-tile bubble map: one bubble per sector, coloured by
    ``color`` and sized by ``size``.

    A bubble map, **not** a filled choropleth -- there is no public GeoJSON for
    informal Gurgaon sector boundaries. Returns a
    ``plotly.graph_objects.Figure``.

    Known issue -- do not re-investigate in a notebook
    -------------------------------------------------
    ``px.scatter_map`` (and the deprecated ``px.scatter_mapbox``) render **blank**
    -- correct colour legend / title / attribution, but no tiles and no bubbles
    -- specifically inside **Jupyter / JupyterLab inline notebook output**. This
    was pinned down with a side-by-side standalone-HTML test: *both* engines draw
    tiles, roads, labels, bubbles and working hover tooltips perfectly **outside**
    Jupyter (a plain browser). So it is a Jupyter inline-renderer bug with
    MapLibre-based figures, not a bug in ``scatter_map`` or in this function.

    Consequence: ``notebooks/19_sector_price_map.ipynb`` embeds
    :func:`build_sector_map_static` (matplotlib) as its committed output, not this
    figure. Keep using ``scatter_map`` here (current, non-deprecated).

    **TODO when the Streamlit app exists:** confirm this figure renders correctly
    *in Streamlit* directly -- Streamlit embeds Plotly differently from a Jupyter
    cell, so it is very likely fine, but verify it there rather than assuming the
    standalone-HTML pass covers it.
    """
    import plotly.express as px  # lazy: the data functions don't need plotly

    fig = px.scatter_map(
        group_df.reset_index(),
        lat="latitude",
        lon="longitude",
        color=color,
        size=size,
        size_max=28,
        hover_name="sector",
        hover_data={"price": ":.2f", "n_properties": True, "latitude": False, "longitude": False},
        color_continuous_scale="Plasma",
        zoom=zoom,
        height=650,
        map_style="open-street-map",
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=48, b=0),
        title="Gurgaon — mean price per sqft by sector (bubble size = mean built-up area)",
    )
    return fig


def build_sector_map_static(
    group_df: pd.DataFrame,
    *,
    color: str = "price_per_sqft",
    size: str = "built_up_area",
    annotate: int = 6,
):
    """Matplotlib fallback for :func:`build_sector_map`.

    Plots sector centroids as a lon/lat scatter, coloured by ``color`` and sized
    by ``size`` -- the same encoding, minus the street basemap. Needed because
    ``plotly``'s ``scatter_map`` renders blank under ``kaleido`` static export
    (a known plotly-7 / MapLibre issue), so this is what the committed notebook
    embeds. The priciest and cheapest ``annotate`` sectors are labelled.
    Aspect is set equal; at ~28.4°N that slightly compresses longitude, fine for
    a rough spatial read.
    """
    import matplotlib.pyplot as plt

    g = group_df.reset_index()
    span = g[size].max() - g[size].min()
    marker_size = 40 + 260 * ((g[size] - g[size].min()) / span if span else 0.5)

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(
        g["longitude"], g["latitude"], c=g[color], s=marker_size,
        cmap="plasma", alpha=0.85, edgecolor="white", linewidth=0.5,
    )
    labelled = pd.concat([g.nlargest(annotate, color), g.nsmallest(annotate, color)])
    for _, r in labelled.iterrows():
        ax.annotate(r["sector"], (r["longitude"], r["latitude"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    fig.colorbar(sc, ax=ax, label=color)
    ax.set_title("Gurgaon — mean price/sqft by sector (no basemap; bubble size = built-up area)")
    fig.tight_layout()
    return fig
