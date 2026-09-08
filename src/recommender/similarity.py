"""Three similarity matrices for Gurgaon apartment projects, and their blend.

Ported from ``notebooks_original/recommender-system.ipynb``, which built three
cosine-similarity matrices and then left several contradictory weighted sums
in the notebook with no conclusion (``30*sim1 + 20*sim2 + 8*sim3`` in one
cell, ``6*sim1 + 5*sim2 + 3*sim3`` in another). This module resolves that.

The three axes
--------------
``facilities`` (was ``cosine_sim1``)
    TF-IDF (1-2 grams) over the ``TopFacilities`` list joined to a string,
    then cosine. "Do these two projects advertise similar amenities."
``structural`` (was ``cosine_sim2``)
    ``PriceDetails`` JSON parsed into per-BHK ``area low/high``,
    ``price low/high`` and ``building_type``; one-hot + ``fillna(0)`` +
    ``StandardScaler``; then cosine. "Do these two projects offer similar
    unit configurations, sizes and price bands."
``location`` (was ``cosine_sim3``)
    ``LocationAdvantages`` parsed into distance-to-landmark in metres
    (~1070 landmark columns), missing filled with a 54 000 m sentinel,
    ``StandardScaler``; then cosine. "Are these two projects near the same
    named landmarks."

Why the raw matrices must be normalised before blending
------------------------------------------------------
Cosine similarity is bounded, but the three matrices are **not on comparable
scales** (off-diagonal, 246x246, measured on the real data):

    axis         range          mean +/- std     % negative   top-5 nbr band
    facilities   [ 0.00, 0.68]  0.072 +/- 0.082      0 %          0.365
    structural   [-0.77, 1.00]  0.033 +/- 0.341     55 %          0.855
    location     [-0.10, 1.00]  0.023 +/- 0.177     79 %          0.282

``structural`` has ~4x the spread of ``facilities`` and ~2x ``location``, so in
an unweighted sum it dominates the ranking purely by variance. ``facilities``
is non-negative while the other two swing both ways, so a naive sum lets a
strong amenity match be cancelled by a structural mismatch. The original
notebook's ``30/20/8`` *looks* like it prioritises text, but weight x std works
out to structure-first anyway -- the stated weights never matched the
behaviour.

Fix: min-max each matrix onto [0, 1] on its off-diagonal values
(:func:`minmax_offdiag`), so every component is a non-negative "how similar on
this axis" score and the weights are real priorities.

The weights
-----------
:data:`DEFAULT_WEIGHTS` = ``structural 0.50 / location 0.30 / facilities 0.20``.

* **structural 0.50** -- what a buyer is actually shopping for (size, config,
  price band); also the axis with the strongest discriminating power here.
* **location 0.30** -- conceptually the #1 price driver, docked only for data
  quality: the landmark signal is 99.2 % sparse and near-binary in this file.
  Raise it once a denser location match (e.g. sector) is available.
* **facilities 0.20** -- real but secondary, a tie-breaker, and partly
  double-counted with price tier (pricier projects list more amenities).

Full derivation, the sensitivity check, and known limitations:
``reports/recommender/blend_weights.md``.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_APPARTMENTS_CSV = REPO_ROOT / "data" / "raw" / "appartments.csv"

# Missing landmark distance -> "very far" sentinel, in metres (from the notebook).
_MISSING_DISTANCE_M = 54_000

# BHK configurations the structural parser looks for, in the notebook's order.
_BHK_CONFIGS = ["1 BHK", "2 BHK", "3 BHK", "4 BHK", "5 BHK", "6 BHK", "1 RK", "Land"]

DEFAULT_WEIGHTS: dict[str, float] = {
    "structural": 0.50,
    "location": 0.30,
    "facilities": 0.20,
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_appartments(path: str | Path = DEFAULT_APPARTMENTS_CSV) -> pd.DataFrame:
    """Load ``appartments.csv`` and drop the embedded repeated-header row.

    Row 22 of the raw scrape is a second header (``PropertyName == "PropertyName"``);
    the original notebook hard-codes ``.drop(22)``. Here it is removed by content
    so the code survives the row moving.
    """
    df = pd.read_csv(path)
    junk = df.index[df["PropertyName"].astype(str).str.strip() == "PropertyName"]
    df = df.drop(index=junk).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Small, testable parsers
# --------------------------------------------------------------------------- #
def parse_facilities(raw: str) -> list[str]:
    """``"['Swimming Pool', 'Salon']"`` -> ``['Swimming Pool', 'Salon']``."""
    return re.findall(r"'(.*?)'", raw or "")


def distance_to_metres(distance_str: str) -> float | None:
    """``'2.5 KM'`` -> ``2500.0``; ``'800 Meter'`` -> ``800.0``; else ``None``."""
    try:
        s = str(distance_str)
        if "Km" in s or "KM" in s:
            return float(s.split()[0]) * 1000
        if "Meter" in s or "meter" in s:
            return float(s.split()[0])
    except (ValueError, IndexError):
        return None
    return None


def parse_price_details(detail_str: str) -> dict:
    """Parse the ``PriceDetails`` JSON blob into flat per-BHK numeric features.

    Returns keys like ``'area low 3 BHK'``, ``'price high 2 BHK'``,
    ``'building type_2 BHK'``. Unparseable input -> ``{}`` (the notebook's
    behaviour). Prices given in lakh (``' L'``) are converted to crore.
    """
    try:
        details = json.loads(str(detail_str).replace("'", '"'))
    except (json.JSONDecodeError, TypeError):
        return {}

    out: dict = {}
    for bhk, detail in details.items():
        out[f"building type_{bhk}"] = detail.get("building_type")

        area_parts = detail.get("area", "").split("-")
        try:
            lo = float(area_parts[0].replace(",", "").replace(" sq.ft.", "").strip())
            if len(area_parts) == 1:
                out[f"area low {bhk}"], out[f"area high {bhk}"] = lo, lo
            elif len(area_parts) == 2:
                hi = float(area_parts[1].replace(",", "").replace(" sq.ft.", "").strip())
                out[f"area low {bhk}"], out[f"area high {bhk}"] = lo, hi
        except (ValueError, IndexError):
            out[f"area low {bhk}"] = out[f"area high {bhk}"] = None

        price_parts = detail.get("price-range", "").split("-")
        if len(price_parts) == 2:
            try:
                p_lo = float(price_parts[0].replace("₹", "").replace(" Cr", "").replace(" L", "").strip())
                p_hi = float(price_parts[1].replace("₹", "").replace(" Cr", "").replace(" L", "").strip())
                if " L" in price_parts[0]:
                    p_lo /= 100
                if " L" in price_parts[1]:
                    p_hi /= 100
                out[f"price low {bhk}"], out[f"price high {bhk}"] = p_lo, p_hi
            except ValueError:
                out[f"price low {bhk}"] = out[f"price high {bhk}"] = None
    return out


# --------------------------------------------------------------------------- #
# The three similarity matrices
# --------------------------------------------------------------------------- #
def facilities_similarity(df: pd.DataFrame) -> np.ndarray:
    """TF-IDF over ``TopFacilities`` -> cosine similarity (n x n, in [0, 1])."""
    facilities_str = df["TopFacilities"].apply(parse_facilities).apply(" ".join)
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = tfidf.fit_transform(facilities_str)
    return cosine_similarity(matrix)


def structural_similarity(df: pd.DataFrame) -> np.ndarray:
    """Per-BHK area / price / building-type features -> cosine similarity.

    One-hot the categorical ``building type_*`` columns, fill missing numerics
    with 0, standardise every column, then cosine. Matches the notebook.
    """
    rows = []
    for _, row in df.iterrows():
        feats = parse_price_details(row["PriceDetails"])
        new_row = {"PropertyName": row["PropertyName"]}
        for cfg in _BHK_CONFIGS:
            new_row[f"building type_{cfg}"] = feats.get(f"building type_{cfg}")
            new_row[f"area low {cfg}"] = feats.get(f"area low {cfg}")
            new_row[f"area high {cfg}"] = feats.get(f"area high {cfg}")
            new_row[f"price low {cfg}"] = feats.get(f"price low {cfg}")
            new_row[f"price high {cfg}"] = feats.get(f"price high {cfg}")
        rows.append(new_row)

    structural = pd.DataFrame(rows).set_index("PropertyName")
    cat_cols = structural.select_dtypes(include=["object"]).columns.tolist()
    encoded = pd.get_dummies(structural, columns=cat_cols, drop_first=True).fillna(0)
    scaled = StandardScaler().fit_transform(encoded)
    return cosine_similarity(scaled)


def location_similarity(df: pd.DataFrame) -> np.ndarray:
    """Distance-to-landmark vectors -> cosine similarity.

    ~1070 landmark columns, ~99 % missing; missing is filled with a
    "very far" sentinel before standardising, so the axis behaves close to a
    binary "share a micro-location fingerprint or not".
    """
    per_project = {}
    for i, row in df.iterrows():
        distances: dict = {}
        try:
            for landmark, raw in ast.literal_eval(row["LocationAdvantages"]).items():
                distances[landmark] = distance_to_metres(raw)
        except (ValueError, SyntaxError):
            pass  # unparseable row -> no landmarks; becomes all-sentinel
        per_project[i] = distances

    location_df = pd.DataFrame.from_dict(per_project, orient="index").fillna(_MISSING_DISTANCE_M)
    scaled = StandardScaler().fit_transform(location_df)
    return cosine_similarity(scaled)


# --------------------------------------------------------------------------- #
# Normalisation + blend
# --------------------------------------------------------------------------- #
def minmax_offdiag(matrix: np.ndarray) -> np.ndarray:
    """Min-max a similarity matrix onto [0, 1] using its off-diagonal values.

    The diagonal (self-similarity) is excluded from the min/max because it is
    a constant that carries no ranking information and would otherwise pin the
    top of the scale. The returned diagonal is clipped into [0, 1] too.
    """
    n = matrix.shape[0]
    off = matrix[~np.eye(n, dtype=bool)]
    lo, hi = float(off.min()), float(off.max())
    if hi == lo:
        return np.zeros_like(matrix)
    return np.clip((matrix - lo) / (hi - lo), 0.0, 1.0)


def blend(
    components: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Weighted sum of min-max-normalised component matrices.

    ``components`` and ``weights`` must have the same keys. Weights are
    renormalised to sum to 1, so the blended score is itself in [0, 1].
    """
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    if set(components) != set(weights):
        raise ValueError(f"components {sorted(components)} != weights {sorted(weights)}")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("weights must sum to a positive number")

    blended = None
    for name, matrix in components.items():
        term = (weights[name] / total) * minmax_offdiag(matrix)
        blended = term if blended is None else blended + term
    return blended


def build_components(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """All three raw (un-normalised) similarity matrices, keyed by axis name."""
    return {
        "facilities": facilities_similarity(df),
        "structural": structural_similarity(df),
        "location": location_similarity(df),
    }
