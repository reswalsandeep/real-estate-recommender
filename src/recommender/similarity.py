"""Three cosine-similarity matrices for Gurgaon apartment projects, and their blend.

- facilities: TF-IDF (1-2 grams) over the TopFacilities list.
- structural: PriceDetails parsed to per-BHK area / price / building_type,
  one-hot + fillna(0) + StandardScaler.
- location: LocationAdvantages distances to ~1070 named landmarks, missing filled
  with a 54000 m sentinel, StandardScaler.

The three matrices are not on comparable scales (off-diagonal, 246x246, real data):

    axis         range          mean +/- std     % negative   top-5 nbr band
    facilities   [ 0.00, 0.68]  0.072 +/- 0.082      0 %          0.365
    structural   [-0.77, 1.00]  0.033 +/- 0.341     55 %          0.855
    location     [-0.10, 1.00]  0.023 +/- 0.177     79 %          0.282

structural has ~4x the spread of facilities, so an unweighted sum is
structure-dominated by variance alone. minmax_offdiag min-maxes each matrix onto
[0, 1] on its off-diagonal so the weights are real priorities.

DEFAULT_WEIGHTS = structural 0.50 / location 0.30 / facilities 0.20. location is
docked from a higher weight because the landmark signal is 99.2% sparse and
near-binary.
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


_DISTANCE_RE = re.compile(
    r"^\s*([\d.]+)\s*(km|kms|m|meter|meters|mtr|mtrs)\.?\s*$", re.IGNORECASE
)


def distance_to_metres(distance_str: str) -> float | None:
    """Parse a distance string to metres.

    Handles the many formats in ``LocationAdvantages``: ``'2.5 KM'``, ``'2.5km'``,
    ``'1.5 kms'``, ``'800 Meter'``, ``'450 m'``, ``'800 M'`` (case-insensitive,
    space optional). Returns ``None`` for time strings (``'10 mins'``,
    ``'5 minutes drive'``) and vague text (``'Close Proximity'``) -- there is no
    distance to extract without a speed assumption.

    (The earlier version only matched ``'Km'``/``'KM'`` and ``'Meter'``/``'meter'``
    and silently dropped ~20 % of the real distances.)
    """
    m = _DISTANCE_RE.match(str(distance_str))
    if not m:
        return None
    try:
        value = float(m.group(1))
    except ValueError:
        return None
    return value * 1000 if m.group(2).lower().startswith("k") else value


_PRICE_TOKEN_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(cr|lakh|lac|l)?\b", re.IGNORECASE)
_LAKH_UNITS = {"l", "lac", "lakh"}


def price_range_to_crore(text: str) -> list[float]:
    """Every rupee amount in a ``price-range`` string, in crore.

    The unit is read **per number**, not per string, so a range that crosses the
    lakh/crore boundary parses correctly:

    - ``'₹ 99 L - 1.37 Cr'``   -> ``[0.99, 1.37]``   (``L`` binds only to ``99``)
    - ``'₹ 1.35 - 6.67 Cr'``   -> ``[1.35, 6.67]``   (bare ``1.35`` inherits ``Cr``)
    - ``'₹ 26.62 - 44.93 L'``  -> ``[0.2662, 0.4493]``
    - ``'₹ 17 L'``             -> ``[0.17]``
    - ``'Price on Request'``   -> ``[]``

    A bare number with no unit of its own inherits the string's one explicit
    unit (bare numbers only ever appear in same-unit ranges); a string with no
    unit at all is assumed crore.

    (The earlier split-on-``-`` parser mishandled two shapes: a lakh range with
    ``L`` only after the second number left the low bound ~100x too large, and
    single-value strings like ``'₹ 17 L'`` were dropped entirely.)
    """
    matches = _PRICE_TOKEN_RE.findall(text or "")
    if not matches:
        return []
    explicit = next((u.lower() for _n, u in matches if u), None)
    values: list[float] = []
    for number, unit in matches:
        unit = (unit or explicit or "cr").lower()
        value = float(number)
        values.append(value / 100 if unit in _LAKH_UNITS else value)
    return values


def parse_price_details(detail_str: str) -> dict:
    """Parse the ``PriceDetails`` JSON blob into flat per-BHK numeric features.

    Returns keys like ``'area low 3 BHK'``, ``'price high 2 BHK'``,
    ``'building type_2 BHK'``. Unparseable input -> ``{}`` (the notebook's
    behaviour). Price strings are parsed by :func:`price_range_to_crore` (lakh
    values converted to crore); ``price low`` / ``price high`` are the min / max
    of the amounts found, or ``None`` when the string carries no number.
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

        prices = price_range_to_crore(detail.get("price-range", ""))
        if prices:
            out[f"price low {bhk}"], out[f"price high {bhk}"] = min(prices), max(prices)
        else:
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
