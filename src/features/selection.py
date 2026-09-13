"""Feature selection: bin luxury_score / floorNum into categories, drop the
numeric sources plus society / price_per_sqft, ordinal-encode the object columns,
drop pooja room / study room / others (a CV-confirmed choice), move price last.

The 6 encoded columns are cast to int (OrdinalEncoder returns float; the committed
CSV is int64). _categorize_luxury / _categorize_floor have no catch-all - a value
outside the bins becomes None, and select_features(strict=True) raises instead.
low_importance_drop= is effectively pinned by the hardcoded OUTPUT_COLUMNS.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

logger = logging.getLogger(__name__)

# dropped up front (cell 5): an id-ish column and the stale price_per_sqft.
PREDROP_COLUMNS = ["society", "price_per_sqft"]

# binned into *_category, then dropped (cells 9-17).
DERIVED_SOURCE_COLUMNS = ["floorNum", "luxury_score"]

# the notebook's feature-selection decision (cells 44-50): confirmed with a
# single 5-fold CV R2 comparison, not an importance threshold.
DEFAULT_LOW_IMPORTANCE_DROP = ("pooja room", "study room", "others")

# object columns present at encode time (cell 19) -- alphabetical categories.
ENCODED_COLUMNS = [
    "property_type", "sector", "balcony",
    "agePossession", "luxury_category", "floor_category",
]

LUXURY_BINS_DOC = "[0, 50) Low, [50, 150) Medium, [150, 175] High"
FLOOR_BINS_DOC = "[0, 2] Low Floor, [3, 10] Mid Floor, [11, 51] High Floor"

OUTPUT_COLUMNS = [
    "property_type", "sector", "bedRoom", "bathroom", "balcony",
    "agePossession", "built_up_area", "servant room", "store room",
    "furnishing_type", "luxury_category", "floor_category", "price",
]


def _categorize_luxury(score: float) -> str | None:
    """Bin ``luxury_score``; ``None`` outside ``[0, 175]`` or for NaN
    (notebook ``categorize_luxury``)."""
    if 0 <= score < 50:
        return "Low"
    if 50 <= score < 150:
        return "Medium"
    if 150 <= score <= 175:
        return "High"
    return None


def _categorize_floor(floor: float) -> str | None:
    """Bin ``floorNum``; ``None`` outside integer ``[0, 51]`` or for NaN
    (notebook ``categorize_floor``)."""
    if 0 <= floor <= 2:
        return "Low Floor"
    if 3 <= floor <= 10:
        return "Mid Floor"
    if 11 <= floor <= 51:
        return "High Floor"
    return None


def _check_binned(df: pd.DataFrame, category_col: str, source_col: str, bins_doc: str) -> None:
    """Raise if any row binned to ``None`` (strict mode), naming the values."""
    bad_mask = df[category_col].isna()
    if not bad_mask.any():
        return
    offending = sorted({repr(v) for v in df.loc[bad_mask, source_col].tolist()})
    raise ValueError(
        f"{bad_mask.sum()} row(s) have a {source_col} outside the bins "
        f"({bins_doc}); they would become None. Offending value(s): "
        f"{', '.join(offending[:10])}"
        f"{' ...' if len(offending) > 10 else ''}. "
        f"Pass strict=False to reproduce the notebook's silent None."
    )


def _ordinal_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal-encode every object column (fresh per-column OrdinalEncoder,
    alphabetical), cast to int."""
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        codes = OrdinalEncoder().fit_transform(df[[col]])
        df[col] = codes.astype(int)
    return df


def select_features(
    df: pd.DataFrame,
    low_importance_drop: tuple[str, ...] = DEFAULT_LOW_IMPORTANCE_DROP,
    strict: bool = True,
) -> pd.DataFrame:
    """Run feature selection on an imputed properties frame (18 cols in, 13 out).

    drop society / price_per_sqft -> add luxury_category / floor_category from
    their numeric sources -> drop floorNum / luxury_score -> ordinal-encode the
    object columns -> drop low_importance_drop -> move price last.

    strict=True (default) raises if any luxury_score / floorNum falls outside its
    bins; strict=False reproduces the notebook's silent None.
    """
    df = df.drop(columns=[c for c in PREDROP_COLUMNS if c in df.columns])

    df = df.assign(
        luxury_category=df["luxury_score"].apply(_categorize_luxury),
        floor_category=df["floorNum"].apply(_categorize_floor),
    )
    if strict:
        _check_binned(df, "luxury_category", "luxury_score", LUXURY_BINS_DOC)
        _check_binned(df, "floor_category", "floorNum", FLOOR_BINS_DOC)

    df = df.drop(columns=DERIVED_SOURCE_COLUMNS)
    df = _ordinal_encode(df)

    # notebook: X_label = drop('price'); export_df = X_label.drop(low_importance);
    # export_df['price'] = y_label  -- i.e. price is removed then re-appended last.
    price = df["price"]
    df = df.drop(columns=["price", *low_importance_drop])
    df["price"] = price

    missing = [c for c in OUTPUT_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"expected columns absent from input: {missing}")
    return df[OUTPUT_COLUMNS].reset_index(drop=True)
