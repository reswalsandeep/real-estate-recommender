"""
Feature selection for the Gurgaon properties pipeline.

``select_features()`` is ported from ``feature-selection.ipynb`` and
value-checked against ``gurgaon_properties_missing_value_imputation.csv`` ->
``gurgaon_properties_post_feature_selection.csv``: **13/13 columns,
3554/3554 rows, 46,202/46,202 cells exact match**; the same check runs in
``notebooks/11_feature_selection.ipynb``.

Unlike ``outlier-treatment.ipynb`` and ``missing-value-imputation.ipynb``,
this notebook **does** have a ``to_csv`` cell and every output column is
produced by a visible cell. The eight feature-importance techniques in the
notebook (cells 22-49: correlation / RF / GB / permutation / LASSO / RFE /
linear weights / SHAP, plus a CV comparison) are **analysis only** -- they
inform the human choice to drop ``pooja room`` / ``study room`` / ``others``
but touch nothing in the export, so they are not ported here; they belong in
``11_feature_selection.ipynb``.

One trivial gap: the six ordinal-encoded columns are ``int64`` in the
committed CSV, but ``OrdinalEncoder`` returns ``float64`` and no cell casts
them. ``select_features`` applies the obvious ``.astype(int)`` -- noted so it
is not mistaken for the notebook's literal code.

Weak spots, flagged rather than silently kept or fixed:

- **The luxury / floor bins have no catch-all.** ``_categorize_luxury`` covers
  ``[0, 175]``; ``_categorize_floor`` covers ``[0, 51]`` (integers). Anything
  outside -- ``luxury_score`` 176, ``floorNum`` 52, a basement floor of ``-1``
  (which ``cleaning._parse_floor_num`` deliberately preserves), a non-integer
  floor, a NaN -- binned to ``None`` in the notebook, which then becomes its
  own ordinal category or breaks the encoder. This data stays inside both
  ranges, so the notebook never hit it. ``select_features(strict=True)`` (the
  default) raises a ``ValueError`` naming the offending value; pass
  ``strict=False`` to reproduce the notebook's silent ``None``.
- **``OrdinalEncoder`` imposes alphabetical order on nominal columns** --
  ``sector`` code 0 = ``"dwarka expressway"``, ``agePossession`` 0 =
  ``"Moderately Old"`` (not age order). The downstream model then treats these
  as ordered ints. This is the smell the untraced ``_v2`` file addresses;
  ``14_model_selection.ipynb`` re-encodes properly in its ``ColumnTransformer``,
  so the export's encoding is a historical artifact. Reproduced as-is.
- **Encoder fit on the full frame**, before any train/test split (cell 19).
- **``price_per_sqft`` dropped here** (cell 5) is the stale one from the
  missing-values stage; its staleness never propagates past this point.
- **``low_importance_drop`` is accepted but effectively pinned.**
  ``select_features`` reindexes to the hard-coded 13-column
  :data:`OUTPUT_COLUMNS` before returning, so any value other than the default
  ``("pooja room", "study room", "others")`` either no-ops (a shorter list
  still yields the same 13 columns) or raises ``KeyError`` (dropping a column
  that ``OUTPUT_COLUMNS`` still names). The parameter reads as configurable but
  only the default works. Documented-not-fixed: zero functional impact today
  (the export is correct), and not worth touching a validated module for API
  honesty alone -- a documented wart, not a bug.
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
    """Ordinal-encode every object column with a fresh per-column
    ``OrdinalEncoder`` (alphabetical categories), then cast to ``int``
    (the committed CSV is integer-typed; ``OrdinalEncoder`` returns float)."""
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
    """Run the feature-selection stage on an imputed properties frame.

    Steps (notebook order): drop ``society`` / ``price_per_sqft`` -> add
    ``luxury_category`` and ``floor_category`` from their numeric sources ->
    drop ``floorNum`` / ``luxury_score`` -> ordinal-encode the object columns
    -> drop the low-importance columns -> move ``price`` to last.

    Parameters
    ----------
    df
        Output of the missing-value stage
        (``gurgaon_properties_missing_value_imputation.csv``), 18 columns.
    low_importance_drop
        Columns removed by the notebook's CV-confirmed selection decision.
    strict
        ``True`` (default): raise ``ValueError`` if any row's ``luxury_score``
        or ``floorNum`` falls outside its bins (would become ``None``).
        ``False``: reproduce the notebook's silent ``None`` (and let it flow
        into the encoder as its own category).

    Returns
    -------
    DataFrame with :data:`OUTPUT_COLUMNS` (13), index reset.
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
