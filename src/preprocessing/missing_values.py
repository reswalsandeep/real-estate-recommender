"""
Missing-value imputation for the Gurgaon properties pipeline.

``impute_missing_values()`` is ported from ``missing-value-imputation.ipynb``
(ran and value-checked against the real
``gurgaon_properties_outlier_treated.csv`` -> ``gurgaon_properties_missing_value_imputation.csv``
pair: **18/18 columns, 3554/3554 rows exact match**; the same check runs in
``notebooks/06_missing_value_imputation.ipynb``).

Ported faithfully -- nothing in the output changes. Notebook decisions that are
worth knowing, documented here rather than silently kept or silently "fixed":

- **floorNum flat fill.** The notebook fills every null ``floorNum`` with a
  constant ``2.0``, ignoring ``property_type`` (a house floor is typically
  0-1). Only 17 rows (0.48 %). Kept as the default; exposed as
  ``floornum_fill=`` so a caller can override without editing this module.
- **The single null-`society` row is dropped, not imputed.** The notebook does
  it by hardcoded ``df.drop(index=[2536])``; that is the only row with a null
  ``society``. Ported here as a content-based "drop rows where ``society`` is
  null" so it survives row reordering -- the same move ``clean_flats`` makes
  for its embedded-header row.
- **built_up_area anomaly override.** After the ratio-based fills, rows with
  ``built_up_area < 2000`` and ``price > 2.5`` Cr get
  ``built_up_area = area`` (the raw column, which is dropped immediately
  after). A blunt heuristic: it fixes genuine garbage (300 sqft at 8 Cr) but
  also shifts some borderline rows where ``built_up_area`` looked plausible.
  Kept -- it is load-bearing for the exact match and nets out positive.
- **price_per_sqft is not recomputed** after ``built_up_area`` is imputed /
  overridden, so it goes stale for those rows. The notebook leaves it; so does
  the real output CSV; so do we. Re-deriving it is not this stage's job.
- **Ratio literals.** ``SUPER_TO_BUILTUP_RATIO`` / ``CARPET_TO_BUILTUP_RATIO``
  are the medians of ``super_built_up_area / built_up_area`` and
  ``carpet_area / built_up_area`` over the rows with all three areas present
  (1.10526 and 0.9). The notebook computes them, prints them, then hardcodes
  ``1.105`` / ``0.9``. Kept as literals -- ``round()`` absorbs the
  1.105-vs-1.10526 gap (verified: still an exact match).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# medians over rows with super_built_up_area, built_up_area and carpet_area all
# present; the notebook hardcodes these rounded values.
SUPER_TO_BUILTUP_RATIO = 1.105
CARPET_TO_BUILTUP_RATIO = 0.9

DEFAULT_FLOORNUM_FILL = 2.0

# built_up_area anomaly override (notebook cell 30): a small area on an
# expensive listing is treated as a scrape error, replaced by the raw `area`.
_ANOMALY_MAX_BUILTUP = 2000
_ANOMALY_MIN_PRICE_CR = 2.5

# agePossession: replace the 'Undefined' sentinel with the mode within each of
# these key groups, tried in order (each stage sees the previous stage's fills).
AGE_CASCADE_KEYS: tuple[tuple[str, ...], ...] = (
    ("sector", "property_type"),
    ("sector",),
    ("property_type",),
)
_AGE_SENTINEL = "Undefined"

# dropped once built_up_area is settled (area, areaWithType, super/carpet,
# area_room_ratio at notebook cell 35; facing at cell 46).
DROPPED_COLUMNS = [
    "area", "areaWithType", "super_built_up_area", "carpet_area",
    "area_room_ratio", "facing",
]

OUTPUT_COLUMNS = [
    "property_type", "society", "sector", "price", "price_per_sqft",
    "bedRoom", "bathroom", "balcony", "floorNum", "agePossession",
    "built_up_area", "study room", "servant room", "store room",
    "pooja room", "others", "furnishing_type", "luxury_score",
]


def _impute_built_up_area(df: pd.DataFrame) -> pd.DataFrame:
    """Fill null ``built_up_area`` from ``super_built_up_area`` / ``carpet_area``,
    then apply the small-area / high-price anomaly override.

    Three disjoint cases cover every null-``built_up_area`` row (no row in this
    dataset has all three area columns null):

    * super + carpet present -> mean of the two back-derived estimates
    * super only              -> ``super / SUPER_TO_BUILTUP_RATIO``
    * carpet only             -> ``carpet / CARPET_TO_BUILTUP_RATIO``

    Then ``built_up_area < 2000`` and ``price > 2.5`` Cr -> ``built_up_area = area``.
    """
    df = df.copy()
    s, b, c = df["super_built_up_area"], df["built_up_area"], df["carpet_area"]

    both = s.notna() & b.isna() & c.notna()
    super_only = s.notna() & b.isna() & c.isna()
    carpet_only = s.isna() & b.isna() & c.notna()

    df.loc[both, "built_up_area"] = np.round(
        ((df.loc[both, "super_built_up_area"] / SUPER_TO_BUILTUP_RATIO)
         + (df.loc[both, "carpet_area"] / CARPET_TO_BUILTUP_RATIO)) / 2
    )
    df.loc[super_only, "built_up_area"] = np.round(
        df.loc[super_only, "super_built_up_area"] / SUPER_TO_BUILTUP_RATIO
    )
    df.loc[carpet_only, "built_up_area"] = np.round(
        df.loc[carpet_only, "carpet_area"] / CARPET_TO_BUILTUP_RATIO
    )

    still_null = int(df["built_up_area"].isna().sum())
    if still_null:
        logger.warning(
            "built_up_area: %d row(s) still null after imputation "
            "(super_built_up_area and carpet_area both missing)", still_null
        )

    anomaly = (df["built_up_area"] < _ANOMALY_MAX_BUILTUP) & (df["price"] > _ANOMALY_MIN_PRICE_CR)
    df.loc[anomaly, "built_up_area"] = df.loc[anomaly, "area"]
    logger.info(
        "built_up_area: filled %d null(s), anomaly-overrode %d row(s)",
        int(both.sum() + super_only.sum() + carpet_only.sum()), int(anomaly.sum()),
    )
    return df


def _impute_floor_num(df: pd.DataFrame, fill_value: float = DEFAULT_FLOORNUM_FILL) -> pd.DataFrame:
    """Fill null ``floorNum`` with a constant. The notebook uses ``2.0`` for
    every row regardless of ``property_type`` -- see the module docstring."""
    df = df.copy()
    n = int(df["floorNum"].isna().sum())
    df["floorNum"] = df["floorNum"].fillna(fill_value)
    logger.info("floorNum: filled %d null(s) with %s", n, fill_value)
    return df


def _drop_null_society_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with a null ``society`` (``society`` is not imputed). The
    notebook hardcodes ``df.drop(index=[2536])``; that is the only such row."""
    mask = df["society"].isna()
    if mask.any():
        logger.info(
            "dropping %d row(s) with null society (positions %s)",
            int(mask.sum()), df.index[mask].tolist(),
        )
    return df[~mask].copy()


def _impute_age_possession(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the ``'Undefined'`` sentinel by a cascade of group-mode fills:
    mode within ``(sector, property_type)``, then ``(sector,)``, then
    ``(property_type,)``. Each stage sees the fills made by the previous one;
    a stage's group mode is taken over the un-imputed snapshot, so a group
    that is still mostly ``'Undefined'`` yields ``'Undefined'`` and defers to
    the next stage (this mirrors the notebook's ``df.apply`` exactly)."""
    df = df.copy()
    for keys in AGE_CASCADE_KEYS:
        target = df["agePossession"] == _AGE_SENTINEL
        if not target.any():
            break
        snapshot = df  # frozen for the duration of the apply below

        def group_mode(row: pd.Series, _keys: tuple[str, ...] = keys) -> object:
            sub = snapshot
            for key in _keys:
                sub = sub[sub[key] == row[key]]
            modes = sub["agePossession"].mode()
            return modes.iloc[0] if not modes.empty else np.nan

        df.loc[target, "agePossession"] = df.loc[target].apply(group_mode, axis=1)
        remaining = int((df["agePossession"] == _AGE_SENTINEL).sum())
        logger.info(
            "agePossession: after (%s) mode-fill, %d still '%s'",
            ", ".join(keys), remaining, _AGE_SENTINEL,
        )
    return df


def impute_missing_values(
    df: pd.DataFrame,
    floornum_fill: float = DEFAULT_FLOORNUM_FILL,
) -> pd.DataFrame:
    """Impute the missing values in an outlier-treated properties frame.

    Steps, in notebook order: (1) fill ``built_up_area`` + anomaly override,
    (2) fill ``floorNum``, (3) drop the null-``society`` row, (4) cascade-fill
    ``agePossession``, (5) drop the spent area / ``facing`` columns.

    Parameters
    ----------
    df
        Output of the outlier-treatment stage
        (``gurgaon_properties_outlier_treated.csv``), 24 columns.
    floornum_fill
        Value for null ``floorNum``. Default ``2.0`` reproduces the notebook;
        override to change only that behaviour.

    Returns
    -------
    DataFrame with :data:`OUTPUT_COLUMNS` (18), no missing values, index reset.
    """
    df = _impute_built_up_area(df)
    df = _impute_floor_num(df, floornum_fill)
    df = _drop_null_society_rows(df)
    df = _impute_age_possession(df)
    df = df.drop(columns=[c for c in DROPPED_COLUMNS if c in df.columns])

    missing = [c for c in OUTPUT_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"expected columns absent from input: {missing}")
    out = df[OUTPUT_COLUMNS].reset_index(drop=True)

    remaining_null = out.isnull().sum()
    if remaining_null.any():
        logger.warning("columns still null after imputation:\n%s",
                       remaining_null[remaining_null > 0])
    return out
